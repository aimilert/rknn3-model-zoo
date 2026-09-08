// Copyright (c) 2025 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifdef ENABLE_YOLOV8_POSTPROCESS

#include "postprocess.h"

#include "rknn3_api.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#ifdef __RT_THREAD__
#include <rtthread.h>
#else
#include <sys/time.h>
#endif

#define MAX_DETECT_NUM MAX_OBJ_NUM  // 最大检测数量

// 动态数组结构体
typedef struct {
    float *data;
    int size;
    int capacity;
} FloatArray;

typedef struct {
    int *data;
    int size;
    int capacity;
} IntArray;

// 初始化动态数组
static void float_array_init(FloatArray *arr, int capacity) {
    arr->data = (float *)malloc(capacity * sizeof(float));
    arr->size = 0;
    arr->capacity = capacity;
}

static void int_array_init(IntArray *arr, int capacity) {
    arr->data = (int *)malloc(capacity * sizeof(int));
    arr->size = 0;
    arr->capacity = capacity;
}

// 释放动态数组
static void float_array_free(FloatArray *arr) {
    if (arr->data) {
        free(arr->data);
        arr->data = NULL;
    }
    arr->size = 0;
    arr->capacity = 0;
}

static void int_array_free(IntArray *arr) {
    if (arr->data) {
        free(arr->data);
        arr->data = NULL;
    }
    arr->size = 0;
    arr->capacity = 0;
}

// 添加元素到动态数组
static void float_array_push(FloatArray *arr, float value) {
    if (arr->size >= arr->capacity) {
        arr->capacity *= 2;
        arr->data = (float *)realloc(arr->data, arr->capacity * sizeof(float));
    }
    arr->data[arr->size++] = value;
}

static void int_array_push(IntArray *arr, int value) {
    if (arr->size >= arr->capacity) {
        arr->capacity *= 2;
        arr->data = (int *)realloc(arr->data, arr->capacity * sizeof(int));
    }
    arr->data[arr->size++] = value;
}

static inline int clamp(float val, int min, int max) { 
    return val > min ? (val < max ? val : max) : min; 
}

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1)
{
    float w = fmaxf(0.f, fminf(xmax0, xmax1) - fmaxf(xmin0, xmin1) + 1.0);
    float h = fmaxf(0.f, fminf(ymax0, ymax1) - fmaxf(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, float *outputLocations, int *classIds, int *order,
               int filterId, float threshold)
{
    for (int i = 0; i < validCount; ++i)
    {
        int n = order[i];
        if (n == -1 || classIds[n] != filterId)
        {
            continue;
        }
        for (int j = i + 1; j < validCount; ++j)
        {
            int m = order[j];
            if (m == -1 || classIds[m] != filterId)
            {
                continue;
            }
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > threshold)
            {
                order[j] = -1;
            }
        }
    }
    return 0;
}

static int quick_sort_indice_inverse(float *input, int left, int right, int *indices)
{
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right)
    {
        key_index = indices[left];
        key = input[left];
        while (low < high)
        {
            while (low < high && input[high] <= key)
            {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key)
            {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}

static inline float __clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { 
    return ((float)qnt - (float)zp) * scale; 
}

static void compute_dfl(float* tensor, int dfl_len, float* box) {
    float *exp_t = (float *)malloc(dfl_len * sizeof(float));

    if (exp_t == NULL) {
        printf("compute_dfl: out of memory\n");
        return;
    }

    for (int b=0; b<4; b++){
        float exp_sum=0;
        float acc_sum=0;
        for (int i=0; i< dfl_len; i++){
            exp_t[i] = expf(tensor[i+b*dfl_len]);
            exp_sum += exp_t[i];
        }
        
        for (int i=0; i< dfl_len; i++){
            acc_sum += exp_t[i]/exp_sum *i;
        }
        box[b] = acc_sum;
    }

    free(exp_t);
}

static int process_i8(int8_t *box_tensor, int32_t box_zp, float box_scale,
                      int8_t *score_tensor, int32_t score_zp, float score_scale,
                      int8_t *score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
                      int grid_h, int grid_w, int stride, int dfl_len,
                      FloatArray *boxes, 
                      FloatArray *objProbs, 
                      IntArray *classId, 
                      float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8_t score_thres_i8 = qnt_f32_to_affine(threshold, score_zp, score_scale);
    int8_t score_sum_thres_i8 = qnt_f32_to_affine(threshold, score_sum_zp, score_sum_scale);
    float *before_dfl = (float *)malloc(dfl_len * 4 * sizeof(float));
    if (before_dfl == NULL) {
        printf("[process_i8] Error: Failed to allocate before_dfl buffer\n");
        return -1;
    }

    float *exp_t = (float *)malloc(dfl_len * sizeof(float));
    if (exp_t == NULL) {
        printf("[process_i8] Error: Failed to allocate exp_t buffer\n");
        free(before_dfl);
        return -1;
    }

    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; j++)
        {
            int offset = i* grid_w + j;
            int max_class_id = -1;

            // 通过 score sum 起到快速过滤的作用
            if (score_sum_tensor != NULL){
                if (score_sum_tensor[offset] < score_sum_thres_i8){
                    continue;
                }
            }

            int8_t max_score = -score_zp;
            for (int c= 0; c< OBJ_CLASS_NUM; c++){
                if ((score_tensor[offset] > score_thres_i8) && (score_tensor[offset] > max_score))
                {
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;
            }

            // compute box
            if (max_score> score_thres_i8){
                offset = i* grid_w + j;
                float box[4];

                if (dfl_len > 1) // yolov8
                {
                    for (int k = 0; k < dfl_len * 4; k++)
                    {
                        before_dfl[k] = deqnt_affine_to_f32(box_tensor[offset], box_zp, box_scale);
                        offset += grid_len;
                    }
                    // compute_dfl
                    {                   
                        for (int b=0; b<4; b++){
                            float exp_sum=0;
                            float acc_sum=0;
                            for (int k =0; k< dfl_len; k++){
                                exp_t[k] = expf(before_dfl[k+b*dfl_len]);
                                exp_sum += exp_t[k];
                            }
                            
                            for (int k=0; k< dfl_len; k++){
                                acc_sum += exp_t[k]/exp_sum *k;
                            }
                            box[b] = acc_sum;
                        }
                    }
                }
                else // yolov6
                {
                    for (int k = 0; k < 4; k++)
                    {
                        box[k] = deqnt_affine_to_f32(box_tensor[offset], box_zp, box_scale);
                        offset += grid_len;
                    }
                }

                float x1,y1,x2,y2,w,h;
                x1 = (-box[0] + j + 0.5)*stride;
                y1 = (-box[1] + i + 0.5)*stride;
                x2 = (box[2] + j + 0.5)*stride;
                y2 = (box[3] + i + 0.5)*stride;
                w = x2 - x1;
                h = y2 - y1;
                float_array_push(boxes, x1);
                float_array_push(boxes, y1);
                float_array_push(boxes, w);
                float_array_push(boxes, h);

                float_array_push(objProbs, deqnt_affine_to_f32(max_score, score_zp, score_scale));
                int_array_push(classId, max_class_id);
                validCount ++;
            }
        }
    }

    free(before_dfl);
    free(exp_t);


    return validCount;
}

static int process_fp32(float *box_tensor, float *score_tensor, float *score_sum_tensor, 
                        int grid_h, int grid_w, int stride, int dfl_len,
                        FloatArray *boxes, 
                        FloatArray *objProbs, 
                        IntArray *classId, 
                        float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;

    float *before_dfl = (float *)malloc(dfl_len * 4 * sizeof(float));
    if (before_dfl == NULL) {
        printf("[process_fp32] Error: Failed to allocate before_dfl buffer\n");
        return -1;
    }

    float *exp_t = (float *)malloc(dfl_len * sizeof(float));
    if (exp_t == NULL) {
        printf("[process_fp32] Error: Failed to allocate exp_t buffer\n");
        free(before_dfl);
        return -1;
    }

    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; j++)
        {
            int offset = i* grid_w + j;
            int max_class_id = -1;

            // 通过 score sum 起到快速过滤的作用
            if (score_sum_tensor != NULL){
                if (score_sum_tensor[offset] < threshold){
                    continue;
                }
            }

            float max_score = 0;
            for (int c= 0; c< OBJ_CLASS_NUM; c++){
                if ((score_tensor[offset] > threshold) && (score_tensor[offset] > max_score))
                {
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;
            }

            // compute box
            if (max_score> threshold) {
                offset = i* grid_w + j;
                float box[4];

                if (dfl_len > 1) { // yolov8
                    for (int k = 0; k < dfl_len * 4; k++) {
                        before_dfl[k] = box_tensor[offset];
                        offset += grid_len;
                    }
                    // compute_dfl - 内嵌实现，避免重复分配内存
                    {
                        for (int b=0; b<4; b++){
                            float exp_sum=0;
                            float acc_sum=0;
                            for (int k=0; k< dfl_len; k++){
                                exp_t[k] = expf(before_dfl[k+b*dfl_len]);
                                exp_sum += exp_t[k];
                            }
                            
                            for (int k=0; k< dfl_len; k++){
                                acc_sum += exp_t[k]/exp_sum *k;
                            }
                            box[b] = acc_sum;
                        }
                    }
                } else { // yolov6
                    for (int k = 0; k < 4; k++) {
                        box[k] = box_tensor[offset];
                        offset += grid_len;
                    }
                }

                float x1, y1, x2, y2, w, h;
                x1 = (-box[0] + j + 0.5)*stride;
                y1 = (-box[1] + i + 0.5)*stride;
                x2 = (box[2] + j + 0.5)*stride;
                y2 = (box[3] + i + 0.5)*stride;
                w = x2 - x1;
                h = y2 - y1;
                float_array_push(boxes, x1);
                float_array_push(boxes, y1);
                float_array_push(boxes, w);
                float_array_push(boxes, h);

                float_array_push(objProbs, max_score);
                int_array_push(classId, max_class_id);
                validCount ++;
            }
        }
    }

    free(before_dfl);
    free(exp_t);
    return validCount;
}

static int process_fp16(float16 *box_tensor, float16 *score_tensor, float16 *score_sum_tensor, 
                        int grid_h, int grid_w, int stride, int dfl_len,
                        FloatArray *boxes, 
                        FloatArray *objProbs, 
                        IntArray *classId, 
                        float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    
    float *before_dfl = (float *)malloc(dfl_len * 4 * sizeof(float));
    if (before_dfl == NULL) {
        printf("[process_fp16] Error: Failed to allocate before_dfl buffer\n");
        return -1;
    }

    float *exp_t = (float *)malloc(dfl_len * sizeof(float));
    if (exp_t == NULL) {
        printf("[process_fp16] Error: Failed to allocate exp_t buffer\n");
        free(before_dfl);
        return -1;
    }

    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; j++)
        {
            int offset = i * grid_w + j;
            int max_class_id = -1;

            // 通过 score sum 起到快速过滤的作用
            if (score_sum_tensor != NULL) {
                float score_sum_fp32 = fp16_to_fp32(score_sum_tensor[offset]);
                if (score_sum_fp32 < threshold) {
                    continue;
                }
            }

            float max_score = 0;
            for (int c = 0; c < OBJ_CLASS_NUM; c++) {
                float score_fp32 = fp16_to_fp32(score_tensor[offset]);
                if ((score_fp32 > threshold) && (score_fp32 > max_score))
                {
                    max_score = score_fp32;
                    max_class_id = c;
                }
                offset += grid_len;
            }

            // compute box
            if (max_score > threshold) {
                offset = i * grid_w + j;
                float box[4];

                if (dfl_len > 1) { // yolov8
                    // 将 float16 转换为 float32 进行 DFL 计算
                    for (int k = 0; k < dfl_len * 4; k++) {
                        before_dfl[k] = fp16_to_fp32(box_tensor[offset]);
                        offset += grid_len;
                    }
                    // compute_dfl - 内嵌实现，避免重复分配内存
                    {
                        for (int b=0; b<4; b++){
                            float exp_sum=0;
                            float acc_sum=0;
                            for (int k=0; k< dfl_len; k++){
                                exp_t[k] = expf(before_dfl[k+b*dfl_len]);
                                exp_sum += exp_t[k];
                            }
                            
                            for (int k=0; k< dfl_len; k++){
                                acc_sum += exp_t[k]/exp_sum *k;
                            }
                            box[b] = acc_sum;
                        }
                    }
                    
                } else { // yolov6
                    for (int k = 0; k < 4; k++) {
                        box[k] = fp16_to_fp32(box_tensor[offset]);
                        offset += grid_len;
                    }
                }

                float x1, y1, x2, y2, w, h;
                x1 = (-box[0] + j + 0.5) * stride;
                y1 = (-box[1] + i + 0.5) * stride;
                x2 = (box[2] + j + 0.5) * stride;
                y2 = (box[3] + i + 0.5) * stride;
                w = x2 - x1;
                h = y2 - y1;
                
                float_array_push(boxes, x1);
                float_array_push(boxes, y1);
                float_array_push(boxes, w);
                float_array_push(boxes, h);

                float_array_push(objProbs, max_score);
                int_array_push(classId, max_class_id);
                validCount++;
            }
        }
    }

    free(before_dfl);
    free(exp_t);
    return validCount;
}

// 获取唯一的类别ID
static int get_unique_classes(int *classIds, int count, int *unique_classes) {
    int unique_count = 0;
    for (int i = 0; i < count; i++) {
        int cls = classIds[i];
        int found = 0;
        for (int j = 0; j < unique_count; j++) {
            if (unique_classes[j] == cls) {
                found = 1;
                break;
            }
        }
        if (!found) {
            unique_classes[unique_count++] = cls;
        }
    }
    return unique_count;
}

// 简化版本的后处理函数，用于插件接口
// 返回值：检测到的对象数量，失败返回 -1
int post_process(rknn3_yolo_postprocess_context *pp_ctx, const rknn3_tensor inputs[], uint32_t n_inputs, rknn3_tensor outputs[], uint32_t n_outputs)
{
    if (!pp_ctx || !inputs || !outputs) {
        printf("[PostProcess] Error: Invalid parameters\n");
        return -1;
    }
    
    FloatArray filterBoxes;
    FloatArray objProbs;
    IntArray classId;

    float_array_init(&filterBoxes, MAX_DETECT_NUM * 4);
    float_array_init(&objProbs, MAX_DETECT_NUM);
    int_array_init(&classId, MAX_DETECT_NUM);
    
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    
    uint32_t model_in_h = pp_ctx->model_in_h;
    float conf_threshold = pp_ctx->conf_threshold;
    float nms_threshold = pp_ctx->nms_threshold;

    // 处理 3 个检测分支
    // 输入假设: inputs[0..8] 分别是 3 个分支的 box, score, score_sum
    uint32_t output_per_branch = n_inputs / 3;

    for (int i = 0; i < 3; i++)
    {
        uint32_t box_idx = i * output_per_branch;
        uint32_t score_idx = i * output_per_branch + 1;
        int score_sum_idx = (output_per_branch == 3) ? (int)(i * output_per_branch + 2) : -1;
        
        if (box_idx >= n_inputs || score_idx >= n_inputs) {
            continue;
        }

        // 获取张量属性
        rknn3_tensor_attr *box_attr = inputs[box_idx].attr;
        rknn3_tensor_attr *score_attr = inputs[score_idx].attr;
        
        if (!box_attr || !score_attr) {
            printf("[PostProcess] Error: Invalid tensor attributes\n");
            continue;
        }

        // 获取网格尺寸
        grid_h = box_attr->shape[2];
        grid_w = box_attr->shape[3];
        stride = model_in_h / grid_h;
        
        // 计算 dfl_len
        int dfl_len = box_attr->shape[1] / 4;

        // 获取数据指针
        void *box_data = inputs[box_idx].mem ? inputs[box_idx].mem->virt_addr : NULL;
        void *score_data = inputs[score_idx].mem ? inputs[score_idx].mem->virt_addr : NULL;
        void *score_sum_data = NULL;
        int32_t score_sum_zp = 0;
        float score_sum_scale = 1.0;
        
        if (score_sum_idx >= 0 && (uint32_t)score_sum_idx < n_inputs) {
            score_sum_data = inputs[score_sum_idx].mem ? inputs[score_sum_idx].mem->virt_addr : NULL;
            if (inputs[score_sum_idx].attr) {
                score_sum_zp = inputs[score_sum_idx].attr->qnt_info.zero_point;
                score_sum_scale = inputs[score_sum_idx].attr->qnt_info.scale;
            }
        }

        if (!box_data || !score_data) {
            printf("[PostProcess] Error: Invalid tensor data\n");
            continue;
        }

        // 判断是否量化
        int is_quant = (box_attr->dtype == RKNN3_TENSOR_INT8 || box_attr->dtype == RKNN3_TENSOR_UINT8);

        if (is_quant && box_attr->dtype == RKNN3_TENSOR_INT8)
        {
            validCount += process_i8((int8_t *)box_data, box_attr->qnt_info.zero_point, box_attr->qnt_info.scale,
                                     (int8_t *)score_data, score_attr->qnt_info.zero_point, score_attr->qnt_info.scale,
                                     (int8_t *)score_sum_data, score_sum_zp, score_sum_scale,
                                     grid_h, grid_w, stride, dfl_len, 
                                     &filterBoxes, &objProbs, &classId, conf_threshold);
        }
        else if (!is_quant && box_attr->dtype == RKNN3_TENSOR_FLOAT32)
        {
            validCount += process_fp32((float *)box_data, (float *)score_data, (float *)score_sum_data,
                                       grid_h, grid_w, stride, dfl_len, 
                                       &filterBoxes, &objProbs, &classId, conf_threshold);
        } else if (!is_quant && box_attr->dtype == RKNN3_TENSOR_FLOAT16)
        {
            validCount += process_fp16((float16 *)box_data, (float16 *)score_data, (float16 *)score_sum_data,
                                       grid_h, grid_w, stride, dfl_len, 
                                       &filterBoxes, &objProbs, &classId, conf_threshold);
        }
        else
        {
            printf("[PostProcess] Error: Unsupported tensor type! dtype=%d\n", box_attr->dtype);
            continue;
        }
    }

    // 如果没有检测到目标
    if (validCount <= 0)
    {
        // printf("[PostProcess] No objects detected\n");
        
        // 在 output 中写入结果计数为 0
        if (n_outputs > 0 && outputs[0].mem && outputs[0].mem->virt_addr) {
            int *count_ptr = (int *)outputs[0].mem->virt_addr;
            *count_ptr = 0;
        }
        
        float_array_free(&filterBoxes);
        float_array_free(&objProbs);
        int_array_free(&classId);
        
        return 0;  // 返回检测到的对象数量：0
    }

    // NMS 处理
    int *indexArray = (int *)malloc(validCount * sizeof(int));
    for (int i = 0; i < validCount; ++i)
    {
        indexArray[i] = i;
    }
    quick_sort_indice_inverse(objProbs.data, 0, validCount - 1, indexArray);

    // 获取唯一的类别
    int *unique_classes = (int *)malloc(OBJ_CLASS_NUM * sizeof(int));
    int unique_count = get_unique_classes(classId.data, validCount, unique_classes);

    for (int i = 0; i < unique_count; i++)
    {
        nms(validCount, filterBoxes.data, classId.data, indexArray, unique_classes[i], nms_threshold);
    }
    free(unique_classes);

    // 将结果保存到 outputs 中
    // outputs[0] 用于存储检测结果
    // 格式: [box1_x, box1_y, box1_w, box1_h, box1_score, box1_class, box2_x, ...]
    int result_count = 0;  // 声明在外层作用域
    
    if (n_outputs > 0 && outputs[0].mem && outputs[0].mem->virt_addr) {
        float *output_ptr = (float *)outputs[0].mem->virt_addr;

        memset(output_ptr, 0, MAX_OBJ_NUM * 6 * sizeof(float));

        for (int i = 0; i < validCount && result_count < MAX_OBJ_NUM; ++i)
        {
            if (indexArray[i] == -1)
            {
                continue;
            }
            int n = indexArray[i];

            float x = filterBoxes.data[n * 4 + 0];
            float y = filterBoxes.data[n * 4 + 1];
            float w = filterBoxes.data[n * 4 + 2];
            float h = filterBoxes.data[n * 4 + 3];
            int cls_id = classId.data[n];
            float score = objProbs.data[i];

            // 每个检测结果: [score, class_id, x1, y1, x2, y2]
            object_detect_result *result = ((object_detect_result *)outputs[0].mem->virt_addr) + result_count;

            result->box.x1 = x;
            result->box.y1 = y;
            result->box.x2 = x + w;
            result->box.y2 = y + h;
            result->score = score;
            result->class_id = (float)cls_id;
            result_count++;
        }
    }

    free(indexArray);
    float_array_free(&filterBoxes);
    float_array_free(&objProbs);
    int_array_free(&classId);

    return result_count;  // 返回检测到的对象数量
}

#endif // ENABLE_YOLOV8_POSTPROCESS
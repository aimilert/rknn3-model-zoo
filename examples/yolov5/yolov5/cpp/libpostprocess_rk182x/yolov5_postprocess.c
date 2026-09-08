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

#ifdef ENABLE_YOLOV5_POSTPROCESS

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

const int anchor[3][6] = {{10, 13, 16, 30, 33, 23},
                          {30, 61, 62, 45, 59, 119},
                          {116, 90, 156, 198, 373, 326}};

 #define PROP_BOX_SIZE (5 + OBJ_CLASS_NUM)


/* ========== 时间统计辅助函数 ========== */

/**
 * 获取当前时间戳（微秒）
 * @return 当前时间戳（微秒）
 */
static inline uint64_t get_time_us(void)
{
#ifdef __RT_THREAD__
    rt_tick_t tick = rt_tick_get();
    return ((uint64_t)tick * 1000000) / RT_TICK_PER_SECOND;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((uint64_t)tv.tv_sec * 1000000) + tv.tv_usec;
#endif
}

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


// YOLOv5版本的process_i8 (使用anchor boxes)
static int process_i8_v5(int8_t *input, int *anchor, int grid_h, int grid_w, int height, int width, int stride,
                         FloatArray *boxes, FloatArray *objProbs, IntArray *classId, float threshold,
                         int32_t zp, float scale)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8_t thres_i8 = qnt_f32_to_affine(threshold, zp, scale);
    for (int a = 0; a < 3; a++)
    {
        for (int i = 0; i < grid_h; i++)
        {
            for (int j = 0; j < grid_w; j++)
            {
                int8_t box_confidence = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
                if (box_confidence >= thres_i8)
                {
                    int offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
                    int8_t *in_ptr = input + offset;
                    float box_x = (deqnt_affine_to_f32(*in_ptr, zp, scale)) * 2.0 - 0.5;
                    float box_y = (deqnt_affine_to_f32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
                    float box_w = (deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
                    float box_h = (deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale)) * 2.0;
                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = box_w * box_w * (float)anchor[a * 2];
                    box_h = box_h * box_h * (float)anchor[a * 2 + 1];
                    box_x -= (box_w / 2.0);
                    box_y -= (box_h / 2.0);

                    int8_t maxClassProbs = in_ptr[5 * grid_len];
                    int maxClassId = 0;
                    for (int k = 1; k < OBJ_CLASS_NUM; ++k)
                    {
                        int8_t prob = in_ptr[(5 + k) * grid_len];
                        if (prob > maxClassProbs)
                        {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }
                    float limit_score = (deqnt_affine_to_f32(maxClassProbs, zp, scale)) * (deqnt_affine_to_f32(box_confidence, zp, scale));
                    if (limit_score >= threshold)
                    {
                        float_array_push(objProbs, limit_score);
                        int_array_push(classId, maxClassId);
                        validCount++;
                        float_array_push(boxes, box_x);
                        float_array_push(boxes, box_y);
                        float_array_push(boxes, box_w);
                        float_array_push(boxes, box_h);
                    }
                }
            }
        }
    }
    return validCount;
}

// YOLOv5版本的process_i8 (使用anchor boxes)
static int process_fp16_v5(float16 *input, int *anchor, int grid_h, int grid_w, int height, int width, int stride,
                         FloatArray *boxes, FloatArray *objProbs, IntArray *classId, float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;

    for (int a = 0; a < 3; a++)
    {
        for (int i = 0; i < grid_h; i++)
        {
            for (int j = 0; j < grid_w; j++)
            {
                float box_confidence = fp16_to_fp32(input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j]);

                if (box_confidence >= threshold)
                {
                    int offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
                    float16 *in_ptr = input + offset;
                    float box_x = (fp16_to_fp32(*in_ptr)) * 2.0 - 0.5;
                    float box_y = (fp16_to_fp32(in_ptr[grid_len])) * 2.0 - 0.5;
                    float box_w = (fp16_to_fp32(in_ptr[2 * grid_len])) * 2.0;   
                    float box_h = (fp16_to_fp32(in_ptr[3 * grid_len])) * 2.0;   
                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = box_w * box_w * (float)anchor[a * 2];
                    box_h = box_h * box_h * (float)anchor[a * 2 + 1];
                    box_x -= (box_w / 2.0);
                    box_y -= (box_h / 2.0);

                    float maxClassProbs = fp16_to_fp32(in_ptr[5 * grid_len]);
                    int maxClassId = 0;
                    for (int k = 1; k < OBJ_CLASS_NUM; ++k)
                    {
                        float prob = fp16_to_fp32(in_ptr[(5 + k) * grid_len]);
                        if (prob > maxClassProbs)
                        {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }
                    float limit_score = maxClassProbs * box_confidence;
                    if (limit_score >= threshold)
                    {
                        float_array_push(objProbs, limit_score);
                        int_array_push(classId, maxClassId);
                        validCount++;
                        float_array_push(boxes, box_x);
                        float_array_push(boxes, box_y);
                        float_array_push(boxes, box_w);
                        float_array_push(boxes, box_h);
                    }
                }
            }
        }
    }
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


    for (int i = 0; i < 3; i++)
    {
        // 获取网格尺寸
        grid_h = inputs[i].attr->shape[2];
        grid_w = inputs[i].attr->shape[3];
        stride = model_in_h / grid_h;
        
        // 判断是否量化
        int is_quant = (inputs[i].attr->dtype == RKNN3_TENSOR_INT8 || inputs[i].attr->dtype == RKNN3_TENSOR_UINT8);

        if (is_quant && inputs[i].attr->dtype == RKNN3_TENSOR_INT8)
        {
            int prev_count = validCount;
            // 每个stride层使用对应的anchor: stride8->anchor[0], stride16->anchor[1], stride32->anchor[2]
            validCount += process_i8_v5((int8_t *)inputs[i].mem->virt_addr, (int *)anchor[i], grid_h, grid_w, model_in_h, model_in_h, stride, 
                                     &filterBoxes, &objProbs, &classId, conf_threshold, inputs[i].attr->qnt_info.zero_point, inputs[i].attr->qnt_info.scale);
        } else if (!is_quant && inputs[i].attr->dtype == RKNN3_TENSOR_FLOAT16)
        {
            validCount += process_fp16_v5((float16 *)inputs[i].mem->virt_addr, (int *)anchor[i], grid_h, grid_w, model_in_h, model_in_h, stride, 
                                     &filterBoxes, &objProbs, &classId, conf_threshold);
        }
        else
        {
            printf("[PostProcess] Error: Unsupported tensor type! dtype=%d\n", inputs[i].attr->dtype);
            continue;
        }
    }

    // printf("[PostProcess] Total objects before NMS: %d\n", validCount);
    
    // 如果没有检测到目标
    if (validCount <= 0)
    {
        // printf("[PostProcess] No objects detected\n");
        
        // 在 output 中写入结果计数为 0
        memset(outputs[0].mem->virt_addr, 0x00, outputs[0].mem->size);
        float_array_free(&filterBoxes);
        float_array_free(&objProbs);
        int_array_free(&classId);
        
        return 0;  // 返回检测到的对象数量：0
    }

    // NMS 处理
    int *indexArray = (int *)malloc(validCount * sizeof(int));
    // 创建一个临时数组用于排序，保持原始objProbs不变
    float *sortedProbs = (float *)malloc(validCount * sizeof(float));
    memcpy(sortedProbs, objProbs.data, validCount * sizeof(float));
    
    for (int i = 0; i < validCount; ++i)
    {
        indexArray[i] = i;
    }
    // 对sortedProbs排序，同时调整indexArray
    quick_sort_indice_inverse(sortedProbs, 0, validCount - 1, indexArray);

    // // 打印排序后的前几个结果（调试用）
    // printf("[PostProcess] Top 5 scores after sorting: ");
    // for (int i = 0; i < 5 && i < validCount; i++) {
    //     printf("%.3f ", sortedProbs[i]);
    // }
    // printf("\n");

    // 获取唯一的类别
    int *unique_classes = (int *)malloc(OBJ_CLASS_NUM * sizeof(int));
    int unique_count = get_unique_classes(classId.data, validCount, unique_classes);
    
    // printf("[PostProcess] Found %d unique classes, applying NMS with threshold=%.2f\n", unique_count, nms_threshold);

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
            // 使用原始索引n从原始objProbs数组获取分数，确保分数与框/类别匹配
            float score = objProbs.data[n];

            // 每个检测结果: [score, class_id, x1, y1, x2, y2]
            object_detect_result *result = ((object_detect_result *)outputs[0].mem->virt_addr) + result_count;

            result->score = score;
            result->class_id = (float)cls_id;
            result->box.x1 = x;
            result->box.y1 = y;
            result->box.x2 = x + w;
            result->box.y2 = y + h;

            // printf("[PostProcess] Result[%d]: box=(%.1f, %.1f, %.1f, %.1f), score=%.3f, class=%d\n", 
            //        result_count, x, y, w, h, score, cls_id);
            result_count++;
        }
    }

    free(sortedProbs);
    free(indexArray);
    float_array_free(&filterBoxes);
    float_array_free(&objProbs);
    int_array_free(&classId);

    // printf("[PostProcess] step 4 time cost = %lu us\n", get_time_us() - start_us);

    return result_count;  // 返回检测到的对象数量
}

#endif // ENABLE_YOLOV5_POSTPROCESS
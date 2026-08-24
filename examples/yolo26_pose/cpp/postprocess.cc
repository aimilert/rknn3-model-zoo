// Copyright (c) 2026 by Rockchip Electronics Co., Ltd. All Rights Reserved.
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

#include "yolo26.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <algorithm>
#include <float.h>
#include <set>
#include <vector>
#define LABEL_NAME_TXT_PATH "./model/coco_80_labels_list.txt"

static char *labels[OBJ_CLASS_NUM];
static char null_label[] = "null";

inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

static char *readLine(FILE *fp, char *buffer, int *len)
{
    int ch;
    int i = 0;
    size_t buff_len = 0;

    buffer = (char *)malloc(buff_len + 1);
    if (!buffer)
        return NULL; // Out of memory

    while ((ch = fgetc(fp)) != '\n' && ch != EOF)
    {
        buff_len++;
        void *tmp = realloc(buffer, buff_len + 1);
        if (tmp == NULL)
        {
            free(buffer);
            return NULL; // Out of memory
        }
        buffer = (char *)tmp;

        buffer[i] = (char)ch;
        i++;
    }
    buffer[i] = '\0';

    *len = buff_len;

    // Detect end
    if (ch == EOF && (i == 0 || ferror(fp)))
    {
        free(buffer);
        return NULL;
    }
    return buffer;
}

static int readLines(const char *fileName, char *lines[], int max_line)
{
    FILE *file = fopen(fileName, "r");
    char *s;
    int i = 0;
    int n = 0;

    if (file == NULL)
    {
        printf("Open %s fail!\n", fileName);
        return -1;
    }

    while ((s = readLine(file, s, &n)) != NULL)
    {
        lines[i++] = s;
        if (i >= max_line)
            break;
    }
    fclose(file);
    return i;
}



static int loadLabelName(const char *locationFilename, char *label[])
{
    printf("load label %s\n", locationFilename);
    readLines(locationFilename, label, OBJ_CLASS_NUM);
    return 0;
}

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1)
{
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, std::vector<float> &outputLocations, const std::vector<int> &classIds, std::vector<int> &order,
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

static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices)
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

inline static int32_t __clip(float val, float min, float max)
{
    int32_t f = (int32_t)(val <= min ? min : (val >= max ? max : val));
    return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }
static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

static void compute_dfl(float* tensor, int dfl_len, float* box){
    for (int b=0; b<4; b++){
        float exp_t[dfl_len];
        float max_v = tensor[b * dfl_len];
        float exp_sum=0;
        float acc_sum=0;
        for (int i=1; i< dfl_len; i++){
            max_v = fmax(max_v, tensor[i+b*dfl_len]);
        }
        for (int i=0; i< dfl_len; i++){
            exp_t[i] = expf(tensor[i+b*dfl_len] - max_v);
            exp_sum += exp_t[i];
        }

        for (int i=0; i< dfl_len; i++){
            acc_sum += exp_t[i]/exp_sum *i;
        }
        box[b] = acc_sum;
    }
}

static int tensor_channel(const rknn3_tensor_attr *attr)
{
    if (attr->n_dims != 4)
    {
        return 0;
    }
    return attr->layout == RKNN3_TENSOR_NHWC ? attr->shape[3] : attr->shape[1];
}

static int tensor_grid_h(const rknn3_tensor_attr *attr)
{
    if (attr->n_dims != 4)
    {
        return 0;
    }
    return attr->layout == RKNN3_TENSOR_NHWC ? attr->shape[1] : attr->shape[2];
}

static int tensor_grid_w(const rknn3_tensor_attr *attr)
{
    if (attr->n_dims != 4)
    {
        return 0;
    }
    return attr->layout == RKNN3_TENSOR_NHWC ? attr->shape[2] : attr->shape[3];
}

static bool infer_tensor_chw(const rknn3_tensor_attr *attr, int *channel, int *grid_h, int *grid_w,
                             int *channel_dim, int *h_dim, int *w_dim)
{
    if (attr->n_dims != 4)
    {
        return false;
    }

    int c_dim = attr->layout == RKNN3_TENSOR_NHWC ? 3 : 1;
    int gh_dim = attr->layout == RKNN3_TENSOR_NHWC ? 1 : 2;
    int gw_dim = attr->layout == RKNN3_TENSOR_NHWC ? 2 : 3;

    for (int d = 1; d < 4; d++)
    {
        if (attr->shape[d] == OBJ_KEYPOINT_NUM * 3)
        {
            c_dim = d;
            break;
        }
    }

    if (attr->shape[c_dim] != OBJ_KEYPOINT_NUM * 3 &&
        attr->shape[c_dim] != OBJ_CLASS_NUM &&
        !(attr->shape[c_dim] >= 4 && attr->shape[c_dim] % 4 == 0))
    {
        for (int d = 1; d < 4; d++)
        {
            if (attr->shape[d] == 4)
            {
                c_dim = d;
                break;
            }
        }
    }

    if (c_dim != (attr->layout == RKNN3_TENSOR_NHWC ? 3 : 1))
    {
        int spatial_dims[2] = {-1, -1};
        int spatial_count = 0;
        for (int d = 1; d < 4; d++)
        {
            if (d != c_dim && spatial_count < 2)
            {
                spatial_dims[spatial_count++] = d;
            }
        }
        if (spatial_count != 2)
        {
            return false;
        }
        gh_dim = spatial_dims[0];
        gw_dim = spatial_dims[1];
    }

    if (channel)
    {
        *channel = attr->shape[c_dim];
    }
    if (grid_h)
    {
        *grid_h = attr->shape[gh_dim];
    }
    if (grid_w)
    {
        *grid_w = attr->shape[gw_dim];
    }
    if (channel_dim)
    {
        *channel_dim = c_dim;
    }
    if (h_dim)
    {
        *h_dim = gh_dim;
    }
    if (w_dim)
    {
        *w_dim = gw_dim;
    }
    return true;
}

static int tensor_channel_infer(const rknn3_tensor_attr *attr)
{
    int channel = 0;
    infer_tensor_chw(attr, &channel, nullptr, nullptr, nullptr, nullptr, nullptr);
    return channel;
}

static int tensor_grid_h_infer(const rknn3_tensor_attr *attr)
{
    int grid_h = 0;
    infer_tensor_chw(attr, nullptr, &grid_h, nullptr, nullptr, nullptr, nullptr);
    return grid_h;
}

static int tensor_grid_w_infer(const rknn3_tensor_attr *attr)
{
    int grid_w = 0;
    infer_tensor_chw(attr, nullptr, nullptr, &grid_w, nullptr, nullptr, nullptr);
    return grid_w;
}

static int tensor_offset(const rknn3_tensor_attr *attr, int c, int y, int x)
{
    int h = 0;
    int w = 0;
    int channel = 0;
    int c_dim = 1;
    int h_dim = 2;
    int w_dim = 3;
    infer_tensor_chw(attr, &channel, &h, &w, &c_dim, &h_dim, &w_dim);
    if (attr->n_stride >= attr->n_dims && attr->n_dims == 4)
    {
        return c * attr->stride[c_dim] + y * attr->stride[h_dim] + x * attr->stride[w_dim];
    }

    if (attr->layout == RKNN3_TENSOR_NHWC)
    {
        return (y * w + x) * channel + c;
    }
    return c * h * w + y * w + x;
}

static int attr_elem_count(const rknn3_tensor_attr *attr)
{
    int size = 1;
    for (uint32_t i = 0; i < attr->n_dims; i++)
    {
        size *= attr->shape[i];
    }
    return size;
}

static bool score_name_has_sigmoid(const rknn3_tensor_attr *attr)
{
    return attr->name != nullptr && strstr(attr->name, "sigmoid") != nullptr;
}

static bool score_tensor_is_logits_i8(int8_t *score_tensor, const rknn3_tensor_attr *score_attr,
                                      int32_t score_zp, float score_scale)
{
    if (score_name_has_sigmoid(score_attr))
    {
        return false;
    }

    int size = attr_elem_count(score_attr);
    for (int i = 0; i < size; i++)
    {
        float score = deqnt_affine_to_f32(score_tensor[i], score_zp, score_scale);
        if (score < 0.0f || score > 1.0f)
        {
            return true;
        }
    }
    return false;
}

static bool score_tensor_is_logits_fp32(float *score_tensor, const rknn3_tensor_attr *score_attr)
{
    if (score_name_has_sigmoid(score_attr))
    {
        return false;
    }

    int size = attr_elem_count(score_attr);
    for (int i = 0; i < size; i++)
    {
        float score = score_tensor[i];
        if (score < 0.0f || score > 1.0f)
        {
            return true;
        }
    }
    return false;
}

static void decode_keypoints_from_raw(const float raw_kpts[OBJ_KEYPOINT_NUM][3],
                                      int grid_y, int grid_x, int stride,
                                      std::vector<float> &keypoints)
{
    for (int k = 0; k < OBJ_KEYPOINT_NUM; k++)
    {
        float x = (raw_kpts[k][0] + grid_x + 0.5f) * stride;
        float y = (raw_kpts[k][1] + grid_y + 0.5f) * stride;
        float conf = sigmoid(raw_kpts[k][2]);

        keypoints.push_back(x);
        keypoints.push_back(y);
        keypoints.push_back(conf);
    }
}

static int process_i8(int8_t *box_tensor, const rknn3_tensor_attr *box_attr, int32_t box_zp, float box_scale,
                      int8_t *score_tensor, const rknn3_tensor_attr *score_attr, int32_t score_zp, float score_scale,
                      int8_t *pose_tensor, const rknn3_tensor_attr *pose_attr, int32_t pose_zp, float pose_scale,
                      int grid_h, int grid_w, int stride, int dfl_len,
                      std::vector<float> &boxes,
                      std::vector<float> &keypoints,
                      std::vector<float> &objProbs,
                      std::vector<int> &classId,
                      float threshold,
                      bool score_is_logits)
{
    int validCount = 0;
    int score_channel = tensor_channel_infer(score_attr);
    int class_count = std::min(score_channel, OBJ_CLASS_NUM);
    float score_cmp_threshold = threshold;
    if (score_is_logits)
    {
        score_cmp_threshold = logf(threshold / (1.0f - threshold));
    }
    int8_t score_thres_i8 = qnt_f32_to_affine(score_cmp_threshold, score_zp, score_scale);

    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; j++)
        {
            int max_class_id = -1;
            int8_t max_score_i8 = -128;

            if (score_channel == 1)
            {
                int score_offset = tensor_offset(score_attr, 0, i, j);
                int8_t score_i8 = score_tensor[score_offset];
                if (score_i8 >= score_thres_i8)
                {
                    max_score_i8 = score_i8;
                    max_class_id = 0;
                }
            }
            else
            {
                for (int c = 0; c < class_count; c++)
                {
                    int score_offset = tensor_offset(score_attr, c, i, j);
                    int8_t score_i8 = score_tensor[score_offset];
                    if ((score_i8 >= score_thres_i8) && (score_i8 > max_score_i8))
                    {
                        max_score_i8 = score_i8;
                        max_class_id = c;
                    }
                }
            }

            // compute box
            if (max_class_id >= 0){
                float max_score = deqnt_affine_to_f32(max_score_i8, score_zp, score_scale);
                if (score_is_logits)
                {
                    max_score = sigmoid(max_score);
                }
                if (max_score <= threshold)
                {
                    continue;
                }

                float box[4];
                if (dfl_len <= 1) {
                    for (int k = 0; k < 4; k++) {
                        int box_offset = tensor_offset(box_attr, k, i, j);
                        box[k] = deqnt_affine_to_f32(box_tensor[box_offset], box_zp, box_scale);
                    }
                } else {
                    float dfl_tensor[dfl_len * 4];
                    for (int k = 0; k < dfl_len * 4; k++) {
                        int box_offset = tensor_offset(box_attr, k, i, j);
                        dfl_tensor[k] = deqnt_affine_to_f32(box_tensor[box_offset], box_zp, box_scale);
                    }
                    compute_dfl(dfl_tensor, dfl_len, box);
                }


                float x1,y1,x2,y2,w,h;
                x1 = (-box[0] + j + 0.5)*stride;
                y1 = (-box[1] + i + 0.5)*stride;
                x2 = (box[2] + j + 0.5)*stride;
                y2 = (box[3] + i + 0.5)*stride;
                w = x2 - x1;
                h = y2 - y1;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);

                objProbs.push_back(max_score);
                classId.push_back(max_class_id);

                float raw_kpts[OBJ_KEYPOINT_NUM][3];
                for (int k = 0; k < OBJ_KEYPOINT_NUM; k++)
                {
                    raw_kpts[k][0] = deqnt_affine_to_f32(pose_tensor[tensor_offset(pose_attr, k * 3 + 0, i, j)], pose_zp, pose_scale);
                    raw_kpts[k][1] = deqnt_affine_to_f32(pose_tensor[tensor_offset(pose_attr, k * 3 + 1, i, j)], pose_zp, pose_scale);
                    raw_kpts[k][2] = deqnt_affine_to_f32(pose_tensor[tensor_offset(pose_attr, k * 3 + 2, i, j)], pose_zp, pose_scale);
                }
                decode_keypoints_from_raw(raw_kpts, i, j, stride, keypoints);
                validCount ++;
            }

        }
    }
    return validCount;
}

static int process_fp32(float *box_tensor, const rknn3_tensor_attr *box_attr,
                        float *score_tensor, const rknn3_tensor_attr *score_attr,
                        float *pose_tensor, const rknn3_tensor_attr *pose_attr,
                        int grid_h, int grid_w, int stride, int dfl_len,
                        std::vector<float> &boxes,
                        std::vector<float> &keypoints,
                        std::vector<float> &objProbs,
                        std::vector<int> &classId,
                        float threshold,
                        bool score_is_logits)
{
    int validCount = 0;
    int score_channel = tensor_channel_infer(score_attr);
    int class_count = std::min(score_channel, OBJ_CLASS_NUM);
    float score_cmp_threshold = threshold;
    if (score_is_logits)
    {
        score_cmp_threshold = logf(threshold / (1.0f - threshold));
    }

    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; j++)
        {
            int max_class_id = -1;
            float max_score = -FLT_MAX;

            if (score_channel == 1)
            {
                int score_offset = tensor_offset(score_attr, 0, i, j);
                float score = score_tensor[score_offset];
                if (score > score_cmp_threshold)
                {
                    max_score = score;
                    max_class_id = 0;
                }
            }
            else
            {
                for (int c = 0; c < class_count; c++)
                {
                    int score_offset = tensor_offset(score_attr, c, i, j);
                    float score = score_tensor[score_offset];
                    if ((score > score_cmp_threshold) && (score > max_score))
                    {
                        max_score = score;
                        max_class_id = c;
                    }
                }
            }

            // compute box
            if (max_class_id >= 0){
                if (score_is_logits)
                {
                    max_score = sigmoid(max_score);
                }
                if (max_score <= threshold)
                {
                    continue;
                }

                float box[4];
                if (dfl_len <= 1) {
                    for (int k=0; k<4; k++){
                        int box_offset = tensor_offset(box_attr, k, i, j);
                        box[k] = box_tensor[box_offset];
                    }
                } else {
                    float dfl_tensor[dfl_len * 4];
                    for (int k=0; k<dfl_len * 4; k++){
                        int box_offset = tensor_offset(box_attr, k, i, j);
                        dfl_tensor[k] = box_tensor[box_offset];
                    }
                    compute_dfl(dfl_tensor, dfl_len, box);
                }

                float x1,y1,x2,y2,w,h;
                x1 = (-box[0] + j + 0.5)*stride;
                y1 = (-box[1] + i + 0.5)*stride;
                x2 = (box[2] + j + 0.5)*stride;
                y2 = (box[3] + i + 0.5)*stride;
                w = x2 - x1;
                h = y2 - y1;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);

                objProbs.push_back(max_score);
                classId.push_back(max_class_id);

                float raw_kpts[OBJ_KEYPOINT_NUM][3];
                for (int k = 0; k < OBJ_KEYPOINT_NUM; k++)
                {
                    raw_kpts[k][0] = pose_tensor[tensor_offset(pose_attr, k * 3 + 0, i, j)];
                    raw_kpts[k][1] = pose_tensor[tensor_offset(pose_attr, k * 3 + 1, i, j)];
                    raw_kpts[k][2] = pose_tensor[tensor_offset(pose_attr, k * 3 + 2, i, j)];
                }
                decode_keypoints_from_raw(raw_kpts, i, j, stride, keypoints);
                validCount ++;
            }
        }
    }
    return validCount;
}

static int convert_fp16_to_fp32(const float16 *src, float *dst, int n_elems)
{
  for (int i = 0; i < n_elems; i++)
  {
    dst[i] = fp16_to_fp32(src[i]);
  }
  return 0;
}

int post_process_after_exYoloPostProcess(rknn_app_context_t *app_ctx, void *outputs, rknn3_tensor_attr output_attrs, letterbox_t *letter_box, object_detect_result_list *od_results)
{
    rknn3_tensor *_outputs = (rknn3_tensor *)outputs;
    bool channels_first = output_attrs.shape[1] >= 6 && output_attrs.shape[2] > output_attrs.shape[1];
    int K = channels_first ? output_attrs.shape[2] : output_attrs.shape[1];
    int det_len = channels_first ? output_attrs.shape[1] : output_attrs.shape[2];
    int size = K * det_len;
    int model_in_w = app_ctx->model_width;
    int model_in_h = app_ctx->model_height;

    rknn3_tensor_type dtype = output_attrs.dtype;
    std::vector<float> outputfp32(size);

    if(dtype == RKNN3_TENSOR_FLOAT16){
        float16 *output_tensor = (float16 *)_outputs[0].mem->virt_addr;
        convert_fp16_to_fp32(output_tensor, outputfp32.data(), size);
    }
    else if(dtype == RKNN3_TENSOR_FLOAT32){
        memcpy(outputfp32.data(), _outputs[0].mem->virt_addr, size*sizeof(float));
    }
    else if(dtype == RKNN3_TENSOR_INT8){
        int8_t *output_tensor = (int8_t *)_outputs[0].mem->virt_addr;
        int32_t zp = output_attrs.qnt_info.zero_point;
        float scale = output_attrs.qnt_info.scale;
        for (int i = 0; i < size; i++)
        {
            outputfp32[i] = deqnt_affine_to_f32(output_tensor[i], zp, scale);
        }
    }
    else{
        return -1;
    }

    auto get_det_value = [&](int det_idx, int channel) -> float {
        if (channels_first)
        {
            return outputfp32[channel * K + det_idx];
        }
        return outputfp32[det_idx * det_len + channel];
    };

    int last_count = 0;
    for(int i = 0; i < K; i++)
    {
        // Support both common layouts:
        //  - [score, cls, x1, y1, x2, y2] (used by some RKNN postprocess plugins)
        //  - [x1, y1, x2, y2, score, cls, ...keypoints] (Ultralytics end2end ONNX `output0`)
        float score = 0.0f;
        int id = -1;
        float x1 = 0.0f, y1 = 0.0f, x2 = 0.0f, y2 = 0.0f;

        float v0 = get_det_value(i, 0);
        float v1 = get_det_value(i, 1);
        float v2 = get_det_value(i, 2);
        float v3 = get_det_value(i, 3);
        float v4 = get_det_value(i, 4);
        float v5 = get_det_value(i, 5);

        bool layout_score_first = (v0 >= 0.0f && v0 <= 1.0f) && (v1 >= -1.0f && v1 <= (float)(OBJ_CLASS_NUM + 1));
        bool layout_score_last  = (v4 >= 0.0f && v4 <= 1.0f) && (v5 >= -1.0f && v5 <= (float)(OBJ_CLASS_NUM + 1));

        if (layout_score_first && !layout_score_last)
        {
            score = v0;
            id = (int)v1;
            x1 = v2;
            y1 = v3;
            x2 = v4;
            y2 = v5;
        }
        else
        {
            // Default to "score last" to avoid dropping all detections when x1<0.
            x1 = v0;
            y1 = v1;
            x2 = v2;
            y2 = v3;
            score = v4;
            id = (int)v5;
        }

        if(score > 0 && id >= 0 && id < OBJ_CLASS_NUM)
        {
            if (last_count >= OBJ_NUMB_MAX_SIZE)
            {
                break;
            }

            x1 -= letter_box->x_pad;
            y1 -= letter_box->y_pad;
            x2 -= letter_box->x_pad;
            y2 -= letter_box->y_pad;

            float obj_conf = score;
            float mapped_x1 = clamp(x1, 0, model_in_w) / letter_box->scale;
            float mapped_y1 = clamp(y1, 0, model_in_h) / letter_box->scale;
            float mapped_x2 = clamp(x2, 0, model_in_w) / letter_box->scale;
            float mapped_y2 = clamp(y2, 0, model_in_h) / letter_box->scale;

            od_results->results[last_count].box.left = (int)mapped_x1;
            od_results->results[last_count].box.top = (int)mapped_y1;
            od_results->results[last_count].box.right = (int)mapped_x2;
            od_results->results[last_count].box.bottom = (int)mapped_y2;
            od_results->results[last_count].box_float[0] = mapped_x1;
            od_results->results[last_count].box_float[1] = mapped_y1;
            od_results->results[last_count].box_float[2] = mapped_x2;
            od_results->results[last_count].box_float[3] = mapped_y2;
            od_results->results[last_count].prop = obj_conf;
            od_results->results[last_count].cls_id = id;
            od_results->results[last_count].keypoint_count = 0;

            int kpt_base = 6;
            if (det_len >= kpt_base + OBJ_KEYPOINT_NUM * 3)
            {
                od_results->results[last_count].keypoint_count = OBJ_KEYPOINT_NUM;
                for (int k = 0; k < OBJ_KEYPOINT_NUM; k++)
                {
                    float kx = get_det_value(i, kpt_base + k * 3 + 0) - letter_box->x_pad;
                    float ky = get_det_value(i, kpt_base + k * 3 + 1) - letter_box->y_pad;
                    float kp = get_det_value(i, kpt_base + k * 3 + 2);
                    float mapped_kx = clamp(kx, 0, model_in_w) / letter_box->scale;
                    float mapped_ky = clamp(ky, 0, model_in_h) / letter_box->scale;
                    od_results->results[last_count].keypoints[k].x = (int)mapped_kx;
                    od_results->results[last_count].keypoints[k].y = (int)mapped_ky;
                    od_results->results[last_count].keypoints[k].x_float = mapped_kx;
                    od_results->results[last_count].keypoints[k].y_float = mapped_ky;
                    od_results->results[last_count].keypoints[k].prop = kp;
                }
            }
            last_count++;
        }
    }
    od_results->count = last_count;
    return 0;
}

static int getTensorSize(rknn3_tensor tensor)
{
    int size = 0;
    for (uint32_t i = 0; i < tensor.attr->n_dims; i++)
    {
        if (i == 0)
            size = tensor.attr->shape[i];
        else
            size *= tensor.attr->shape[i];
    }
    return size;
}

int post_process(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, object_detect_result_list *od_results)
{
    rknn3_tensor *_outputs = (rknn3_tensor *)outputs;

    typedef struct {
        int box_idx;
        int score_idx;
        int pose_idx;
        int grid_h;
        int grid_w;
    } yolo_branch_t;

    std::vector<float> filterBoxes;
    std::vector<float> filterPoses;
    std::vector<float> objProbs;
    std::vector<int> classId;
    filterBoxes.reserve(1024 * 4);
    filterPoses.reserve(1024 * OBJ_KEYPOINT_NUM * 3);
    objProbs.reserve(1024);
    classId.reserve(1024);
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int model_in_w = app_ctx->model_width;
    int model_in_h = app_ctx->model_height;

    memset(od_results, 0, sizeof(object_detect_result_list));

    std::vector<yolo_branch_t> branches;
    int output_count = (int)app_ctx->io_num.n_output;
    for (int idx = 0; idx < output_count; idx++)
    {
        const rknn3_tensor_attr *attr = app_ctx->outputs[idx].attr;
        if (attr->n_dims != 4)
        {
            continue;
        }

        int channel = tensor_channel_infer(attr);
        int out_grid_h = tensor_grid_h_infer(attr);
        int out_grid_w = tensor_grid_w_infer(attr);
        int branch_id = -1;
        for (int i = 0; i < (int)branches.size(); i++)
        {
            if (branches[i].grid_h == out_grid_h && branches[i].grid_w == out_grid_w)
            {
                branch_id = i;
                break;
            }
        }
        if (branch_id < 0)
        {
            yolo_branch_t branch;
            branch.box_idx = -1;
            branch.score_idx = -1;
            branch.pose_idx = -1;
            branch.grid_h = out_grid_h;
            branch.grid_w = out_grid_w;
            branches.push_back(branch);
            branch_id = (int)branches.size() - 1;
        }

        if (channel == OBJ_CLASS_NUM || channel == 1)
        {
            branches[branch_id].score_idx = idx;
        }
        else if (channel == OBJ_KEYPOINT_NUM * 3)
        {
            branches[branch_id].pose_idx = idx;
        }
        else if (channel >= 4 && channel % 4 == 0)
        {
            branches[branch_id].box_idx = idx;
        }
    }

    for (int i = 0; i < (int)branches.size();)
    {
        if (branches[i].box_idx < 0 || branches[i].score_idx < 0 || branches[i].pose_idx < 0)
        {
            branches.erase(branches.begin() + i);
            continue;
        }
        i++;
    }

    std::sort(branches.begin(), branches.end(), [](const yolo_branch_t &a, const yolo_branch_t &b) {
        return a.grid_h > b.grid_h;
    });

    if (branches.empty())
    {
        printf("Unsupported YOLO26 pose output layout, output count=%d\n", app_ctx->io_num.n_output);
        return -1;
    }

    for (int i = 0; i < (int)branches.size(); i++)
    {
        int box_idx = branches[i].box_idx;
        int score_idx = branches[i].score_idx;
        int pose_idx = branches[i].pose_idx;

        if (box_idx < 0 || score_idx < 0 || pose_idx < 0)
        {
            printf("Unsupported YOLO26 branch %d output layout\n", i);
            return -1;
        }

        int dfl_len = tensor_channel_infer(app_ctx->outputs[box_idx].attr) / 4;

        grid_h = tensor_grid_h_infer(app_ctx->outputs[box_idx].attr);
        grid_w = tensor_grid_w_infer(app_ctx->outputs[box_idx].attr);

        stride = model_in_h / grid_h;

        if (app_ctx->is_quant)
        {
            bool score_is_logits = score_tensor_is_logits_i8((int8_t *)_outputs[score_idx].mem->virt_addr,
                                                             app_ctx->outputs[score_idx].attr,
                                                             app_ctx->outputs[score_idx].attr->qnt_info.zero_point,
                                                             app_ctx->outputs[score_idx].attr->qnt_info.scale);
            // write_tensor_out(&_outputs[box_idx], file_path[i]);
            validCount += process_i8((int8_t *)_outputs[box_idx].mem->virt_addr, app_ctx->outputs[box_idx].attr, app_ctx->outputs[box_idx].attr->qnt_info.zero_point, app_ctx->outputs[box_idx].attr->qnt_info.scale,
                                     (int8_t *)_outputs[score_idx].mem->virt_addr, app_ctx->outputs[score_idx].attr, app_ctx->outputs[score_idx].attr->qnt_info.zero_point, app_ctx->outputs[score_idx].attr->qnt_info.scale,
                                     (int8_t *)_outputs[pose_idx].mem->virt_addr, app_ctx->outputs[pose_idx].attr, app_ctx->outputs[pose_idx].attr->qnt_info.zero_point, app_ctx->outputs[pose_idx].attr->qnt_info.scale,
                                     grid_h, grid_w, stride, dfl_len,
                                     filterBoxes, filterPoses, objProbs, classId, conf_threshold, score_is_logits);
        }
        else
        {
            float *box_tensor = nullptr;
            float *score_tensor = nullptr;
            float *pose_tensor = nullptr;
            bool box_tensor_allocated = false;
            bool score_tensor_allocated = false;
            bool pose_tensor_allocated = false;

            //convert box from fp16 to fp32
            if(_outputs[box_idx].attr->dtype == RKNN3_TENSOR_FLOAT16){
                int size = getTensorSize(_outputs[box_idx]);
                box_tensor = (float*) malloc(size * sizeof(float));
                convert_fp16_to_fp32((float16*)_outputs[box_idx].mem->virt_addr, box_tensor, size);
                box_tensor_allocated = true;
            }
            else {
                box_tensor = (float *)_outputs[box_idx].mem->virt_addr;
            }

            //convert score from fp16 to fp32
            if(_outputs[score_idx].attr->dtype == RKNN3_TENSOR_FLOAT16){
                int size = getTensorSize(_outputs[score_idx]);
                score_tensor = (float*) malloc(size * sizeof(float));
                convert_fp16_to_fp32((float16*)_outputs[score_idx].mem->virt_addr, score_tensor, size);
                score_tensor_allocated = true;
            }
            else {
                score_tensor = (float *)_outputs[score_idx].mem->virt_addr;
            }

            //convert pose from fp16 to fp32
            if(_outputs[pose_idx].attr->dtype == RKNN3_TENSOR_FLOAT16){
                int size = getTensorSize(_outputs[pose_idx]);
                pose_tensor = (float*) malloc(size * sizeof(float));
                convert_fp16_to_fp32((float16*)_outputs[pose_idx].mem->virt_addr, pose_tensor, size);
                pose_tensor_allocated = true;
            }
            else {
                pose_tensor = (float *)_outputs[pose_idx].mem->virt_addr;
            }

            bool score_is_logits = score_tensor_is_logits_fp32(score_tensor, app_ctx->outputs[score_idx].attr);
            validCount += process_fp32(box_tensor, app_ctx->outputs[box_idx].attr,
                                       score_tensor, app_ctx->outputs[score_idx].attr,
                                       pose_tensor, app_ctx->outputs[pose_idx].attr,
                                       grid_h, grid_w, stride, dfl_len,
                                       filterBoxes, filterPoses, objProbs, classId, conf_threshold, score_is_logits);
            if(box_tensor_allocated)
            {
                free(box_tensor);
            }
            if(score_tensor_allocated)
            {
                free(score_tensor);
            }
            if(pose_tensor_allocated)
            {
                free(pose_tensor);
            }
        }
    }

    // no object detect

    if (validCount <= 0)
    {
        return 0;
    }
    std::vector<int> indexArray(validCount);
    for (int i = 0; i < validCount; ++i)
    {
        indexArray[i] = i;
    }
    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

    bool class_appeared[OBJ_CLASS_NUM] = {false};
    for (int i = 0; i < validCount; ++i)
    {
        if (classId[i] >= 0 && classId[i] < OBJ_CLASS_NUM)
        {
            class_appeared[classId[i]] = true;
        }
    }

    for (int c = 0; c < OBJ_CLASS_NUM; c++)
    {
        if (class_appeared[c])
        {
            nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
        }
    }

    int last_count = 0;
    od_results->count = 0;

    /* box valid detect target */
    for (int i = 0; i < validCount; ++i)
    {
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE)
        {
            continue;
        }
        int n = indexArray[i];

        float x1 = filterBoxes[n * 4 + 0] - letter_box->x_pad;
        float y1 = filterBoxes[n * 4 + 1] - letter_box->y_pad;
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n];
        float obj_conf = objProbs[i];
        float mapped_x1 = clamp(x1, 0, model_in_w) / letter_box->scale;
        float mapped_y1 = clamp(y1, 0, model_in_h) / letter_box->scale;
        float mapped_x2 = clamp(x2, 0, model_in_w) / letter_box->scale;
        float mapped_y2 = clamp(y2, 0, model_in_h) / letter_box->scale;

        od_results->results[last_count].box.left = (int)mapped_x1;
        od_results->results[last_count].box.top = (int)mapped_y1;
        od_results->results[last_count].box.right = (int)mapped_x2;
        od_results->results[last_count].box.bottom = (int)mapped_y2;
        od_results->results[last_count].box_float[0] = mapped_x1;
        od_results->results[last_count].box_float[1] = mapped_y1;
        od_results->results[last_count].box_float[2] = mapped_x2;
        od_results->results[last_count].box_float[3] = mapped_y2;
        od_results->results[last_count].prop = obj_conf;
        od_results->results[last_count].cls_id = id;
        od_results->results[last_count].keypoint_count = OBJ_KEYPOINT_NUM;
        for (int k = 0; k < OBJ_KEYPOINT_NUM; k++)
        {
            float kx = filterPoses[n * OBJ_KEYPOINT_NUM * 3 + k * 3 + 0] - letter_box->x_pad;
            float ky = filterPoses[n * OBJ_KEYPOINT_NUM * 3 + k * 3 + 1] - letter_box->y_pad;
            float kp = filterPoses[n * OBJ_KEYPOINT_NUM * 3 + k * 3 + 2];
            float mapped_kx = clamp(kx, 0, model_in_w) / letter_box->scale;
            float mapped_ky = clamp(ky, 0, model_in_h) / letter_box->scale;
            od_results->results[last_count].keypoints[k].x = (int)mapped_kx;
            od_results->results[last_count].keypoints[k].y = (int)mapped_ky;
            od_results->results[last_count].keypoints[k].x_float = mapped_kx;
            od_results->results[last_count].keypoints[k].y_float = mapped_ky;
            od_results->results[last_count].keypoints[k].prop = kp;
        }
        last_count++;
    }
    od_results->count = last_count;
    return 0;
}

int init_post_process()
{
    int ret = 0;
    ret = loadLabelName(LABEL_NAME_TXT_PATH, labels);
    if (ret < 0)
    {
        printf("Load %s failed!\n", LABEL_NAME_TXT_PATH);
        return -1;
    }
    return 0;
}

const char *coco_cls_to_name(int cls_id)
{

    if (cls_id >= OBJ_CLASS_NUM)
    {
        return null_label;
    }

    if (labels[cls_id])
    {
        return labels[cls_id];
    }

    return null_label;
}

void deinit_post_process()
{
    for (int i = 0; i < OBJ_CLASS_NUM; i++)
    {
        if (labels[i] != nullptr)
        {
            free(labels[i]);
            labels[i] = nullptr;
        }
    }
}

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

static int tensor_offset(const rknn3_tensor_attr *attr, int c, int y, int x)
{
    int h = tensor_grid_h(attr);
    int w = tensor_grid_w(attr);
    int channel = tensor_channel(attr);
    if (attr->n_stride >= attr->n_dims && attr->n_dims == 4)
    {
        int c_dim = attr->layout == RKNN3_TENSOR_NHWC ? 3 : 1;
        int h_dim = attr->layout == RKNN3_TENSOR_NHWC ? 1 : 2;
        int w_dim = attr->layout == RKNN3_TENSOR_NHWC ? 2 : 3;
        return c * attr->stride[c_dim] + y * attr->stride[h_dim] + x * attr->stride[w_dim];
    }

    if (attr->layout == RKNN3_TENSOR_NHWC)
    {
        return (y * w + x) * channel + c;
    }
    return c * h * w + y * w + x;
}

static void tensor_chw_strides(const rknn3_tensor_attr *attr, int *c_stride, int *h_stride, int *w_stride)
{
    int h = tensor_grid_h(attr);
    int w = tensor_grid_w(attr);
    int channel = tensor_channel(attr);
    if (attr->n_stride >= attr->n_dims && attr->n_dims == 4)
    {
        int c_dim = attr->layout == RKNN3_TENSOR_NHWC ? 3 : 1;
        int h_dim = attr->layout == RKNN3_TENSOR_NHWC ? 1 : 2;
        int w_dim = attr->layout == RKNN3_TENSOR_NHWC ? 2 : 3;
        *c_stride = attr->stride[c_dim];
        *h_stride = attr->stride[h_dim];
        *w_stride = attr->stride[w_dim];
        return;
    }

    if (attr->layout == RKNN3_TENSOR_NHWC)
    {
        *c_stride = 1;
        *h_stride = w * channel;
        *w_stride = channel;
        return;
    }

    *c_stride = h * w;
    *h_stride = w;
    *w_stride = 1;
}

static float tensor_value_to_f32(const rknn3_tensor *tensor, int offset)
{
    const rknn3_tensor_attr *attr = tensor->attr;
    if (attr->dtype == RKNN3_TENSOR_FLOAT16)
    {
        return fp16_to_fp32(((float16 *)tensor->mem->virt_addr)[offset]);
    }
    if (attr->dtype == RKNN3_TENSOR_FLOAT32)
    {
        return ((float *)tensor->mem->virt_addr)[offset];
    }
    if (attr->dtype == RKNN3_TENSOR_INT8)
    {
        return deqnt_affine_to_f32(((int8_t *)tensor->mem->virt_addr)[offset],
                                   attr->qnt_info.zero_point, attr->qnt_info.scale);
    }
    return 0.0f;
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

static bool score_tensor_qrange_is_probability(const rknn3_tensor_attr *attr, int32_t zp, float scale)
{
    if (attr->dtype != RKNN3_TENSOR_INT8 || scale <= 0.0f)
    {
        return false;
    }

    float qmin = deqnt_affine_to_f32((int8_t)-128, zp, scale);
    float qmax = deqnt_affine_to_f32((int8_t)127, zp, scale);
    return qmin >= -0.01f && qmax <= 1.01f;
}

static bool score_tensor_is_logits_i8(int8_t *score_tensor, const rknn3_tensor_attr *score_attr,
                                      int32_t score_zp, float score_scale)
{
    if (score_name_has_sigmoid(score_attr))
    {
        return false;
    }
    if (score_tensor_qrange_is_probability(score_attr, score_zp, score_scale))
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




static int convert_proto_tensor_to_f32(const rknn3_tensor *proto_tensor,
                                       std::vector<float> &proto_data,
                                       int *proto_h,
                                       int *proto_w)
{
    int proto_c = tensor_channel(proto_tensor->attr);
    *proto_h = tensor_grid_h(proto_tensor->attr);
    *proto_w = tensor_grid_w(proto_tensor->attr);
    if (proto_c != OBJ_MASK_DIM || *proto_h <= 0 || *proto_w <= 0)
    {
        return -1;
    }

    proto_data.resize(OBJ_MASK_DIM * (*proto_h) * (*proto_w));
    int c_stride = 0;
    int h_stride = 0;
    int w_stride = 0;
    tensor_chw_strides(proto_tensor->attr, &c_stride, &h_stride, &w_stride);

    const rknn3_tensor_attr *attr = proto_tensor->attr;
    for (int c = 0; c < OBJ_MASK_DIM; c++)
    {
        for (int y = 0; y < *proto_h; y++)
        {
            for (int x = 0; x < *proto_w; x++)
            {
                int proto_offset = c * c_stride + y * h_stride + x * w_stride;
                int dst_offset = c * (*proto_h) * (*proto_w) + y * (*proto_w) + x;
                if (attr->dtype == RKNN3_TENSOR_INT8)
                {
                    proto_data[dst_offset] =
                        deqnt_affine_to_f32(((int8_t *)proto_tensor->mem->virt_addr)[proto_offset],
                                            attr->qnt_info.zero_point, attr->qnt_info.scale);
                }
                else if (attr->dtype == RKNN3_TENSOR_FLOAT16)
                {
                    proto_data[dst_offset] = fp16_to_fp32(((float16 *)proto_tensor->mem->virt_addr)[proto_offset]);
                }
                else if (attr->dtype == RKNN3_TENSOR_FLOAT32)
                {
                    proto_data[dst_offset] = ((float *)proto_tensor->mem->virt_addr)[proto_offset];
                }
                else
                {
                    return -1;
                }
            }
        }
    }
    return 0;
}

static int build_instance_mask(const float *proto_data,
                               int proto_h,
                               int proto_w,
                               const float *coeff,
                               int src_w,
                               int src_h,
                               const int *x_to_px,
                               const int *y_to_py,
                               std::vector<unsigned char> &proto_mask,
                               object_detect_result *result)
{
    if (proto_data == nullptr || proto_h <= 0 || proto_w <= 0 ||
        src_w <= 0 || src_h <= 0 || x_to_px == nullptr || y_to_py == nullptr)
    {
        return -1;
    }

    int left = clamp((float)result->box.left, 0, src_w);
    int top = clamp((float)result->box.top, 0, src_h);
    int right = clamp((float)result->box.right, 0, src_w);
    int bottom = clamp((float)result->box.bottom, 0, src_h);
    if (left >= right || top >= bottom)
    {
        return 0;
    }

    int mask_w = right - left;
    int mask_h = bottom - top;
    result->mask.width = mask_w;
    result->mask.height = mask_h;
    result->mask.data = (unsigned char *)malloc((size_t)mask_w * mask_h);
    if (result->mask.data == nullptr)
    {
        return -1;
    }

    int proto_left = x_to_px[left];
    int proto_right = x_to_px[right - 1];
    int proto_top = y_to_py[top];
    int proto_bottom = y_to_py[bottom - 1];
    if (proto_left > proto_right)
    {
        std::swap(proto_left, proto_right);
    }
    if (proto_top > proto_bottom)
    {
        std::swap(proto_top, proto_bottom);
    }

    float mask_thresh_logit = logf(SEG_MASK_THRESH / (1.0f - SEG_MASK_THRESH));
    int proto_plane = proto_h * proto_w;
    proto_mask.resize(proto_plane);
    for (int py = proto_top; py <= proto_bottom; py++)
    {
        for (int px = proto_left; px <= proto_right; px++)
        {
            int proto_idx = py * proto_w + px;
            float mask_logit = 0.0f;
            for (int c = 0; c < OBJ_MASK_DIM; c++)
            {
                mask_logit += coeff[c] * proto_data[c * proto_plane + proto_idx];
            }
            proto_mask[proto_idx] = mask_logit > mask_thresh_logit ? 255 : 0;
        }
    }

    for (int y = top; y < bottom; y++)
    {
        int py = y_to_py[y];
        const unsigned char *proto_row = proto_mask.data() + py * proto_w;
        unsigned char *mask_row = result->mask.data + (y - top) * mask_w;
        for (int x = left; x < right; x++)
        {
            int px = x_to_px[x];
            mask_row[x - left] = proto_row[px];
        }
    }

    return 0;
}

static int build_instance_mask_i8(const rknn3_tensor *proto_tensor,
                                  const int8_t *coeff,
                                  int32_t coeff_zp,
                                  float coeff_scale,
                                  int src_w,
                                  int src_h,
                                  const int *x_to_px,
                                  const int *y_to_py,
                                  std::vector<unsigned char> &proto_mask,
                                  object_detect_result *result)
{
    int proto_c = tensor_channel(proto_tensor->attr);
    int proto_h = tensor_grid_h(proto_tensor->attr);
    int proto_w = tensor_grid_w(proto_tensor->attr);
    if (proto_tensor->attr->dtype != RKNN3_TENSOR_INT8 || proto_c != OBJ_MASK_DIM ||
        proto_h <= 0 || proto_w <= 0 || src_w <= 0 || src_h <= 0 ||
        x_to_px == nullptr || y_to_py == nullptr)
    {
        return -1;
    }
    float dot_scale = coeff_scale * proto_tensor->attr->qnt_info.scale;
    if (dot_scale <= 0.0f)
    {
        return -1;
    }

    int left = clamp((float)result->box.left, 0, src_w);
    int top = clamp((float)result->box.top, 0, src_h);
    int right = clamp((float)result->box.right, 0, src_w);
    int bottom = clamp((float)result->box.bottom, 0, src_h);
    if (left >= right || top >= bottom)
    {
        return 0;
    }

    int mask_w = right - left;
    int mask_h = bottom - top;
    result->mask.width = mask_w;
    result->mask.height = mask_h;
    result->mask.data = (unsigned char *)malloc((size_t)mask_w * mask_h);
    if (result->mask.data == nullptr)
    {
        return -1;
    }

    int proto_left = x_to_px[left];
    int proto_right = x_to_px[right - 1];
    int proto_top = y_to_py[top];
    int proto_bottom = y_to_py[bottom - 1];
    if (proto_left > proto_right)
    {
        std::swap(proto_left, proto_right);
    }
    if (proto_top > proto_bottom)
    {
        std::swap(proto_top, proto_bottom);
    }

    int proto_c_stride = 0;
    int proto_h_stride = 0;
    int proto_w_stride = 0;
    tensor_chw_strides(proto_tensor->attr, &proto_c_stride, &proto_h_stride, &proto_w_stride);

    const int8_t *proto_data = (const int8_t *)proto_tensor->mem->virt_addr;
    int32_t proto_zp = proto_tensor->attr->qnt_info.zero_point;
    float mask_thresh_logit = logf(SEG_MASK_THRESH / (1.0f - SEG_MASK_THRESH));
    int32_t acc_threshold = (int32_t)floorf(mask_thresh_logit / dot_scale);

    int32_t coeff_delta[OBJ_MASK_DIM];
    int proto_c_offsets[OBJ_MASK_DIM];
    for (int c = 0; c < OBJ_MASK_DIM; c++)
    {
        coeff_delta[c] = (int32_t)coeff[c] - coeff_zp;
        proto_c_offsets[c] = c * proto_c_stride;
    }

    proto_mask.resize((size_t)proto_h * proto_w);
    for (int py = proto_top; py <= proto_bottom; py++)
    {
        int proto_y_base = py * proto_h_stride;
        for (int px = proto_left; px <= proto_right; px++)
        {
            int proto_base = proto_y_base + px * proto_w_stride;
            int32_t acc = 0;
            for (int c = 0; c < OBJ_MASK_DIM; c++)
            {
                int proto_offset = proto_c_offsets[c] + proto_base;
                acc += coeff_delta[c] * ((int32_t)proto_data[proto_offset] - proto_zp);
            }
            proto_mask[py * proto_w + px] = acc > acc_threshold ? 255 : 0;
        }
    }

    for (int y = top; y < bottom; y++)
    {
        int py = y_to_py[y];
        const unsigned char *proto_row = proto_mask.data() + py * proto_w;
        unsigned char *mask_row = result->mask.data + (y - top) * mask_w;
        for (int x = left; x < right; x++)
        {
            int px = x_to_px[x];
            mask_row[x - left] = proto_row[px];
        }
    }

    return 0;
}

static int process_i8(int8_t *box_tensor, const rknn3_tensor_attr *box_attr, int32_t box_zp, float box_scale,
                      int8_t *score_tensor, const rknn3_tensor_attr *score_attr, int32_t score_zp, float score_scale,
                      int8_t *mask_tensor, const rknn3_tensor_attr *mask_attr, int32_t mask_zp, float mask_scale,
                      int grid_h, int grid_w, int stride, int dfl_len,
                      std::vector<float> &boxes,
                      std::vector<float> &objProbs,
                      std::vector<int> &classId,
                      std::vector<int8_t> &masks_i8,
                      std::vector<int32_t> &maskZps,
                      std::vector<float> &maskScales,
                      float threshold,
                      bool score_is_logits)
{
    int validCount = 0;
    int class_count = std::min(tensor_channel(score_attr), OBJ_CLASS_NUM);
    int score_c_stride = 0;
    int score_h_stride = 0;
    int score_w_stride = 0;
    int box_c_stride = 0;
    int box_h_stride = 0;
    int box_w_stride = 0;
    int mask_c_stride = 0;
    int mask_h_stride = 0;
    int mask_w_stride = 0;
    tensor_chw_strides(score_attr, &score_c_stride, &score_h_stride, &score_w_stride);
    tensor_chw_strides(box_attr, &box_c_stride, &box_h_stride, &box_w_stride);
    tensor_chw_strides(mask_attr, &mask_c_stride, &mask_h_stride, &mask_w_stride);
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
            int score_base = i * score_h_stride + j * score_w_stride;
            for (int c= 0; c< class_count; c++){
                int score_offset = score_base + c * score_c_stride;
                int8_t score_i8 = score_tensor[score_offset];
                if((score_i8 >= score_thres_i8) && (score_i8 > max_score_i8))
                {
                    max_score_i8 = score_i8;
                    max_class_id = c;
                }
            }

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
                int box_base = i * box_h_stride + j * box_w_stride;
                if (dfl_len <= 1) {
                    for (int k = 0; k < 4; k++) {
                        int box_offset = box_base + k * box_c_stride;
                        box[k] = deqnt_affine_to_f32(box_tensor[box_offset], box_zp, box_scale);
                    }
                } else {
                    float dfl_tensor[dfl_len * 4];
                    for (int k = 0; k < dfl_len * 4; k++) {
                        int box_offset = box_base + k * box_c_stride;
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

                int mask_base = i * mask_h_stride + j * mask_w_stride;
                for (int c = 0; c < OBJ_MASK_DIM; c++)
                {
                    int mask_offset = mask_base + c * mask_c_stride;
                    masks_i8.push_back(mask_tensor[mask_offset]);
                }
                maskZps.push_back(mask_zp);
                maskScales.push_back(mask_scale);
                validCount ++;
            }
        }
    }
    return validCount;
}

static int process_fp32(float *box_tensor, const rknn3_tensor_attr *box_attr,
                        float *score_tensor, const rknn3_tensor_attr *score_attr,
                        float *mask_tensor, const rknn3_tensor_attr *mask_attr,
                        int grid_h, int grid_w, int stride, int dfl_len,
                        std::vector<float> &boxes,
                        std::vector<float> &objProbs,
                        std::vector<int> &classId,
                        std::vector<float> &masks,
                        float threshold,
                        bool score_is_logits)
{
    int validCount = 0;
    int class_count = std::min(tensor_channel(score_attr), OBJ_CLASS_NUM);
    int score_c_stride = 0;
    int score_h_stride = 0;
    int score_w_stride = 0;
    int box_c_stride = 0;
    int box_h_stride = 0;
    int box_w_stride = 0;
    int mask_c_stride = 0;
    int mask_h_stride = 0;
    int mask_w_stride = 0;
    tensor_chw_strides(score_attr, &score_c_stride, &score_h_stride, &score_w_stride);
    tensor_chw_strides(box_attr, &box_c_stride, &box_h_stride, &box_w_stride);
    tensor_chw_strides(mask_attr, &mask_c_stride, &mask_h_stride, &mask_w_stride);
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
            int score_base = i * score_h_stride + j * score_w_stride;
            for (int c= 0; c< class_count; c++){
                int score_offset = score_base + c * score_c_stride;
                float score = score_tensor[score_offset];
                if((score > score_cmp_threshold) && (score > max_score))
                {
                    max_score = score;
                    max_class_id = c;
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
                int box_base = i * box_h_stride + j * box_w_stride;
                if (dfl_len <= 1) {
                    for (int k=0; k<4; k++){
                        int box_offset = box_base + k * box_c_stride;
                        box[k] = box_tensor[box_offset];
                    }
                } else {
                    float dfl_tensor[dfl_len * 4];
                    for (int k=0; k<dfl_len * 4; k++){
                        int box_offset = box_base + k * box_c_stride;
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

                int mask_base = i * mask_h_stride + j * mask_w_stride;
                for (int c = 0; c < OBJ_MASK_DIM; c++)
                {
                    int mask_offset = mask_base + c * mask_c_stride;
                    masks.push_back(mask_tensor[mask_offset]);
                }
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

    release_object_detect_result_list(od_results);
    memset(od_results, 0, sizeof(object_detect_result_list));

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

    release_object_detect_result_list(od_results);
    memset(od_results, 0, sizeof(object_detect_result_list));


    static thread_local std::vector<float> filterBoxes;
    static thread_local std::vector<float> objProbs;
    static thread_local std::vector<int> classId;
    static thread_local std::vector<float> masks;
    static thread_local std::vector<int8_t> masks_i8;
    static thread_local std::vector<int32_t> maskZps;
    static thread_local std::vector<float> maskScales;
    static thread_local std::vector<float> protoData;
    static thread_local std::vector<unsigned char> protoMaskScratch;
    static thread_local std::vector<int> indexArray;
    static thread_local std::vector<int> xToProto;
    static thread_local std::vector<int> yToProto;
    filterBoxes.clear();
    objProbs.clear();
    classId.clear();
    masks.clear();
    masks_i8.clear();
    maskZps.clear();
    maskScales.clear();
    protoData.clear();
    indexArray.clear();
    xToProto.clear();
    yToProto.clear();
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int model_in_w = app_ctx->model_width;
    int model_in_h = app_ctx->model_height;
    int max_candidates = std::max(1024,
                                  (model_in_w / 8) * (model_in_h / 8) +
                                  (model_in_w / 16) * (model_in_h / 16) +
                                  (model_in_w / 32) * (model_in_h / 32));
    filterBoxes.reserve((size_t)max_candidates * 4);
    objProbs.reserve(max_candidates);
    classId.reserve(max_candidates);
    masks.reserve((size_t)max_candidates * OBJ_MASK_DIM);
    masks_i8.reserve((size_t)max_candidates * OBJ_MASK_DIM);
    maskZps.reserve(max_candidates);
    maskScales.reserve(max_candidates);

    if (app_ctx->io_num.n_output != 10)
    {
        printf("Unsupported YOLO26 segment output count: %d, expected 10\n", app_ctx->io_num.n_output);
        return -1;
    }
    int output_per_branch = 3;
    for (int i = 0; i < 3; i++)
    {
        int box_idx = -1;
        int score_idx = -1;
        int mask_ids = -1;
        int base_idx = i * output_per_branch;
        for (int j = 0; j < output_per_branch; j++)
        {
            int idx = base_idx + j;
            int channel = tensor_channel(app_ctx->outputs[idx].attr);
            if (channel == OBJ_CLASS_NUM)
            {
                score_idx = idx;
            }
            else if (channel == 32)
            {
                mask_ids = idx;
            }
            else if (channel >= 4 && channel % 4 == 0)
            {
                box_idx = idx;
            }
        }

        if (box_idx < 0 || score_idx < 0 || mask_ids < 0)
        {
            printf("Unsupported YOLO26 branch %d output layout\n", i);
            return -1;
        }

        int dfl_len = tensor_channel(app_ctx->outputs[box_idx].attr) / 4;

        grid_h = tensor_grid_h(app_ctx->outputs[box_idx].attr);
        grid_w = tensor_grid_w(app_ctx->outputs[box_idx].attr);

        stride = model_in_h / grid_h;

        if (app_ctx->is_quant)
        {

            bool score_is_logits = score_tensor_is_logits_i8((int8_t *)_outputs[score_idx].mem->virt_addr,
                                                             app_ctx->outputs[score_idx].attr,
                                                             app_ctx->outputs[score_idx].attr->qnt_info.zero_point,
                                                             app_ctx->outputs[score_idx].attr->qnt_info.scale);
            validCount += process_i8((int8_t *)_outputs[box_idx].mem->virt_addr, app_ctx->outputs[box_idx].attr, app_ctx->outputs[box_idx].attr->qnt_info.zero_point, app_ctx->outputs[box_idx].attr->qnt_info.scale,
                                     (int8_t *)_outputs[score_idx].mem->virt_addr, app_ctx->outputs[score_idx].attr, app_ctx->outputs[score_idx].attr->qnt_info.zero_point, app_ctx->outputs[score_idx].attr->qnt_info.scale,
                                     (int8_t *)_outputs[mask_ids].mem->virt_addr, app_ctx->outputs[mask_ids].attr, app_ctx->outputs[mask_ids].attr->qnt_info.zero_point, app_ctx->outputs[mask_ids].attr->qnt_info.scale,
                                     grid_h, grid_w, stride, dfl_len,
                                     filterBoxes, objProbs, classId, masks_i8, maskZps, maskScales, conf_threshold, score_is_logits);
        }
        else
        {
            float *box_tensor = nullptr;
            float *score_tensor = nullptr;
            float *mask_tensor = nullptr;

            //convert box from fp16 to fp32
            if(_outputs[box_idx].attr->dtype == RKNN3_TENSOR_FLOAT16){
                int size = getTensorSize(_outputs[box_idx]);
                box_tensor = (float*) malloc(size * sizeof(float));
                convert_fp16_to_fp32((float16*)_outputs[box_idx].mem->virt_addr, box_tensor, size);
            }
            else {
                box_tensor = (float *)_outputs[box_idx].mem->virt_addr;
            }

            //convert score from fp16 to fp32
            if(_outputs[score_idx].attr->dtype == RKNN3_TENSOR_FLOAT16){
                int size = getTensorSize(_outputs[score_idx]);
                score_tensor = (float*) malloc(size * sizeof(float));
                convert_fp16_to_fp32((float16*)_outputs[score_idx].mem->virt_addr, score_tensor, size);
            }
            else {
                score_tensor = (float *)_outputs[score_idx].mem->virt_addr;
            }


            if(_outputs[mask_ids].attr->dtype == RKNN3_TENSOR_FLOAT16){
                int size = getTensorSize(_outputs[mask_ids]);
                mask_tensor = (float*) malloc(size * sizeof(float));
                convert_fp16_to_fp32((float16*)_outputs[mask_ids].mem->virt_addr, mask_tensor, size);
            }
            else {
                mask_tensor = (float *)_outputs[mask_ids].mem->virt_addr;
            }

            bool score_is_logits = score_tensor_is_logits_fp32(score_tensor, app_ctx->outputs[score_idx].attr);
            validCount += process_fp32(box_tensor, app_ctx->outputs[box_idx].attr,
                                       score_tensor, app_ctx->outputs[score_idx].attr,
                                       mask_tensor, app_ctx->outputs[mask_ids].attr,
                                       grid_h, grid_w, stride, dfl_len,
                                       filterBoxes, objProbs, classId, masks, conf_threshold, score_is_logits);
            if (_outputs[box_idx].attr->dtype == RKNN3_TENSOR_FLOAT16)
            {
                free(box_tensor);
            }
            if (_outputs[score_idx].attr->dtype == RKNN3_TENSOR_FLOAT16)
            {
                free(score_tensor);
            }
            if (_outputs[mask_ids].attr->dtype == RKNN3_TENSOR_FLOAT16)
            {
                free(mask_tensor);
            }
        }
    }

    // no object detect

    if (validCount <= 0)
    {
        return 0;
    }

    rknn3_tensor *proto_tensor = &_outputs[9];
    int proto_h = tensor_grid_h(proto_tensor->attr);
    int proto_w = tensor_grid_w(proto_tensor->attr);
    if (proto_h <= 0 || proto_w <= 0)
    {
        return -1;
    }
    bool use_i8_mask = app_ctx->is_quant && proto_tensor->attr->dtype == RKNN3_TENSOR_INT8;
    if (!use_i8_mask && convert_proto_tensor_to_f32(proto_tensor, protoData, &proto_h, &proto_w) != 0)
    {
        return -1;
    }

    int src_w = (int)roundf((model_in_w - 2.0f * letter_box->x_pad) / letter_box->scale);
    int src_h = (int)roundf((model_in_h - 2.0f * letter_box->y_pad) / letter_box->scale);
    if (src_w <= 0 || src_h <= 0)
    {
        return -1;
    }

    xToProto.resize(src_w);
    for (int x = 0; x < src_w; x++)
    {
        float model_x = x * letter_box->scale + letter_box->x_pad;
        xToProto[x] = clamp(model_x * proto_w / model_in_w, 0, proto_w - 1);
    }

    yToProto.resize(src_h);
    for (int y = 0; y < src_h; y++)
    {
        float model_y = y * letter_box->scale + letter_box->y_pad;
        yToProto[y] = clamp(model_y * proto_h / model_in_h, 0, proto_h - 1);
    }

    indexArray.resize(validCount);
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

        object_detect_result *result = &od_results->results[last_count];
        float mapped_x1 = clamp(x1, 0, model_in_w) / letter_box->scale;
        float mapped_y1 = clamp(y1, 0, model_in_h) / letter_box->scale;
        float mapped_x2 = clamp(x2, 0, model_in_w) / letter_box->scale;
        float mapped_y2 = clamp(y2, 0, model_in_h) / letter_box->scale;
        result->box.left = (int)mapped_x1;
        result->box.top = (int)mapped_y1;
        result->box.right = (int)mapped_x2;
        result->box.bottom = (int)mapped_y2;
        result->box_float[0] = mapped_x1;
        result->box_float[1] = mapped_y1;
        result->box_float[2] = mapped_x2;
        result->box_float[3] = mapped_y2;
        result->prop = objProbs[i];
        result->cls_id = classId[n];
        if (use_i8_mask)
        {
            build_instance_mask_i8(proto_tensor, &masks_i8[n * OBJ_MASK_DIM], maskZps[n], maskScales[n],
                                   src_w, src_h, xToProto.data(), yToProto.data(), protoMaskScratch, result);
        }
        else
        {
            build_instance_mask(protoData.data(), proto_h, proto_w, &masks[n * OBJ_MASK_DIM],
                                src_w, src_h, xToProto.data(), yToProto.data(), protoMaskScratch, result);
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

void release_object_detect_result_list(object_detect_result_list *od_results)
{
    if (od_results == nullptr)
    {
        return;
    }

    for (int i = 0; i < od_results->count && i < OBJ_NUMB_MAX_SIZE; i++)
    {
        if (od_results->results[i].mask.data != nullptr)
        {
            free(od_results->results[i].mask.data);
            od_results->results[i].mask.data = nullptr;
        }
        od_results->results[i].mask.width = 0;
        od_results->results[i].mask.height = 0;
    }
    od_results->count = 0;
}

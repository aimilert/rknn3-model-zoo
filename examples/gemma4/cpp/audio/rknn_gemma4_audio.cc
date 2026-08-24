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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <complex>
#include <string>
#include <vector>

#include "pocketfft_hdronly.h"

#include "rknn_gemma4_audio.h"
#include "common.h"
#include "file_utils.h"
#include "image_utils.h"
#include "audio_utils.h"

#define DEBUG_AUDIO 0 // 0: no debug, 1: debug

int init_gemma4_audio(rknn_gemma4_audio_context* audio_ctx, const char* model_path, const char* weight_path, uint32_t core_mask)
{
    int ret;
    rknn3_context ctx = 0;
    rknn3_config config;
    rknn3_shape_info *shape_infos = NULL;

    if (!audio_ctx) {
        printf("Error: init_gemma4_audio: audio_ctx is NULL!\n");
        return -1;
    }
    if (!model_path || !weight_path) {
        printf("Error: init_gemma4_audio: model_path or weight_path is NULL!\n");
        return -1;
    }

    memset(&config, 0, sizeof(config));
    config.run_core_mask = core_mask;
    // config.user_mem_internal = 1; // 使用用户管理的internal内存

    rknn3_devices devs;
    memset(&devs, 0, sizeof(devs));
    ret = rknn3_find_devices(&devs);
    if (ret != RKNN3_SUCCESS || devs.n_devices == 0) {
        printf("rknn3_find_devices fail! ret=%d, n_devices=%d\n", ret, devs.n_devices);
        return -1;
    }
    rknn3_init_extend init_extend;
    init_extend.device_id = devs.devices[0].id;
    printf("device id: %s, type: %s\n",  devs.devices[0].id, devs.devices[0].type);

    // RKNN Init
    ret = rknn3_init(&ctx, &init_extend);
    if (ret < 0) {
        printf("rknn_init fail ret=%d\n", ret);
        return ret;
    }

    // Load RKNN Model
    ret = rknn3_load_model_from_path(ctx, model_path, weight_path);
    if (ret < 0) {
        printf("rknn_load_model failed! ret=%d\n", ret);
        goto err_ctx;
    }

    //Init RKNN Model
    ret = rknn3_model_init(ctx, &config);
    if (ret < 0) {
        printf("rknn_model_init failed! ret=%d\n", ret);
        goto err_ctx;
    }

    // Get Model Input Output Number
    rknn3_input_output_num io_num;
    ret = rknn3_query(ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        printf("rknn_query fail! ret=%d\n", ret);
        goto err_ctx;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    rknn3_shape_config shape_config;
    ret = rknn3_query(ctx, RKNN3_QUERY_DYNAMIC_SHAPE_CONFIG, &shape_config, sizeof(shape_config));
    if (ret != RKNN3_SUCCESS) {
        printf("Query dynamic shape config failed! ret=%d\n", ret);
        // 如果不支持动态形状，使用静态形状
        shape_config.n_shapes = 1;
        shape_config.current_shape_id = 0;
    } else {
        printf("Model supports %d shape combinations\n", shape_config.n_shapes);
        printf("Current shape ID: %d\n", shape_config.current_shape_id);
    }

    // Guard: n_shapes == 0 is invalid.
    if (shape_config.n_shapes == 0) {
        printf("Error: shape_config.n_shapes is 0!\n");
        ret = -1;
        goto err_ctx;
    }
    if (shape_config.n_shapes > MAX_SHAPE_ID) {
        printf("Error: shape_config.n_shapes=%d exceeds MAX_SHAPE_ID=%d!\n", shape_config.n_shapes, MAX_SHAPE_ID);
        ret = -1;
        goto err_ctx;
    }

    shape_infos = (rknn3_shape_info *)malloc(sizeof(rknn3_shape_info) * shape_config.n_shapes);
    if (!shape_infos) {
        printf("Error: malloc shape_infos failed!\n");
        ret = -1;
        goto err_ctx;
    }
    memset(shape_infos, 0, sizeof(rknn3_shape_info) * shape_config.n_shapes);

    // 为每个形状信息分配输入输出属性内存
    for (uint32_t i = 0; i < shape_config.n_shapes; i++) {
        shape_infos[i].shape_id = i;
        shape_infos[i].input_attrs = (rknn3_tensor_attr *)malloc(sizeof(rknn3_tensor_attr) * io_num.n_input);
        shape_infos[i].output_attrs = (rknn3_tensor_attr *)malloc(sizeof(rknn3_tensor_attr) * io_num.n_output);
        if (!shape_infos[i].input_attrs || !shape_infos[i].output_attrs) {
            printf("Error: malloc input_attrs/output_attrs for shape %u failed!\n", i);
            ret = -1;
            goto err_shape_infos;
        }
    }

    // 查询所有形状信息
    ret = rknn3_query(ctx, RKNN3_QUERY_DYNAMIC_SHAPE_INFO, shape_infos, sizeof(rknn3_shape_info) * shape_config.n_shapes);
    if (ret != RKNN3_SUCCESS) {
        printf("Query dynamic shape info failed! ret=%d\n", ret);
        goto err_shape_infos;
    }

    // 打印所有形状信息
    for (uint32_t i = 0; i < shape_config.n_shapes; i++) {
        printf("Shape %d (ID: %d)%s:\n", i, shape_infos[i].shape_id, shape_infos[i].is_default ? " [Default]" : "");
        for (uint32_t j = 0; j < shape_infos[i].n_inputs; j++) {
            rknn3_tensor_attr *attr = &shape_infos[i].input_attrs[j];
            printf("  Input %d (%s): [", attr->index, attr->name);
            for (uint32_t k = 0; k < attr->n_dims; k++) {
                printf("%d%s", attr->shape[k], (k < attr->n_dims - 1) ? ", " : "");
            }
            printf("] Aligned size: %lu bytes\n", attr->aligned_size);
        }
    }

    audio_ctx->rknn_ctx = ctx;
    audio_ctx->io_num = io_num;
    audio_ctx->n_shapes = shape_config.n_shapes;

    // 检查attrs, 并根据shape获取相关参数
    for (int i = 0; i < shape_config.n_shapes; i++) {
        rknn3_tensor_attr* input_attrs = shape_infos[i].input_attrs;
        rknn3_tensor_attr* output_attrs = shape_infos[i].output_attrs;
        if (input_attrs[0].layout == RKNN3_TENSOR_UNDEFINED) {
            // Gemma4 audio model takes input_features in HF Gemma3n natural layout:
            //   shape = [batch=1, n_frame (T), n_mels]
            // (this is the same shape that the in-graph `input_features.unsqueeze(1)`
            //  -> Conv2d would consume internally).
            audio_ctx->n_frame[i] = input_attrs[0].shape[1];
            audio_ctx->n_mels     = input_attrs[0].shape[2];
            printf("shape %d: input n_frame=%d, n_mels=%d\n", i, audio_ctx->n_frame[i], audio_ctx->n_mels);
        } else {
            printf("model input 0 layout is not UNDEFINED! shape=%d actual layout=%d (%s), name=%s\n",
                   i, input_attrs[0].layout,
                   rknn3_get_layout_string(input_attrs[0].layout), input_attrs[0].name);
            ret = -1;
            goto err_shape_infos;
        }
        if (input_attrs[1].layout == RKNN3_TENSOR_UNDEFINED) {
            audio_ctx->padded_mask_size[i] = input_attrs[1].shape[input_attrs[1].n_dims-1];
            printf("shape %d: input padded_mask_size=%d\n", i, audio_ctx->padded_mask_size[i]);
        } else {
            printf("model input 1 layout is not UNDEFINED! shape=%d actual layout=%d (%s), name=%s\n",
                   i, input_attrs[1].layout,
                   rknn3_get_layout_string(input_attrs[1].layout), input_attrs[1].name);
            ret = -1;
            goto err_shape_infos;
        }
        if (io_num.n_input > 2) {
            if (input_attrs[2].layout == RKNN3_TENSOR_UNDEFINED) {
                audio_ctx->attn_mask_size[i] = input_attrs[2].shape[input_attrs[2].n_dims-1];
                printf("shape %d: input attn_mask_size=%d\n", i, audio_ctx->attn_mask_size[i]);
            } else {
                printf("model input 2 layout is not UNDEFINED! shape=%d actual layout=%d (%s), name=%s\n",
                       i, input_attrs[2].layout,
                       rknn3_get_layout_string(input_attrs[2].layout), input_attrs[2].name);
                ret = -1;
                goto err_shape_infos;
            }
        } else {
            audio_ctx->attn_mask_size[i] = 0;
        }
        if (output_attrs[0].layout == RKNN3_TENSOR_UNDEFINED) {
            int n_dims = output_attrs[0].n_dims;
            audio_ctx->embeds_dim0[i] = output_attrs[0].shape[n_dims-2];
            audio_ctx->embeds_dim1 = output_attrs[0].shape[n_dims-1];
            printf("shape %d: output audio embeds dim0=%d,  audio embeds dim1=%d\n", i, audio_ctx->embeds_dim0[i], audio_ctx->embeds_dim1);
        } else {
            printf("model output 0 layout is not UNDEFINED! shape=%d actual layout=%d (%s), name=%s\n",
                   i, output_attrs[0].layout,
                   rknn3_get_layout_string(output_attrs[0].layout), output_attrs[0].name);
            ret = -1;
            goto err_shape_infos;
        }
    }

    audio_ctx->shape_infos = shape_infos;

    audio_ctx->inputs = (rknn3_tensor*)malloc(io_num.n_input * sizeof(rknn3_tensor));
    audio_ctx->outputs = (rknn3_tensor*)malloc(io_num.n_output * sizeof(rknn3_tensor));
    if (!audio_ctx->inputs || !audio_ctx->outputs) {
        printf("Error: malloc inputs/outputs failed!\n");
        ret = -1;
        goto err_io_alloc;
    }
    memset(audio_ctx->inputs, 0, io_num.n_input * sizeof(rknn3_tensor));
    memset(audio_ctx->outputs, 0, io_num.n_output * sizeof(rknn3_tensor));

    for (int i = 0; i < audio_ctx->io_num.n_input; i++) {
        audio_ctx->inputs[i].mem = rknn3_create_mem(ctx,
            shape_infos[0].input_attrs[i].aligned_size,
            shape_infos[0].input_attrs[i].core_id,
            RKNN3_FLAG_MEMORY_CACHEABLE);
        if (!audio_ctx->inputs[i].mem) {
            printf("Error: rknn3_create_mem for input[%d] failed!\n", i);
            ret = -1;
            goto err_io_mem;
        }
    }
    for (int i = 0; i < audio_ctx->io_num.n_output; i++) {
        audio_ctx->outputs[i].mem = rknn3_create_mem(ctx,
            shape_infos[0].output_attrs[i].aligned_size,
            shape_infos[0].output_attrs[i].core_id,
            RKNN3_FLAG_MEMORY_CACHEABLE);
        if (!audio_ctx->outputs[i].mem) {
            printf("Error: rknn3_create_mem for output[%d] failed!\n", i);
            ret = -1;
            goto err_io_mem;
        }
    }

    return ret;

err_io_mem:
    // Rollback: destroy all successfully created mem objects (NULL-safe).
    for (int i = 0; i < audio_ctx->io_num.n_input; i++) {
        if (audio_ctx->inputs[i].mem)
            rknn3_destroy_mem(ctx, audio_ctx->inputs[i].mem);
    }
    for (int i = 0; i < audio_ctx->io_num.n_output; i++) {
        if (audio_ctx->outputs[i].mem)
            rknn3_destroy_mem(ctx, audio_ctx->outputs[i].mem);
    }
err_io_alloc:
    if(audio_ctx->inputs) {
        free(audio_ctx->inputs);
        audio_ctx->inputs = NULL;
    }
    if (audio_ctx->outputs) {
        free(audio_ctx->outputs);
        audio_ctx->outputs = NULL;
    }
err_shape_infos:
    if (shape_infos) {
        for (uint32_t i = 0; i < shape_config.n_shapes; i++) {
            if (shape_infos[i].input_attrs)  free(shape_infos[i].input_attrs);
            if (shape_infos[i].output_attrs) free(shape_infos[i].output_attrs);
        }
        free(shape_infos);
    }
err_ctx:
    if (ctx) {
        rknn3_destroy(ctx);
    }
    return ret;
}

int release_gemma4_audio(rknn_gemma4_audio_context* audio_ctx)
{
    if (!audio_ctx) {
        printf("Error: release_gemma4_audio: audio_ctx is NULL!\n");
        return -1;
    }

    for (int i = 0; i < audio_ctx->io_num.n_input; i++) {
        if (audio_ctx->inputs[i].mem)
            rknn3_destroy_mem(audio_ctx->rknn_ctx, audio_ctx->inputs[i].mem);
    }
    for (int i = 0; i < audio_ctx->io_num.n_output; i++) {
        if (audio_ctx->outputs[i].mem)
            rknn3_destroy_mem(audio_ctx->rknn_ctx, audio_ctx->outputs[i].mem);
    }
    free(audio_ctx->inputs);
    free(audio_ctx->outputs);
    audio_ctx->inputs = NULL;
    audio_ctx->outputs = NULL;

    for (uint32_t i = 0; i < audio_ctx->n_shapes; i++) {
        if (audio_ctx->shape_infos[i].input_attrs)     free(audio_ctx->shape_infos[i].input_attrs);
        if (audio_ctx->shape_infos[i].output_attrs)    free(audio_ctx->shape_infos[i].output_attrs);
    }
    free(audio_ctx->shape_infos);
    audio_ctx->shape_infos = NULL;

    if (audio_ctx->rknn_ctx) {
        rknn3_destroy(audio_ctx->rknn_ctx);
        audio_ctx->rknn_ctx = 0;
    }
    return 0;
}

void vecfp32_to_fp16(std::vector<float> &vec, float16 *dst) {
    for (int i = 0; i < vec.size(); i++) {
        dst[i] = fp32_to_fp16(vec[i]);
    }
}

// Gemma-4 audio encoder preprocessing.
//
// All parameters below mirror HuggingFace Transformers' Gemma4AudioFeatureExtractor
// (transformers/models/gemma4/feature_extraction_gemma4.py), NOT the older
// Gemma3nAudioFeatureExtractor. The two are different feature extractors:
//
//   Gemma3n  Gemma4 (this file)
//   ---------------------------  -----------------------
//   frame_length = 512            frame_length = 320     (20 ms @ 16 kHz)
//   fft_length   = 1024 (overdr.) fft_length   = 512     (no overdrive)
//   f_min        = 125 Hz         f_min        = 0   Hz
//   f_max        = 7600 Hz        f_max        = 8000 Hz
//   preemphasis  = 0.97 (HTK)     preemphasis  = 0.0     (no preemph at all)
//   semicausal pad: none          left-pads frame_length//2 = 160 zeros
//   log: log(max(mel, floor))     log: log(mel + floor)
//   masked frames: log(floor)     masked frames: 0.0     (multiplied by mask)
#define GEMMA4_SAMPLE_RATE       16000
#define GEMMA4_HOP_LENGTH        160     // 10 ms @ 16 kHz
#define GEMMA4_FRAME_LENGTH      320     // 20 ms @ 16 kHz
#define GEMMA4_FFT_LENGTH        512     // 2^ceil(log2(320)) = 512, no overdrive
#define GEMMA4_F_MIN             0.0f
#define GEMMA4_F_MAX             8000.0f
#define GEMMA4_PREEMPHASIS       0.0f    // Gemma-4 does NOT do pre-emphasis
#define GEMMA4_MEL_FLOOR         1e-3f   // log(0 + 1e-3) = -6.9078 for silence
#define GEMMA4_PAD_TO_MULTIPLE   128     // SequenceFeatureExtractor.pad default
#define GEMMA4_SEMI_CAUSAL_LEFT  (GEMMA4_FRAME_LENGTH / 2)  // = 160

// ---------------------------------------------------------------------------
// Debug helpers: dump rknn3_tensor_attr + a small slice of the underlying data
// so we can sanity-check inputs / outputs around rknn3_run() against the
// reference Python pipeline.
// ---------------------------------------------------------------------------

static size_t gemma4_dtype_elem_size(int dtype) {
    // Mirrors the table in main.cc's get_dtype_elem_size.
    switch (dtype) {
        case 0:  return 4;  // FLOAT32
        case 1:  return 2;  // FLOAT16
        case 2:  return 1;  // INT8
        case 3:  return 1;  // UINT8
        case 4:  return 2;  // INT16
        case 5:  return 2;  // UINT16
        case 6:  return 4;  // INT32
        case 7:  return 4;  // UINT32
        case 8:  return 8;  // INT64
        case 9:  return 8;  // UINT64
        case 10: return 1;  // BOOL
        case 11: return 1;  // INT4
        case 12: return 1;  // FLOAT8E4M3FN
        case 13: return 2;  // BFLOAT16
        case 14: return 1;  // FLOAT8E8M0
        case 15: return 1;  // FLOAT4E2M1
        default: return 1;
    }
}

static void gemma4_print_tensor_attr(const char* tag, int idx, const rknn3_tensor_attr* attr)
{
    char shape_buf[128];  shape_buf[0] = '\0';
    char stride_buf[128]; stride_buf[0] = '\0';
    int  off = 0;
    for (uint32_t j = 0; j < attr->n_dims; j++) {
        off += snprintf(shape_buf + off, sizeof(shape_buf) - off,
                        "%d%s", attr->shape[j], (j + 1 < attr->n_dims) ? "," : "");
        if (off >= (int)sizeof(shape_buf)) break;
    }
    off = 0;
    for (uint32_t j = 0; j < attr->n_stride; j++) {
        off += snprintf(stride_buf + off, sizeof(stride_buf) - off,
                        "%d%s", attr->stride[j], (j + 1 < attr->n_stride) ? "," : "");
        if (off >= (int)sizeof(stride_buf)) break;
    }
    printf("[%s][%d] name=%s shape=[%s] stride=[%s] aligned=%lu B "
           "layout=%s dtype=%s qnt=%s scale=%g zp=%d core_id=%d\n",
           tag, idx,
           attr->name, shape_buf, stride_buf, attr->aligned_size,
           rknn3_get_layout_string(attr->layout),
           rknn3_get_type_string(attr->dtype),
           rknn3_get_qnt_type_string(attr->qnt_type),
           attr->qnt_info.scale, attr->qnt_info.zero_point,
           attr->core_id);
}

// Print the first `head` and last `tail` elements of a tensor's memory.
// Fp16 / Fp32 / int8 / int16 / int32 are decoded; everything else is hex-dumped.
static void gemma4_dump_tensor_data(const char* tag, int idx,
                                    const rknn3_tensor_attr* attr,
                                    const void* virt_addr,
                                    int head, int tail)
{
    size_t n = 1;
    for (uint32_t j = 0; j < attr->n_dims; j++) {
        size_t d = (attr->shape[j] > 0) ? (size_t)attr->shape[j] : (size_t)1;
        n *= d;
    }
    if (n == 0 || virt_addr == NULL) {
        printf("[%s][%d] data: <empty>\n", tag, idx);
        return;
    }
    if ((size_t)(head + tail) > n) { head = (int)n; tail = 0; }

    auto print_one = [&](size_t k) {
        switch (attr->dtype) {
        case 0: { // FLOAT32
            float v = ((const float*)virt_addr)[k];
            printf("%g", v);
            break;
        }
        case 1: { // FLOAT16
            float16 raw = ((const float16*)virt_addr)[k];
            printf("%g", fp16_to_fp32(raw));
            break;
        }
        case 2: { // INT8
            printf("%d", ((const int8_t*)virt_addr)[k]);
            break;
        }
        case 3: { // UINT8
            printf("%u", ((const uint8_t*)virt_addr)[k]);
            break;
        }
        case 4: { // INT16
            printf("%d", ((const int16_t*)virt_addr)[k]);
            break;
        }
        case 6: { // INT32
            printf("%d", ((const int32_t*)virt_addr)[k]);
            break;
        }
        case 13: { // BFLOAT16: bits = (uint16<<16)
            uint32_t bits = (uint32_t)((const uint16_t*)virt_addr)[k] << 16;
            float v;
            memcpy(&v, &bits, sizeof(v));
            printf("%g", v);
            break;
        }
        default: {
            size_t elem_sz = gemma4_dtype_elem_size(attr->dtype);
            const uint8_t* p = (const uint8_t*)virt_addr + k * elem_sz;
            printf("0x");
            for (size_t b = 0; b < elem_sz; b++) printf("%02x", p[b]);
            break;
        }
        }
    };

    printf("[%s][%d] data first %d:[", tag, idx, head);
    for (int k = 0; k < head; k++) {
        if (k) printf(", ");
        print_one((size_t)k);
    }
    printf("]");
    if (tail > 0) {
        printf(" last %d:[", tail);
        for (int k = 0; k < tail; k++) {
            if (k) printf(", ");
            print_one(n - (size_t)tail + (size_t)k);
        }
        printf("]");
    }
    printf(" (n=%zu)\n", n);
}

// One-shot dump for an entire array of inputs / outputs.
static void gemma4_dump_io(const char* tag,
                           rknn3_tensor* tensors, uint32_t n_tensors,
                           int head, int tail)
{
    for (uint32_t i = 0; i < n_tensors; i++) {
        if (tensors[i].attr) {
            gemma4_print_tensor_attr(tag, (int)i, tensors[i].attr);
        }
        if (tensors[i].mem) {
            gemma4_dump_tensor_data(tag, (int)i, tensors[i].attr,
                                    tensors[i].mem->virt_addr, head, tail);
        }
    }
}

int get_shape_id(rknn_gemma4_audio_context* audio_ctx, int audio_len) {
    if (!audio_ctx) {
        printf("Error: get_shape_id: audio_ctx is NULL!\n");
        return -1;
    }
    if (audio_ctx->n_shapes <= 0) {
        printf("Error: get_shape_id called with n_shapes=%d!\n", audio_ctx->n_shapes);
        return -1;
    }
    int n_frame_ = (audio_len + GEMMA4_HOP_LENGTH - 1) / GEMMA4_HOP_LENGTH;
    int i = audio_ctx->n_shapes - 1;
    for (i = audio_ctx->n_shapes - 1; i >= 0; i--) {
        if (n_frame_ <= audio_ctx->n_frame[i]) {
            break;
        }
    }
    return i < 0 ? 0 : i;
}

int get_n_audio(rknn_gemma4_audio_context* audio_ctx, int audio_len) {
    if (!audio_ctx) {
        printf("Error: get_n_audio: audio_ctx is NULL!\n");
        return -1;
    }
    int shape_id = get_shape_id(audio_ctx, audio_len);
    if (shape_id < 0) {
        printf("Error: get_n_audio: get_shape_id returned invalid shape_id=%d\n", shape_id);
        return -1;
    }
    return audio_ctx->embeds_dim0[shape_id];
}

// Build a USM-style mel filterbank matrix in row-major [n_freqs, n_mels] layout.
// Identical math to HF's create_fb_matrix (HTK mel scale, no Slaney normalisation).
static void build_gemma4_mel_filters(float* fb,
                                     int n_freqs,
                                     int n_mels,
                                     float f_min,
                                     float f_max,
                                     int sample_rate,
                                     int fft_length)
{
    if (!fb) {
        printf("Error: build_gemma4_mel_filters: fb is NULL!\n");
        return;
    }
    if (n_freqs <= 0 || n_mels <= 0 || sample_rate <= 0 || fft_length <= 0) {
        printf("Error: build_gemma4_mel_filters: invalid params (n_freqs=%d, n_mels=%d, sample_rate=%d, fft_length=%d)\n",
               n_freqs, n_mels, sample_rate, fft_length);
        return;
    }

    std::vector<float> all_freqs(n_freqs);
    const float bin_hz = (float)sample_rate / (float)fft_length;
    for (int i = 0; i < n_freqs; i++) {
        all_freqs[i] = i * bin_hz;
    }

    const float m_min = 2595.0f * log10f(1.0f + f_min / 700.0f);
    const float m_max = 2595.0f * log10f(1.0f + f_max / 700.0f);

    std::vector<float> f_pts(n_mels + 2);
    for (int i = 0; i < n_mels + 2; i++) {
        float m = m_min + (m_max - m_min) * (float)i / (float)(n_mels + 1);
        f_pts[i] = 700.0f * (powf(10.0f, m / 2595.0f) - 1.0f);
    }

    std::vector<float> f_diff(n_mels + 1);
    for (int i = 0; i < n_mels + 1; i++) {
        f_diff[i] = f_pts[i + 1] - f_pts[i];
    }

    for (int k = 0; k < n_freqs; k++) {
        for (int m = 0; m < n_mels; m++) {
            float down = (all_freqs[k] - f_pts[m])     / f_diff[m];
            float up   = (f_pts[m + 2] - all_freqs[k]) / f_diff[m + 1];
            float val  = std::min(down, up);
            fb[k * n_mels + m] = val < 0.0f ? 0.0f : val;
        }
    }
}

// Gemma-4 log-mel preprocessing 
// Gemma4AudioFeatureExtractor._extract_spectrogram + __call__.
//
// Pipeline per the HF reference:
//   1. Pad raw waveform on the right to a multiple of 128 (pad_to_multiple_of).
//   2. Prepend frame_length//2 = 160 zeros (semicausal time padding) so that
//      the first STFT frame is centered at t=0.
//   3. Unfold with size = frame_length + 1 = 321, step = hop_length = 160.
//      => num_frames = (L_total - 321) / 160 + 1
//   4. preemph == 0 path: drop the trailing extra sample, frames = unfolded[..., :frame_length].
//   5. Multiply by periodic Hann window of length 320 (divisor N).
//   6. Real FFT of length fft_length = 512, magnitude = |stft|.
//   7. mel = magnitude @ mel_filters,  log_mel = log(mel + mel_floor).
//      NOTE: HF uses log(mel + floor), NOT log(max(mel, floor)). Subtle but
//      different for non-zero but small mel energies.
//   8. mask[i] = attention_mask[i*hop + frame_length] (in semicausal coords),
//      i.e. frame i is "valid" iff i*hop + frame_length < pad_left + L_orig.
//   9. Output := log_mel * mask, i.e. invalid frames are zeroed (NOT log_floor).
//
// Output layout: padded_feature[t * n_mels + m] (time-major, n_mels contiguous),
// matching the RKNN model input shape [1, n_frame (T), n_mels].
static void gemma4_audio_preprocess(audio_buffer_t* audio,
                                    const float* mel_filters,  // [n_freqs, n_mels]
                                    int n_mels,
                                    int n_frame,
                                    std::vector<float>& padded_feature,
                                    int* actual_len)
{
    const int frame_length = GEMMA4_FRAME_LENGTH;
    const int fft_length   = GEMMA4_FFT_LENGTH;
    const int hop_length   = GEMMA4_HOP_LENGTH;
    const int n_freqs      = fft_length / 2 + 1;
    const int pad_left     = GEMMA4_SEMI_CAUSAL_LEFT;          // 160
    const int pad_multiple = GEMMA4_PAD_TO_MULTIPLE;           // 128
    const float mel_floor  = GEMMA4_MEL_FLOOR;

    const int audio_length = (audio && audio->data) ? audio->num_frames : 0;

    if (!mel_filters) {
        printf("Error: gemma4_audio_preprocess: mel_filters is NULL!\n");
        if (actual_len) *actual_len = 0;
        return;
    }
    if (!actual_len) {
        printf("Error: gemma4_audio_preprocess: actual_len is NULL!\n");
        return;
    }

    // Step 1 + 2: build the [pad_left + L_pad128]-sample buffer.
    //   - Indices [0, pad_left)                 -> 0 (semicausal pad)
    //   - Indices [pad_left, pad_left+L_orig)   -> raw audio
    //   - Indices [pad_left+L_orig, end)        -> 0 (right pad to multiple of 128)
    const int L_pad128 = ((audio_length + pad_multiple - 1) / pad_multiple) * pad_multiple;
    const int L_total  = pad_left + L_pad128;

    // We also need ≥ (n_frame-1)*hop + frame_length samples in the working
    // buffer because the model's bucket size n_frame may exceed the audio's
    // natural number of mel frames; the trailing frames will compute on zeros
    // and then be masked out below.
    const int min_buf  = (n_frame > 0) ? ((n_frame - 1) * hop_length + frame_length + 1) : 0;
    const int buf_size = std::max(L_total, min_buf) + 16;     // small safety margin

    std::vector<float> padded_audio(buf_size, 0.0f);
    if (audio_length > 0) {
        std::copy(audio->data, audio->data + audio_length,
                  padded_audio.begin() + pad_left);
    }

    // Step 8: number of "valid" mel frames -- frame i is valid iff its mask
    // sample lands inside the original audio. Equivalent to:
    //   i*hop + frame_length < pad_left + L_orig
    //   <=> i < (L_orig + pad_left - frame_length) / hop
    // Number of valid i = ceil((L_orig + pad_left - frame_length) / hop).
    int n_valid = 0;
    {
        long rhs = (long)audio_length + pad_left - frame_length;
        if (rhs > 0) {
            n_valid = (int)((rhs + hop_length - 1) / hop_length);
        }
    }
    *actual_len = std::min(n_frame, n_valid);

    // Periodic Hann window of length frame_length with divisor N. Equivalent
    // to numpy.hanning(N+1)[:N], which is what HF's window_function returns.
    std::vector<float> window(frame_length);
    for (int i = 0; i < frame_length; i++) {
        window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)frame_length));
    }

    std::vector<float> fft_in(fft_length, 0.0f);
    std::vector<std::complex<float>> fft_out(n_freqs);
    pocketfft::shape_t fft_shape = {static_cast<size_t>(fft_length)};
    pocketfft::stride_t fft_stride_in = {sizeof(float)};
    pocketfft::stride_t fft_stride_out = {sizeof(std::complex<float>)};

    std::vector<float> magnitude(n_freqs);

    // Pre-fill the model input with 0.0. Frames with index >= actual_len are
    // padded silence and HF zeros them out via `speech * mask[..., None]`.
    std::fill(padded_feature.begin(), padded_feature.end(), 0.0f);

    for (int i = 0; i < n_frame; i++) {
        const int start = i * hop_length;  // already in semicausal coords

        // Window (no preemph for Gemma-4) the first frame_length samples;
        // unfold size is frame_length+1 but with preemph==0 we drop the last.
        for (int k = 0; k < frame_length; k++) {
            fft_in[k] = padded_audio[start + k] * window[k];
        }
        for (int k = frame_length; k < fft_length; k++) {
            fft_in[k] = 0.0f;  // implicit np.fft.rfft(..., n=fft_length)
        }

        pocketfft::r2c(fft_shape, fft_stride_in, fft_stride_out, 0, true,
                       fft_in.data(), fft_out.data(), 1.0f);

        for (int k = 0; k < n_freqs; k++) {
            const float re = fft_out[k].real();
            const float im = fft_out[k].imag();
            magnitude[k] = sqrtf(re * re + im * im);  // |stft|, not power
        }

        // Output is stored time-major: padded_feature[t * n_mels + m].
        // mel_filters layout is row-major [n_freqs, n_mels].
        const size_t row_off = (size_t)i * (size_t)n_mels;
        const bool   valid   = (i < *actual_len);
        if (!valid) {
            // Already zero-filled; skip the mel reduction.
            continue;
        }
        for (int m = 0; m < n_mels; m++) {
            float sum = 0.0f;
            for (int k = 0; k < n_freqs; k++) {
                sum += magnitude[k] * mel_filters[k * n_mels + m];
            }
            // HF: log(mel + mel_floor). NOT log(max(mel, mel_floor)).
            padded_feature[row_off + (size_t)m] = logf(sum + mel_floor);
        }
    }
}

int inference_gemma4_audio(rknn_gemma4_audio_context* audio_ctx, audio_buffer_t* audio, float16* audio_embeds, int* n_valid_audio)
{
    if ((!audio_ctx) || (!audio)) {
        printf("audio_ctx or audio is NULL");
        return -1;
    }

    int shape_id = get_shape_id(audio_ctx, audio->num_frames);
    if (shape_id < 0) {
        printf("Error: inference_gemma4_audio: get_shape_id returned invalid shape_id=%d\n", shape_id);
        return -1;
    }
    int n_mels = audio_ctx->n_mels;
    int n_frame = audio_ctx->n_frame[shape_id];
    int attn_mask_size = audio_ctx->attn_mask_size[shape_id];

    // Build the USM-style mel filterbank on the fly. Layout: [n_freqs, n_mels].
    const int n_freqs = GEMMA4_FFT_LENGTH / 2 + 1;
    std::vector<float> mel_filters((size_t)n_freqs * (size_t)n_mels);
    build_gemma4_mel_filters(mel_filters.data(),
                             n_freqs, n_mels,
                             GEMMA4_F_MIN, GEMMA4_F_MAX,
                             GEMMA4_SAMPLE_RATE, GEMMA4_FFT_LENGTH);

    // USM-style log-mel feature extraction.
    // RKNN model input shape is [1, n_frame, n_mels].
    printf("input size: [T=%d, n_mels=%d] (frame_length=%d, fft_length=%d, hop=%d)\n",
           n_frame, n_mels, GEMMA4_FRAME_LENGTH, GEMMA4_FFT_LENGTH, GEMMA4_HOP_LENGTH);
    std::vector<float> padded_feature((size_t)n_frame * (size_t)n_mels, 0.0f);
    int actual_len = 0;
    gemma4_audio_preprocess(audio, mel_filters.data(), n_mels, n_frame, padded_feature, &actual_len);

    // Report the actual valid audio token counts.
    //
    // - actual_len: number of valid mel frames before the audio encoder, computed
    //   the same way as HF Gemma4AudioFeatureExtractor:
    //     L_pad128 = ceil(L_orig / 128) * 128
    //     L_total  = L_pad128 + frame_length//2  (semicausal left pad = 160)
    //     n_frames = (L_total - (frame_length+1)) / hop + 1
    //     valid    = ceil((L_orig + frame_length//2 - frame_length) / hop)
    //   This MUST equal the first-dim valid count of HF's `input_features_mask`
    //   for the same wav (e.g. for the 1.536 s / 24576-sample demo: 153).
    // - valid_audio_tokens: number of tokens after the audio encoder's
    //   subsample_conv_projection. The encoder has two Conv2d layers with
    //   stride=(2,2) and the corresponding mask is downsampled by mask[:, ::2]
    //   twice -> overall 4x temporal subsampling (ceil-div by 2 twice).
    //   This is what should be passed to the LLM as `n_audio_tokens` if you
    //   want to consume only the non-padding portion of the audio embeds.
    int valid_audio_tokens = ((actual_len + 1) / 2 + 1) / 2;
    int total_audio_tokens = audio_ctx->embeds_dim0[shape_id];
    printf("audio waveform: %d samples (%.3fs @16kHz), "
           "actual mel frames=%d / %d, valid audio tokens=%d / total %d (after 4x subsampling)\n",
           audio->num_frames, (float)audio->num_frames / 16000.0f,
           actual_len, n_frame, valid_audio_tokens, total_audio_tokens);

    if (n_valid_audio) {
        *n_valid_audio = valid_audio_tokens;
    }

    // padded_mask: 1 for valid frames, 0 for silence-padded frames.
    // (Matches HF Gemma3n's `input_features_mask`: True over valid mel frames,
    //  False over the silence-padded tail. The model treats it as fp16 and
    //  multiplies the post-conv hidden_states by it.)
    std::vector<float> padded_mask(n_frame, 0.0f);
    for (int i = 0; i < std::min(actual_len, n_frame); i++) {
        padded_mask[i] = 1.0f;
    }
    {
        // Boundary sanity check: mask must be 1.0 at [actual_len-1] and
        // 0.0 at [actual_len] (if both indices are in-range).
        int b0 = actual_len - 1;
        int b1 = actual_len;
        float vb0 = (b0 >= 0 && b0 < n_frame) ? padded_mask[b0] : -1.0f;
        float vb1 = (b1 >= 0 && b1 < n_frame) ? padded_mask[b1] : -1.0f;
        printf("padded_mask size: %ld, head=[%g,%g,%g,%g], boundary[%d]=%g [%d]=%g, tail=[%g,%g,%g,%g]\n",
               padded_mask.size(),
               padded_mask[0], padded_mask[1], padded_mask[2], padded_mask[3],
               b0, vb0, b1, vb1,
               padded_mask[n_frame - 4], padded_mask[n_frame - 3],
               padded_mask[n_frame - 2], padded_mask[n_frame - 1]);
    }

    // Optional 3rd input: dense bidirectional attention mask. If the RKNN model has
    // baked the chunked-blocked mask transformation into its graph, the simple
    // "all-zero (= attend everywhere)" mask used here is appropriate. If not, the
    // model export needs to be revisited to embed the mask reshape (see notes).
    std::vector<float> attention_mask;
    if (audio_ctx->io_num.n_input > 2 && attn_mask_size > 0) {
        attention_mask.assign((size_t)attn_mask_size * (size_t)attn_mask_size, 0.0f);
        printf("attention_mask size: %ld\n", attention_mask.size());
    }

    vecfp32_to_fp16(padded_feature, (float16*)(audio_ctx->inputs[0].mem->virt_addr));
    vecfp32_to_fp16(padded_mask,    (float16*)(audio_ctx->inputs[1].mem->virt_addr));

#if DEBUG_AUDIO
    // dump input features and input mask
    {
        float16* input_feature = (float16*)audio_ctx->inputs[0].mem->virt_addr;
        float16* input_mask = (float16*)audio_ctx->inputs[1].mem->virt_addr;
        FILE *fp = fopen("input_features_rk3588.bin", "wb");
        fwrite(input_feature, sizeof(float16), n_frame * n_mels, fp);
        fclose(fp);
        fp = fopen("input_features_mask_rk3588.bin", "wb");
        fwrite(input_mask, sizeof(float16), n_frame, fp);
        fclose(fp);
    }
#endif

    if (audio_ctx->io_num.n_input > 2) {
        vecfp32_to_fp16(attention_mask, (float16*)(audio_ctx->inputs[2].mem->virt_addr));
    }

    for (int i = 0; i < audio_ctx->io_num.n_input; i++) {
        audio_ctx->inputs[i].attr = &(audio_ctx->shape_infos[shape_id].input_attrs[i]);
    }
    for (int i = 0; i < audio_ctx->io_num.n_output; i++) {
        audio_ctx->outputs[i].attr = &(audio_ctx->shape_infos[shape_id].output_attrs[i]);
    }

    printf("rknn3_set_shape, shape_id = %d\n", shape_id);
    int ret = rknn3_set_shape(audio_ctx->rknn_ctx, shape_id);
    if (ret < 0) {
        printf("rknn3_set_shape fail! ret=%d\n", ret);
        return ret;
    }

    // Sync Inputs
    for (int i = 0; i < audio_ctx->io_num.n_input; i++)
    {
        ret = rknn3_mem_sync(audio_ctx->rknn_ctx, audio_ctx->inputs[i].mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
        if (ret < 0)
        {
            printf("rknn3_mem_sync input[%d] fail! ret=%d\n", i, ret);
            return ret;
        }
    }

#if DEBUG_AUDIO
    // ---- Debug: dump inputs right before rknn3_run ----
    printf("=== gemma4 audio inputs (before rknn3_run, shape_id=%d) ===\n", shape_id);
    gemma4_dump_io("in", audio_ctx->inputs, audio_ctx->io_num.n_input, /*head=*/8, /*tail=*/4);
#endif

    // Run
    ret = rknn3_run(audio_ctx->rknn_ctx, audio_ctx->inputs, audio_ctx->io_num.n_input, audio_ctx->outputs, audio_ctx->io_num.n_output);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return ret;
    }

    // Sync Outputs
    for (int i = 0; i < audio_ctx->io_num.n_output; i++)
    {
        ret = rknn3_mem_sync(audio_ctx->rknn_ctx, audio_ctx->outputs[i].mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
        if (ret < 0)
        {
            printf("rknn3_mem_sync output[%d] fail! ret=%d\n", i, ret);
            return ret;
        }
    }

#if DEBUG_AUDIO
    // ---- Debug: dump outputs after rknn3_run ----
    printf("=== gemma4 audio outputs (after rknn3_run, shape_id=%d) ===\n", shape_id);
    gemma4_dump_io("out", audio_ctx->outputs, audio_ctx->io_num.n_output, /*head=*/8, /*tail=*/4);
#endif

    // Copy actual NPU output into audio_embeds. Size in bytes = n_audio_tokens * embed_dim * sizeof(fp16).
    size_t embeds_size = (size_t)audio_ctx->embeds_dim0[shape_id]
                       * (size_t)audio_ctx->embeds_dim1
                       * sizeof(float16);
    memcpy(audio_embeds, (float16*)audio_ctx->outputs[0].mem->virt_addr, embeds_size);

    return ret;
}

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
#include <stdint.h>
#include "float16.h"
#include "rknn3_api.h"
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>

#include "sensevoice_small.h"


static void dump_tensor_attr(rknn3_tensor_attr* attrs)
{
    std::string shape_str = "";
    for (int j = 0; j < attrs->n_dims; j++) {
      shape_str += std::to_string(attrs->shape[j]);
      if (j < attrs->n_dims - 1) {
        shape_str += ", ";
      }
    }
  
    std::string stride_str = "";
    for (int j = 0; j < attrs->n_stride; j++) {
      stride_str += std::to_string(attrs->stride[j]);
      if (j < attrs->n_stride - 1) {
        stride_str += ", ";
      }
    }
  
    printf("Tensor: name=%s, n_dims=%d, shape=[%s], stride=[%s], aligned_size=%ld, layout=%s, dtype=%s, core_id=%d, "
           "qnt_type=%s\n",
           attrs->name, attrs->n_dims, shape_str.c_str(), stride_str.c_str(), attrs->aligned_size, rknn3_get_layout_string(attrs->layout),
           rknn3_get_type_string(attrs->dtype), attrs->core_id, rknn3_get_qnt_type_string(attrs->qnt_type));
}


int init_sensevoice_small_model(const char *model_path, const char* weight_path, rknn_app_context_t *app_ctx, uint32_t core_mask)
{
    int ret;
    int model_len = 0;
    char *model;
    rknn3_context ctx = 0;

    rknn3_config config;
    memset(&config, 0, sizeof(config));
    // 0xff
    config.run_core_mask = core_mask;

    // RKNN Init
    ret = rknn3_init(&ctx, NULL);
    if (ret < 0)
    {
        printf("rknn_init fail ret=%d\n", ret);
        return ret;
    }

    // Load RKNN Model
    ret = rknn3_load_model_from_path(ctx, model_path, weight_path);
    if (ret < 0)
    {
        printf("load_model fail!\n");
        return -1;
    }

    //Init RKNN Model
    ret = rknn3_model_init(ctx, &config);
    if (ret < 0) {
        printf("rknn_model_init failed! ret=%d\n", ret);
        return ret;
    }

    // Get Model Input Output Number
    rknn3_input_output_num io_num;
    ret = rknn3_query(ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return ret;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);


    // Get Model Input Info
    printf("input tensors:\n");
    rknn3_tensor_attr input_attrs[io_num.n_input];
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return ret;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    printf("output tensors:\n");
    rknn3_tensor_attr output_attrs[io_num.n_output];
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return ret;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // Set to context
    app_ctx->inputs = (rknn3_tensor*)malloc(io_num.n_input * sizeof(rknn3_tensor));
    app_ctx->outputs = (rknn3_tensor*)malloc(io_num.n_output * sizeof(rknn3_tensor));
    app_ctx->rknn_ctx = ctx;
    app_ctx->io_num = io_num;
    for (int i = 0; i < app_ctx->io_num.n_input; i++) {
        app_ctx->inputs[i].mem  = rknn3_create_mem(ctx, input_attrs[i].aligned_size, input_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        app_ctx->inputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        memcpy(app_ctx->inputs[i].attr, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
    }
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        app_ctx->outputs[i].mem  = rknn3_create_mem(ctx, output_attrs[i].aligned_size, output_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        app_ctx->outputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        memcpy(app_ctx->outputs[i].attr, &(output_attrs[i]), sizeof(rknn3_tensor_attr));
    }

    if (output_attrs[0].qnt_type == RKNN3_TENSOR_PER_LAYER_ASYMMETRIC && output_attrs[0].dtype == RKNN3_TENSOR_INT8)
    {
        app_ctx->is_quant = true;
    }
    else
    {
        app_ctx->is_quant = false;
    }

    if (input_attrs[0].layout == RKNN3_TENSOR_NHWC)
    {
        printf("model is NHWC input fmt\n");
        app_ctx->model_channel = input_attrs[0].shape[3];
        app_ctx->model_height = input_attrs[0].shape[1];
        app_ctx->model_width = input_attrs[0].shape[2];
    }
    else
    {
        printf("model is NCHW input fmt\n");
        app_ctx->model_height = input_attrs[0].shape[1];
        app_ctx->model_width = input_attrs[0].shape[2];
        app_ctx->model_channel = input_attrs[0].shape[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n",
           app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);

    return 0;
}

int release_sensevoice_small_model(rknn_app_context_t *app_ctx)
{
    for (int i = 0; i < app_ctx->io_num.n_input; i++) {
        if (app_ctx->inputs[i].mem) {
            rknn3_destroy_mem(app_ctx->rknn_ctx, app_ctx->inputs[i].mem);
        }
        if (app_ctx->inputs[i].attr != NULL) {
            free(app_ctx->inputs[i].attr);
            app_ctx->inputs[i].attr = NULL;
        }
    }
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        if (app_ctx->outputs[i].mem) {
            rknn3_destroy_mem(app_ctx->rknn_ctx, app_ctx->outputs[i].mem);
        }
        if (app_ctx->outputs[i].attr != NULL) {
            free(app_ctx->outputs[i].attr);
            app_ctx->outputs[i].attr = NULL;
        }
    }
    if (app_ctx->rknn_ctx != 0)
    {
        rknn3_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    return 0;
}

int inference_sensevoice_small_model(rknn_app_context_t *app_ctx, float* embedding_matrix, audio_buffer_t audio, VocabEntry *vocabs, std::vector<std::string> &recognized_text)
{
    int ret = 0;

    float* features = NULL;
    int feature_len = 0;
    ret = sense_voice_audio_preprocess(audio.data, audio.num_frames, MODEL_INPUT_SAMPLES, &features, &feature_len);
    if (ret < 0)
    {
        printf("audio_preprocess fail! ret=%d\n", ret);
        return -1;
    }

    int D_EMB = EMBEDDING_FEAT_DIM;
    int T_FEAT = feature_len / D_EMB;
    int T_TOTAL = T_FEAT + 4;

    float* total_feature = (float*)malloc(T_TOTAL * D_EMB * sizeof(float));
    float* dst = total_feature;
    float* language_query = (float*)malloc(1 * EMBEDDING_FEAT_DIM * sizeof(float));
    int language_query_id = DEFAULT_LANGUAGE_ID; 

    memcpy(language_query, embedding_matrix + language_query_id * EMBEDDING_FEAT_DIM,
            EMBEDDING_FEAT_DIM * sizeof(float));
    float* event_emo_query = (float*)malloc(2 * EMBEDDING_FEAT_DIM * sizeof(float));
    memcpy(event_emo_query, embedding_matrix + 1 * EMBEDDING_FEAT_DIM, 2 * EMBEDDING_FEAT_DIM * sizeof(float));
    float* text_norm_query = (float*)malloc(1 * EMBEDDING_FEAT_DIM * sizeof(float));
    memcpy(text_norm_query, embedding_matrix + TEXTNORM_ID * EMBEDDING_FEAT_DIM,
            EMBEDDING_FEAT_DIM * sizeof(float));
    memcpy(dst, language_query, D_EMB * sizeof(float));
    dst += D_EMB;
    memcpy(dst, event_emo_query, 2 * D_EMB * sizeof(float));
    dst += 2 * D_EMB;
    memcpy(dst, text_norm_query, D_EMB * sizeof(float));
    dst += D_EMB;
    memcpy(dst, features, T_FEAT * D_EMB * sizeof(float));
    free(language_query);
    free(event_emo_query);
    free(text_norm_query);

    std::vector<float16> input_x = fp32_array_to_fp16(total_feature, NUM_ELEMENTS);
    std::vector<int8_t> input_x_mask;
    for(int i = 0; i < SHAPE[1]; i++){
        input_x_mask.push_back(1);
    }

    // Set Input Data
    memcpy(app_ctx->inputs[0].mem->virt_addr, input_x.data(), NUM_ELEMENTS*sizeof(float16));
    memcpy(app_ctx->inputs[1].mem->virt_addr, input_x_mask.data(), SHAPE[1]*sizeof(int8_t));

    // Sync input data to device (cacheable memory requires explicit sync)
    for (int i = 0; i < app_ctx->io_num.n_input; i++)
    {
        ret = rknn3_mem_sync(app_ctx->rknn_ctx, app_ctx->inputs[i].mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
        if (ret < 0)
        {
            printf("rknn3_mem_sync input[%d] fail! ret=%d\n", i, ret);
            free(total_feature);
            free(features);
            return ret;
        }
    }

    // Run
    printf("-->rknn_run\n");
    ret = rknn3_run(app_ctx->rknn_ctx, app_ctx->inputs, app_ctx->io_num.n_input, app_ctx->outputs, app_ctx->io_num.n_output);
    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        free(total_feature);
        free(features);
        return ret;
    }

    // Sync output data from device (cacheable memory requires explicit sync)
    for (int i = 0; i < app_ctx->io_num.n_output; i++)
    {
        ret = rknn3_mem_sync(app_ctx->rknn_ctx, app_ctx->outputs[i].mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
        if (ret < 0)
        {
            printf("rknn3_mem_sync output[%d] fail! ret=%d\n", i, ret);
            free(total_feature);
            free(features);
            return ret;
        }
    }

    rknn3_tensor_attr output_attrs;
    output_attrs.index = 0;
    ret = rknn3_query(app_ctx->rknn_ctx, RKNN3_QUERY_OUTPUT_ATTR, &output_attrs, sizeof(rknn3_tensor_attr));
    if (ret < 0)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        free(total_feature);
        free(features);
        return ret;
    }

    float16* out_data = (float16*)app_ctx->outputs[0].mem->virt_addr;
    std::vector<float> out_vec = fp16_array_to_fp32(out_data, OUTPUT_NUM_ELEMENTS);

    int prev_id = -1;
    int blank_id = 0;
    std::vector<int> tokens_int;

    for (int t = 0; t < SHAPE[1]; t++) {
        float* frame_probs = out_vec.data() + t * VOCAB_NUM;
        int y = argmax(frame_probs);
        if (y != blank_id && y != prev_id) {
            tokens_int.push_back(y);
        }
        prev_id = y;
    }

    for (int32_t i = 0; i < tokens_int.size(); ++i) {
        int token_id = tokens_int[i];
        std::string text = vocabs[token_id].token;
        replace_substr(text, "▁", " ");
        recognized_text.push_back(text);
    }

    free(total_feature);
    free(features);

    return ret;
}

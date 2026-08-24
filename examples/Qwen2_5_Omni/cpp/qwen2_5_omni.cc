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

#include "qwen2_5_omni.h"
#include "common.h"
#include "rknn3_api.h"
#include "image_utils.h"
#include "time_utils.h"

int init_internal_share(rknn_app_context_t* app_ctx, uint32_t core_mask_vision, uint32_t core_mask_audio, uint32_t core_mask_llm)
{
    int ret = -1;

    uint32_t core_num_vision = 0;
    uint32_t core_num_audio = 0;
    uint32_t core_num_llm = 0;
    ret = rknn3_query(app_ctx->vision.rknn_ctx, RKNN3_QUERY_CORE_NUMBER, &core_num_vision, sizeof(core_num_vision));
    if (ret < 0) {
        printf("rknn3_query failed! ret=%d\n", ret);
        return ret;
    }
    ret = rknn3_query(app_ctx->audio.rknn_ctx, RKNN3_QUERY_CORE_NUMBER, &core_num_audio, sizeof(core_num_audio));
    if (ret < 0) {
        printf("rknn3_query failed! ret=%d\n", ret);
        return ret;
    }
    ret = rknn3_query(app_ctx->llm.rknn_ctx, RKNN3_QUERY_CORE_NUMBER, &core_num_llm, sizeof(core_num_llm));
    if (ret < 0) {
        printf("rknn3_query failed! ret=%d\n", ret);
        return ret;
    }

    uint32_t core_num_vision_ = 0;
    uint32_t core_num_audio_ = 0;
    uint32_t core_num_llm_ = 0;
    for (int i = 0; i < 32; i++) {
        if (core_mask_vision & (1 << i))    core_num_vision_++;
    }
    for (int i = 0; i < 32; i++) {
        if (core_mask_audio & (1 << i))    core_num_audio_++;
    }
    for (int i = 0; i < 32; i++) {
        if (core_mask_llm & (1 << i))    core_num_llm_++;
    }
    if (core_num_vision_ != core_num_vision) {
        printf("the core_mask_vision = %x is not match the core_num_vision = %d!\n", core_mask_vision, core_num_vision);
        return -1;
    }
    if (core_num_audio_ != core_num_audio) {
        printf("the core_mask_audio = %x is not match the core_num_audio = %d!\n", core_mask_audio, core_num_audio);
        return -1;
    }
    if (core_num_llm_ != core_num_llm) {
        printf("the core_mask_llm = %x is not match the core_num_llm = %d!\n", core_mask_llm, core_num_llm);
        return -1;
    }

    rknn3_core_mem_size* core_mem_sizes_vision = (rknn3_core_mem_size*)malloc(sizeof(rknn3_core_mem_size) * core_num_vision);
    if (!core_mem_sizes_vision) {
        printf("Failed to allocate memory for core_mem_sizes_vision\n");
        return ret;
    }
    rknn3_core_mem_size* core_mem_sizes_audio = (rknn3_core_mem_size*)malloc(sizeof(rknn3_core_mem_size) * core_num_audio);
    if (!core_mem_sizes_audio) {
        printf("Failed to allocate memory for core_mem_sizes_audio\n");
        return ret;
    }
    rknn3_core_mem_size* core_mem_sizes_llm = (rknn3_core_mem_size*)malloc(sizeof(rknn3_core_mem_size) * core_num_llm);
    if (!core_mem_sizes_llm) {
        printf("Failed to allocate memory for core_mem_sizes_llm\n");
        return ret;
    }
    ret = rknn3_query(app_ctx->vision.rknn_ctx, RKNN3_QUERY_CORE_MEM_SIZE, core_mem_sizes_vision, sizeof(rknn3_core_mem_size) * core_num_vision);
    if (ret < 0) {
        printf("rknn3_query core memory size failed! ret=%d\n", ret);
        return ret;
    }
    ret = rknn3_query(app_ctx->audio.rknn_ctx, RKNN3_QUERY_CORE_MEM_SIZE, core_mem_sizes_audio, sizeof(rknn3_core_mem_size) * core_num_audio);
    if (ret < 0) {
        printf("rknn3_query core memory size failed! ret=%d\n", ret);
        return ret;
    }
    ret = rknn3_query(app_ctx->llm.rknn_ctx, RKNN3_QUERY_CORE_MEM_SIZE, core_mem_sizes_llm, sizeof(rknn3_core_mem_size) * core_num_llm);
    if (ret < 0) {
        printf("rknn3_query core memory size failed! ret=%d\n", ret);
        return ret;
    }

    rknn3_core_mem_size* core_mem_sizes_max = NULL;
    if (core_num_vision >= core_num_audio) {
        for (int i = 0; i < core_num_audio; i++) {
            core_mem_sizes_vision[i].internal_size = std::max(core_mem_sizes_vision[i].internal_size, core_mem_sizes_audio[i].internal_size);
        }
        core_mem_sizes_max = core_mem_sizes_vision;
    } else {
        for (int i = 0; i < core_num_vision; i++) {
            core_mem_sizes_audio[i].internal_size = std::max(core_mem_sizes_audio[i].internal_size, core_mem_sizes_vision[i].internal_size);
        }
        core_mem_sizes_max = core_mem_sizes_audio;
    }
    uint32_t core_num_max = std::max(core_num_vision, core_num_audio);

    if (core_num_max >= core_num_llm) {
        for (int i = 0; i < core_num_llm; i++) {
            core_mem_sizes_max[i].internal_size = std::max(core_mem_sizes_max[i].internal_size, core_mem_sizes_llm[i].internal_size);
        }
    } else {
        for (int i = 0; i < core_num_max; i++) {
            core_mem_sizes_llm[i].internal_size = std::max(core_mem_sizes_llm[i].internal_size, core_mem_sizes_max[i].internal_size);
        }
        core_mem_sizes_max = core_mem_sizes_llm;
        core_num_max = core_num_llm;
    }

    app_ctx->internal_mems = (rknn3_tensor_mem**)malloc(sizeof(rknn3_tensor_mem*) * core_num_max);
    if (!app_ctx->internal_mems) {
        printf("Failed to allocate memory for app_ctx->internal_mems array\n");
        return -1;
    }
    app_ctx->n_internal_mems = core_num_max;

    for (uint32_t i = 0; i < core_num_max; i++) {
        if (core_mem_sizes_max[i].internal_size > 0) {
            app_ctx->internal_mems[i] = rknn3_create_mem(app_ctx->llm.rknn_ctx, core_mem_sizes_max[i].internal_size, core_mem_sizes_max[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
            if (!app_ctx->internal_mems[i]) {
                return -1;
            }
        }
    }

    free(core_mem_sizes_vision);
    free(core_mem_sizes_audio);
    free(core_mem_sizes_llm);

    ret = rknn3_set_internal_mem(app_ctx->vision.rknn_ctx, app_ctx->internal_mems, core_num_vision);
    if (ret < 0) {
        printf("rknn3_set_internal_mem failed! ret=%d\n", ret);
        return ret;
    }
    ret = rknn3_set_internal_mem(app_ctx->audio.rknn_ctx, app_ctx->internal_mems, core_num_audio);
    if (ret < 0) {
        printf("rknn3_set_internal_mem failed! ret=%d\n", ret);
        return ret;
    }
    ret = rknn3_set_internal_mem(app_ctx->llm.rknn_ctx, app_ctx->internal_mems, core_num_llm);
    if (ret < 0) {
        printf("rknn3_set_internal_mem failed! ret=%d\n", ret);
        return ret;
    }

    return ret;
}

int release_internal_share(rknn_app_context_t* app_ctx)
{
    if (app_ctx->internal_mems) {
        for (int i = 0; i < app_ctx->n_internal_mems; i++) {
            if (app_ctx->internal_mems[i]) {
                rknn3_destroy_mem(app_ctx->llm.rknn_ctx, app_ctx->internal_mems[i]);
                app_ctx->internal_mems[i] = NULL;
            }
        }
        free(app_ctx->internal_mems);
        app_ctx->internal_mems = NULL;
        app_ctx->n_internal_mems = 0;
    }
    return 0;
}

int init_qwen2_5_omni_model(rknn_app_context_t* app_ctx, const char* llm_model_path, const char* llm_weight_path,
                    const char* vision_model_path, const char* vision_weight_path, const char* audio_model_path, const char* audio_weight_path,
                    rknn3_llm_param* params, int n_params, RKLLMCallback& callback, uint32_t llm_core_mask, uint32_t vision_core_mask, uint32_t audio_core_mask)
{
    int ret = 0;

    printf("--> init qwen2_5_omni vision model\n");
    ret = init_qwen2_5_omni_vision(&(app_ctx->vision), vision_model_path, vision_weight_path, vision_core_mask);
    if (ret < 0)
    {
        printf("rknn_init qwen2_5_omni vision model fail! ret=%d\n", ret);
        return ret;
    }

    printf("--> init qwen2_5_omni audio model\n");
    ret = init_qwen2_5_omni_audio(&(app_ctx->audio), audio_model_path, audio_weight_path, audio_core_mask);
    if (ret < 0)
    {
        printf("rknn_init qwen2_5_omni audio model fail! ret=%d\n", ret);
        return ret;
    }

    printf("--> init qwen2_5_omni llm model\n");
    ret = init_qwen2_5_omni_llm(&(app_ctx->llm), llm_model_path, llm_weight_path, params, n_params, callback, llm_core_mask);
    if (ret < 0)
    {
        printf("rknn_init qwen2_5_omni llm model fail! ret=%d\n", ret);
        return ret;
    }

    printf("--> init internal share\n");
    ret = init_internal_share(app_ctx, vision_core_mask, audio_core_mask, llm_core_mask);
    if (ret < 0)
    {
        printf("qwen2_5_omni llm/vision/audio internal memory share fail! ret=%d\n", ret);
        return ret;
    }

    return ret;
}

int release_qwen2_5_omni_model(rknn_app_context_t* app_ctx)
{
    release_internal_share(app_ctx);
    release_qwen2_5_omni_vision(&(app_ctx->vision));
    release_qwen2_5_omni_audio(&(app_ctx->audio));
    release_qwen2_5_omni_llm(&(app_ctx->llm));
    return 0;
}

int inference_qwen2_5_omni_model(rknn_app_context_t* app_ctx, image_buffer_t* img, float16* img_embeds, audio_buffer_t* audio, float16* audio_embeds,
                            rknn3_llm_multimodal_tensor tensor, int n_inputs, rknn_perf_metrics_t* perf)
{
    int ret;

    if (!app_ctx) {
        printf("app_ctx is NULL");
        return -1;
    }

    if (tensor.image.n_image) {
        printf("--> inference qwen2_5_omni vision model\n");
        int start_us = getCurrentTimeUs();
        ret = inference_qwen2_5_omni_vision(&(app_ctx->vision), img, img_embeds);
        perf->vision_latency = getCurrentTimeUs()-start_us;  
        if (ret != 0)
        {
            printf("inference qwen2_5_omni vision model fail! ret=%d\n", ret);
            return ret;
        }
    }

    if (tensor.audio.n_audio) {
        printf("--> inference qwen2_5_omni audio model\n");
        int start_us = getCurrentTimeUs();
        ret = inference_qwen2_5_omni_audio(&(app_ctx->audio), audio, audio_embeds);
        perf->audio_latency = getCurrentTimeUs()-start_us;  
        if (ret != 0)
        {
            printf("inference qwen2_5_omni audio model fail! ret=%d\n", ret);
            return ret;
        }
    }

    printf("--> inference qwen2_5_omni llm model\n");  
    ret = inference_qwen2_5_omni_llm(&(app_ctx->llm), tensor, n_inputs, perf);
    if (ret != 0)
    {
        printf("inference qwen2_5_omni llm model fail! ret=%d\n", ret);
        return ret;
    }

    return ret;
}
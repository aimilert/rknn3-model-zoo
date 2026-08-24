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
#include <algorithm>

#include "qwen3_asr.h"
#include "common.h"
#include "rknn3_api.h"
#include "time_utils.h"

int init_internal_share(rknn_app_context_t* app_ctx, uint32_t core_mask_audio, uint32_t core_mask_llm)
{
    int ret = -1;

    uint32_t core_num_audio = 0;
    uint32_t core_num_llm = 0;
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

    uint32_t core_num_audio_ = 0;
    uint32_t core_num_llm_ = 0;
    for (int i = 0; i < 32; i++) {
        if (core_mask_audio & (1 << i))    core_num_audio_++;
    }
    for (int i = 0; i < 32; i++) {
        if (core_mask_llm & (1 << i))    core_num_llm_++;
    }
    if (core_num_audio_ != core_num_audio) {
        printf("the core_mask_audio = %x is not match the core_num_audio = %d!\n", core_mask_audio, core_num_audio);
        return -1;
    }
    if (core_num_llm_ != core_num_llm) {
        printf("the core_mask_llm = %x is not match the core_num_llm = %d!\n", core_mask_llm, core_num_llm);
        return -1;
    }

    rknn3_core_mem_size* core_mem_sizes_audio = (rknn3_core_mem_size*)malloc(sizeof(rknn3_core_mem_size) * core_num_audio);
    if (!core_mem_sizes_audio) {
        printf("Failed to allocate memory for core_mem_sizes_audio\n");
        return ret;
    }
    rknn3_core_mem_size* core_mem_sizes_llm = (rknn3_core_mem_size*)malloc(sizeof(rknn3_core_mem_size) * core_num_llm);
    if (!core_mem_sizes_llm) {
        printf("Failed to allocate memory for core_mem_sizes_llm\n");
        free(core_mem_sizes_audio);
        return ret;
    }
    ret = rknn3_query(app_ctx->audio.rknn_ctx, RKNN3_QUERY_CORE_MEM_SIZE, core_mem_sizes_audio, sizeof(rknn3_core_mem_size) * core_num_audio);
    if (ret < 0) {
        printf("rknn3_query core memory size failed! ret=%d\n", ret);
        free(core_mem_sizes_audio);
        free(core_mem_sizes_llm);
        return ret;
    }
    ret = rknn3_query(app_ctx->llm.rknn_ctx, RKNN3_QUERY_CORE_MEM_SIZE, core_mem_sizes_llm, sizeof(rknn3_core_mem_size) * core_num_llm);
    if (ret < 0) {
        printf("rknn3_query core memory size failed! ret=%d\n", ret);
        free(core_mem_sizes_audio);
        free(core_mem_sizes_llm);
        return ret;
    }

    rknn3_core_mem_size* core_mem_sizes_max = core_mem_sizes_audio;
    uint32_t core_num_max = core_num_audio;

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
        free(core_mem_sizes_audio);
        free(core_mem_sizes_llm);
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

    free(core_mem_sizes_audio);
    free(core_mem_sizes_llm);

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

int init_qwen3_asr_model(rknn_app_context_t* app_ctx, const char* llm_model_path, const char* llm_weight_path,
                    const char* audio_model_path, const char* audio_weight_path,
                    rknn3_llm_param* params, int n_params, RKLLMCallback& callback, uint32_t llm_core_mask, uint32_t audio_core_mask)
{
    int ret = 0;

    printf("--> init qwen3_asr audio model\n");
    ret = init_qwen3_asr_audio(&(app_ctx->audio), audio_model_path, audio_weight_path, audio_core_mask);
    if (ret < 0)
    {
        printf("rknn_init qwen3_asr audio model fail! ret=%d\n", ret);
        return ret;
    }

    printf("--> init qwen3_asr llm model\n");
    ret = init_qwen3_asr_llm(&(app_ctx->llm), llm_model_path, llm_weight_path, params, n_params, callback, llm_core_mask);
    if (ret < 0)
    {
        printf("rknn_init qwen3_asr llm model fail! ret=%d\n", ret);
        return ret;
    }

    printf("--> init internal share\n");
    ret = init_internal_share(app_ctx, audio_core_mask, llm_core_mask);
    if (ret < 0)
    {
        printf("qwen3_asr llm/audio internal memory share fail! ret=%d\n", ret);
        return ret;
    }

    Tokenizer* tokenizer = (Tokenizer*)callback.tokenizer_userdata;
    struct embedding_info *embedding_info = (struct embedding_info *)callback.embed_userdata;

    const char *prefix_text = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n";
    int32_t prefix_tokens[PREFIX_N_TOKENS];
    tokenizer->Tokenize(prefix_text, strlen(prefix_text), prefix_tokens, PREFIX_N_TOKENS);
    prefix_tokens[PREFIX_N_TOKENS-1] = AUDIO_START_TOKEN;

    int32_t postfix_tokens[POSTFIX_N_TOKENS];
    const char *postfix_text = "<|im_end|>\n<|im_start|>assistant\n";
    tokenizer->Tokenize(postfix_text, strlen(postfix_text), postfix_tokens+1, POSTFIX_N_TOKENS);
    postfix_tokens[0] = AUDIO_END_TOKEN;

    app_ctx->prefix_embeds = (float16*)malloc(PREFIX_N_TOKENS * app_ctx->audio.embeds_shape[1] * sizeof(float16));
    for (int i = 0; i < PREFIX_N_TOKENS; i++) {
        memcpy(app_ctx->prefix_embeds + i * app_ctx->audio.embeds_shape[1], embedding_info->embedding_data + prefix_tokens[i] * embedding_info->embedding_dim, embedding_info->embedding_dim * sizeof(float16));
    }
    printf("prefix_embeds[0]=%f prefix_embeds[1]=%f prefix_embeds[2]=%f prefix_embeds[-1]=%f \n", fp16_to_fp32(app_ctx->prefix_embeds[0]), fp16_to_fp32(app_ctx->prefix_embeds[1]), fp16_to_fp32(app_ctx->prefix_embeds[2]), fp16_to_fp32(app_ctx->prefix_embeds[PREFIX_N_TOKENS * app_ctx->audio.embeds_shape[1]-1]));

    app_ctx->postfix_embeds = (float16*)malloc(POSTFIX_N_TOKENS * app_ctx->audio.embeds_shape[1] * sizeof(float16));
    for (int i = 0; i < POSTFIX_N_TOKENS; i++) {
        memcpy(app_ctx->postfix_embeds + i * app_ctx->audio.embeds_shape[1], embedding_info->embedding_data + postfix_tokens[i] * embedding_info->embedding_dim, embedding_info->embedding_dim * sizeof(float16));
    }
    printf("postfix_embeds[0]=%f postfix_embeds[1]=%f postfix_embeds[2]=%f postfix_embeds[-1]=%f \n", fp16_to_fp32(app_ctx->postfix_embeds[0]), fp16_to_fp32(app_ctx->postfix_embeds[1]), fp16_to_fp32(app_ctx->postfix_embeds[2]), fp16_to_fp32(app_ctx->postfix_embeds[POSTFIX_N_TOKENS * app_ctx->audio.embeds_shape[1]-1]));

    return ret;
}

int release_qwen3_asr_model(rknn_app_context_t* app_ctx)
{
    release_internal_share(app_ctx);
    release_qwen3_asr_audio(&(app_ctx->audio));
    release_qwen3_asr_llm(&(app_ctx->llm));
    free(app_ctx->prefix_embeds);
    free(app_ctx->postfix_embeds);
    return 0;
}

int inference_qwen3_asr_model(rknn_app_context_t* app_ctx, audio_buffer_t* audio, float16* audio_embeds, size_t n_audio_tokens,
                            int n_inputs, rknn_perf_metrics_t* perf)
{
    int ret;

    if (!app_ctx) {
        printf("app_ctx is NULL\n");
        return -1;
    }

    printf("--> inference qwen3_asr audio model\n");
    int start_us = getCurrentTimeUs();
    ret = inference_qwen3_asr_audio(&(app_ctx->audio), audio, audio_embeds);
    perf->audio_latency = getCurrentTimeUs()-start_us;  
    if (ret != 0)
    {
        printf("inference qwen3_asr audio model fail! ret=%d\n", ret);
        return ret;
    }

    float16 *embed = nullptr;
    size_t n_tokens = n_audio_tokens+PREFIX_N_TOKENS+POSTFIX_N_TOKENS;
    embed = (float16 *)malloc(n_tokens * app_ctx->audio.embeds_shape[1] * sizeof(float16));
    if (!embed)
    {
      printf("malloc embed failed\n");
      return -1;
    }
    memcpy(embed, app_ctx->prefix_embeds, PREFIX_N_TOKENS * app_ctx->audio.embeds_shape[1] * sizeof(float16));
    memcpy(embed+PREFIX_N_TOKENS*app_ctx->audio.embeds_shape[1], audio_embeds, n_audio_tokens * app_ctx->audio.embeds_shape[1] * sizeof(float16));
    memcpy(embed+(PREFIX_N_TOKENS+n_audio_tokens)*app_ctx->audio.embeds_shape[1], app_ctx->postfix_embeds, POSTFIX_N_TOKENS * app_ctx->audio.embeds_shape[1] * sizeof(float16));

    rknn3_llm_tensor tensor;
    memset(&tensor, 0, sizeof(rknn3_llm_tensor));
    tensor = {.name = NULL, .prompt = NULL, .embed = embed, .tokens = NULL, .n_tokens = n_tokens, .enable_thinking = false};

    printf("--> inference qwen3_asr llm model\n");  
    ret = inference_qwen3_asr_llm(&(app_ctx->llm), tensor, n_inputs, perf);
    if (ret != 0)
    {
        printf("inference qwen3_asr llm model fail! ret=%d\n", ret);
        free(embed);
        return ret;
    }
    free(embed);

    return ret;
}

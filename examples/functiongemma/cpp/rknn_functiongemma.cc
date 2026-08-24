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

#include "rknn_functiongemma.h"
#include "common.h"
#include "nlohmann/json.hpp"

#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

int init_functiongemma_model(rknn_functiongemma_app_context* app_ctx,
                      const char* llm_model_path, const char* llm_weight_path,
                      rknn3_llm_param* params, int n_params,
                      RKLLMCallback& callback, uint32_t llm_core_mask,
                      const char* safetensors_path,
                      Tokenizer* tokenizer, embedding_info* token_embedding,
                      input_cb_userdata* input_cb_data)
{
    (void)safetensors_path;
    (void)tokenizer;
    (void)token_embedding;
    (void)input_cb_data;

    int ret = 0;

    printf("--> init functiongemma llm model\n");
    ret = init_functiongemma_llm(&app_ctx->llm, llm_model_path, llm_weight_path, params, n_params, &callback, llm_core_mask);
    if (ret < 0) {
        printf("init_functiongemma_llm failed! ret=%d\n", ret);
        return ret;
    }

    return ret;
}


int release_functiongemma_model(rknn_functiongemma_app_context* app_ctx)
{
    release_functiongemma_llm(&app_ctx->llm);
    return 0;
}


int inference_functiongemma_model(rknn_functiongemma_app_context* app_ctx,
                           rknn3_llm_multimodal_tensor tensor,
                           int32_t max_new_tokens,
                           rknn_perf_metrics_t* perf)
{
    int ret = 0;

    printf("--> inference functiongemma llm model\n");
    ret = inference_functiongemma_llm(&app_ctx->llm, tensor, max_new_tokens, perf);
    if (ret != 0) {
        printf("inference_functiongemma_llm failed! ret=%d\n", ret);
        return ret;
    }

    return ret;
}

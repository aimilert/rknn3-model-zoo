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


#ifndef _RKNN_DEMO_QWEN3_ASR_H_
#define _RKNN_DEMO_QWEN3_ASR_H_

#include "rknn3_api.h"
#include "common.h"
#include "rknn_qwen3_asr_audio.h"
#include "rknn_qwen3_asr_llm.h"
#include "time_utils.h"

#define PREFIX_N_TOKENS 9
#define POSTFIX_N_TOKENS 6
#define AUDIO_START_TOKEN 151669
#define AUDIO_END_TOKEN 151670

// llm
extern const rknn3_sampling_params SAMPLE_PARAMS;

struct embedding_info
{
  int      fd;
  float16* embedding_data;
  int      embedding_dim;
  int      vocab_size;
};

typedef struct {
    rknn_qwen3_asr_llm_context llm;
    rknn_qwen3_asr_audio_context audio;
    int n_internal_mems;
    rknn3_tensor_mem** internal_mems;
    float16* prefix_embeds;
    float16* postfix_embeds;
} rknn_app_context_t;

int init_qwen3_asr_model(rknn_app_context_t* app_ctx, const char* llm_model_path, const char* llm_weight_path, 
                        const char* audio_model_path, const char* audio_weight_path,
                        rknn3_llm_param* params, int n_params, RKLLMCallback& callback, uint32_t llm_core_mask, uint32_t audio_core_mask);

int release_qwen3_asr_model(rknn_app_context_t* app_ctx);

int inference_qwen3_asr_model(rknn_app_context_t* app_ctx, audio_buffer_t* audio, float16* audio_embeds, size_t n_audio_tokens,
                             int n_inputs, rknn_perf_metrics_t* perf);

#endif //_RKNN_DEMO_QWEN3_ASR_H_
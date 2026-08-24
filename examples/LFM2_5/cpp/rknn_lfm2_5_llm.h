// Copyright (c) 2026 by Rockchip Electronics Co., Ltd. All Rights Reserved.
// Licensed under the Apache License, Version 2.0

#ifndef _RKNN_DEMO_LFM2_LLM_UTILS_H_
#define _RKNN_DEMO_LFM2_LLM_UTILS_H_

#include "rknn3_api.h"
#include "Tokenizer.h"
#include "common.h"

#define LFM2_MAX_CONTEXT_LEN      4096
#define LFM2_MAX_NEW_TOKENS       2048
#define LFM2_DEFAULT_NEW_TOKENS   1024

typedef struct
{
    rknn3_context rknn_ctx;
    rknn3_session *rknn_sess;
    bool initialized;
} rknn_lfm2_llm_context;

int init_lfm2_llm(rknn_lfm2_llm_context *ctx,
                  const char *model_path,
                  const char *weight_path,
                  rknn3_llm_param *params,
                  int n_params,
                  RKLLMCallback &callback,
                  uint32_t core_mask);

int inference_lfm2_llm(rknn_lfm2_llm_context *ctx,
                       rknn3_llm_input *inputs,
                       uint32_t n_inputs,
                       uint32_t max_new_tokens,
                       bool keep_history,
                       rknn_perf_metrics_t *perf);

int clear_lfm2_cache(rknn_lfm2_llm_context *ctx);

int release_lfm2_llm(rknn_lfm2_llm_context *ctx);

#endif // _RKNN_DEMO_LFM2_LLM_UTILS_H_

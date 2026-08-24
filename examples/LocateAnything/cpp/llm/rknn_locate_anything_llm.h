// Copyright (c) 2025 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");

#ifndef _RKNN_DEMO_LOCATE_ANYTHING_LLM_H_
#define _RKNN_DEMO_LOCATE_ANYTHING_LLM_H_

#include "Tokenizer.h"
#include "common.h"
#include "rknn3_api.h"

extern const char* system_prompt;
extern const char* prompt_prefix;
extern const char* prompt_postfix;

#define MAX_NEW_TOKENS 1024
#define MAX_CONTEXT_LEN 1024

typedef struct {
    rknn3_context rknn_ctx;
    rknn3_session* rknn_sess;
    rknn3_llm_config llm_config;
} rknn_locate_anything_llm_context;

int init_locate_anything_llm(rknn_locate_anything_llm_context* llm_ctx,
                             const char* model_path,
                             const char* weight_path,
                             rknn3_llm_param* params,
                             int n_params,
                             RKLLMCallback& callback,
                             uint32_t core_mask);

int release_locate_anything_llm(rknn_locate_anything_llm_context* llm_ctx);

int inference_locate_anything_llm(rknn_locate_anything_llm_context* llm_ctx,
                                  rknn3_llm_multimodal_tensor tensor,
                                  int n_inputs,
                                  rknn_perf_metrics_t* perf);

#endif  // _RKNN_DEMO_LOCATE_ANYTHING_LLM_H_

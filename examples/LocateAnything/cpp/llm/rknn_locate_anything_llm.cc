// Copyright (c) 2025 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rknn_locate_anything_llm.h"
#include "time_utils.h"

static rknn3_init_extend make_rknn3_init_extend(const char* tag)
{
    static char selected_device_id[RKNN3_MAX_DEV_LEN] = {0};
    rknn3_init_extend init_extend;
    memset(&init_extend, 0, sizeof(init_extend));

    const char* env_device_id = getenv("RKNN3_DEVICE_ID");
    if (env_device_id && env_device_id[0] != '\0') {
        snprintf(selected_device_id, sizeof(selected_device_id), "%s", env_device_id);
        init_extend.device_id = selected_device_id;
        printf("[%s] use RKNN3_DEVICE_ID=%s\n", tag, selected_device_id);
        return init_extend;
    }

    rknn3_devices devices;
    memset(&devices, 0, sizeof(devices));
    int ret = rknn3_find_devices(&devices);
    if (ret == 0 && devices.n_devices > 1) {
        printf("[%s] multiple RKNN3 devices found, using device_id=%s\n", tag, devices.devices[0].id);
        snprintf(selected_device_id, sizeof(selected_device_id), "%s", devices.devices[0].id);
        init_extend.device_id = selected_device_id;
    }
    return init_extend;
}

int init_locate_anything_llm(rknn_locate_anything_llm_context* llm_ctx,
                             const char* model_path,
                             const char* weight_path,
                             rknn3_llm_param* params,
                             int n_params,
                             RKLLMCallback& callback,
                             uint32_t core_mask)
{
    int ret = 0;
    rknn3_context ctx = 0;
    rknn3_session* session = NULL;

    rknn3_config config;
    memset(&config, 0, sizeof(config));
    config.run_core_mask = core_mask;
    config.user_mem_internal = 1;

    rknn3_init_extend init_extend = make_rknn3_init_extend("LLMInit");
    ret = rknn3_init(&ctx, init_extend.device_id ? &init_extend : NULL);
    if (ret < 0) {
        printf("rknn3_init llm failed! ret=%d\n", ret);
        return ret;
    }

    ret = rknn3_load_model_from_path(ctx, model_path, weight_path);
    if (ret < 0) {
        printf("rknn3_load_model_from_path llm failed! ret=%d\n", ret);
        goto out_destroy;
    }

    ret = rknn3_model_init(ctx, &config);
    if (ret < 0) {
        printf("rknn3_model_init llm failed! ret=%d\n", ret);
        goto out_destroy;
    }

    ret = rknn3_query(ctx, RKNN3_QUERY_LLM_CONFIG, &llm_ctx->llm_config, sizeof(llm_ctx->llm_config));
    if (ret < 0) {
        printf("rknn3_query RKNN3_QUERY_LLM_CONFIG failed! ret=%d\n", ret);
        goto out_destroy;
    }

    if (params[0].max_context_len <= 0) {
        params[0].max_context_len = llm_ctx->llm_config.max_ctx_len;
    }

    session = rknn3_session_init(ctx, params, n_params);
    if (!session) {
        printf("rknn3_session_init llm failed\n");
        ret = -1;
        goto out_destroy;
    }

    ret = rknn3_session_set_chat_template(session, system_prompt, prompt_prefix, prompt_postfix);
    if (ret < 0) {
        printf("rknn3_session_set_chat_template failed! ret=%d\n", ret);
        goto out_destroy_session;
    }

    ret = rknn3_session_set_callback(session, &callback);
    if (ret < 0) {
        printf("rknn3_session_set_callback failed! ret=%d\n", ret);
        goto out_destroy_session;
    }

    llm_ctx->rknn_ctx = ctx;
    llm_ctx->rknn_sess = session;
    return ret;

out_destroy_session:
    rknn3_session_destroy(session);
out_destroy:
    rknn3_destroy(ctx);
    return ret;
}

int release_locate_anything_llm(rknn_locate_anything_llm_context* llm_ctx)
{
    if (llm_ctx->rknn_sess) {
        rknn3_session_destroy(llm_ctx->rknn_sess);
        llm_ctx->rknn_sess = NULL;
    }
    if (llm_ctx->rknn_ctx != 0) {
        rknn3_destroy(llm_ctx->rknn_ctx);
        llm_ctx->rknn_ctx = 0;
    }
    return 0;
}

int inference_locate_anything_llm(rknn_locate_anything_llm_context* llm_ctx,
                                  rknn3_llm_multimodal_tensor tensor,
                                  int n_inputs,
                                  rknn_perf_metrics_t* perf)
{
    if (!llm_ctx || !llm_ctx->rknn_sess) {
        printf("llm_ctx or rknn_sess is NULL\n");
        return -1;
    }

    rknn3_llm_input inputs[n_inputs];
    rknn3_llm_infer_param llm_infer_param;
    memset(inputs, 0, sizeof(inputs));
    memset(&llm_infer_param, 0, sizeof(llm_infer_param));

    llm_infer_param.keep_history = 0;
    llm_infer_param.max_new_tokens = MAX_NEW_TOKENS;

    inputs[0].input_type = RKNN3_LLM_INPUT_MULTIMODAL;
    inputs[0].multimodal_input = tensor;

    printf("rknn3_session_run\n");
    perf->llm_start_time = getCurrentTimeUs();
    int ret = rknn3_session_run(llm_ctx->rknn_sess, inputs, n_inputs, &llm_infer_param);
    perf->llm_end_time = getCurrentTimeUs();
    if (ret < 0) {
        printf("rknn3_session_run failed! ret=%d\n", ret);
        return ret;
    }

    RKLLMRunState state = {0};
    ret = rknn3_session_query_state(llm_ctx->rknn_sess, &state);
    if (ret < 0) {
        printf("rknn3_session_query_state failed! ret=%d\n", ret);
        return ret;
    }
    perf->n_decode_tokens = state.n_decode_tokens;
    perf->n_prefill_tokens = state.n_prefill_tokens;
    return ret;
}

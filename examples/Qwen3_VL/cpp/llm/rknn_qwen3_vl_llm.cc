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
#include <string>
#include "time_utils.h"

#include "rknn_qwen3_vl_llm.h"

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
    if (ret == 0) {
        if (devices.n_devices > 1) {
            printf("[%s] multiple RKNN3 devices found, using device_id=%s\n",
                   tag, devices.devices[0].id);
            snprintf(selected_device_id, sizeof(selected_device_id), "%s", devices.devices[0].id);
            init_extend.device_id = selected_device_id;
        }
    }
    return init_extend;
}

int init_qwen3_vl_llm(rknn_qwen3_vl_llm_context* llm_ctx, const char* model_path, const char* weight_path, rknn3_llm_param* params, int n_params, RKLLMCallback& callback, uint32_t core_mask, int* deepstack_aligned_size)
{
    int ret;
    rknn3_context  ctx     = 0;
    rknn3_session* session = NULL;

    rknn3_config config;
    memset(&config, 0, sizeof(config));
    config.run_core_mask = core_mask;
    config.user_mem_internal = 1; // 使用用户管理的internal内存
    rknn3_tensor_attr deepstack_attrs[3];

    // RKNN Init
    rknn3_init_extend init_extend = make_rknn3_init_extend("LLMInit");
    ret = rknn3_init(&ctx, init_extend.device_id ? &init_extend : NULL);
    if (ret < 0)
    {
        printf("rknn_init fail ret=%d\n", ret);
        return ret;
    }

    // Load RKNN Model
    ret = rknn3_load_model_from_path(ctx, model_path, weight_path);
    if (ret < 0) {
        printf("rknn_load_model failed! ret=%d\n", ret);
        return ret;
    }

    //Init RKNN Model
    ret = rknn3_model_init(ctx, &config);
    if (ret < 0) {
        printf("rknn_model_init failed! ret=%d\n", ret);
        return ret;
    }

    // RKNN Session Init
    session = rknn3_session_init(ctx, params, n_params);
    if (!session)
    {
        printf("Failed to initialize test session\n");
        return -1;
    }

    // Set Chat Template
    ret = rknn3_session_set_chat_template(session, system_prompt, prompt_prefix, prompt_postfix);
    if (ret < 0)
    {
        printf("Failed to set chat template\n");
        return -1;
    }

    // Set Callback
    ret = rknn3_session_set_callback(session, &(callback));
    if (ret < 0)
    {
        printf("Failed to set callback\n");
        return -1;
    }

    for(int i = 0; i < 3; i++) {
        deepstack_attrs[i].index = 2 + i; //deepstack的index为2、3、 4,可以通过rknn3_query查询所有input_attrs定位到deepstack的index
        ret = rknn3_query(ctx, RKNN3_QUERY_INPUT_ATTR, &(deepstack_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret < 0) {
          printf("rknn_query fail! ret=%d\n", ret);
          return -1;
        }

        llm_ctx->deepstack_tensor[i].mem = rknn3_create_mem(ctx, *deepstack_aligned_size, deepstack_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (!llm_ctx->deepstack_tensor[i].mem) {
          printf("fail to create aux_input_tensor.mem!\n");
          return -1;
        }
    }

    // Set to context
    llm_ctx->rknn_ctx = ctx;
    llm_ctx->rknn_sess = session;

    return ret;
}

int release_qwen3_vl_llm(rknn_qwen3_vl_llm_context* llm_ctx)
{
    for(int i = 0; i < 3; i++) {
        if (llm_ctx->deepstack_tensor[i].mem) {
          rknn3_destroy_mem(llm_ctx->rknn_ctx, llm_ctx->deepstack_tensor[i].mem);
          llm_ctx->deepstack_tensor[i].mem = nullptr;
        }
    }

    if (llm_ctx->rknn_sess) {
        rknn3_session_destroy(llm_ctx->rknn_sess);
        llm_ctx->rknn_sess = NULL;
    }

    if (llm_ctx->rknn_ctx != 0)
    {
        rknn3_destroy(llm_ctx->rknn_ctx);
        llm_ctx->rknn_ctx = 0;
    }
    return 0;
}

int inference_qwen3_vl_llm(rknn_qwen3_vl_llm_context* llm_ctx, rknn3_llm_multimodal_tensor tensor, int n_inputs, rknn_perf_metrics_t* perf)
{
    if ((!llm_ctx) || !(llm_ctx->rknn_sess))
    {
        printf("llm_ctx or rknn_session is NULL");
        return -1;
    }
    
    int ret;
    rknn3_llm_input inputs[n_inputs];
    rknn3_llm_infer_param llm_infer_param;

    memset(inputs, 0, sizeof(inputs));
    memset(&(llm_infer_param), 0, sizeof(llm_infer_param));

    llm_infer_param.keep_history = 0;
    llm_infer_param.max_new_tokens = MAX_NEW_TOKENS;


    // Set Input Data
    inputs[0].input_type = RKNN3_LLM_INPUT_MULTIMODAL;
    inputs[0].multimodal_input = tensor;
    for (int i = 0; i < 3; ++i)
    {
        inputs[i + 1].input_type = RKNN3_LLM_INPUT_AUX;
        inputs[i + 1].aux_input  = llm_ctx->deepstack_tensor[i];
    }
    
    // Run
    printf("rknn_session_run\n");
    perf->llm_start_time = getCurrentTimeUs();
    ret = rknn3_session_run(llm_ctx->rknn_sess, inputs, n_inputs, &llm_infer_param);
    perf->llm_end_time = getCurrentTimeUs();
    if (ret < 0)
    {
        printf("rknn_session_run fail! ret=%d\n", ret);
        return ret;
    }
    
    // Query State
    RKLLMRunState state = {0};
    ret = rknn3_session_query_state(llm_ctx->rknn_sess, &state);
    if (ret < 0)
    {
        printf("rknn_session_query_state fail! ret=%d\n", ret);
        return ret;
    }   
    perf->n_decode_tokens = state.n_decode_tokens;
    perf->n_prefill_tokens = state.n_prefill_tokens;



    return ret;
}

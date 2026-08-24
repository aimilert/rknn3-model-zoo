// Copyright (c) 2025 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");

#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"
#include "image_utils.h"
#include "locate_anything.h"
#include "rknn3_api.h"
#include "time_utils.h"

static int release_internal_share(rknn_app_context_t* app_ctx);

static int init_internal_share(rknn_app_context_t* app_ctx, uint32_t core_mask_vision, uint32_t core_mask_llm)
{
    int ret = -1;
    uint32_t core_num_vision = 0;
    uint32_t core_num_llm = 0;
    int* llm_to_vision = NULL;
    rknn3_tensor_mem** internal_mems_vision = NULL;
    rknn3_tensor_mem** internal_mems_llm = NULL;
    int core_num_same = 0;
    int idx = 0;
    ret = rknn3_query(app_ctx->vision.rknn_ctx, RKNN3_QUERY_CORE_NUMBER, &core_num_vision, sizeof(core_num_vision));
    if (ret < 0) {
        printf("rknn3_query vision core number failed! ret=%d\n", ret);
        return ret;
    }
    ret = rknn3_query(app_ctx->llm.rknn_ctx, RKNN3_QUERY_CORE_NUMBER, &core_num_llm, sizeof(core_num_llm));
    if (ret < 0) {
        printf("rknn3_query llm core number failed! ret=%d\n", ret);
        return ret;
    }

    uint32_t selected_vision = 0;
    uint32_t selected_llm = 0;
    for (int i = 0; i < 32; i++) {
        if (core_mask_vision & (1U << i)) selected_vision++;
        if (core_mask_llm & (1U << i)) selected_llm++;
    }
    if (selected_vision != core_num_vision) {
        printf("vision core mask 0x%x does not match model core_num=%u\n", core_mask_vision, core_num_vision);
        return -1;
    }
    if (selected_llm != core_num_llm) {
        printf("llm core mask 0x%x does not match model core_num=%u\n", core_mask_llm, core_num_llm);
        return -1;
    }

    rknn3_core_mem_size* core_mem_sizes_vision = (rknn3_core_mem_size*)malloc(sizeof(rknn3_core_mem_size) * core_num_vision);
    rknn3_core_mem_size* core_mem_sizes_llm = (rknn3_core_mem_size*)malloc(sizeof(rknn3_core_mem_size) * core_num_llm);
    if (!core_mem_sizes_vision || !core_mem_sizes_llm) {
        printf("allocate core memory size buffers failed\n");
        free(core_mem_sizes_vision);
        free(core_mem_sizes_llm);
        return -1;
    }

    ret = rknn3_query(app_ctx->vision.rknn_ctx, RKNN3_QUERY_CORE_MEM_SIZE, core_mem_sizes_vision, sizeof(rknn3_core_mem_size) * core_num_vision);
    if (ret < 0) {
        printf("rknn3_query vision core memory size failed! ret=%d\n", ret);
        goto out;
    }
    ret = rknn3_query(app_ctx->llm.rknn_ctx, RKNN3_QUERY_CORE_MEM_SIZE, core_mem_sizes_llm, sizeof(rknn3_core_mem_size) * core_num_llm);
    if (ret < 0) {
        printf("rknn3_query llm core memory size failed! ret=%d\n", ret);
        goto out;
    }

    llm_to_vision = (int*)malloc(sizeof(int) * core_num_llm);
    internal_mems_vision = (rknn3_tensor_mem**)calloc(core_num_vision, sizeof(rknn3_tensor_mem*));
    internal_mems_llm = (rknn3_tensor_mem**)calloc(core_num_llm, sizeof(rknn3_tensor_mem*));
    if (!llm_to_vision || !internal_mems_vision || !internal_mems_llm) {
        printf("allocate internal share helper buffers failed\n");
        ret = -1;
        free(llm_to_vision);
        free(internal_mems_vision);
        free(internal_mems_llm);
        goto out;
    }
    for (uint32_t i = 0; i < core_num_llm; i++) {
        llm_to_vision[i] = -1;
    }

    for (uint32_t i = 0; i < core_num_vision; i++) {
        for (uint32_t j = 0; j < core_num_llm; j++) {
            if (core_mem_sizes_vision[i].core_id == core_mem_sizes_llm[j].core_id) {
                uint64_t internal_size = std::max(core_mem_sizes_vision[i].internal_size, core_mem_sizes_llm[j].internal_size);
                core_mem_sizes_vision[i].internal_size = internal_size;
                core_mem_sizes_llm[j].internal_size = internal_size;
                core_num_same++;
                llm_to_vision[j] = i;
                break;
            }
        }
    }

    app_ctx->n_internal_mems = core_num_vision + core_num_llm - core_num_same;
    app_ctx->internal_mems = (rknn3_tensor_mem**)calloc(app_ctx->n_internal_mems, sizeof(rknn3_tensor_mem*));
    if (!app_ctx->internal_mems) {
        printf("allocate app internal memory list failed\n");
        ret = -1;
        free(llm_to_vision);
        free(internal_mems_vision);
        free(internal_mems_llm);
        goto out;
    }

    for (uint32_t i = 0; i < core_num_vision; i++) {
        internal_mems_vision[i] = rknn3_create_mem(app_ctx->vision.rknn_ctx, core_mem_sizes_vision[i].internal_size, core_mem_sizes_vision[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (!internal_mems_vision[i]) {
            ret = -1;
            goto share_out;
        }
        app_ctx->internal_mems[idx++] = internal_mems_vision[i];
    }
    for (uint32_t i = 0; i < core_num_llm; i++) {
        if (llm_to_vision[i] != -1) {
            internal_mems_llm[i] = internal_mems_vision[llm_to_vision[i]];
            continue;
        }
        internal_mems_llm[i] = rknn3_create_mem(app_ctx->vision.rknn_ctx, core_mem_sizes_llm[i].internal_size, core_mem_sizes_llm[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (!internal_mems_llm[i]) {
            ret = -1;
            goto share_out;
        }
        app_ctx->internal_mems[idx++] = internal_mems_llm[i];
    }

    ret = rknn3_set_internal_mem(app_ctx->vision.rknn_ctx, internal_mems_vision, core_num_vision);
    if (ret < 0) {
        printf("rknn3_set_internal_mem vision failed! ret=%d\n", ret);
        goto share_out;
    }
    ret = rknn3_set_internal_mem(app_ctx->llm.rknn_ctx, internal_mems_llm, core_num_llm);
    if (ret < 0) {
        printf("rknn3_set_internal_mem llm failed! ret=%d\n", ret);
    }

share_out:
    free(llm_to_vision);
    free(internal_mems_vision);
    free(internal_mems_llm);
    if (ret < 0) {
        release_internal_share(app_ctx);
    }
out:
    free(core_mem_sizes_vision);
    free(core_mem_sizes_llm);
    return ret;
}

static int release_internal_share(rknn_app_context_t* app_ctx)
{
    if (app_ctx->internal_mems) {
        for (int i = 0; i < app_ctx->n_internal_mems; i++) {
            if (app_ctx->internal_mems[i]) {
                rknn3_destroy_mem(app_ctx->vision.rknn_ctx, app_ctx->internal_mems[i]);
                app_ctx->internal_mems[i] = NULL;
            }
        }
        free(app_ctx->internal_mems);
        app_ctx->internal_mems = NULL;
        app_ctx->n_internal_mems = 0;
    }
    return 0;
}

int init_locate_anything_model(rknn_app_context_t* app_ctx,
                               const char* llm_model_path,
                               const char* llm_weight_path,
                               const char* vision_model_path,
                               const char* vision_weight_path,
                               rknn3_llm_param* params,
                               int n_params,
                               RKLLMCallback& callback,
                               uint32_t vision_core_mask,
                               uint32_t llm_core_mask)
{
    int ret = 0;
    printf("--> init LocateAnything vision model\n");
    ret = init_locate_anything_vision(&app_ctx->vision, vision_model_path, vision_weight_path, vision_core_mask, app_ctx->model_width, app_ctx->model_height);
    if (ret < 0) {
        printf("init LocateAnything vision model failed! ret=%d\n", ret);
        return ret;
    }

    printf("--> init LocateAnything llm model\n");
    ret = init_locate_anything_llm(&app_ctx->llm, llm_model_path, llm_weight_path, params, n_params, callback, llm_core_mask);
    if (ret < 0) {
        printf("init LocateAnything llm model failed! ret=%d\n", ret);
        release_locate_anything_vision(&app_ctx->vision);
        return ret;
    }

    printf("--> init internal share\n");
    ret = init_internal_share(app_ctx, vision_core_mask, llm_core_mask);
    if (ret < 0) {
        printf("LocateAnything llm/vision internal memory share failed! ret=%d\n", ret);
        release_internal_share(app_ctx);
        release_locate_anything_llm(&app_ctx->llm);
        release_locate_anything_vision(&app_ctx->vision);
    }
    return ret;
}

int release_locate_anything_model(rknn_app_context_t* app_ctx)
{
    release_internal_share(app_ctx);
    release_locate_anything_vision(&app_ctx->vision);
    release_locate_anything_llm(&app_ctx->llm);
    return 0;
}

int inference_locate_anything_model(rknn_app_context_t* app_ctx,
                                    image_buffer_t* img,
                                    float16* img_embeds,
                                    rknn3_llm_multimodal_tensor tensor,
                                    int n_inputs,
                                    rknn_perf_metrics_t* perf)
{
    if (!app_ctx || !img) {
        printf("app_ctx or img is NULL\n");
        return -1;
    }

    printf("--> inference LocateAnything vision model\n");
    int start_us = getCurrentTimeUs();
    int ret = inference_locate_anything_vision(&app_ctx->vision, img, img_embeds);
    perf->vision_latency = getCurrentTimeUs() - start_us;
    if (ret != 0) {
        printf("inference LocateAnything vision model failed! ret=%d\n", ret);
        return ret;
    }

    printf("--> inference LocateAnything llm model\n");
    ret = inference_locate_anything_llm(&app_ctx->llm, tensor, n_inputs, perf);
    if (ret != 0) {
        printf("inference LocateAnything llm model failed! ret=%d\n", ret);
    }
    return ret;
}

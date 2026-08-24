// Copyright (c) 2025 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#include "common.h"
#include "image_utils.h"
#include "rknn_locate_anything_vision.h"

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

static void dump_tensor_attr(rknn3_tensor_attr* attrs)
{
    std::string shape_str;
    for (int i = 0; i < attrs->n_dims; i++) {
        shape_str += std::to_string(attrs->shape[i]);
        if (i + 1 < attrs->n_dims) {
            shape_str += ", ";
        }
    }
    std::string stride_str;
    for (int i = 0; i < attrs->n_stride; i++) {
        stride_str += std::to_string(attrs->stride[i]);
        if (i + 1 < attrs->n_stride) {
            stride_str += ", ";
        }
    }

    printf("Tensor: name=%s, n_dims=%d, shape=[%s], stride=[%s], aligned_size=%ld, layout=%s, dtype=%s, core_id=%d, qnt_type=%s\n",
           attrs->name,
           attrs->n_dims,
           shape_str.c_str(),
           stride_str.c_str(),
           attrs->aligned_size,
           rknn3_get_layout_string(attrs->layout),
           rknn3_get_type_string(attrs->dtype),
           attrs->core_id,
           rknn3_get_qnt_type_string(attrs->qnt_type));
}

static size_t tensor_elem_count(const rknn3_tensor_attr* attr)
{
    size_t elems = 1;
    for (int i = 0; i < attr->n_dims; i++) {
        elems *= attr->shape[i];
    }
    return elems;
}

int init_locate_anything_vision(rknn_locate_anything_vision_context* vision_ctx,
                                const char* model_path,
                                const char* weight_path,
                                uint32_t core_mask,
                                uint32_t model_width,
                                uint32_t model_height)
{
    int ret = 0;
    rknn3_context ctx = 0;
    rknn3_config config;
    memset(&config, 0, sizeof(config));
    config.run_core_mask = core_mask;
    config.user_mem_internal = 1;

    rknn3_init_extend init_extend = make_rknn3_init_extend("VisionInit");
    ret = rknn3_init(&ctx, init_extend.device_id ? &init_extend : NULL);
    if (ret < 0) {
        printf("rknn3_init vision failed! ret=%d\n", ret);
        return ret;
    }

    ret = rknn3_load_model_from_path(ctx, model_path, weight_path);
    if (ret < 0) {
        printf("rknn3_load_model_from_path vision failed! ret=%d\n", ret);
        rknn3_destroy(ctx);
        return ret;
    }

    ret = rknn3_model_init(ctx, &config);
    if (ret < 0) {
        printf("rknn3_model_init vision failed! ret=%d\n", ret);
        rknn3_destroy(ctx);
        return ret;
    }

    rknn3_input_output_num io_num;
    ret = rknn3_query(ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        printf("rknn3_query in/out num failed! ret=%d\n", ret);
        rknn3_destroy(ctx);
        return ret;
    }
    printf("vision input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    rknn3_tensor_attr input_attrs[io_num.n_input];
    printf("vision input tensors:\n");
    for (int i = 0; i < io_num.n_input; i++) {
        memset(&input_attrs[i], 0, sizeof(rknn3_tensor_attr));
        input_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn3_tensor_attr));
        if (ret < 0) {
            printf("rknn3_query input attr failed! ret=%d\n", ret);
            rknn3_destroy(ctx);
            return ret;
        }
        dump_tensor_attr(&input_attrs[i]);
    }

    rknn3_tensor_attr output_attrs[io_num.n_output];
    printf("vision output tensors:\n");
    for (int i = 0; i < io_num.n_output; i++) {
        memset(&output_attrs[i], 0, sizeof(rknn3_tensor_attr));
        output_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn3_tensor_attr));
        if (ret < 0) {
            printf("rknn3_query output attr failed! ret=%d\n", ret);
            rknn3_destroy(ctx);
            return ret;
        }
        dump_tensor_attr(&output_attrs[i]);
    }

    int width = model_width ? (int)model_width : LOCATE_ANYTHING_DEFAULT_WIDTH;
    int height = model_height ? (int)model_height : LOCATE_ANYTHING_DEFAULT_HEIGHT;
    if (width % LOCATE_ANYTHING_PATCH_SIZE != 0 || height % LOCATE_ANYTHING_PATCH_SIZE != 0) {
        printf("model_width/model_height must be divisible by %d, got %dx%d\n",
               LOCATE_ANYTHING_PATCH_SIZE, width, height);
        ret = -1;
        rknn3_destroy(ctx);
        return ret;
    }
    int grid_h = height / LOCATE_ANYTHING_PATCH_SIZE;
    int grid_w = width / LOCATE_ANYTHING_PATCH_SIZE;
    if (input_attrs[0].n_dims != 4 || input_attrs[0].shape[0] != 1 ||
        input_attrs[0].shape[1] != (uint32_t)height ||
        input_attrs[0].shape[2] != (uint32_t)width || input_attrs[0].shape[3] != 3) {
        printf("image input shape should be NHWC [1, %d, %d, 3], got n_dims=%d shape=[%u, %u, %u, %u].\n",
               height,
               width,
               input_attrs[0].n_dims,
               input_attrs[0].shape[0],
               input_attrs[0].shape[1],
               input_attrs[0].shape[2],
               input_attrs[0].shape[3]);
        ret = -1;
        goto cleanup;
    }
    if (input_attrs[0].dtype != RKNN3_TENSOR_UINT8) {
        printf("image input dtype should be uint8, got %s\n",
               rknn3_get_type_string(input_attrs[0].dtype));
        ret = -1;
        goto cleanup;
    }

    vision_ctx->inputs = (rknn3_tensor*)calloc(io_num.n_input, sizeof(rknn3_tensor));
    vision_ctx->outputs = (rknn3_tensor*)calloc(io_num.n_output, sizeof(rknn3_tensor));
    if (!vision_ctx->inputs || !vision_ctx->outputs) {
        printf("alloc vision tensor holders failed\n");
        ret = -1;
        goto cleanup;
    }

    vision_ctx->rknn_ctx = ctx;
    vision_ctx->io_num = io_num;
    vision_ctx->model_channel = 3;
    vision_ctx->model_width = width;
    vision_ctx->model_height = height;
    vision_ctx->grid_h = grid_h;
    vision_ctx->grid_w = grid_w;

    for (int i = 0; i < io_num.n_input; i++) {
        vision_ctx->inputs[i].mem = rknn3_create_mem(ctx, input_attrs[i].aligned_size, input_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        vision_ctx->inputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        memcpy(vision_ctx->inputs[i].attr, &input_attrs[i], sizeof(rknn3_tensor_attr));
    }
    for (int i = 0; i < io_num.n_output; i++) {
        vision_ctx->outputs[i].mem = rknn3_create_mem(ctx, output_attrs[i].aligned_size, output_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        vision_ctx->outputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        memcpy(vision_ctx->outputs[i].attr, &output_attrs[i], sizeof(rknn3_tensor_attr));
    }

    vision_ctx->embeds_shape = vision_ctx->outputs[0].attr->shape;
    vision_ctx->embeds_ndims = vision_ctx->outputs[0].attr->n_dims;
    return ret;

cleanup:
    if (vision_ctx->inputs) {
        free(vision_ctx->inputs);
        vision_ctx->inputs = NULL;
    }
    if (vision_ctx->outputs) {
        free(vision_ctx->outputs);
        vision_ctx->outputs = NULL;
    }
    if (ctx != 0) {
        rknn3_destroy(ctx);
    }
    return ret;
}

int release_locate_anything_vision(rknn_locate_anything_vision_context* vision_ctx)
{
    if (vision_ctx->inputs) {
        for (int i = 0; i < vision_ctx->io_num.n_input; i++) {
            if (vision_ctx->inputs[i].mem) {
                rknn3_destroy_mem(vision_ctx->rknn_ctx, vision_ctx->inputs[i].mem);
            }
            if (vision_ctx->inputs[i].attr) {
                free(vision_ctx->inputs[i].attr);
            }
        }
        free(vision_ctx->inputs);
        vision_ctx->inputs = NULL;
    }
    if (vision_ctx->outputs) {
        for (int i = 0; i < vision_ctx->io_num.n_output; i++) {
            if (vision_ctx->outputs[i].mem) {
                rknn3_destroy_mem(vision_ctx->rknn_ctx, vision_ctx->outputs[i].mem);
            }
            if (vision_ctx->outputs[i].attr) {
                free(vision_ctx->outputs[i].attr);
            }
        }
        free(vision_ctx->outputs);
        vision_ctx->outputs = NULL;
    }
    if (vision_ctx->rknn_ctx != 0) {
        rknn3_destroy(vision_ctx->rknn_ctx);
        vision_ctx->rknn_ctx = 0;
    }
    return 0;
}

int inference_locate_anything_vision(rknn_locate_anything_vision_context* vision_ctx,
                                     image_buffer_t* img,
                                     float16* img_embeds)
{
    if (!vision_ctx || !img || !img_embeds) {
        printf("vision_ctx, img or img_embeds is NULL\n");
        return -1;
    }

    int ret = 0;
    image_buffer_t dst_img;
    memset(&dst_img, 0, sizeof(image_buffer_t));
    dst_img.width = vision_ctx->model_width;
    dst_img.height = vision_ctx->model_height;
    dst_img.format = IMAGE_FORMAT_RGB888;
    dst_img.size = get_image_size(&dst_img);
    dst_img.virt_addr = (unsigned char*)malloc(dst_img.size);
    if (!dst_img.virt_addr) {
        printf("malloc resized image buffer failed, size=%d\n", dst_img.size);
        return -1;
    }

    ret = convert_image(img, &dst_img, NULL, NULL, 0);
    if (ret < 0) {
        printf("convert_image failed! ret=%d\n", ret);
        goto out;
    }

    memcpy(vision_ctx->inputs[0].mem->virt_addr, dst_img.virt_addr, dst_img.size);

    if (vision_ctx->io_num.n_input >= 2) {
        if (vision_ctx->inputs[1].attr->dtype != RKNN3_TENSOR_INT64) {
            printf("grid_hws input dtype should be int64, got %s\n",
                   rknn3_get_type_string(vision_ctx->inputs[1].attr->dtype));
            ret = -1;
            goto out;
        }
        int64_t* grid_hws = (int64_t*)vision_ctx->inputs[1].mem->virt_addr;
        grid_hws[0] = vision_ctx->grid_h;
        grid_hws[1] = vision_ctx->grid_w;
    }

    for (int i = 0; i < vision_ctx->io_num.n_input; i++) {
        ret = rknn3_mem_sync(vision_ctx->rknn_ctx, vision_ctx->inputs[i].mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
        if (ret != RKNN3_SUCCESS) {
            printf("rknn3_mem_sync input[%d] failed! ret=%d\n", i, ret);
            goto out;
        }
    }

    ret = rknn3_run(vision_ctx->rknn_ctx, vision_ctx->inputs, vision_ctx->io_num.n_input, vision_ctx->outputs, vision_ctx->io_num.n_output);
    if (ret < 0) {
        printf("rknn3_run vision failed! ret=%d\n", ret);
        goto out;
    }

    for (int i = 0; i < vision_ctx->io_num.n_output; i++) {
        ret = rknn3_mem_sync(vision_ctx->rknn_ctx, vision_ctx->outputs[i].mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
        if (ret != RKNN3_SUCCESS) {
            printf("rknn3_mem_sync output[%d] failed! ret=%d\n", i, ret);
            goto out;
        }
    }

    if (vision_ctx->outputs[0].attr->dtype == RKNN3_TENSOR_FLOAT16) {
        memcpy(img_embeds, vision_ctx->outputs[0].mem->virt_addr, vision_ctx->outputs[0].mem->size);
    } else if (vision_ctx->outputs[0].attr->dtype == RKNN3_TENSOR_FLOAT32) {
        size_t elems = tensor_elem_count(vision_ctx->outputs[0].attr);
        float* src = (float*)vision_ctx->outputs[0].mem->virt_addr;
        for (size_t i = 0; i < elems; i++) {
            img_embeds[i] = fp32_to_fp16(src[i]);
        }
    } else {
        printf("unsupported vision output dtype: %s\n", rknn3_get_type_string(vision_ctx->outputs[0].attr->dtype));
        ret = -1;
    }

out:
    if (dst_img.virt_addr) {
        free(dst_img.virt_addr);
    }
    return ret;
}

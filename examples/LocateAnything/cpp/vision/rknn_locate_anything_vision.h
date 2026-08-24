// Copyright (c) 2025 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");

#ifndef _RKNN_DEMO_LOCATE_ANYTHING_VISION_H_
#define _RKNN_DEMO_LOCATE_ANYTHING_VISION_H_

#include "common.h"
#include "rknn3_api.h"

#define LOCATE_ANYTHING_PATCH_SIZE 14
#define LOCATE_ANYTHING_MERGE_H 2
#define LOCATE_ANYTHING_MERGE_W 2
#define LOCATE_ANYTHING_DEFAULT_WIDTH 448
#define LOCATE_ANYTHING_DEFAULT_HEIGHT 448

typedef struct {
    rknn3_context rknn_ctx;
    rknn3_input_output_num io_num;
    rknn3_tensor* inputs;
    rknn3_tensor* outputs;

    int model_channel;
    int model_height;
    int model_width;
    int grid_h;
    int grid_w;

    uint32_t* embeds_shape;
    uint32_t embeds_ndims;
} rknn_locate_anything_vision_context;

int init_locate_anything_vision(rknn_locate_anything_vision_context* vision_ctx,
                                const char* model_path,
                                const char* weight_path,
                                uint32_t core_mask,
                                uint32_t model_width,
                                uint32_t model_height);

int release_locate_anything_vision(rknn_locate_anything_vision_context* vision_ctx);

int inference_locate_anything_vision(rknn_locate_anything_vision_context* vision_ctx,
                                     image_buffer_t* img,
                                     float16* img_embeds);

#endif  // _RKNN_DEMO_LOCATE_ANYTHING_VISION_H_

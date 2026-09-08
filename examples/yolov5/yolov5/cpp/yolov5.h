// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
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


#ifndef _RKNN_DEMO_YOLOV5_H_
#define _RKNN_DEMO_YOLOV5_H_

#include "rknn3_api.h"
#include "common.h"

typedef struct {
    rknn3_context rknn_ctx;
    rknn3_input_output_num io_num;
    rknn3_tensor* inputs;
    rknn3_tensor* outputs;

    int model_channel;
    int model_width;
    int model_height;
    bool is_quant;

    bool use_postprocess_plugin;

    // If true, suppress per-frame/init informational prints (used by
    // multi-channel benchmark where 32 contexts would flood stdout).
    bool quiet;
} rknn_app_context_t;

// Phase-level timing for one inference call.
typedef struct {
    double preprocess_ms;
    double inference_ms;
    double postprocess_ms;
    double total_ms;
} rknn_yolov5_times;

#include "postprocess.h"

// Original API kept for compatibility (single-device, verbose).
int init_yolov5_model(const char* model_path, const char* weight_path, rknn_app_context_t* app_ctx, uint32_t core_mask, const char* postprocess_plugin_path);

// Extended init: allows binding to a specific RK182x device via device_id
// (NULL = default/first device) and quiet mode (no per-channel prints).
int init_yolov5_model_ex(const char* model_path, const char* weight_path, rknn_app_context_t* app_ctx,
                         uint32_t core_mask, const char* postprocess_plugin_path, const char* device_id, bool quiet);

int release_yolov5_model(rknn_app_context_t* app_ctx);

// times may be NULL. If non-NULL, the phase timings of this call are written back.
int inference_yolov5_model(rknn_app_context_t* app_ctx, image_buffer_t* img, object_detect_result_list* od_results, rknn_yolov5_times* times);

#endif //_RKNN_DEMO_YOLOV5_H_
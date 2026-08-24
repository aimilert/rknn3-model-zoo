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


#ifndef _RKNN_DEMO_SENSEVOICE_SMALL_H_
#define _RKNN_DEMO_SENSEVOICE_SMALL_H_

#include "rknn3_api.h"
#include "process.h"


// 输入shape: [1,170,560]
const size_t SHAPE[3] = {1, 170, 560};
const size_t NUM_ELEMENTS = SHAPE[0] * SHAPE[1] * SHAPE[2];
const size_t OUTPUT_NUM_ELEMENTS = 1 * 170 * 25055;

typedef struct {
    rknn3_context rknn_ctx;
    rknn3_input_output_num io_num;
    rknn3_tensor* inputs;
    rknn3_tensor* outputs;

    int model_channel;
    int model_width;
    int model_height;
    bool is_quant;
} rknn_app_context_t;


int init_sensevoice_small_model(const char* model_path, const char* weight_path, rknn_app_context_t* app_ctx, uint32_t core_mask);

int release_sensevoice_small_model(rknn_app_context_t* app_ctx);

int inference_sensevoice_small_model(rknn_app_context_t *app_ctx, float* embedding_matrix, audio_buffer_t audio, VocabEntry *vocabs, std::vector<std::string> &recognized_text);

#endif //_RKNN_DEMO_SENSEVOICE_SMALL_H_
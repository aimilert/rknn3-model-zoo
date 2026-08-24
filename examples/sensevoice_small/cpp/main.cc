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

#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <iostream>
#include "audio_utils.h"
#include "sensevoice_small.h"
#include "process.h"


int main(int argc, char **argv)
{
    if (argc != 5) {
        printf("%s <model_path> <weight_path> <audio_path> <core_mask>\n", argv[0]);
        return -1;
    }

    const char* model_path  = argv[1];
    const char* weight_path = argv[2];
    const char* audio_path  = argv[3];
    uint32_t core_mask = strtoul(argv[4], nullptr, 16);

    int ret;
    float* embedding_matrix = NULL;
    std::vector<std::string> recognized_text;

    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    long duration = 0;
    VocabEntry vocab[VOCAB_NUM];
    audio_buffer_t audio;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_ctx));
    memset(vocab, 0, sizeof(vocab));
    memset(&audio, 0, sizeof(audio_buffer_t));

    ret = read_audio(audio_path, &audio);
    if (ret != 0)
    {
        printf("read audio fail! ret=%d audio_path=%s\n", ret, audio_path);
        goto out;
    }

    if (audio.num_channels == 2)
    {
        ret = convert_channels(&audio);
        if (ret != 0)
        {
            printf("convert channels fail! ret=%d audio_path=%s\n", ret, audio_path);
            goto out;
        }
    }

    if (audio.sample_rate != SAMPLE_RATE)
    {
        ret = resample_audio(&audio, audio.sample_rate, SAMPLE_RATE);
        if (ret != 0)
        {
            printf("resample audio fail! ret=%d audio_path=%s\n", ret, audio_path);
            goto out;
        }
    }

    embedding_matrix = (float*)malloc(N_EMBEDDING * EMBEDDING_FEAT_DIM * sizeof(float));
    ret = read_lines(EMBEDDING_MATRIX_PATH, embedding_matrix, N_EMBEDDING * EMBEDDING_FEAT_DIM);
    if (ret != 0) {
        printf("read embedding_matrix fail! ret=%d embedding_matrix_path=%s\n", ret, EMBEDDING_MATRIX_PATH);
        goto out;
    }

    ret = read_vocab(VOCAB_PATH, vocab);
    if (ret != 0)
    {
        printf("read vocab fail! ret=%d vocab_path=%s\n", ret, VOCAB_PATH);
        goto out;
    }

    printf("--> init model\n");
    ret = init_sensevoice_small_model(model_path, weight_path, &rknn_app_ctx, core_mask);
    
    if (ret != 0)
    {
        printf("init_model fail! ret=%d model_path=%s\n", ret, model_path);
        goto out;
    }


    start = std::chrono::high_resolution_clock::now();
    printf("--> inference model\n");
    ret = inference_sensevoice_small_model(&rknn_app_ctx, embedding_matrix, audio, vocab, recognized_text);
    if (ret != 0)
    {
        printf("inference_model fail! ret=%d\n", ret);
    }
    printf("--> inference model done\n");

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Inference time: " << duration << " ms" << std::endl;

    std::cout << "\nSensevoice_small output: ";
    for (const auto &str : recognized_text)
    {
        std::cout << str;
    }
    std::cout << std::endl;

out:

    if (audio.data)
    {
        free(audio.data);
    }

    if (embedding_matrix)
    {
        free(embedding_matrix);
    }

    for (int i = 0; i < VOCAB_NUM; i++)
    {
        if (vocab[i].token)
        {
            free(vocab[i].token);
            vocab[i].token = NULL;
        }
    }

    ret = release_sensevoice_small_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release_model fail! ret=%d\n", ret);
    }

    return 0;
}

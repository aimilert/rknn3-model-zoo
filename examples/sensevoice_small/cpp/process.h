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

#ifndef _RKNN_DEMO_SENSEVOICE_SMALL_PROCESS_H_
#define _RKNN_DEMO_SENSEVOICE_SMALL_PROCESS_H_

#include "kaldi-native-fbank/csrc/online-feature.h"
#include "float16.h"
#include "audio_utils.h"

#define VOCAB_NUM 25055
#define SAMPLE_RATE 16000
#define N_EMBEDDING 16
#define EMBEDDING_FEAT_DIM 560
#define MODEL_INPUT_SAMPLES 10*16000
#define DEFAULT_LANGUAGE_ID 0  // {"auto": 0, "zh": 3, "en":4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
#define TEXTNORM_ID 14  // {"withitn": 14, "woitn": 15}
#define VOCAB_PATH "./model/tokens.txt"
#define EMBEDDING_MATRIX_PATH "./model/embedding_matrix.txt"

typedef struct
{
    int index;
    char *token;
} VocabEntry;

int sense_voice_audio_preprocess(float *pcmdata_f, int frames_num, int model_in_n_samples, float** output, int* output_len);
std::vector<float> apply_lfr(const std::vector<float>& in);
void apply_cmvn(std::vector<float>* v);
int argmax(float *array);
void replace_substr(std::string &str, const std::string &from, const std::string &to);
int read_vocab(const char *fileName, VocabEntry *vocab);
int read_lines(const char *fileName, float *data, int max_lines);
void pad_or_trim(const std::vector<float> &array, std::vector<float> &result, int array_shape, int length);
std::vector<float16> fp32_array_to_fp16(const float* src, size_t len);
std::vector<float> fp16_array_to_fp32(const float16* src, size_t len);
#endif //_RKNN_DEMO_SENSEVOICE_SMALL_PROCESS_H_
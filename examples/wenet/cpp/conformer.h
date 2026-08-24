#ifndef _RKNN_DEMO_CONFORMER_H_
#define _RKNN_DEMO_CONFORMER_H_

#include "rknn3_api.h"
#include "audio_utils.h"
#include <iostream>
#include <vector>
#include <string>
#include "process.h"

typedef enum {
    CONFORMER_MODEL_ENCODER = 0,
    CONFORMER_MODEL_CTC,
    CONFORMER_MODEL_DECODER,
} conformer_model_type_t;

typedef struct
{
    rknn3_context rknn_ctx;
    rknn3_input_output_num io_num;
    rknn3_tensor *inputs;
    rknn3_tensor *outputs;
} rknn_app_context_t;

typedef struct
{
    rknn_app_context_t encoder_context;
    rknn_app_context_t ctc_context;
    rknn_app_context_t decoder_context;
} rknn_conformer_context_t;

int init_conformer_model(const char *model_path, const char *weight_path, rknn_app_context_t *app_ctx, uint32_t core_mask, conformer_model_type_t model_type);
int inference_conformer_model(rknn_conformer_context_t *app_ctx, audio_buffer_t audio, VocabEntry *vocab, std::string &result_text);
int release_conformer_model(rknn_app_context_t *app_ctx);

#endif //_RKNN_DEMO_CONFORMER_H_

#ifndef __VLM_H__
#define __VLM_H__

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>

#include "paddleocr_vl.h"
#include "image_utils.h"


#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include "time_utils.h"


const rknn3_sampling_params SAMPLE_PARAMS = {
  .top_k = 1,
  .top_p = 0.9,
  .temperature = 0.0f,
  .repeat_penalty = 1.1f,
  .frequency_penalty = 0.0f, 
  .presence_penalty = 0.0f
};

const char* system_prompt  = "";
const char* prompt_prefix  = "<|begin_of_sentence|>User: ";
const char* prompt_postfix = "\nAssistant:\n";

int64_t first_token;
bool first_decode = true;

struct embedding_info
{
  int      fd;
  float16* embedding_data;
  int      embedding_dim;
  int      vocab_size;
};

rknn_perf_metrics_t perf;

// RKNN Context
rknn_app_context_t rknn_app_ctx;

// Tokenizer
Tokenizer* tokenizer;
VocabInfo vocab_info;

// Embedding
struct embedding_info embedding_info;
struct stat           emb_st;

// LLM Param
int n_params = 1;
rknn3_llm_param params;

// Input Image
image_buffer_t src_image;

// Image Embed
size_t vision_embed_elems = 1;
size_t img_embed_elems    = 1;
float16* vision_embeds;
float16* img_embeds;

// LLM Multi Model Tensor
int n_inputs = 1;
rknn3_llm_multimodal_tensor tensor;

// Callback
RKLLMCallback callback;


std::string output_str;

#ifdef __cplusplus
extern "C" {
#endif

int release_model();
int inference_model(const char* image_path, const char* prompt);
int init_model(const char *vision_model_path, const char *vision_weight_path, const char *position_embedding_path, const char *llm_model_path, const char *llm_weight_path, const char *tokenizer_path, const char *embedding_path, const char *mlpar_model_path, const char *mlpar_weight_path, uint32_t vision_core_mask, uint32_t mlpar_core_mask, uint32_t llm_core_mask, uint32_t model_width, uint32_t model_height);
int get_result(char** result);


#ifdef __cplusplus
}
#endif

#endif
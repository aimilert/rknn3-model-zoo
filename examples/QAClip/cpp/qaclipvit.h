#ifndef _QACLIPVIT_H_
#define _QACLIPVIT_H_

#include "rknn_qaclipvit_embedding.h"
#include "rknn_qaclipvit_vision.h"
#include "image_utils.h"
#include "Tokenizer.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//2.6592
#define LOGIT_SCALE_EXP 99.9946
typedef struct {
    int fd;
    float16* embedding_data;
    int vocab_size;
    int embedding_dim;
    size_t file_size;
} embedding_info_t;

// qaclipvit完整上下文结构体
typedef struct {
    qaclipvit_text_context_t text_ctx;      // 文本模型上下文
    qaclipvit_vision_context_t vision_ctx;   // 视觉模型上下文
    Tokenizer* tokenizer;          // 外部GGUF分词器
    embedding_info_t embedding_info;         // 嵌入信息
    uint32_t model_width;                    // 模型宽度
    uint32_t model_height;                   // 模型高度
} rknn_qaclipvit_context_t;

// 相似度计算结果
typedef struct {
    float logit_one_image;          // 相似度得分
    float16* text_embedding;        // 文本嵌入向量(512维)
    float16* image_embedding;       // 图像嵌入向量(512维)
    int embedding_size;             // 嵌入维度(512)
} qaclipvit_similarity_result_t;

// 函数声明
int init_qaclipvit_model(const char* text_model_path, const char* text_weight_path, const char* embedding_path, const char* tokenizer_path,
                        const char* vision_model_path, const char* vision_weight_path,
                        uint32_t core_mask, rknn_qaclipvit_context_t* ctx);

int release_qaclipvit_model(rknn_qaclipvit_context_t* ctx);

int inference_qaclipvit_model(rknn_qaclipvit_context_t* ctx, 
                             const char* text, 
                             image_buffer_t* image_input,
                             qaclipvit_similarity_result_t* result);

// 工具函数
float calculate_similarity(float16* text_embed, float16* image_embed, int embed_dim);
void free_similarity_result(qaclipvit_similarity_result_t* result);

#ifdef __cplusplus
}
#endif

#endif // _QACLIPVIT_H_
#ifndef _QACLIPVIT_EMBEDDING_H_
#define _QACLIPVIT_EMBEDDING_H_

#include "rknn3_api.h"
#include "common.h"
#define MAX_SEQ_LEN 352
#define EMBEDDING_DIM 768
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    rknn3_context rknn_ctx;
    rknn3_input_output_num io_num;
    rknn3_tensor* inputs;
    rknn3_tensor* outputs;
    
    int model_channel;
    int model_width;
    int model_height;
    bool is_quant;
} qaclipvit_text_context_t;

// 嵌入结果结构体
typedef struct {
    float16* embedding;        // 512维嵌入向量
    int embedding_size;        // 嵌入向量大小
} qaclipvit_result_t;

// 输入数据结构体
typedef struct {
    float16* input_embeds;     // [1, 64, 512] 输入嵌入
    float16* attention_mask;   // [1, 64] 注意力掩码
    int32_t* position_ids;     // [1, 64] 位置ID
    int32_t* Th;               // [1] 阈值参数
    int32_t* Tc;               // [1] 阈值参数
    int32_t* Ts;               // [1] 阈值参数
    int32_t* Tsr;              // [1] 阈值参数
} qaclipvit_input_data_t;

// 函数声明
int init_qaclipvit_embedding_model(const char *model_path, const char* weight_path, 
                                  qaclipvit_text_context_t *app_ctx, uint32_t core_mask);
int release_qaclipvit_embedding_model(qaclipvit_text_context_t *app_ctx);
int inference_qaclipvit_embedding_model(qaclipvit_text_context_t *app_ctx, 
                                       qaclipvit_input_data_t *input_data, 
                                       qaclipvit_result_t *result);
int set_qaclipvit_input_data(qaclipvit_text_context_t *app_ctx, 
                            qaclipvit_input_data_t *input_data);
int process_qaclipvit_output(qaclipvit_text_context_t *app_ctx, 
                            qaclipvit_result_t *result);

#ifdef __cplusplus
}
#endif

#endif // _QACLIPVIT_EMBEDDING_H_
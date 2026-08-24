#ifndef _QACLIPVIT_VISION_H_
#define _QACLIPVIT_VISION_H_

#include "rknn3_api.h"
#include "common.h"
#include "image_utils.h"

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
} qaclipvit_vision_context_t;

// qaclipvit Vision推理结果结构体
typedef struct {
    float* logits_per_text;    // 文本-图像相似度logits
    int logits_size;           // logits数组大小
    float16* image_embeddings;   // 图像嵌入特征（可选）
    int embedding_size;        // 嵌入特征维度
} qaclipvit_vision_result_t;

// 函数声明
int init_qaclipvit_vision_model(const char *model_path, const char* weight_path, 
                               qaclipvit_vision_context_t *app_ctx, uint32_t core_mask);
int release_qaclipvit_vision_model(qaclipvit_vision_context_t *app_ctx);
int inference_qaclipvit_vision_model(qaclipvit_vision_context_t *app_ctx, 
                                    image_buffer_t *image_input,
                                    qaclipvit_vision_result_t *result);

#ifdef __cplusplus
}
#endif

#endif // _QACLIPVIT_VISION_H_
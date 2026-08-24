#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <math.h>

#include "qaclipvit.h"
#include "file_utils.h"
#include "float16.h"
#include "Tokenizer.h"

void softmax_fun(float* logits, float* probs, int n) {
    if (n <= 0 || !logits || !probs) return;
    
    // 1. 找到最大值（防止数值溢出）
    float max_val = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
        }
    }
    
    // 2. 计算指数和
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] = expf(logits[i] - max_val);
        sum += probs[i];
    }
    
    // 3. 归一化
    for (int i = 0; i < n; i++) {
        probs[i] /= sum;
    }
}

int main(int argc, char** argv) {
    if (argc != 9) {
        printf("Usage: %s <text_model> <text_weight> <embedding> <tokenizer> "
               "<vision_model> <vision_weight> <image_path> <core_mask>\n", argv[0]);
        printf("Example: %s qaclipvit_text.rknn weight_text.bin embedding.bin "
               "qaclipvit_text.tokenizer.gguf "
               "qaclipvit_vision.rknn weight_vision.bin test.jpg 1\n", argv[0]);
        return -1;
    }
    
    const char* text_model_path = argv[1];
    const char* text_weight_path = argv[2];
    const char* embedding_path = argv[3];
    const char* tokenizer_path = argv[4];
    const char* vision_model_path = argv[5];
    const char* vision_weight_path = argv[6];
    const char* image_path = argv[7];
    uint32_t core_mask = strtoul(argv[8], NULL, 16);
    
    int ret = 0;
    rknn_qaclipvit_context_t ctx;
    memset(&ctx, 0, sizeof(rknn_qaclipvit_context_t));
    
    qaclipvit_similarity_result_t result;
    memset(&result, 0, sizeof(qaclipvit_similarity_result_t));
    
    image_buffer_t image_input;
    memset(&image_input, 0, sizeof(image_buffer_t));
    
    printf("=== chinese_clip Complete Demo ===\n");
    
    // 读取输入图像
    printf("--> Reading input image\n");
    ret = read_image(image_path, &image_input);
    if (ret != 0) {
        printf("Failed to read image: %s\n", image_path);
        return -1;
    }
    printf("Image loaded: %dx%d, format: %d\n", 
           image_input.width, image_input.height, image_input.format);
    
    // 初始化模型
    printf("--> Initializing chinese_clip model\n");
    ret = init_qaclipvit_model(text_model_path, text_weight_path, embedding_path, tokenizer_path,
                              vision_model_path, vision_weight_path, core_mask, &ctx);
    if (ret != 0) {
        printf("Model initialization failed\n");
        free(image_input.virt_addr);
        return ret;
    }
    
    float logits_per_image[5];
    float probs[5];
    // 执行推理
    printf("--> Running complete inference\n");
    
    const char* test_text[2] = {"一只小狗在沙滩上跳跃。", "宇航员在月球上喝酒"};
    ret = inference_qaclipvit_model(&ctx, test_text[0], &image_input, &result);
    if (ret != 0) {
        printf("Inference failed\n");
        release_qaclipvit_model(&ctx);
        free(image_input.virt_addr);
        return ret;
    }
    logits_per_image[0] = result.logit_one_image;
    
    // 打印embedding结果
    printf("Text embedding sample (first 5 elements):\n");
    for (int i = 0; i < 5; i++) {
        printf("  [%d]: %.6f\n", i, fp16_to_fp32(result.text_embedding[i]));
    }

    ret = inference_qaclipvit_model(&ctx, test_text[1], &image_input, &result);
    if (ret != 0) {
        printf("Inference failed\n");
        release_qaclipvit_model(&ctx);
        free(image_input.virt_addr);
        return ret;
    }
    logits_per_image[1] = result.logit_one_image;
    int n = 2;
    softmax_fun(logits_per_image, probs, n);
    for(int i = 0; i < n; i++){
        printf("%s :prob %f\n",test_text[i],probs[i]);
    }
    
    // 打印embedding结果
    printf("Text embedding sample (first 5 elements):\n");
    for (int i = 0; i < 5; i++) {
        printf("  [%d]: %.6f\n", i, fp16_to_fp32(result.text_embedding[i]));
    }
    
    printf("Image embedding sample (first 5 elements):\n");
    for (int i = 0; i < 5; i++) {
        printf("  [%d]: %.6f\n", i, fp16_to_fp32(result.image_embedding[i]));
    }
    
    // 保存结果到文件
    // printf("--> Saving results\n");
    // FILE* text_embed_file = fopen("text_embedding.bin", "wb");
    // FILE* image_embed_file = fopen("image_embedding.bin", "wb");
    
    // if (text_embed_file) {
    //     fwrite(result.text_embedding, sizeof(float16), result.embedding_size, text_embed_file);
    //     fclose(text_embed_file);
    //     printf("Text embedding saved to text_embedding.bin\n");
    // }
    
    // if (image_embed_file) {
    //     fwrite(result.image_embedding, sizeof(float16), result.embedding_size, image_embed_file);
    //     fclose(image_embed_file);
    //     printf("Image embedding saved to image_embedding.bin\n");
    // }
    

    // 清理资源
    printf("--> Cleaning up\n");
    free_similarity_result(&result);
    release_qaclipvit_model(&ctx);
    free(image_input.virt_addr);
    
    printf("=== chinese_clip Demo Completed Successfully ===\n");
    return 0;
}
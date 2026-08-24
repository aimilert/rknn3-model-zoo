#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <math.h>
#include <sys/time.h>

#include "rknn_qaclipvit_vision.h"
#include "common.h"
#include "file_utils.h"
#include "image_utils.h"
#include "float16.h"

static int convert_fp16_to_fp32(const float16* src, float* dst, int n_elems)
{
  for (int i = 0; i < n_elems; i++) {
    dst[i] = fp16_to_fp32(src[i]);
  }
  return 0;
}


static void dump_tensor_attr(rknn3_tensor_attr* attrs)
{
    std::string shape_str = "";
    for (int j = 0; j < attrs->n_dims; j++) {
      shape_str += std::to_string(attrs->shape[j]);
      if (j < attrs->n_dims - 1) {
        shape_str += ", ";
      }
    }
  
    std::string stride_str = "";
    for (int j = 0; j < attrs->n_stride; j++) {
      stride_str += std::to_string(attrs->stride[j]);
      if (j < attrs->n_stride - 1) {
        stride_str += ", ";
      }
    }
  
    printf("Tensor: name=%s, n_dims=%d, shape=[%s], stride=[%s], aligned_size=%ld, layout=%s, dtype=%s, core_id=%d, "
           "qnt_type=%s\n",
           attrs->name, attrs->n_dims, shape_str.c_str(), stride_str.c_str(), attrs->aligned_size, 
           rknn3_get_layout_string(attrs->layout), rknn3_get_type_string(attrs->dtype), 
           attrs->core_id, rknn3_get_qnt_type_string(attrs->qnt_type));
}

int init_qaclipvit_vision_model(const char *model_path, const char* weight_path, 
                               qaclipvit_vision_context_t *app_ctx, uint32_t core_mask)
{
    int ret;
    rknn3_context ctx = 0;

    rknn3_config config;
    memset(&config, 0, sizeof(config));
    config.run_core_mask = core_mask;

    // RKNN Init
    ret = rknn3_init(&ctx, NULL);
    if (ret < 0)
    {
        printf("rknn_init fail ret=%d\n", ret);
        return ret;
    }

    // Load RKNN Model
    ret = rknn3_load_model_from_path(ctx, model_path, weight_path);
    if (ret < 0)
    {
        printf("load_model fail!\n");
        return -1;
    }

    // Init RKNN Model
    ret = rknn3_model_init(ctx, &config);
    if (ret < 0) {
        printf("rknn_model_init failed! ret=%d\n", ret);
        return ret;
    }

    // Get Model Input Output Number
    rknn3_input_output_num io_num;
    ret = rknn3_query(ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return ret;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Get Model Input Info
    printf("input tensors:\n");
    rknn3_tensor_attr input_attrs[io_num.n_input];
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return ret;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    printf("output tensors:\n");
    rknn3_tensor_attr output_attrs[io_num.n_output];
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return ret;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // Set to context
    app_ctx->inputs = (rknn3_tensor*)malloc(io_num.n_input * sizeof(rknn3_tensor));
    app_ctx->outputs = (rknn3_tensor*)malloc(io_num.n_output * sizeof(rknn3_tensor));
    app_ctx->rknn_ctx = ctx;
    app_ctx->io_num = io_num;
    
    for (int i = 0; i < app_ctx->io_num.n_input; i++) {
        app_ctx->inputs[i].mem  = rknn3_create_mem(ctx, input_attrs[i].aligned_size, input_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        app_ctx->inputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        memcpy(app_ctx->inputs[i].attr, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
    }
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        app_ctx->outputs[i].mem  = rknn3_create_mem(ctx, output_attrs[i].aligned_size, output_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        app_ctx->outputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        memcpy(app_ctx->outputs[i].attr, &(output_attrs[i]), sizeof(rknn3_tensor_attr));
    }

    // 设置模型输入尺寸（假设第一个输入是图像）
    if (input_attrs[0].layout == RKNN3_TENSOR_NHWC)
    {
        printf("model is NHWC input fmt\n");
        app_ctx->model_channel = input_attrs[0].shape[3];
        app_ctx->model_height = input_attrs[0].shape[1];
        app_ctx->model_width = input_attrs[0].shape[2];
    }
    else
    {
        printf("model is NCHW input fmt\n");
        app_ctx->model_height = input_attrs[0].shape[2];
        app_ctx->model_width = input_attrs[0].shape[3];
        app_ctx->model_channel = input_attrs[0].shape[1];
    }
    printf("model image input height=%d, width=%d, channel=%d\n",
           app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);

    return 0;
}

int release_qaclipvit_vision_model(qaclipvit_vision_context_t *app_ctx)
{
    for (int i = 0; i < app_ctx->io_num.n_input; i++) {
        if (app_ctx->inputs[i].mem) {
            rknn3_destroy_mem(app_ctx->rknn_ctx, app_ctx->inputs[i].mem);
        }
        if (app_ctx->inputs[i].attr != NULL) {
            free(app_ctx->inputs[i].attr);
            app_ctx->inputs[i].attr = NULL;
        }
    }
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        if (app_ctx->outputs[i].mem) {
            rknn3_destroy_mem(app_ctx->rknn_ctx, app_ctx->outputs[i].mem);
        }
        if (app_ctx->outputs[i].attr != NULL) {
            free(app_ctx->outputs[i].attr);
            app_ctx->outputs[i].attr = NULL;
        }
    }
    if (app_ctx->rknn_ctx != 0)
    {
        rknn3_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    
    free(app_ctx->inputs);
    free(app_ctx->outputs);
    
    return 0;
}

// 清理处理图像的函数
static void cleanup_processed_img(image_buffer_t *processed_img) {
    if (processed_img->virt_addr != NULL) {
        free(processed_img->virt_addr);
        processed_img->virt_addr = NULL;
    }
}

int inference_qaclipvit_vision_model(qaclipvit_vision_context_t *app_ctx, 
                                    image_buffer_t *image_input,
                                    qaclipvit_vision_result_t *result)
{
    int ret = 0;
    image_buffer_t processed_img;
    letterbox_t letter_box;
    int bg_color = 114;

    // 时延计算变量
    struct timeval start_time, end_time;
    double preprocess_time, inference_time, postprocess_time;

    if ((!app_ctx) || (!image_input) || (!result))
    {
        return -1;
    }

    memset(result, 0, sizeof(qaclipvit_vision_result_t));
    memset(&letter_box, 0, sizeof(letterbox_t));
    memset(&processed_img, 0, sizeof(image_buffer_t));

    // 记录预处理开始时间
    gettimeofday(&start_time, NULL);

    // 预处理图像输入
    processed_img.width = app_ctx->model_width;
    processed_img.height = app_ctx->model_height;
    processed_img.format = IMAGE_FORMAT_RGB888;
    processed_img.size = get_image_size(&processed_img);
    processed_img.virt_addr = (unsigned char *)malloc(processed_img.size);
    if (processed_img.virt_addr == NULL)
    {
        printf("malloc image buffer size:%d fail!\n", processed_img.size);
        return -1;
    }

    ret = convert_image(image_input, &processed_img, NULL, NULL, 0);
    if (ret < 0)
    {
        printf("convert_image fail! ret=%d\n", ret);
        cleanup_processed_img(&processed_img);
        return ret;
    }

    // FILE* raw_file = fopen("processed_image.raw", "wb");
    // if (raw_file) {
    //     fwrite(image_input->virt_addr, 1, image_input->size, raw_file);
    //     fclose(raw_file);
    //     printf("RAW 数据已保存: processed_image.raw (%zu bytes)\n", image_input->size);
    // }
    // 设置图像输入数据（假设第一个输入是图像）
    memcpy(app_ctx->inputs[0].mem->virt_addr, (uint8_t*)processed_img.virt_addr, processed_img.size);
    // memcpy(app_ctx->inputs[0].mem->virt_addr, (uint8_t*)image_input->virt_addr, processed_img.size);

    // 记录预处理结束时间
    gettimeofday(&end_time, NULL);
    preprocess_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + (end_time.tv_usec - start_time.tv_usec) / 1000.0;
    printf("Pre-process time: %.2f ms\n", preprocess_time);

    // 记录推理开始时间
    gettimeofday(&start_time, NULL);

    // Sync inputs
    for (int i = 0; i < app_ctx->io_num.n_input; i++) {
        ret = rknn3_mem_sync(app_ctx->rknn_ctx, app_ctx->inputs[i].mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
        if (ret < 0) {
            printf("rknn3_mem_sync input[%d] fail! ret=%d\n", i, ret);
            cleanup_processed_img(&processed_img);
            return ret;
        }
    }

    // Run Inference
    ret = rknn3_run(app_ctx->rknn_ctx, app_ctx->inputs, app_ctx->io_num.n_input, 
                   app_ctx->outputs, app_ctx->io_num.n_output);
    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        cleanup_processed_img(&processed_img);
        return ret;
    }

    // Sync outputs
    for (int i = 0; i < app_ctx->io_num.n_output; i++)
    {
        ret = rknn3_mem_sync(app_ctx->rknn_ctx, app_ctx->outputs[i].mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
        if (ret < 0)
        {
            printf("rknn3_mem_sync output[%d] fail! ret=%d\n", i, ret);
            cleanup_processed_img(&processed_img);
            return ret;
        }
    }

    // 记录推理结束时间
    gettimeofday(&end_time, NULL);
    inference_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + (end_time.tv_usec - start_time.tv_usec) / 1000.0;
    printf("Inference time: %.2f ms\n", inference_time);

    // 记录后处理开始时间
    gettimeofday(&start_time, NULL);

    // 处理后处理输出
    rknn3_tensor_attr output_attr;
    output_attr.index = 0;
    ret = rknn3_query(app_ctx->rknn_ctx, RKNN3_QUERY_OUTPUT_ATTR, &output_attr, sizeof(rknn3_tensor_attr));
    if (ret < 0)
    {
        printf("rknn_query output attr fail! ret=%d\n", ret);
        cleanup_processed_img(&processed_img);
        return ret;
    }

    // 获取输出数据
    const float16* output_data = (const float16*)app_ctx->outputs[0].mem->virt_addr;

    int output_size = 1;
    for (int i = 0; i < output_attr.n_dims; i++) {
        output_size *= output_attr.shape[i];
    }

    // 分配结果内存
    result->image_embeddings = (float16*)malloc(output_size * sizeof(float16));
    if (result->image_embeddings == NULL) {
        printf("malloc logits buffer fail!\n");
        ret = -1;
        cleanup_processed_img(&processed_img);
        return ret;
    }
    memcpy(result->image_embeddings, app_ctx->outputs[0].mem->virt_addr, 
        output_size * sizeof(float16));
    // convert_fp16_to_fp32( output_data, result->image_embeddings, output_size);

    // 记录后处理结束时间
    gettimeofday(&end_time, NULL);
    postprocess_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + (end_time.tv_usec - start_time.tv_usec) / 1000.0;
    printf("Post-process time: %.2f ms\n", postprocess_time);
    printf("Total time: %.2f ms\n", preprocess_time + inference_time + postprocess_time);

    // 清理处理图像
    cleanup_processed_img(&processed_img);

    return ret;
}
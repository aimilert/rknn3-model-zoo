#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <math.h>
#include <sys/time.h>

#include "rknn_qaclipvit_embedding.h"
#include "common.h"
#include "file_utils.h"
#include "float16.h"

// 计算张量逻辑数据大小（基于 shape × dtype 元素大小，不包含对齐填充）
static size_t get_tensor_data_size(rknn3_tensor_attr* attr) {
    size_t size = 1;
    for (int i = 0; i < attr->n_dims; i++) {
        size *= attr->shape[i];
    }
    switch (attr->dtype) {
        case RKNN3_TENSOR_FLOAT16: size *= 2; break;
        case RKNN3_TENSOR_FLOAT32: size *= 4; break;
        case RKNN3_TENSOR_INT32:   size *= 4; break;
        case RKNN3_TENSOR_INT8:    size *= 1; break;
        case RKNN3_TENSOR_UINT8:   size *= 1; break;
        default: break;
    }
    return size;
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

int init_qaclipvit_embedding_model(const char *model_path, const char* weight_path,
                                 qaclipvit_text_context_t *app_ctx, uint32_t core_mask)
{
    int ret = -1;
    rknn3_context ctx = 0;
    rknn3_tensor_attr* input_attrs = NULL;
    rknn3_tensor_attr* output_attrs = NULL;

    if (app_ctx == NULL) {
        return -1;
    }
    memset(app_ctx, 0, sizeof(qaclipvit_text_context_t));

    rknn3_config config;
    memset(&config, 0, sizeof(config));
    config.run_core_mask = core_mask;

    // RKNN Init
    ret = rknn3_init(&ctx, NULL);
    if (ret < 0)
    {
        printf("rknn_init fail ret=%d\n", ret);
        goto cleanup;
    }
    app_ctx->rknn_ctx = ctx;

    // Load RKNN Model
    ret = rknn3_load_model_from_path(ctx, model_path, weight_path);
    if (ret < 0)
    {
        printf("load_model fail! ret=%d\n", ret);
        goto cleanup;
    }

    // Init RKNN Model
    ret = rknn3_model_init(ctx, &config);
    if (ret < 0) {
        printf("rknn_model_init failed! ret=%d\n", ret);
        goto cleanup;
    }

    // Get Model Input Output Number
    rknn3_input_output_num io_num;
    ret = rknn3_query(ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        goto cleanup;
    }
    app_ctx->io_num = io_num;
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Get Model Input Info
    input_attrs = (rknn3_tensor_attr*)malloc(io_num.n_input * sizeof(rknn3_tensor_attr));
    if (input_attrs == NULL) {
        printf("malloc input_attrs failed!\n");
        ret = -1;
        goto cleanup;
    }
    memset(input_attrs, 0, io_num.n_input * sizeof(rknn3_tensor_attr));
    printf("input tensors:\n");
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            goto cleanup;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    output_attrs = (rknn3_tensor_attr*)malloc(io_num.n_output * sizeof(rknn3_tensor_attr));
    if (output_attrs == NULL) {
        printf("malloc output_attrs failed!\n");
        ret = -1;
        goto cleanup;
    }
    memset(output_attrs, 0, io_num.n_output * sizeof(rknn3_tensor_attr));
    printf("output tensors:\n");
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            goto cleanup;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // Set to context
    app_ctx->inputs = (rknn3_tensor*)malloc(io_num.n_input * sizeof(rknn3_tensor));
    app_ctx->outputs = (rknn3_tensor*)malloc(io_num.n_output * sizeof(rknn3_tensor));
    if (app_ctx->inputs == NULL || app_ctx->outputs == NULL) {
        printf("malloc input/output tensors failed!\n");
        ret = -1;
        goto cleanup;
    }
    memset(app_ctx->inputs, 0, io_num.n_input * sizeof(rknn3_tensor));
    memset(app_ctx->outputs, 0, io_num.n_output * sizeof(rknn3_tensor));

    for (int i = 0; i < app_ctx->io_num.n_input; i++) {
        app_ctx->inputs[i].mem = rknn3_create_mem(ctx, input_attrs[i].aligned_size,
                                                 input_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (app_ctx->inputs[i].mem == NULL) {
            printf("rknn3_create_mem input[%d] failed!\n", i);
            ret = -1;
            goto cleanup;
        }

        app_ctx->inputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        if (app_ctx->inputs[i].attr == NULL) {
            printf("malloc input attr[%d] failed!\n", i);
            ret = -1;
            goto cleanup;
        }
        memcpy(app_ctx->inputs[i].attr, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
    }
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        app_ctx->outputs[i].mem = rknn3_create_mem(ctx, output_attrs[i].aligned_size,
                                                  output_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (app_ctx->outputs[i].mem == NULL) {
            printf("rknn3_create_mem output[%d] failed!\n", i);
            ret = -1;
            goto cleanup;
        }

        app_ctx->outputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        if (app_ctx->outputs[i].attr == NULL) {
            printf("malloc output attr[%d] failed!\n", i);
            ret = -1;
            goto cleanup;
        }
        memcpy(app_ctx->outputs[i].attr, &(output_attrs[i]), sizeof(rknn3_tensor_attr));
        // app_ctx->outputs[i].attr->dtype = RKNN3_TENSOR_FLOAT16;
    }

    printf("MetaCLIP2 embedding model initialized successfully\n");
    free(input_attrs);
    free(output_attrs);
    return 0;

cleanup:
    if (input_attrs) {
        free(input_attrs);
    }
    if (output_attrs) {
        free(output_attrs);
    }
    release_qaclipvit_embedding_model(app_ctx);
    return ret;
}

int release_qaclipvit_embedding_model(qaclipvit_text_context_t *app_ctx)
{
    if (app_ctx == NULL) {
        return 0;
    }

    if (app_ctx->inputs) {
        for (int i = 0; i < app_ctx->io_num.n_input; i++) {
            if (app_ctx->inputs[i].mem) {
                rknn3_destroy_mem(app_ctx->rknn_ctx, app_ctx->inputs[i].mem);
                app_ctx->inputs[i].mem = NULL;
            }
            if (app_ctx->inputs[i].attr != NULL) {
                free(app_ctx->inputs[i].attr);
                app_ctx->inputs[i].attr = NULL;
            }
        }
        free(app_ctx->inputs);
        app_ctx->inputs = NULL;
    }

    if (app_ctx->outputs) {
        for (int i = 0; i < app_ctx->io_num.n_output; i++) {
            if (app_ctx->outputs[i].mem) {
                rknn3_destroy_mem(app_ctx->rknn_ctx, app_ctx->outputs[i].mem);
                app_ctx->outputs[i].mem = NULL;
            }
            if (app_ctx->outputs[i].attr != NULL) {
                free(app_ctx->outputs[i].attr);
                app_ctx->outputs[i].attr = NULL;
            }
        }
        free(app_ctx->outputs);
        app_ctx->outputs = NULL;
    }

    if (app_ctx->rknn_ctx != 0)
    {
        rknn3_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    memset(&app_ctx->io_num, 0, sizeof(app_ctx->io_num));
    return 0;
}

int set_qaclipvit_input_data(qaclipvit_text_context_t *app_ctx, 
                           qaclipvit_input_data_t *input_data)
{
    if (!app_ctx || !input_data) {
        return -1;
    }

    // 设置input_embeds [1, 77, EMBEDDING_DIM] FP16
    if (input_data->input_embeds) {
        size_t data_size = get_tensor_data_size(app_ctx->inputs[0].attr);
        memcpy(app_ctx->inputs[0].mem->virt_addr, input_data->input_embeds, data_size);
    }

    // 设置attention_mask [1, 77] FP16
    if (input_data->attention_mask) {
        size_t data_size = get_tensor_data_size(app_ctx->inputs[1].attr);
        memcpy(app_ctx->inputs[1].mem->virt_addr, input_data->attention_mask, data_size);
    }

    // 设置position_ids [1, MAX_SEQ_LEN] INT32
    if (input_data->position_ids) {
        size_t data_size = get_tensor_data_size(app_ctx->inputs[2].attr);
        memcpy(app_ctx->inputs[2].mem->virt_addr, input_data->position_ids, data_size);
    }

    // 设置Th, Tc, Ts, Tsr [1] INT32
    if (input_data->Th) {
        ((int*)(app_ctx->inputs[3].mem->virt_addr))[0] = ((int*)input_data->Th)[0];
        printf("Setting Th input is %d\n", ((int*)input_data->Th)[0]);
        // memcpy(app_ctx->inputs[4].mem->virt_addr, input_data->Th, 
        //        app_ctx->inputs[4].attr->aligned_size);
    }
    if (input_data->Tc) {
        ((int*)(app_ctx->inputs[4].mem->virt_addr))[0] = ((int*)input_data->Tc)[0];
        // printf("Setting Tc input is %d\n", ((int*)input_data->Tc)[0]);
        // memcpy(app_ctx->inputs[5].mem->virt_addr, input_data->Tc, 
        //        app_ctx->inputs[5].attr->aligned_size);
    }
    if (input_data->Ts) {
        ((int*)(app_ctx->inputs[5].mem->virt_addr))[0] = ((int*)input_data->Ts)[0];
        printf("Setting Ts input is %d\n", ((int*)input_data->Ts)[0]);
        // memcpy(app_ctx->inputs[6].mem->virt_addr, input_data->Ts, 
        //        app_ctx->inputs[6].attr->aligned_size);
    }
    if (input_data->Tsr) {
        ((int*)(app_ctx->inputs[6].mem->virt_addr))[0] = ((int*)input_data->Tsr)[0];
        printf("Setting Tsr input is %d\n", ((int*)input_data->Tsr)[0]);
        // memcpy(app_ctx->inputs[7].mem->virt_addr, input_data->Tsr, 
        //        app_ctx->inputs[7].attr->aligned_size);
    }

    return 0;
}

int process_qaclipvit_output(qaclipvit_text_context_t *app_ctx, 
                           qaclipvit_result_t *result)
{
    if (!app_ctx || !result) {
        return -1;
    }

    // 输出形状为 [1, EMBEDDING_DIM] FP16
    result->embedding_size = EMBEDDING_DIM;
    result->embedding = (float16*)malloc(result->embedding_size * sizeof(float16));
    
    if (!result->embedding) {
        printf("Failed to allocate memory for embedding result\n");
        return -1;
    }

    // 复制输出数据
    memcpy(result->embedding, app_ctx->outputs[0].mem->virt_addr, 
           result->embedding_size * sizeof(float16));

    return 0;
}

int inference_qaclipvit_embedding_model(qaclipvit_text_context_t *app_ctx, 
                                      qaclipvit_input_data_t *input_data, 
                                      qaclipvit_result_t *result)
{
    int ret = 0;
    
    // 时延计算变量
    struct timeval start_time, end_time;
    double preprocess_time, inference_time, postprocess_time;

    if ((!app_ctx) || (!input_data) || (!result)) {
        return -1;
    }

    memset(result, 0, sizeof(qaclipvit_result_t));

    // 记录预处理开始时间
    gettimeofday(&start_time, NULL);
    // 设置输入数据
    ret = set_qaclipvit_input_data(app_ctx, input_data);
    if (ret < 0) {
        printf("set_qaclipvit_input_data fail! ret=%d\n", ret);
        return ret;
    }
    // 记录预处理结束时间，计算时延
    gettimeofday(&end_time, NULL);
    preprocess_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + 
                     (end_time.tv_usec - start_time.tv_usec) / 1000.0;
    printf("Pre-process time: %.2f ms\n", preprocess_time);

    // 记录推理开始时间
    gettimeofday(&start_time, NULL);

    // Sync inputs
    for (int i = 0; i < app_ctx->io_num.n_input; i++) {
        printf("mem sync input[%d]\n", i);
        ret = rknn3_mem_sync(app_ctx->rknn_ctx, app_ctx->inputs[i].mem, 
                           RKNN3_MEMORY_SYNC_TO_DEVICE);
        if (ret < 0) {
            printf("rknn3_mem_sync input[%d] fail! ret=%d\n", i, ret);
            return ret;
        }
    }

    // Run Inference
    // RKNN3_TENSOR_FLOAT16
    ret = rknn3_run(app_ctx->rknn_ctx, app_ctx->inputs, app_ctx->io_num.n_input, 
                   app_ctx->outputs, app_ctx->io_num.n_output);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return ret;
    }
    // Sync outputs
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        ret = rknn3_mem_sync(app_ctx->rknn_ctx, app_ctx->outputs[i].mem, 
                           RKNN3_MEMORY_SYNC_FROM_DEVICE);
        if (ret < 0) {
            printf("rknn3_mem_sync output[%d] fail! ret=%d\n", i, ret);
            return ret;
        }
    }
    // 记录推理结束时间，计算时延
    gettimeofday(&end_time, NULL);
    inference_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + 
                    (end_time.tv_usec - start_time.tv_usec) / 1000.0;
    printf("Inference time: %.2f ms\n", inference_time);

    // 记录后处理开始时间
    gettimeofday(&start_time, NULL);

    // 处理输出
    ret = process_qaclipvit_output(app_ctx, result);
    if (ret < 0) {
        printf("process_qaclipvit_output fail! ret=%d\n", ret);
        return ret;
    }

    // 记录后处理结束时间，计算时延
    gettimeofday(&end_time, NULL);
    postprocess_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + 
                      (end_time.tv_usec - start_time.tv_usec) / 1000.0;
    printf("Post-process time: %.2f ms\n", postprocess_time);
    printf("Total time: %.2f ms\n", preprocess_time + inference_time + postprocess_time);

    return ret;
}
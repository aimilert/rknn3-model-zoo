// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <math.h>
#include <sys/time.h>

#include "yolov5.h"
#include "common.h"
#include "file_utils.h"
#include "image_utils.h"

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
           attrs->name, attrs->n_dims, shape_str.c_str(), stride_str.c_str(), attrs->aligned_size, rknn3_get_layout_string(attrs->layout),
           rknn3_get_type_string(attrs->dtype), attrs->core_id, rknn3_get_qnt_type_string(attrs->qnt_type));
}


int init_yolov5_model(const char *model_path, const char* weight_path, rknn_app_context_t *app_ctx, uint32_t core_mask, const char* postprocess_plugin_path)
{
    return init_yolov5_model_ex(model_path, weight_path, app_ctx, core_mask, postprocess_plugin_path, NULL, false);
}

int init_yolov5_model_ex(const char *model_path, const char* weight_path, rknn_app_context_t *app_ctx, uint32_t core_mask, const char* postprocess_plugin_path, const char* device_id, bool quiet)
{
    int ret;
    int model_len = 0;
    char *model;
    rknn3_context ctx = 0;

    rknn3_config config;
    memset(&config, 0, sizeof(config));
    config.run_core_mask = core_mask;

    // RKNN Init
    rknn3_init_extend ext;
    memset(&ext, 0, sizeof(ext));
    ext.device_id = (char*)device_id;   // NULL -> default / first device
    ret = rknn3_init(&ctx, device_id ? &ext : NULL);
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

    //Init RKNN Model
    ret = rknn3_model_init(ctx, &config);
    if (ret < 0) {
        printf("rknn_model_init failed! ret=%d\n", ret);
        return ret;
    }

    if (postprocess_plugin_path != NULL) {
        app_ctx->use_postprocess_plugin = true;

        ret = rknn3_register_custom_ops_plugins(ctx, postprocess_plugin_path, strlen(postprocess_plugin_path));
        if (ret != RKNN3_SUCCESS)
        {
          printf("rknn3_register_custom_ops_plugins failed! ret=%d\n", ret);
          rknn3_destroy(ctx);
          return -1;
        }
        if (!quiet)
            printf("rknn3_register_custom_ops_plugins success\n");
    }
    else {
        app_ctx->use_postprocess_plugin = false;
    }

    // Get Model Input Output Number
    rknn3_input_output_num io_num;
    rknn3_query_cmd cmd = app_ctx->use_postprocess_plugin ? RKNN3_QUERY_POSTPROCESS_IN_OUT_NUM : RKNN3_QUERY_IN_OUT_NUM;
    ret = rknn3_query(ctx, cmd, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return ret;
    }
    if (!quiet)
        printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);


    // Get Model Input Info
    if (!quiet)
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
        if (!quiet)
            dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    if (!quiet)
        printf("output tensors:\n");
    rknn3_tensor_attr output_attrs[io_num.n_output];
    for (int i = 0; i < io_num.n_output; i++)
    {
        rknn3_query_cmd cmd = app_ctx->use_postprocess_plugin ? RKNN3_QUERY_POSTPROCESS_OUTPUT_ATTR : RKNN3_QUERY_OUTPUT_ATTR;
        output_attrs[i].index = i;
        ret = rknn3_query(ctx, cmd, &(output_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return ret;
        }
        if (!quiet)
            dump_tensor_attr(&(output_attrs[i]));
    }

    // Set to context
    app_ctx->inputs = (rknn3_tensor*)malloc(io_num.n_input * sizeof(rknn3_tensor));
    app_ctx->outputs = (rknn3_tensor*)malloc(io_num.n_output * sizeof(rknn3_tensor));
    app_ctx->rknn_ctx = ctx;
    app_ctx->io_num = io_num;
    app_ctx->quiet = quiet;
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

    // TODO
    if (output_attrs[0].qnt_type == RKNN3_TENSOR_PER_LAYER_ASYMMETRIC && output_attrs[0].dtype == RKNN3_TENSOR_INT8)
    {
        app_ctx->is_quant = true;
    }
    else
    {
        app_ctx->is_quant = false;
    }

    if (input_attrs[0].layout == RKNN3_TENSOR_NHWC)
    {
        if (!quiet)
            printf("model is NHWC input fmt\n");
        app_ctx->model_channel = input_attrs[0].shape[3];
        app_ctx->model_height = input_attrs[0].shape[1];
        app_ctx->model_width = input_attrs[0].shape[2];
    }
    else
    {
        if (!quiet)
            printf("model is NCHW input fmt\n");
        app_ctx->model_height = input_attrs[0].shape[1];
        app_ctx->model_width = input_attrs[0].shape[2];
        app_ctx->model_channel = input_attrs[0].shape[3];
    }
    if (!quiet)
        printf("model input height=%d, width=%d, channel=%d\n",
               app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);

    return 0;
}

int release_yolov5_model(rknn_app_context_t *app_ctx)
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
    return 0;
}

int inference_yolov5_model(rknn_app_context_t *app_ctx, image_buffer_t *img, object_detect_result_list *od_results, rknn_yolov5_times *times)
{
    bool quiet = app_ctx ? app_ctx->quiet : false;
    int ret = 0;
    image_buffer_t dst_img;
    letterbox_t letter_box;
    const float nms_threshold = NMS_THRESH;      // 默认的NMS阈值
    const float box_conf_threshold = BOX_THRESH; // 默认的置信度阈值
    int bg_color = 114;
    rknn3_tensor_attr output_attrs;
    rknn3_query_cmd query_cmd = app_ctx->use_postprocess_plugin ? RKNN3_QUERY_POSTPROCESS_OUTPUT_ATTR : RKNN3_QUERY_OUTPUT_ATTR;

    // 时延计算变量
    struct timeval start_time, end_time;
    double preprocess_time, inference_time, postprocess_time;

    if ((!app_ctx) || (!img) || (!od_results))
    {
        return -1;
    }

    memset(od_results, 0x00, sizeof(*od_results));
    memset(&letter_box, 0, sizeof(letterbox_t));
    memset(&dst_img, 0, sizeof(image_buffer_t));

    // 记录预处理开始时间
    gettimeofday(&start_time, NULL);

    // Pre Process
    dst_img.width = app_ctx->model_width;
    dst_img.height = app_ctx->model_height;
    dst_img.format = IMAGE_FORMAT_RGB888;
    dst_img.size = get_image_size(&dst_img);
    dst_img.virt_addr = (unsigned char *)malloc(dst_img.size);
    if (dst_img.virt_addr == NULL)
    {
        printf("malloc buffer size:%d fail!\n", dst_img.size);
        return -1;
    }

    // letterbox
    ret = convert_image_with_letterbox(img, &dst_img, &letter_box, bg_color);
    if (ret < 0)
    {
        printf("convert_image_with_letterbox fail! ret=%d\n", ret);
        return -1;
    }

    // Set Input Data
    memcpy(app_ctx->inputs[0].mem->virt_addr, (uint8_t*)dst_img.virt_addr, dst_img.size);

    // 记录预处理结束时间，计算时延
    gettimeofday(&end_time, NULL);
    preprocess_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + (end_time.tv_usec - start_time.tv_usec) / 1000.0;
    if (!quiet)
        printf("Pre-process time: %.2f ms\n", preprocess_time);

    // 记录推理开始时间
    gettimeofday(&start_time, NULL);

    // Sync inputs
    for (int i = 0; i < app_ctx->io_num.n_input; i++)
    {
        ret = rknn3_mem_sync(app_ctx->rknn_ctx, app_ctx->inputs[i].mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
        if (ret < 0)
        {
            printf("rknn3_mem_sync input[%d] fail! ret=%d\n", i, ret);
            goto out;
        }
    }

    // Run
    ret = rknn3_run(app_ctx->rknn_ctx, app_ctx->inputs, app_ctx->io_num.n_input, app_ctx->outputs, app_ctx->io_num.n_output);
    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        goto out;
    }

    // Sync outputs
    for (int i = 0; i < app_ctx->io_num.n_output; i++)
    {
        ret = rknn3_mem_sync(app_ctx->rknn_ctx, app_ctx->outputs[i].mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
        if (ret < 0)
        {
            printf("rknn3_mem_sync output[%d] fail! ret=%d\n", i, ret);
            goto out;
        }
    }

    // 记录推理结束时间，计算时延
    gettimeofday(&end_time, NULL);
    inference_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + (end_time.tv_usec - start_time.tv_usec) / 1000.0;
    if (!quiet)
        printf("Inference time: %.2f ms\n", inference_time);

    // 记录后处理开始时间
    gettimeofday(&start_time, NULL);

    output_attrs.index = 0;
    ret = rknn3_query(app_ctx->rknn_ctx, query_cmd, &output_attrs, sizeof(rknn3_tensor_attr));
    if (ret < 0)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        goto out;
    }

    if(app_ctx->io_num.n_output == 1 && output_attrs.n_dims == 3 && output_attrs.shape[2] == 6)
    {
        post_process_after_exYoloPostProcess(app_ctx, app_ctx->outputs, output_attrs, &letter_box, od_results);
    }
    else
    {
        // Post Process
        post_process(app_ctx, app_ctx->outputs, &letter_box, box_conf_threshold, nms_threshold, od_results);
    }

    // 记录后处理结束时间，计算时延
    gettimeofday(&end_time, NULL);
    postprocess_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + (end_time.tv_usec - start_time.tv_usec) / 1000.0;

    if (times != NULL) {
        times->preprocess_ms = preprocess_time;
        times->inference_ms = inference_time;
        times->postprocess_ms = postprocess_time;
        times->total_ms = preprocess_time + inference_time + postprocess_time;
    }

    if (!quiet) {
        printf("Post-process time: %.2f ms\n", postprocess_time);
        printf("Total time: %.2f ms\n", preprocess_time + inference_time + postprocess_time);
    }

out:
    if (dst_img.virt_addr != NULL)
    {
        free(dst_img.virt_addr);
        dst_img.virt_addr = NULL;
    }

    return ret;
}
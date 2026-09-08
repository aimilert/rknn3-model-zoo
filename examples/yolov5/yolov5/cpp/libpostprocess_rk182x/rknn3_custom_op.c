#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "rknn3_api.h"
#include "postprocess.h"

#ifdef __RT_THREAD__
#include <rtthread.h>
#else
#include <sys/time.h>
#endif


/* ========== 插件实现 ========== */

static void dump_tensor_attr(rknn3_tensor_attr* attrs)
{
    char shape_str[256] = {0};
    char stride_str[256] = {0};
    char temp[64];
    
    for (uint32_t j = 0; j < attrs->n_dims; j++) {
        snprintf(temp, sizeof(temp), "%d", attrs->shape[j]);
        strcat(shape_str, temp);
        if (j < attrs->n_dims - 1) {
            strcat(shape_str, ", ");
        }
    }
  
    for (uint32_t j = 0; j < attrs->n_stride; j++) {
        snprintf(temp, sizeof(temp), "%lu", (unsigned long)attrs->stride[j]);
        strcat(stride_str, temp);
        if (j < attrs->n_stride - 1) {
            strcat(stride_str, ", ");
        }
    }
  
    printf("Tensor: name=%s, n_dims=%d, shape=[%s], stride=[%s], aligned_size=%ld, layout=%s, dtype=%s, core_id=%d, "
           "qnt_type=%s\n",
           attrs->name, attrs->n_dims, shape_str, stride_str, attrs->aligned_size, rknn3_get_layout_string(attrs->layout),
           rknn3_get_type_string(attrs->dtype), attrs->core_id, rknn3_get_qnt_type_string(attrs->qnt_type));
}

/* 初始化函数 - 符合 rknn3_custom_op 接口 */
static int yolo_postprocess_plugin_init(rknn3_custom_op_context *op_ctx)
{
    int ret = 0;
    
    if (!op_ctx) {
        printf("[Plugin] Error: Invalid op_ctx!\n");
        return -1;
    }
    
    printf("[Plugin] Initializing plugin...\n");
    
    /* 分配私有数据结构体 */
    rknn3_yolo_postprocess_context *pp_ctx = (rknn3_yolo_postprocess_context*)malloc(sizeof(rknn3_yolo_postprocess_context));
    if (!pp_ctx) {
        printf("[Plugin] Error: Failed to allocate private data!\n");
        return -1;
    }
    
    memset(pp_ctx, 0, sizeof(rknn3_yolo_postprocess_context));
       
    /* 设置默认参数 */
    pp_ctx->conf_threshold = BOX_THRESH;
    pp_ctx->nms_threshold = NMS_THRESH;

    // Get Model Input Output Number
    rknn3_input_output_num io_num;
    ret = rknn3_query(op_ctx->rknn_ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        free(pp_ctx);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Get Model Input Info
    printf("input tensors:\n");
    rknn3_tensor_attr *input_attrs = (rknn3_tensor_attr *)malloc(io_num.n_input * sizeof(rknn3_tensor_attr));
    memset(input_attrs, 0, io_num.n_input * sizeof(rknn3_tensor_attr));
    
    for (uint32_t i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn3_query(op_ctx->rknn_ctx, RKNN3_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            free(pp_ctx);
            free(input_attrs);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // 假设输入只有一个
    if(input_attrs[0].layout == RKNN3_TENSOR_NHWC) {
        pp_ctx->model_in_w = input_attrs[0].shape[2];
        pp_ctx->model_in_h = input_attrs[0].shape[1];
    } else if(input_attrs[0].layout == RKNN3_TENSOR_NCHW) {
        pp_ctx->model_in_w = input_attrs[0].shape[3];
        pp_ctx->model_in_h = input_attrs[0].shape[2];
    } else {
        printf("input layout not supported! layout=%d\n", input_attrs[0].layout);
        free(pp_ctx);
        free(input_attrs);
        return -1;
    }

    free(input_attrs);

    /* 保存私有数据到上下文 */
    op_ctx->priv_data = pp_ctx;

    printf("[Plugin] Initialized with rknn_ctx=%p, model_in_w=%d, model_in_h=%d, conf_threshold=%f, nms_threshold=%f\n", 
           (void*)(uintptr_t)op_ctx->rknn_ctx, pp_ctx->model_in_w, pp_ctx->model_in_h, 
           pp_ctx->conf_threshold, pp_ctx->nms_threshold);

    return 0;
}

/* 反初始化函数 - 符合 rknn3_custom_op 接口 */
static int yolo_postprocess_plugin_deinit(rknn3_custom_op_context *op_ctx)
{
    if (!op_ctx) {
        printf("[Plugin] Error: Invalid op_ctx!\n");
        return -1;
    }
    
    if (op_ctx->priv_data) {
        free(op_ctx->priv_data);
        op_ctx->priv_data = NULL;
    }
    
    printf("[Plugin] De-initialized successfully!\n");
    
    return 0;
}

/* 计算函数 - 符合 rknn3_custom_op 接口 */
static int yolo_postprocess_plugin_compute(rknn3_custom_op_context *op_ctx, rknn3_tensor *inputs, uint32_t n_inputs, 
                             rknn3_tensor *outputs, uint32_t n_outputs)
{   
    // 参数校验
    if (!op_ctx) {
        printf("[Plugin] Error: Invalid op_ctx!\n");
        return -1;
    }
    
    if (!op_ctx->priv_data) {
        printf("[Plugin] Error: Private data not initialized!\n");
        return -1;
    }
    
    if (n_inputs <= 0 || n_outputs <= 0 || inputs == NULL || outputs == NULL) {
        printf("[Plugin] Error: Invalid input or output!\n");
        return -1;
    }

    if (inputs[0].attr->layout != RKNN3_TENSOR_NCHW) {
        printf("[Plugin] Error: Input layout not supported! layout=%d\n", inputs[0].attr->layout);
        return -1;
    }


    // 调用实际的后处理函数
    rknn3_yolo_postprocess_context *pp_ctx = (rknn3_yolo_postprocess_context*)op_ctx->priv_data;
    int result_count = post_process(pp_ctx, inputs, n_inputs, outputs, n_outputs);
    
    if (result_count >= 0) {
        return 0;
    } else {
        printf("[Plugin] PostProcess Failed! result_count=%d\n", result_count);
        return result_count;
    }

    return 0;
}

/* 获取输入和输出张量属性 - 符合 rknn3_custom_op 接口 */
static int yolo_postprocess_plugin_get_attrs(rknn3_custom_op_context *op_ctx, 
                                rknn3_tensor_attr *input_attrs, uint32_t n_inputs,
                                rknn3_tensor_attr *output_attrs, uint32_t n_outputs)
{
    if (!op_ctx) {
        printf("[Plugin] Error: Invalid op_ctx!\n");
        return -1;
    }
    
    if (input_attrs == NULL) {
        printf("[Plugin] Error: input_attrs is NULL\n");
        return -1;
    }

    if (n_inputs <= 0) {
        printf("[Plugin] Error: Invalid input count!, got %d\n", n_inputs);
        return -1;
    }

    if (n_outputs != 1) {
        printf("[Plugin] Error: Invalid output count!, expect 1, but got %d\n", n_outputs);
        return -1;
    }
    
    if (!output_attrs) {
        printf("[Plugin] Error: output_attrs is NULL\n");
        return -1;
    }
    
    output_attrs[0].index = 0;
    strncpy(output_attrs[0].name, "output", RKNN3_MAX_NAME_LEN - 1);
    output_attrs[0].n_dims = 3;
    output_attrs[0].shape[0] = input_attrs[0].shape[0]; // batch
    output_attrs[0].shape[1] = MAX_OBJ_NUM;
    output_attrs[0].shape[2] = 6; // class id, prop, box
    output_attrs[0].dtype = RKNN3_TENSOR_FLOAT32;
    output_attrs[0].layout = RKNN3_TENSOR_NCHW;
    output_attrs[0].qnt_type = RKNN3_TENSOR_QNT_NONE;
    output_attrs[0].qnt_info.zero_point = 0;
    output_attrs[0].qnt_info.scale = 1.0;
    output_attrs[0].core_id = input_attrs[0].core_id;
    output_attrs[0].n_stride = output_attrs[0].n_dims;
    output_attrs[0].stride[0] = output_attrs[0].shape[0] * output_attrs[0].shape[1] * output_attrs[0].shape[2];
    output_attrs[0].stride[1] = output_attrs[0].shape[1] * output_attrs[0].shape[2];
    output_attrs[0].stride[2] = output_attrs[0].shape[2];
    output_attrs[0].n_elems = output_attrs[0].shape[0] * output_attrs[0].shape[1] * output_attrs[0].shape[2];
    output_attrs[0].aligned_size = output_attrs[0].n_elems * sizeof(float);
    
    return 0;
}

/* 获取输出数量 - 符合 rknn3_custom_op 接口 */
static int yolo_postprocess_plugin_get_output_num(rknn3_custom_op_context *op_ctx)
{
    return 1;  // YOLO 后处理输出1个张量
}


/* 插件结构体实例 - 符合 rknn3_custom_op 定义 */
static rknn3_custom_op yolo_postprocess_op = {
    .op_type = "YoloPostprocess",
    .plugin_type = RKNN3_OP_PLUGIN_TYPE_POSTPROCESS,
    .target = RKNN3_OP_TARGET_TYPE_CPU,
    .version = 1,
    .author = "Rockchip",
    .description = "YOLO Postprocess Plugin for RKNN3",
    
    /* 回调函数 */
    .init = yolo_postprocess_plugin_init,
    .prepare = NULL,  // 可选
    .compute = yolo_postprocess_plugin_compute,
    .deinit = yolo_postprocess_plugin_deinit,
    
    /* 后处理插件专用接口 */
    .get_output_num = yolo_postprocess_plugin_get_output_num,
    .get_attrs = yolo_postprocess_plugin_get_attrs,
};

/**
 * 所有注册的 custom op 数组（以 NULL 结尾）
 * 要添加更多 op，只需在数组中添加即可
 */
static rknn3_custom_op* registered_ops[] = {
    &yolo_postprocess_op,
    // &another_op,  // 如果有第二个 op，取消注释
    // &third_op,    // 如果有第三个 op，取消注释
    NULL  // 数组结尾标记，必须保留
};

rknn3_custom_op* rknn3_register_custom_ops_plugin(int op_index)
{
    printf("[Plugin] rknn3_register_custom_ops_plugin() called\n");

    if (op_index < 0) {
        printf("[Plugin] Error: Invalid index!, got %d\n", op_index);
        return NULL;
    }

    if (op_index >= sizeof(registered_ops) / sizeof(registered_ops[0])) {
        printf("[Plugin] Error: Invalid index!, got %d\n", op_index);
        return NULL;
    }
    return registered_ops[op_index];
}


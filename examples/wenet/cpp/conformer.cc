#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <algorithm>
#include <numeric>
#include <vector>
#include <string>
#include <limits>

#include "conformer.h"
#include "process.h"
#include "float16.h"

// ============================================================================
// Utility
// ============================================================================
// #include <fstream>

// bool write_data_to_file(const std::vector<float>& data, const std::string& filename) {
//     // 检查空数据（可选：根据需求决定是否允许空文件）
//     if (data.empty()) {
//         return false; // 或改为 return true; 允许创建空文件
//     }

//     std::ofstream file(filename, std::ios::binary | std::ios::out);
//     if (!file.is_open()) {
//         return false; // 文件无法创建/打开
//     }

//     // 直接写入连续内存块（float 数组的二进制表示）
//     file.write(
//         reinterpret_cast<const char*>(data.data()),
//         static_cast<std::streamsize>(data.size() * sizeof(float))
//     );

//     // 检查写入状态
//     const bool success = file.good();
//     file.close();
//     return success;
// }

static void dump_tensor_attr(rknn3_tensor_attr *attr)
{
    std::string shape_str = "";
    for (int j = 0; j < attr->n_dims; j++)
    {
        shape_str += std::to_string(attr->shape[j]);
        if (j < attr->n_dims - 1)
            shape_str += ", ";
    }
    printf("  index=%d, name=%s, n_dims=%d, shape=[%s], n_elems=%d, "
           "aligned_size=%zu, fmt=%s, type=%s, qnt_type=%s, core_id=%d\n",
           attr->index, attr->name, attr->n_dims, shape_str.c_str(),
           attr->n_elems, (size_t)attr->aligned_size,
           rknn3_get_layout_string(attr->layout),
           rknn3_get_type_string(attr->dtype),
           rknn3_get_qnt_type_string(attr->qnt_type), attr->core_id);
}

static int NCHW_fp16_to_NC1HWC2_fp16(const float16 *src, float16 *dst, int batch, int h, int w, int channel, int sub_c, int align_stride,
                                     int align_hw)
{
    printf("NCHW_fp16_to_NC1HWC2_fp16\n");
    printf("batch=%d, h=%d, w=%d, channel=%d, sub_c=%d, align_stride=%d, "
           "align_hw=%d\n",
           batch, h, w, channel, sub_c, align_stride, align_hw);

    int hw = w * h;
    int align_c = (channel + sub_c - 1) / sub_c * sub_c;
    memset(dst, 0, batch * align_c * align_hw * sizeof(float16));
    for (int b = 0; b < batch; b++)
    {
        const float16 *src_b = src + b * channel * hw;
        float16 *dst_b = dst + b * align_c * align_hw;
        for (int c = 0; c < channel; ++c)
        {
            int plane = c / sub_c;
            float16 *dstPlane = plane * align_hw * sub_c + dst_b;
            int offset = c % sub_c;
            for (int cur_h = 0; cur_h < h; ++cur_h)
                for (int cur_w = 0; cur_w < w; ++cur_w)
                {
                    int cur_hw = cur_h * align_stride + cur_w;
                    dstPlane[sub_c * cur_hw + offset] = (src_b[c * hw + cur_h * w + cur_w]);
                }
        }
    }

    return 0;
}

// 添加计算余弦相似度的函数
static float cosine_similarity(const float* a, const float* b, size_t size)
{
  float dot_product = 0.0f;
  float norm_a      = 0.0f;
  float norm_b      = 0.0f;
  for (size_t i = 0; i < size; i++) {
    float val_a = a[i];
    float val_b = b[i];

    if (!isfinite(val_a) || !isfinite(val_b))
      return 0.0f;

    dot_product += val_a * val_b;
    norm_a += val_a * val_a;
    norm_b += val_b * val_b;
  }

  if (norm_a < FLT_EPSILON || norm_b < FLT_EPSILON) {
    return 0.0f;
  }

  float norm_product = sqrtf(norm_a) * sqrtf(norm_b);

  if (norm_product < FLT_EPSILON) {
    return 0.0f;
  }

  float similarity = dot_product / norm_product;

  // 限制在 [-1.0, 1.0] 范围内
  if (similarity > 1.0f)
    similarity = 1.0f;
  if (similarity < -1.0f)
    similarity = -1.0f;

  return similarity;
}

// Load float32 data from a binary file. Returns number of floats read, or -1 on error.
static int load_bin_float32(const char *path, float *buf, int max_elems)
{
    FILE *fp = fopen(path, "rb");
    if (!fp)
    {
        printf("[WARN] load_bin_float32: cannot open %s\n", path);
        return -1;
    }
    int n = (int)fread(buf, sizeof(float), max_elems, fp);
    fclose(fp);
    return n;
}

static int getAlignHW(int hw)
{
  if (hw == 1) {
    return 1;
  } else {
    return (hw + 3) / 4 * 4;
  }
}

// Validate tensor attributes match expected model type
static int validate_tensor_attrs(conformer_model_type_t model_type,
                                  const std::vector<rknn3_tensor_attr> &input_attrs,
                                  const std::vector<rknn3_tensor_attr> &output_attrs)
{
    int n_input = (int)input_attrs.size();
    int n_output = (int)output_attrs.size();

    switch (model_type)
    {
    case CONFORMER_MODEL_ENCODER:
        // Expected: 3 inputs (chunk, att_cache, cnn_cache), 3 outputs (enc_out, att_cache, cnn_cache)
        if (n_input != 3 || n_output != 3)
        {
            printf("[VALIDATE] Encoder expects 3 inputs + 3 outputs, got %d + %d\n", n_input, n_output);
            return -1;
        }
        // input[0] = chunk: shape (1, 67, 80), float32 or float16
        if (input_attrs[0].n_dims != 3 ||
            input_attrs[0].shape[1] != DECODING_WINDOW ||
            input_attrs[0].shape[2] != N_MELS)
        {
            printf("[VALIDATE] Encoder input[0] (chunk) shape mismatch: expected (1,%d,%d)\n",
                   DECODING_WINDOW, N_MELS);
            return -1;
        }
        if (input_attrs[0].dtype != RKNN3_TENSOR_FLOAT32 &&
            input_attrs[0].dtype != RKNN3_TENSOR_FLOAT16)
        {
            printf("[VALIDATE] Encoder input[0] (chunk) dtype must be float32 or float16\n");
            return -1;
        }
        // att_cache: input[1] and output[1] must have matching n_elems
        if (input_attrs[1].n_elems != output_attrs[1].n_elems)
        {
            printf("[VALIDATE] Encoder att_cache size mismatch: input=%d, output=%d\n",
                   input_attrs[1].n_elems, output_attrs[1].n_elems);
            return -1;
        }
        // cnn_cache: input[2] and output[2] must have matching n_elems
        if (input_attrs[2].n_elems != output_attrs[2].n_elems)
        {
            printf("[VALIDATE] Encoder cnn_cache size mismatch: input=%d, output=%d\n",
                   input_attrs[2].n_elems, output_attrs[2].n_elems);
            return -1;
        }
        // output[0] = enc_out: shape (1, 16, 256)
        if (output_attrs[0].n_dims != 3 ||
            output_attrs[0].shape[2] != OUTPUT_SIZE)
        {
            printf("[VALIDATE] Encoder output[0] (enc_out) shape mismatch: expected (*,*,%d)\n", OUTPUT_SIZE);
            return -1;
        }
        break;

    case CONFORMER_MODEL_CTC:
        // Expected: 1 input (hidden), 1 output (probs)
        if (n_input != 1 || n_output != 1)
        {
            printf("[VALIDATE] CTC expects 1 input + 1 output, got %d + %d\n", n_input, n_output);
            return -1;
        }
        // input[0] = hidden: shape (1, 16, 256)
        if (input_attrs[0].n_dims != 3 ||
            input_attrs[0].shape[2] != OUTPUT_SIZE)
        {
            printf("[VALIDATE] CTC input[0] (hidden) shape mismatch: expected (*,*,%d)\n", OUTPUT_SIZE);
            return -1;
        }
        if (input_attrs[0].dtype != RKNN3_TENSOR_FLOAT32 &&
            input_attrs[0].dtype != RKNN3_TENSOR_FLOAT16)
        {
            printf("[VALIDATE] CTC input[0] (hidden) dtype must be float32 or float16\n");
            return -1;
        }
        // output[0] = probs: shape (1, 16, 4233)
        if (output_attrs[0].n_dims != 3 ||
            output_attrs[0].shape[2] != VOCAB_SIZE)
        {
            printf("[VALIDATE] CTC output[0] (probs) shape mismatch: expected (*,*,%d)\n", VOCAB_SIZE);
            return -1;
        }
        break;

    case CONFORMER_MODEL_DECODER:
        // Expected: 2 inputs (hyps, encoder_out), 2 outputs (score, r_score)
        if (n_input != 2 || n_output != 2)
        {
            printf("[VALIDATE] Decoder expects 2 inputs + 2 outputs, got %d + %d\n", n_input, n_output);
            return -1;
        }
        // input[0] = hyps: shape (1, 13), int32 or int64
        if (input_attrs[0].n_dims != 2 ||
            input_attrs[0].shape[1] != MAX_HYP_LEN)
        {
            printf("[VALIDATE] Decoder input[0] (hyps) shape mismatch: expected (*,%d)\n", MAX_HYP_LEN);
            return -1;
        }
        if (input_attrs[0].dtype != RKNN3_TENSOR_INT32 &&
            input_attrs[0].dtype != RKNN3_TENSOR_INT64)
        {
            printf("[VALIDATE] Decoder input[0] (hyps) dtype must be int32 or int64\n");
            return -1;
        }
        // input[1] = encoder_out: shape (1, 200, 256)
        if (input_attrs[1].n_dims != 3 ||
            input_attrs[1].shape[1] != ENCODER_OUT_LEN ||
            input_attrs[1].shape[2] != OUTPUT_SIZE)
        {
            printf("[VALIDATE] Decoder input[1] (encoder_out) shape mismatch: expected (1,%d,%d)\n",
                   ENCODER_OUT_LEN, OUTPUT_SIZE);
            return -1;
        }
        // output[0] = score: shape (1, 13, 4233)
        if (output_attrs[0].n_dims != 3 ||
            output_attrs[0].shape[1] != MAX_HYP_LEN ||
            output_attrs[0].shape[2] != VOCAB_SIZE)
        {
            printf("[VALIDATE] Decoder output[0] (score) shape mismatch: expected (1,%d,%d)\n",
                   MAX_HYP_LEN, VOCAB_SIZE);
            return -1;
        }
        // output[1] = r_score: shape (1, 13, 4233)
        if (output_attrs[1].n_dims != 3 ||
            output_attrs[1].shape[1] != MAX_HYP_LEN ||
            output_attrs[1].shape[2] != VOCAB_SIZE)
        {
            printf("[VALIDATE] Decoder output[1] (r_score) shape mismatch: expected (1,%d,%d)\n",
                   MAX_HYP_LEN, VOCAB_SIZE);
            return -1;
        }
        break;
    }

    printf("[VALIDATE] Model type %d: tensor shapes OK\n", (int)model_type);
    return 0;
}

int init_conformer_model(const char *model_path, const char *weight_path,
                         rknn_app_context_t *app_ctx, uint32_t core_mask,
                         conformer_model_type_t model_type)
{
    int ret;
    rknn3_context ctx = 0;

    // Validate input parameters
    if (model_path == NULL || weight_path == NULL || app_ctx == NULL)
    {
        printf("init_conformer_model: invalid parameter (NULL pointer)\n");
        return -1;
    }

    rknn3_config config;
    memset(&config, 0, sizeof(config));
    config.run_core_mask = core_mask;

    ret = rknn3_init(&ctx, NULL);
    if (ret < 0)
    {
        printf("rknn3_init fail! ret=%d\n", ret);
        return ret;
    }

    ret = rknn3_load_model_from_path(ctx, model_path, weight_path);
    if (ret < 0)
    {
        printf("rknn3_load_model_from_path fail! ret=%d model=%s\n", ret, model_path);
        rknn3_destroy(ctx);
        return ret;
    }

    ret = rknn3_model_init(ctx, &config);
    if (ret < 0)
    {
        printf("rknn3_model_init failed! ret=%d\n", ret);
        rknn3_destroy(ctx);
        return ret;
    }

    rknn3_input_output_num io_num;
    ret = rknn3_query(ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn3_query fail! ret=%d\n", ret);
        rknn3_destroy(ctx);
        return ret;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Query input attrs
    printf("input tensors:\n");
    std::vector<rknn3_tensor_attr> input_attrs(io_num.n_input);
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret < 0)
        {
            printf("rknn3_query input fail! ret=%d\n", ret);
            rknn3_destroy(ctx);
            return ret;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Query output attrs
    printf("output tensors:\n");
    std::vector<rknn3_tensor_attr> output_attrs(io_num.n_output);
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret < 0)
        {
            printf("rknn3_query output fail! ret=%d\n", ret);
            rknn3_destroy(ctx);
            return ret;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // Validate tensor shapes/dtypes/layouts match expected model type
    ret = validate_tensor_attrs(model_type, input_attrs, output_attrs);
    if (ret < 0)
    {
        printf("validate_tensor_attrs failed! model_type=%d\n", (int)model_type);
        rknn3_destroy(ctx);
        return ret;
    }

    app_ctx->rknn_ctx = ctx;
    app_ctx->io_num = io_num;

    // Allocate input tensors
    app_ctx->inputs = (rknn3_tensor *)malloc(io_num.n_input * sizeof(rknn3_tensor));
    if (app_ctx->inputs == NULL)
    {
        printf("malloc inputs failed!\n");
        rknn3_destroy(ctx);
        return -1;
    }
    memset(app_ctx->inputs, 0, io_num.n_input * sizeof(rknn3_tensor));

    // Allocate output tensors
    app_ctx->outputs = (rknn3_tensor *)malloc(io_num.n_output * sizeof(rknn3_tensor));
    if (app_ctx->outputs == NULL)
    {
        printf("malloc outputs failed!\n");
        if (app_ctx->inputs) free(app_ctx->inputs);
        rknn3_destroy(ctx);
        return -1;
    }
    memset(app_ctx->outputs, 0, io_num.n_output * sizeof(rknn3_tensor));

    for (int i = 0; i < io_num.n_input; i++)
    {
        app_ctx->inputs[i].mem = rknn3_create_mem(ctx, input_attrs[i].aligned_size,
                                                   input_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (app_ctx->inputs[i].mem == NULL)
        {
            printf("rknn3_create_mem for input[%d] failed!\n", i);
            for (int j = 0; j < i; j++)
            {
                free(app_ctx->inputs[j].attr);
                rknn3_destroy_mem(ctx, app_ctx->inputs[j].mem);
            }
            free(app_ctx->inputs);
            app_ctx->inputs = NULL;
            free(app_ctx->outputs);
            app_ctx->outputs = NULL;
            rknn3_destroy(ctx);
            app_ctx->rknn_ctx = 0;
            return -1;
        }
        app_ctx->inputs[i].attr = (rknn3_tensor_attr *)malloc(sizeof(rknn3_tensor_attr));
        if (app_ctx->inputs[i].attr == NULL)
        {
            printf("malloc input attr failed!\n");
            // Release current inputs[i].mem before cleanup
            rknn3_destroy_mem(ctx, app_ctx->inputs[i].mem);
            app_ctx->inputs[i].mem = NULL;
            for (int j = 0; j < i; j++)
            {
                free(app_ctx->inputs[j].attr);
                rknn3_destroy_mem(ctx, app_ctx->inputs[j].mem);
            }
            free(app_ctx->inputs);
            app_ctx->inputs = NULL;
            free(app_ctx->outputs);
            app_ctx->outputs = NULL;
            rknn3_destroy(ctx);
            app_ctx->rknn_ctx = 0;
            return -1;
        }
        memcpy(app_ctx->inputs[i].attr, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
    }


    for (int i = 0; i < io_num.n_output; i++)
    {
        app_ctx->outputs[i].mem = rknn3_create_mem(ctx, output_attrs[i].aligned_size,
                                                    output_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (app_ctx->outputs[i].mem == NULL)
        {
            printf("rknn3_create_mem for output[%d] failed!\n", i);
            for (int j = 0; j < io_num.n_input; j++)
            {
                free(app_ctx->inputs[j].attr);
                rknn3_destroy_mem(ctx, app_ctx->inputs[j].mem);
            }
            for (int j = 0; j < i; j++)
            {
                free(app_ctx->outputs[j].attr);
                rknn3_destroy_mem(ctx, app_ctx->outputs[j].mem);
            }
            free(app_ctx->inputs);
            app_ctx->inputs = NULL;
            free(app_ctx->outputs);
            app_ctx->outputs = NULL;
            rknn3_destroy(ctx);
            app_ctx->rknn_ctx = 0;
            return -1;
        }
        app_ctx->outputs[i].attr = (rknn3_tensor_attr *)malloc(sizeof(rknn3_tensor_attr));
        if (app_ctx->outputs[i].attr == NULL)
        {
            printf("malloc output attr failed!\n");
            // Release current outputs[i].mem before cleanup
            rknn3_destroy_mem(ctx, app_ctx->outputs[i].mem);
            app_ctx->outputs[i].mem = NULL;
            for (int j = 0; j < io_num.n_input; j++)
            {
                free(app_ctx->inputs[j].attr);
                rknn3_destroy_mem(ctx, app_ctx->inputs[j].mem);
            }
            for (int j = 0; j < i; j++)
            {
                free(app_ctx->outputs[j].attr);
                rknn3_destroy_mem(ctx, app_ctx->outputs[j].mem);
            }
            free(app_ctx->inputs);
            app_ctx->inputs = NULL;
            free(app_ctx->outputs);
            app_ctx->outputs = NULL;
            rknn3_destroy(ctx);
            app_ctx->rknn_ctx = 0;
            return -1;
        }
        memcpy(app_ctx->outputs[i].attr, &(output_attrs[i]), sizeof(rknn3_tensor_attr));
    }

    return 0;
}

int release_conformer_model(rknn_app_context_t *app_ctx)
{
    if (app_ctx == NULL)
    {
        return -1;
    }

    if (app_ctx->inputs)
    {
        for (int i = 0; i < app_ctx->io_num.n_input; i++)
        {
            if (app_ctx->inputs[i].mem)
            {
                rknn3_destroy_mem(app_ctx->rknn_ctx, app_ctx->inputs[i].mem);
                app_ctx->inputs[i].mem = NULL;
            }
            if (app_ctx->inputs[i].attr)
            {
                free(app_ctx->inputs[i].attr);
                app_ctx->inputs[i].attr = NULL;
            }
        }
        free(app_ctx->inputs);
        app_ctx->inputs = NULL;
    }

    if (app_ctx->outputs)
    {
        for (int i = 0; i < app_ctx->io_num.n_output; i++)
        {
            if (app_ctx->outputs[i].mem)
            {
                rknn3_destroy_mem(app_ctx->rknn_ctx, app_ctx->outputs[i].mem);
                app_ctx->outputs[i].mem = NULL;
            }
            if (app_ctx->outputs[i].attr)
            {
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

    // Reset io_num to prevent issues with double-release or corrupted contexts
    memset(&app_ctx->io_num, 0, sizeof(app_ctx->io_num));

    return 0;
}

// ============================================================================
// Encoder inference 
// ============================================================================

static int run_encoder(rknn_app_context_t *app_ctx)
{
    int ret;

    // Sync all inputs
    for (int i = 0; i < app_ctx->io_num.n_input; i++)
    {
        ret = rknn3_mem_sync(app_ctx->rknn_ctx, app_ctx->inputs[i].mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
        if (ret < 0)
        {
            printf("rknn3_mem_sync encoder input[%d] fail! ret=%d\n", i, ret);
            return ret;
        }
    }

    // Run
    ret = rknn3_run(app_ctx->rknn_ctx, app_ctx->inputs, app_ctx->io_num.n_input,
                    app_ctx->outputs, app_ctx->io_num.n_output);
    if (ret < 0)
    {
        printf("rknn3_run encoder fail! ret=%d\n", ret);
        return ret;
    }

    // Sync all outputs
    for (int i = 0; i < app_ctx->io_num.n_output; i++)
    {
        ret = rknn3_mem_sync(app_ctx->rknn_ctx, app_ctx->outputs[i].mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
        if (ret < 0)
        {
            printf("rknn3_mem_sync encoder output[%d] fail! ret=%d\n", i, ret);
            return ret;
        }
    }

    // Feedback caches: encoder outputs -> inputs for next chunk
    if (app_ctx->io_num.n_input >= 3 && app_ctx->io_num.n_output >= 3)
    {
        // --- att_cache: output[1] -> input[1] ---
        {
            rknn3_tensor_attr *src_attr = app_ctx->outputs[1].attr;
            rknn3_tensor_attr *dst_attr = app_ctx->inputs[1].attr;

            if (src_attr->layout == RKNN3_TENSOR_NCHW && dst_attr->layout == RKNN3_TENSOR_NC1HWC2)
            {
                int batch       = src_attr->shape[0];
                int channel     = src_attr->shape[1];
                int h           = src_attr->shape[2];
                int w           = src_attr->shape[3];
                int c2          = dst_attr->shape[4];
                int align_hw = getAlignHW(dst_attr->shape[2] * dst_attr->shape[3]);
                int align_stride = align_hw / h;

                float16 *src_fp16 = (float16 *)app_ctx->outputs[1].mem->virt_addr;
                float16 *dst_fp16 = (float16 *)app_ctx->inputs[1].mem->virt_addr;
                NCHW_fp16_to_NC1HWC2_fp16(src_fp16, dst_fp16, batch, h, w, channel, c2, align_stride, align_hw);
            }
            else
            {
                memcpy(app_ctx->inputs[1].mem->virt_addr,
                       app_ctx->outputs[1].mem->virt_addr,
                       app_ctx->inputs[1].attr->aligned_size);
            }
        }

        // --- cnn_cache: output[2] -> input[2] ---
        {
            rknn3_tensor_attr *src_attr = app_ctx->outputs[2].attr;
            rknn3_tensor_attr *dst_attr = app_ctx->inputs[2].attr;

            if (src_attr->layout == RKNN3_TENSOR_NCHW && dst_attr->layout == RKNN3_TENSOR_NC1HWC2)
            {
                int batch       = src_attr->shape[0];
                int channel     = src_attr->shape[1];
                int h           = src_attr->shape[2];
                int w           = src_attr->shape[3];
                int c2          = dst_attr->shape[4];
                int align_hw = getAlignHW(dst_attr->shape[2] * dst_attr->shape[3]);
                int align_stride = align_hw / h;

                float16 *src_fp16 = (float16 *)app_ctx->outputs[2].mem->virt_addr;
                float16 *dst_fp16 = (float16 *)app_ctx->inputs[2].mem->virt_addr;
                NCHW_fp16_to_NC1HWC2_fp16(src_fp16, dst_fp16, batch, h, w, channel, c2, align_stride, align_hw);
            }
            else
            {
                memcpy(app_ctx->inputs[2].mem->virt_addr,
                       app_ctx->outputs[2].mem->virt_addr,
                       app_ctx->inputs[2].attr->aligned_size);
            }
        }
    }

    return 0;
}

// ============================================================================
// CTC inference
// ============================================================================

static int run_ctc(rknn_app_context_t *app_ctx, float *enc_out_fp32, int enc_out_elems)
{
    int ret;

    // Copy encoder output to CTC input (handle FP16)
    if (app_ctx->inputs[0].attr->dtype == RKNN3_TENSOR_FLOAT16)
    {
        float16 *dst = (float16 *)app_ctx->inputs[0].mem->virt_addr;
        for (int i = 0; i < enc_out_elems; i++)
            dst[i] = fp32_to_fp16(enc_out_fp32[i]);
    }
    else
    {
        memcpy(app_ctx->inputs[0].mem->virt_addr, enc_out_fp32, enc_out_elems * sizeof(float));
    }

    // Sync input
    ret = rknn3_mem_sync(app_ctx->rknn_ctx, app_ctx->inputs[0].mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
    if (ret < 0)
    {
        printf("rknn3_mem_sync ctc input fail! ret=%d\n", ret);
        return ret;
    }

    // Run
    ret = rknn3_run(app_ctx->rknn_ctx, app_ctx->inputs, app_ctx->io_num.n_input,
                    app_ctx->outputs, app_ctx->io_num.n_output);
    if (ret < 0)
    {
        printf("rknn3_run ctc fail! ret=%d\n", ret);
        return ret;
    }

    // Sync output
    ret = rknn3_mem_sync(app_ctx->rknn_ctx, app_ctx->outputs[0].mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
    if (ret < 0)
    {
        printf("rknn3_mem_sync ctc output fail! ret=%d\n", ret);
        return ret;
    }

    return 0;
}

// ============================================================================
// Decoder inference
// ============================================================================

static int run_decoder(rknn_app_context_t *app_ctx)
{
    int ret;

    // Sync inputs
    for (int i = 0; i < app_ctx->io_num.n_input; i++)
    {
        ret = rknn3_mem_sync(app_ctx->rknn_ctx, app_ctx->inputs[i].mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
        if (ret < 0)
        {
            printf("rknn3_mem_sync decoder input[%d] fail! ret=%d\n", i, ret);
            return ret;
        }
    }

    // Run
    ret = rknn3_run(app_ctx->rknn_ctx, app_ctx->inputs, app_ctx->io_num.n_input,
                    app_ctx->outputs, app_ctx->io_num.n_output);
    if (ret < 0)
    {
        printf("rknn3_run decoder fail! ret=%d\n", ret);
        return ret;
    }

    // Sync outputs
    for (int i = 0; i < app_ctx->io_num.n_output; i++)
    {
        ret = rknn3_mem_sync(app_ctx->rknn_ctx, app_ctx->outputs[i].mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
        if (ret < 0)
        {
            printf("rknn3_mem_sync decoder output[%d] fail! ret=%d\n", i, ret);
            return ret;
        }
    }

    return 0;
}

// ============================================================================
// Attention rescoring
// ============================================================================

static std::vector<int> attention_rescoring(rknn_conformer_context_t *app_ctx,
                                            std::vector<CtcPrefix> &nbest,
                                            float *all_enc, int enc_len,
                                            VocabEntry *vocab)
{
    auto N = nbest.size();
    if (N == 0)
        return {};

    const auto L = MAX_HYP_LEN;           

    rknn_app_context_t *dec_ctx = &app_ctx->decoder_context;

    // Build hyps: (N, L) filled with eos, then fill tokens
    std::vector<int64_t> hyps(N * L, EOS_ID);
    for (int i = 0; i < N; i++)
    {
        hyps[i * L + 0] = SOS_ID;
        int tok_len = std::min((int)nbest[i].tokens.size(), MAX_HYP_LEN - 1);
        for (int j = 0; j < tok_len; j++)
        {
            hyps[i * L + j + 1] = nbest[i].tokens[j];
        }
    }

    // Set decoder input[0] = hyps
    if (dec_ctx->inputs[0].attr->dtype == RKNN3_TENSOR_INT32)
    {
        int32_t *dst = (int32_t *)dec_ctx->inputs[0].mem->virt_addr;
        for (int i = 0; i < N * L; i++)
            dst[i] = (int32_t)hyps[i];
    }
    else
    {
        memcpy(dec_ctx->inputs[0].mem->virt_addr, hyps.data(), N * L * sizeof(int64_t));
    }

    // Set decoder input[1] = all_enc (handle FP16)
    int enc_total = 1 * enc_len * OUTPUT_SIZE;
    if (dec_ctx->inputs[1].attr->dtype == RKNN3_TENSOR_FLOAT16)
    {
        float16 *dst = (float16 *)dec_ctx->inputs[1].mem->virt_addr;
        for (int i = 0; i < enc_total; i++)
            dst[i] = fp32_to_fp16(all_enc[i]);
    }
    else
    {
        memcpy(dec_ctx->inputs[1].mem->virt_addr, all_enc, enc_total * sizeof(float));
    }

    // Run decoder
    int ret = run_decoder(dec_ctx);
    if (ret < 0)
    {
        printf("run_decoder fail in attention_rescoring! ret=%d\n", ret);
        return nbest[0].tokens;
    }

    // Convert decoder outputs to float and apply log (softmax -> log)
    const auto V = VOCAB_SIZE;
    const auto out_elems = N * L * V;

    float *scores = (float *)malloc(out_elems * sizeof(float));
    if (scores==NULL)
    {
        printf("scores is null and malloc fail for scores!\n");
        return nbest[0].tokens;
    }
    float *r_scores = NULL;

    if (dec_ctx->outputs[0].attr->dtype == RKNN3_TENSOR_FLOAT16)
    {
        float16 *src = (float16 *)dec_ctx->outputs[0].mem->virt_addr;
        for (int i = 0; i < out_elems; i++)
            scores[i] = logf(fp16_to_fp32(src[i]));
    }
    else
    {
        float *src = (float *)dec_ctx->outputs[0].mem->virt_addr;
        for (int i = 0; i < out_elems; i++)
            scores[i] = logf(src[i]);
    }

    if (dec_ctx->io_num.n_output > 1)
    {
        r_scores = (float *)malloc(out_elems * sizeof(float));
        if (dec_ctx->outputs[1].attr->dtype == RKNN3_TENSOR_FLOAT16)
        {
            float16 *src = (float16 *)dec_ctx->outputs[1].mem->virt_addr;
            for (int i = 0; i < out_elems; i++)
                r_scores[i] = logf(fp16_to_fp32(src[i]));
        }
        else
        {
            float *src = (float *)dec_ctx->outputs[1].mem->virt_addr;
            for (int i = 0; i < out_elems; i++)
                r_scores[i] = logf(src[i]);
        }
    }

    // Score each hypothesis
    float best_sc = -std::numeric_limits<float>::infinity();
    int best_i = 0;

    for (int i = 0; i < N; i++)
    {
        int tok_len = std::min((int)nbest[i].tokens.size(), MAX_HYP_LEN - 1);
        float sc = 0.0f;

        // Forward score
        for (int j = 0; j < tok_len; j++)
        {
            int w = nbest[i].tokens[j];
            sc += scores[i * L * V + j * V + w];
        }
        sc += scores[i * L * V + tok_len * V + EOS_ID];

        // Reverse score
        if (REVERSE_WEIGHT > 0 && r_scores != NULL)
        {
            float rsc = 0.0f;
            for (int j = 0; j < tok_len; j++)
            {
                int w = nbest[i].tokens[j];
                rsc += r_scores[i * L * V + (tok_len - j - 1) * V + w];
            }
            rsc += r_scores[i * L * V + tok_len * V + EOS_ID];
            sc = sc * (1.0f - REVERSE_WEIGHT) + rsc * REVERSE_WEIGHT;
        }

        sc += nbest[i].score * CTC_WEIGHT;

        if (sc > best_sc)
        {
            best_sc = sc;
            best_i = i;
        }
    }

    std::vector<int> result = nbest[best_i].tokens;

    if (scores)
        free(scores);
    if (r_scores)
        free(r_scores);

    return result;
}

// ============================================================================
// Main inference pipeline
// ============================================================================

int inference_conformer_model(rknn_conformer_context_t *app_ctx,
                              audio_buffer_t audio, VocabEntry *vocab,
                              std::string &result_text)
{
    int ret;
    rknn_app_context_t *enc_ctx = &app_ctx->encoder_context;
    rknn_app_context_t *ctc_ctx = &app_ctx->ctc_context;

    bool enc_out_fp16 = (enc_ctx->outputs[0].attr->dtype == RKNN3_TENSOR_FLOAT16);
    int enc_out_T = enc_ctx->outputs[0].attr->shape[1]; // e.g. 16
    int enc_out_D = enc_ctx->outputs[0].attr->shape[2]; // e.g. 256
    int enc_out_elems = enc_ctx->outputs[0].attr->n_elems;

    // Initialize encoder caches (att_cache + cnn_cache) to zero
    for (int i = 1; i < enc_ctx->io_num.n_input; i++)
    {
        memset(enc_ctx->inputs[i].mem->virt_addr, 0, enc_ctx->inputs[i].attr->aligned_size);
    }

    // Allocate buffers
    float *enc_chunk_fp32 = (float *)malloc(enc_out_elems * sizeof(float));
    std::vector<float> all_enc_buf;
    std::vector<float> ctc_out_buf;
    int total_enc_T = 0;
    int total_ctc_T = 0;

    // Compute fbank 
    knf::FbankOptions fbank_opts;
    fbank_opts.frame_opts.samp_freq = SAMPLE_RATE;
    fbank_opts.mel_opts.num_bins = N_MELS;
    fbank_opts.mel_opts.high_freq = -400;
    fbank_opts.frame_opts.dither = 0;
    fbank_opts.frame_opts.snip_edges = false;
    knf::OnlineFbank fbank(fbank_opts);

    // Scale audio to int-16 range 
    for (int i = 0; i < audio.num_frames; i++)
        audio.data[i] *= (1 << 15);

    fbank.AcceptWaveform(SAMPLE_RATE, audio.data, audio.num_frames);
    fbank.InputFinished();
    int T = fbank.NumFramesReady();

    // Pad tail with decoding_window zero frames (matches Python:
    // feat = np.concatenate([feat, np.zeros((decoding_window, 80))])
    int padded_T = T + DECODING_WINDOW;
    float *feat_buf = (float *)calloc(padded_T * N_MELS, sizeof(float));
    for (int i = 0; i < T; i++)
    {
        const float *frame = fbank.GetFrame(i);
        memcpy(feat_buf + i * N_MELS, frame, N_MELS * sizeof(float));
    }
    // frames T..padded_T-1 are already zero from calloc

    int num_chunks = std::max(1, (T - DECODING_WINDOW) / STRIDE + 1);
    printf("\n[STREAMING] %d fbank frames => ~%d chunks\n", T, num_chunks);

    // Allocate fbank chunk buffer
    float *fbank_chunk = (float *)malloc(DECODING_WINDOW * N_MELS * sizeof(float));

    // Streaming loop 
    int cid = 0;
    while (true) {

        int start = cid * STRIDE;
        if (start >= T)
            break;

        // Extract chunk: feat_buf[start : start+DECODING_WINDOW] (always 67 frames)
        memcpy(fbank_chunk, feat_buf + start * N_MELS, DECODING_WINDOW * N_MELS * sizeof(float));

        // Set encoder input[0] = chunk (handle FP16)
        if (enc_ctx->inputs[0].attr->dtype == RKNN3_TENSOR_FLOAT16)
        {
            float16 *dst = (float16 *)enc_ctx->inputs[0].mem->virt_addr;
            for (int i = 0; i < DECODING_WINDOW * N_MELS; i++)
            {
                dst[i] = fp32_to_fp16(fbank_chunk[i]);
            }
        }
        else
        {
            float *dst = (float *)enc_ctx->inputs[0].mem->virt_addr;
            memcpy(dst, fbank_chunk, DECODING_WINDOW * N_MELS * sizeof(float));
        }

        // Run encoder
        ret = run_encoder(enc_ctx);
        if (ret < 0)
        {
            printf("run_encoder fail at chunk %d! ret=%d\n", cid, ret);
            goto out;
        }

        // Read encoder output -> fp32
        if (enc_out_fp16)
        {
            float16 *src = (float16 *)enc_ctx->outputs[0].mem->virt_addr;
            for (int i = 0; i < enc_out_elems; i++)
                enc_chunk_fp32[i] = fp16_to_fp32(src[i]);
        }
        else
        {
            memcpy(enc_chunk_fp32, enc_ctx->outputs[0].mem->virt_addr, enc_out_elems * sizeof(float));
        }

        // Accumulate encoder output (1, T', D)
        all_enc_buf.insert(all_enc_buf.end(), enc_chunk_fp32, enc_chunk_fp32 + enc_out_elems);

        // Run CTC
        ret = run_ctc(ctc_ctx, enc_chunk_fp32, enc_out_elems);
        if (ret < 0)
        {
            printf("run_ctc fail at chunk %d! ret=%d\n", cid, ret);
            goto out;
        }

        // Read CTC output -> fp32, convert softmax to log
        {
            int ctc_elems = ctc_ctx->outputs[0].attr->n_elems;
            size_t old_size = ctc_out_buf.size();
            ctc_out_buf.resize(old_size + ctc_elems);
            float *ctc_dst = ctc_out_buf.data() + old_size;
            if (ctc_ctx->outputs[0].attr->dtype == RKNN3_TENSOR_FLOAT16)
            {
                float16 *src = (float16 *)ctc_ctx->outputs[0].mem->virt_addr;
                for (int i = 0; i < ctc_elems; i++)
                    ctc_dst[i] = logf(fp16_to_fp32(src[i]));
            }
            else
            {
                float *src = (float *)ctc_ctx->outputs[0].mem->virt_addr;
                for (int i = 0; i < ctc_elems; i++)
                    ctc_dst[i] = logf(src[i]);
            }
            total_ctc_T += enc_out_T;
        }

        // Partial CTC greedy decode for display
        {
            float *ctc_chunk = ctc_out_buf.data() + (total_ctc_T - enc_out_T) * VOCAB_SIZE;
            std::vector<int> partial_ids = ctc_greedy_search(ctc_chunk, enc_out_T, VOCAB_SIZE);
            std::string partial_txt = "";
            for (int t : partial_ids)
            {
                partial_txt += vocab[t].token;
            }
            printf("  chunk %3d: fbank [%d,%d), enc %d frames, partial: %s\n",
                   cid, start, start + DECODING_WINDOW, enc_out_T,
                   partial_txt.empty() ? "(silence)" : partial_txt.c_str());
        }

        total_enc_T += enc_out_T;
        cid++;
    }

    printf("[INFO] encoder output total length: %d\n", total_enc_T);

    // -- CTC prefix beam search --
    {
        std::vector<CtcPrefix> nbest = ctc_prefix_beam_search(ctc_out_buf.data(), total_ctc_T, VOCAB_SIZE, BEAM_SIZE);
        if (nbest.empty())
        {
            printf("[WARN] CTC beam search returned empty\n");
            result_text = "";
            goto out;
        }

        // Print top 3 nbest
        for (int rank = 0; rank < std::min(3, (int)nbest.size()); rank++)
        {
            std::string txt = "";
            for (int t : nbest[rank].tokens)
                txt += vocab[t].token;
            printf("  CTC nbest[%d]: score=%.2f  %s\n", rank, nbest[rank].score, txt.c_str());
        }

        // -- Pad all_enc to ENCODER_OUT_LEN --
        {
            int enc_len = total_enc_T;
            float *padded_enc = NULL;

            if (enc_len < ENCODER_OUT_LEN)
            {
                padded_enc = (float *)calloc(1 * ENCODER_OUT_LEN * enc_out_D, sizeof(float));
                memcpy(padded_enc, all_enc_buf.data(), enc_len * enc_out_D * sizeof(float));
            }
            else
            {
                enc_len = ENCODER_OUT_LEN;
                padded_enc = (float *)malloc(1 * ENCODER_OUT_LEN * enc_out_D * sizeof(float));
                memcpy(padded_enc, all_enc_buf.data(), ENCODER_OUT_LEN * enc_out_D * sizeof(float));
            }

            // Attention rescoring
            std::vector<int> token_ids = attention_rescoring(app_ctx, nbest, padded_enc, enc_len, vocab);

            // Build result text
            result_text = "";
            for (int t : token_ids)
            {
                if (vocab[t].token)
                    result_text += vocab[t].token;
                else
                    result_text += "<unk>";
            }

            free(padded_enc);
        }
    }

out:
    if (feat_buf)
        free(feat_buf);
    if (enc_chunk_fp32)
        free(enc_chunk_fp32);
    if (fbank_chunk)
        free(fbank_chunk);

    return ret;
}

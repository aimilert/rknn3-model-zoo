// Copyright (c) 2026 by Rockchip Electronics Co., Ltd. All Rights Reserved.

#include <float.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "common.h"
#include "file_utils.h"
#include "image_utils.h"
#include "rknn3_api.h"

typedef struct {
    double copy_ms;
    double sync_to_ms;
    double execute_ms;
    double sync_from_ms;
} da3_run_timing_t;

typedef struct {
    const void* data;
    rknn3_tensor_type dtype;
    size_t n_elems;
} da3_host_input_t;

typedef struct {
    rknn3_context ctx;
    uint32_t core_num;
    uint32_t core_mask;
    rknn3_input_output_num io_num;
    rknn3_tensor_attr* input_attrs;
    rknn3_tensor_attr* output_attrs;
    rknn3_tensor* inputs;
    rknn3_tensor* outputs;
} da3_model_ctx_t;

typedef struct {
    rknn3_context owner_ctx;
    uint32_t n_mems;
    rknn3_tensor_mem** mems;
} da3_internal_mem_pool_t;

// RKNN model lifecycle and tensor transfer helpers.
static double get_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static size_t get_dtype_size(rknn3_tensor_type dtype)
{
    switch (dtype) {
    case RKNN3_TENSOR_FLOAT32:
    case RKNN3_TENSOR_INT32:
    case RKNN3_TENSOR_UINT32:
        return 4;
    case RKNN3_TENSOR_FLOAT16:
    case RKNN3_TENSOR_INT16:
    case RKNN3_TENSOR_UINT16:
    case RKNN3_TENSOR_BFLOAT16:
        return 2;
    case RKNN3_TENSOR_INT8:
    case RKNN3_TENSOR_UINT8:
    case RKNN3_TENSOR_BOOL:
        return 1;
    case RKNN3_TENSOR_INT64:
    case RKNN3_TENSOR_UINT64:
        return 8;
    default:
        return 0;
    }
}

static size_t get_tensor_bytes(const rknn3_tensor_attr* attr)
{
    return (size_t)attr->n_elems * get_dtype_size(attr->dtype);
}

static void print_tensor_attr(const rknn3_tensor_attr* attr)
{
    uint32_t i;
    printf("Tensor: index=%u name=%s dims=[", attr->index, attr->name);
    for (i = 0; i < attr->n_dims; ++i) {
        printf("%u%s", attr->shape[i], i + 1 == attr->n_dims ? "" : ",");
    }
    printf("] n_elems=%u aligned_size=%llu dtype=%s layout=%s core_id=%d\n", attr->n_elems,
           (unsigned long long)attr->aligned_size, rknn3_get_type_string(attr->dtype),
           rknn3_get_layout_string(attr->layout), attr->core_id);
}

static int check_tensor_shape(const rknn3_tensor_attr* attr, const uint32_t* shape, uint32_t n_dims)
{
    uint32_t i;
    if (attr->n_dims != n_dims) {
        return 0;
    }
    for (i = 0; i < n_dims; ++i) {
        if (attr->shape[i] != shape[i]) {
            return 0;
        }
    }
    return 1;
}

static char* make_weight_path(const char* model_path)
{
    const char* suffix = ".rknn";
    size_t path_len = strlen(model_path);
    size_t suffix_len = strlen(suffix);
    char* weight_path = (char*)malloc(path_len + 8);

    if (weight_path == NULL) {
        return NULL;
    }
    strcpy(weight_path, model_path);
    if (path_len >= suffix_len && strcmp(model_path + path_len - suffix_len, suffix) == 0) {
        strcpy(weight_path + path_len - suffix_len, ".weight");
    } else {
        strcat(weight_path, ".weight");
    }
    return weight_path;
}

static void da3_model_release(da3_model_ctx_t* model)
{
    uint32_t i;
    if (model == NULL) {
        return;
    }
    if (model->ctx != 0) {
        for (i = 0; i < model->io_num.n_input; ++i) {
            if (model->inputs != NULL && model->inputs[i].mem != NULL) {
                rknn3_destroy_mem(model->ctx, model->inputs[i].mem);
            }
        }
        for (i = 0; i < model->io_num.n_output; ++i) {
            if (model->outputs != NULL && model->outputs[i].mem != NULL) {
                rknn3_destroy_mem(model->ctx, model->outputs[i].mem);
            }
        }
        rknn3_destroy(model->ctx);
    }
    free(model->input_attrs);
    free(model->output_attrs);
    free(model->inputs);
    free(model->outputs);
    memset(model, 0, sizeof(*model));
}

static int da3_model_init(da3_model_ctx_t* model, const char* model_path, int use_user_internal_mem)
{
    rknn3_config config;
    uint32_t i;
    char* weight_path;
    int ret;

    memset(model, 0, sizeof(*model));
    weight_path = make_weight_path(model_path);
    if (weight_path == NULL) {
        return -1;
    }

    ret = rknn3_init(&model->ctx, NULL);
    if (ret != RKNN3_SUCCESS) {
        printf("rknn3_init failed ret=%d\n", ret);
        goto fail;
    }
    ret = rknn3_load_model_from_path(model->ctx, model_path, weight_path);
    if (ret != RKNN3_SUCCESS) {
        printf("rknn3_load_model_from_path failed ret=%d model=%s weight=%s\n", ret, model_path,
               weight_path);
        goto fail;
    }
    ret =
        rknn3_query(model->ctx, RKNN3_QUERY_CORE_NUMBER, &model->core_num, sizeof(model->core_num));
    if (ret != RKNN3_SUCCESS || model->core_num == 0 || model->core_num > 32) {
        printf("RKNN3_QUERY_CORE_NUMBER failed ret=%d core_num=%u model=%s\n", ret, model->core_num,
               model_path);
        if (ret == RKNN3_SUCCESS) {
            ret = -1;
        }
        goto fail;
    }
    model->core_mask =
        model->core_num == 32 ? UINT32_MAX : (uint32_t)((1ULL << model->core_num) - 1ULL);
    printf("model=%s core_num=%u auto core_mask=0x%x\n", model_path, model->core_num,
           model->core_mask);

    memset(&config, 0, sizeof(config));
    config.run_core_mask = model->core_mask;
    config.user_mem_internal = use_user_internal_mem ? 1 : 0;
    ret = rknn3_model_init(model->ctx, &config);
    if (ret != RKNN3_SUCCESS) {
        printf("rknn3_model_init failed ret=%d model=%s\n", ret, model_path);
        goto fail;
    }
    ret = rknn3_query(model->ctx, RKNN3_QUERY_IN_OUT_NUM, &model->io_num, sizeof(model->io_num));
    if (ret != RKNN3_SUCCESS) {
        printf("RKNN3_QUERY_IN_OUT_NUM failed ret=%d\n", ret);
        goto fail;
    }

    model->input_attrs =
        (rknn3_tensor_attr*)calloc(model->io_num.n_input, sizeof(*model->input_attrs));
    model->output_attrs =
        (rknn3_tensor_attr*)calloc(model->io_num.n_output, sizeof(*model->output_attrs));
    model->inputs = (rknn3_tensor*)calloc(model->io_num.n_input, sizeof(*model->inputs));
    model->outputs = (rknn3_tensor*)calloc(model->io_num.n_output, sizeof(*model->outputs));
    if (model->input_attrs == NULL || model->output_attrs == NULL || model->inputs == NULL ||
        model->outputs == NULL) {
        ret = -1;
        goto fail;
    }

    printf("model=%s input_num=%u output_num=%u\n", model_path, model->io_num.n_input,
           model->io_num.n_output);
    printf("inputs:\n");
    for (i = 0; i < model->io_num.n_input; ++i) {
        model->input_attrs[i].index = i;
        ret = rknn3_query(model->ctx, RKNN3_QUERY_INPUT_ATTR, &model->input_attrs[i],
                          sizeof(model->input_attrs[i]));
        if (ret != RKNN3_SUCCESS) {
            printf("RKNN3_QUERY_INPUT_ATTR[%u] failed ret=%d\n", i, ret);
            goto fail;
        }
        print_tensor_attr(&model->input_attrs[i]);
        model->inputs[i].attr = &model->input_attrs[i];
        model->inputs[i].mem =
            rknn3_create_mem(model->ctx, model->input_attrs[i].aligned_size,
                             model->input_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (model->inputs[i].mem == NULL) {
            printf("rknn3_create_mem input[%u] failed\n", i);
            ret = -1;
            goto fail;
        }
    }

    printf("outputs:\n");
    for (i = 0; i < model->io_num.n_output; ++i) {
        model->output_attrs[i].index = i;
        ret = rknn3_query(model->ctx, RKNN3_QUERY_OUTPUT_ATTR, &model->output_attrs[i],
                          sizeof(model->output_attrs[i]));
        if (ret != RKNN3_SUCCESS) {
            printf("RKNN3_QUERY_OUTPUT_ATTR[%u] failed ret=%d\n", i, ret);
            goto fail;
        }
        print_tensor_attr(&model->output_attrs[i]);
        model->outputs[i].attr = &model->output_attrs[i];
        model->outputs[i].mem =
            rknn3_create_mem(model->ctx, model->output_attrs[i].aligned_size,
                             model->output_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (model->outputs[i].mem == NULL) {
            printf("rknn3_create_mem output[%u] failed\n", i);
            ret = -1;
            goto fail;
        }
    }
    free(weight_path);
    return RKNN3_SUCCESS;

fail:
    free(weight_path);
    da3_model_release(model);
    return ret;
}

static void da3_internal_mem_pool_release(da3_internal_mem_pool_t* pool)
{
    uint32_t i;

    if (pool == NULL) {
        return;
    }
    for (i = 0; i < pool->n_mems; ++i) {
        if (pool->mems != NULL && pool->mems[i] != NULL) {
            rknn3_destroy_mem(pool->owner_ctx, pool->mems[i]);
        }
    }
    free(pool->mems);
    memset(pool, 0, sizeof(*pool));
}

static int da3_init_internal_mem_share(da3_model_ctx_t** models, uint32_t n_models,
                                       da3_internal_mem_pool_t* pool)
{
    rknn3_core_mem_size** model_sizes = NULL;
    rknn3_tensor_mem*** bindings = NULL;
    rknn3_tensor_mem* mem_by_core[32] = {0};
    uint64_t max_size_by_core[32] = {0};
    uint32_t model_index;
    uint32_t core_index;
    uint32_t core_id;
    uint32_t pool_index = 0;
    int ret = -1;

    if (models == NULL || n_models == 0 || pool == NULL || models[0] == NULL) {
        return -1;
    }
    memset(pool, 0, sizeof(*pool));
    model_sizes = (rknn3_core_mem_size**)calloc(n_models, sizeof(*model_sizes));
    bindings = (rknn3_tensor_mem***)calloc(n_models, sizeof(*bindings));
    if (model_sizes == NULL || bindings == NULL) {
        goto out;
    }

    for (model_index = 0; model_index < n_models; ++model_index) {
        da3_model_ctx_t* model = models[model_index];
        model_sizes[model_index] =
            (rknn3_core_mem_size*)calloc(model->core_num, sizeof(*model_sizes[model_index]));
        bindings[model_index] =
            (rknn3_tensor_mem**)calloc(model->core_num, sizeof(*bindings[model_index]));
        if (model_sizes[model_index] == NULL || bindings[model_index] == NULL) {
            goto out;
        }
        ret = rknn3_query(model->ctx, RKNN3_QUERY_CORE_MEM_SIZE, model_sizes[model_index],
                          sizeof(*model_sizes[model_index]) * model->core_num);
        if (ret != RKNN3_SUCCESS) {
            printf("RKNN3_QUERY_CORE_MEM_SIZE model[%u] failed ret=%d\n", model_index, ret);
            goto out;
        }
        for (core_index = 0; core_index < model->core_num; ++core_index) {
            core_id = model_sizes[model_index][core_index].core_id;
            if (core_id >= 32) {
                printf("invalid internal memory core_id=%u\n", core_id);
                ret = -1;
                goto out;
            }
            if (model_sizes[model_index][core_index].internal_size > max_size_by_core[core_id]) {
                max_size_by_core[core_id] = model_sizes[model_index][core_index].internal_size;
            }
        }
    }

    for (core_id = 0; core_id < 32; ++core_id) {
        if (max_size_by_core[core_id] != 0) {
            ++pool->n_mems;
        }
    }
    pool->owner_ctx = models[0]->ctx;
    pool->mems = (rknn3_tensor_mem**)calloc(pool->n_mems, sizeof(*pool->mems));
    if (pool->mems == NULL) {
        ret = -1;
        goto out;
    }
    for (core_id = 0; core_id < 32; ++core_id) {
        if (max_size_by_core[core_id] == 0) {
            continue;
        }
        mem_by_core[core_id] = rknn3_create_mem(pool->owner_ctx, max_size_by_core[core_id], core_id,
                                                RKNN3_FLAG_MEMORY_CACHEABLE);
        if (mem_by_core[core_id] == NULL) {
            printf("create shared internal memory core=%u size=%llu failed\n", core_id,
                   (unsigned long long)max_size_by_core[core_id]);
            ret = -1;
            goto out;
        }
        pool->mems[pool_index++] = mem_by_core[core_id];
        printf("shared internal memory core=%u size=%llu\n", core_id,
               (unsigned long long)max_size_by_core[core_id]);
    }

    for (model_index = 0; model_index < n_models; ++model_index) {
        da3_model_ctx_t* model = models[model_index];
        for (core_index = 0; core_index < model->core_num; ++core_index) {
            core_id = model_sizes[model_index][core_index].core_id;
            bindings[model_index][core_index] = mem_by_core[core_id];
        }
        ret = rknn3_set_internal_mem(model->ctx, bindings[model_index], model->core_num);
        if (ret != RKNN3_SUCCESS) {
            printf("rknn3_set_internal_mem model[%u] failed ret=%d\n", model_index, ret);
            goto out;
        }
    }
    ret = RKNN3_SUCCESS;

out:
    if (model_sizes != NULL) {
        for (model_index = 0; model_index < n_models; ++model_index) {
            free(model_sizes[model_index]);
        }
    }
    if (bindings != NULL) {
        for (model_index = 0; model_index < n_models; ++model_index) {
            free(bindings[model_index]);
        }
    }
    free(model_sizes);
    free(bindings);
    if (ret != RKNN3_SUCCESS) {
        da3_internal_mem_pool_release(pool);
    }
    return ret;
}

static int copy_input_to_tensor(const da3_host_input_t* input, const rknn3_tensor_attr* attr,
                                rknn3_tensor_mem* mem)
{
    size_t bytes;
    uint32_t i;

    if (input->data == NULL || mem == NULL || mem->virt_addr == NULL ||
        input->n_elems != attr->n_elems) {
        printf("invalid input tensor data or elems, got=%zu expected=%u\n", input->n_elems,
               attr->n_elems);
        return -1;
    }
    bytes = get_tensor_bytes(attr);
    if (bytes == 0 || bytes > mem->size) {
        printf("invalid input tensor bytes=%zu mem_size=%llu\n", bytes,
               (unsigned long long)mem->size);
        return -1;
    }
    if (input->dtype == attr->dtype) {
        memcpy(mem->virt_addr, input->data, bytes);
        return RKNN3_SUCCESS;
    }
    if (input->dtype == RKNN3_TENSOR_FLOAT32 && attr->dtype == RKNN3_TENSOR_FLOAT16) {
        const float* src = (const float*)input->data;
        float16* dst = (float16*)mem->virt_addr;
        for (i = 0; i < attr->n_elems; ++i) {
            dst[i] = fp32_to_fp16(src[i]);
        }
        return RKNN3_SUCCESS;
    }
    if (input->dtype == RKNN3_TENSOR_FLOAT16 && attr->dtype == RKNN3_TENSOR_FLOAT32) {
        const float16* src = (const float16*)input->data;
        float* dst = (float*)mem->virt_addr;
        for (i = 0; i < attr->n_elems; ++i) {
            dst[i] = fp16_to_fp32(src[i]);
        }
        return RKNN3_SUCCESS;
    }
    printf("unsupported input dtype conversion %s -> %s\n", rknn3_get_type_string(input->dtype),
           rknn3_get_type_string(attr->dtype));
    return -1;
}

static int da3_model_run(da3_model_ctx_t* model, const da3_host_input_t* host_inputs,
                         uint32_t n_inputs, da3_run_timing_t* timing)
{
    uint32_t i;
    int ret;
    double t0;

    if (n_inputs != model->io_num.n_input) {
        printf("input count mismatch got=%u expected=%u\n", n_inputs, model->io_num.n_input);
        return -1;
    }
    for (i = 0; i < n_inputs; ++i) {
        t0 = get_time_ms();
        ret = copy_input_to_tensor(&host_inputs[i], &model->input_attrs[i], model->inputs[i].mem);
        if (ret != RKNN3_SUCCESS) {
            return ret;
        }
        timing->copy_ms += get_time_ms() - t0;

        t0 = get_time_ms();
        ret = rknn3_mem_sync(model->ctx, model->inputs[i].mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
        if (ret != RKNN3_SUCCESS) {
            printf("rknn3_mem_sync input[%u] failed ret=%d\n", i, ret);
            return ret;
        }
        timing->sync_to_ms += get_time_ms() - t0;
    }

    t0 = get_time_ms();
    ret = rknn3_run(model->ctx, model->inputs, model->io_num.n_input, model->outputs,
                    model->io_num.n_output);
    timing->execute_ms += get_time_ms() - t0;
    if (ret != RKNN3_SUCCESS) {
        printf("rknn3_run failed ret=%d\n", ret);
        return ret;
    }

    for (i = 0; i < model->io_num.n_output; ++i) {
        t0 = get_time_ms();
        ret = rknn3_mem_sync(model->ctx, model->outputs[i].mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
        if (ret != RKNN3_SUCCESS) {
            printf("rknn3_mem_sync output[%u] failed ret=%d\n", i, ret);
            return ret;
        }
        timing->sync_from_ms += get_time_ms() - t0;
    }
    return RKNN3_SUCCESS;
}

static int copy_output_to_fp32(const da3_model_ctx_t* model, uint32_t index, float* dst,
                               size_t dst_elems)
{
    const rknn3_tensor_attr* attr;
    const void* src;
    uint32_t i;

    if (index >= model->io_num.n_output || dst == NULL) {
        return -1;
    }
    attr = &model->output_attrs[index];
    src = model->outputs[index].mem->virt_addr;
    if (src == NULL || dst_elems != attr->n_elems) {
        return -1;
    }
    if (attr->dtype == RKNN3_TENSOR_FLOAT32) {
        memcpy(dst, src, dst_elems * sizeof(float));
        return RKNN3_SUCCESS;
    }
    if (attr->dtype == RKNN3_TENSOR_FLOAT16) {
        const float16* src_fp16 = (const float16*)src;
        for (i = 0; i < attr->n_elems; ++i) {
            dst[i] = fp16_to_fp32(src_fp16[i]);
        }
        return RKNN3_SUCCESS;
    }
    printf("unsupported output dtype=%s\n", rknn3_get_type_string(attr->dtype));
    return -1;
}

static int copy_output_to_host(const da3_model_ctx_t* model, uint32_t index, void** dst,
                               size_t* bytes)
{
    size_t output_bytes;
    void* output;

    if (index >= model->io_num.n_output || dst == NULL || bytes == NULL) {
        return -1;
    }
    output_bytes = get_tensor_bytes(&model->output_attrs[index]);
    if (output_bytes == 0 || output_bytes > model->outputs[index].mem->size) {
        return -1;
    }
    output = malloc(output_bytes);
    if (output == NULL) {
        return -1;
    }
    memcpy(output, model->outputs[index].mem->virt_addr, output_bytes);
    *dst = output;
    *bytes = output_bytes;
    return RKNN3_SUCCESS;
}

// Image preprocessing and depth visualization helpers.
static int preprocess_image_rgb(const char* path, uint8_t* output, int target_height,
                                int target_width)
{
    image_buffer_t source;
    image_buffer_t target;
    int ret;

    memset(&source, 0, sizeof(source));
    memset(&target, 0, sizeof(target));
    ret = read_image(path, &source);
    if (ret != 0) {
        printf("read_image failed path=%s ret=%d\n", path, ret);
        return ret;
    }
    if (source.format != IMAGE_FORMAT_RGB888 || source.width <= 0 || source.height <= 0 ||
        source.virt_addr == NULL) {
        printf("unsupported image format or size path=%s\n", path);
        free(source.virt_addr);
        return -1;
    }

    target.width = target_width;
    target.height = target_height;
    target.format = IMAGE_FORMAT_RGB888;
    target.size = target_width * target_height * 3;
    target.virt_addr = output;
    ret = convert_image(&source, &target, NULL, NULL, 0);
    if (ret != 0) {
        printf("image resize failed path=%s ret=%d\n", path, ret);
    }

    free(source.virt_addr);
    return ret;
}

static void print_tensor_summary(const char* name, const float* data, size_t count)
{
    size_t i;
    double sum = 0.0;
    float min_value = FLT_MAX;
    float max_value = -FLT_MAX;

    for (i = 0; i < count; ++i) {
        sum += data[i];
        if (data[i] < min_value)
            min_value = data[i];
        if (data[i] > max_value)
            max_value = data[i];
    }
    printf("%s: elems=%zu min=%f max=%f mean=%f\n", name, count, min_value, max_value, sum / count);
}

static int compare_float(const void* lhs, const void* rhs)
{
    float a = *(const float*)lhs;
    float b = *(const float*)rhs;
    return (a > b) - (a < b);
}

static void turbo_color(float value, uint8_t* rgb)
{
    float x;
    float x2;
    float x3;
    float x4;
    float x5;
    float red;
    float green;
    float blue;

    if (value < 0.0f)
        value = 0.0f;
    if (value > 1.0f)
        value = 1.0f;
    x = value;
    x2 = x * x;
    x3 = x2 * x;
    x4 = x3 * x;
    x5 = x4 * x;
    red = 0.13572138f + 4.61539260f * x - 42.66032258f * x2 + 132.13108234f * x3 -
          152.94239396f * x4 + 59.28637943f * x5;
    green = 0.09140261f + 2.19418839f * x + 4.84296658f * x2 - 14.18503333f * x3 +
            4.27729857f * x4 + 2.82956604f * x5;
    blue = 0.10667330f + 12.64194608f * x - 60.58204836f * x2 + 110.36276771f * x3 -
           89.90310912f * x4 + 27.34824973f * x5;
    red = fminf(fmaxf(red, 0.0f), 1.0f);
    green = fminf(fmaxf(green, 0.0f), 1.0f);
    blue = fminf(fmaxf(blue, 0.0f), 1.0f);
    rgb[0] = (uint8_t)(red * 255.0f + 0.5f);
    rgb[1] = (uint8_t)(green * 255.0f + 0.5f);
    rgb[2] = (uint8_t)(blue * 255.0f + 0.5f);
}

static void build_depth_valid_mask(const uint8_t* rgb, uint32_t height, uint32_t width,
                                   uint8_t* mask)
{
    const uint32_t dark_threshold = 16;
    const uint32_t min_column_pixels = height / 20 > 0 ? height / 20 : 1;
    const uint32_t min_row_pixels = width / 20 > 0 ? width / 20 : 1;
    uint32_t left = 0;
    uint32_t right = width;
    uint32_t top = 0;
    uint32_t bottom = height;
    uint32_t x;
    uint32_t y;

    while (left < right) {
        uint32_t non_black = 0;
        for (y = 0; y < height; ++y) {
            const uint8_t* pixel = rgb + ((size_t)y * width + left) * 3;
            if (pixel[0] > dark_threshold || pixel[1] > dark_threshold ||
                pixel[2] > dark_threshold) {
                ++non_black;
            }
        }
        if (non_black >= min_column_pixels)
            break;
        ++left;
    }
    while (right > left) {
        uint32_t non_black = 0;
        x = right - 1;
        for (y = 0; y < height; ++y) {
            const uint8_t* pixel = rgb + ((size_t)y * width + x) * 3;
            if (pixel[0] > dark_threshold || pixel[1] > dark_threshold ||
                pixel[2] > dark_threshold) {
                ++non_black;
            }
        }
        if (non_black >= min_column_pixels)
            break;
        --right;
    }
    while (top < bottom) {
        uint32_t non_black = 0;
        for (x = 0; x < width; ++x) {
            const uint8_t* pixel = rgb + ((size_t)top * width + x) * 3;
            if (pixel[0] > dark_threshold || pixel[1] > dark_threshold ||
                pixel[2] > dark_threshold) {
                ++non_black;
            }
        }
        if (non_black >= min_row_pixels)
            break;
        ++top;
    }
    while (bottom > top) {
        uint32_t non_black = 0;
        y = bottom - 1;
        for (x = 0; x < width; ++x) {
            const uint8_t* pixel = rgb + ((size_t)y * width + x) * 3;
            if (pixel[0] > dark_threshold || pixel[1] > dark_threshold ||
                pixel[2] > dark_threshold) {
                ++non_black;
            }
        }
        if (non_black >= min_row_pixels)
            break;
        --bottom;
    }

    if (right - left < width / 2 || bottom - top < height / 2) {
        left = 0;
        right = width;
        top = 0;
        bottom = height;
    }
    memset(mask, 0, (size_t)height * width);
    for (y = top; y < bottom; ++y) {
        memset(mask + (size_t)y * width + left, 1, right - left);
    }
    printf("depth visualization valid region: x=[%u,%u) y=[%u,%u)\n", left, right, top, bottom);
}

static int save_depth_heatmaps(const char* output_dir, const float* depth,
                               const uint8_t* valid_masks, int views, uint32_t height,
                               uint32_t width)
{
    const size_t pixels = (size_t)height * width;
    const size_t total_pixels = (size_t)views * pixels;
    const uint32_t montage_columns = views < 5 ? (uint32_t)views : 5;
    const uint32_t montage_rows = ((uint32_t)views + montage_columns - 1) / montage_columns;
    const uint32_t montage_width = montage_columns * width;
    const uint32_t montage_height = montage_rows * height;
    float* valid_values = NULL;
    uint8_t* rgb = NULL;
    uint8_t* montage = NULL;
    size_t valid_count = 0;
    float lower;
    float upper;
    int view;
    int ret = -1;

    if (mkdir(output_dir, 0755) != 0 && errno != EEXIST) {
        printf("create output directory failed path=%s errno=%d\n", output_dir, errno);
        return -1;
    }
    valid_values = (float*)malloc(total_pixels * sizeof(float));
    rgb = (uint8_t*)malloc(pixels * 3);
    montage = (uint8_t*)calloc((size_t)montage_width * montage_height, 3);
    if (valid_values == NULL || rgb == NULL || montage == NULL) {
        printf("depth heatmap buffer allocation failed\n");
        goto out;
    }

    for (view = 0; view < views; ++view) {
        const float* view_depth = depth + (size_t)view * pixels;
        const uint8_t* view_mask = valid_masks + (size_t)view * pixels;
        size_t i;

        for (i = 0; i < pixels; ++i) {
            if (view_mask[i] && view_depth[i] > 0.0f && isfinite(view_depth[i])) {
                valid_values[valid_count++] = 1.0f / view_depth[i];
            }
        }
    }
    if (valid_count <= 10) {
        printf("all depth views have too few valid pixels\n");
        goto out;
    }
    qsort(valid_values, valid_count, sizeof(float), compare_float);
    lower = valid_values[(size_t)(0.02 * (valid_count - 1))];
    upper = valid_values[(size_t)(0.98 * (valid_count - 1))];
    if (upper <= lower)
        upper = lower + 1e-6f;
    printf("shared inverse-depth visualization range: [%.6f, %.6f]\n", lower, upper);

    for (view = 0; view < views; ++view) {
        const float* view_depth = depth + (size_t)view * pixels;
        const uint8_t* view_mask = valid_masks + (size_t)view * pixels;
        image_buffer_t image = {0};
        char path[PATH_MAX];
        size_t i;
        uint32_t row;
        uint32_t montage_x = ((uint32_t)view % montage_columns) * width;
        uint32_t montage_y = ((uint32_t)view / montage_columns) * height;

        for (i = 0; i < pixels; ++i) {
            if (view_mask[i] && view_depth[i] > 0.0f && isfinite(view_depth[i])) {
                float inverse_depth = 1.0f / view_depth[i];
                float normalized = (inverse_depth - lower) / (upper - lower);
                turbo_color(normalized, rgb + i * 3);
            } else {
                memset(rgb + i * 3, 0, 3);
            }
        }
        for (row = 0; row < height; ++row) {
            memcpy(montage + ((size_t)(montage_y + row) * montage_width + montage_x) * 3,
                   rgb + (size_t)row * width * 3, (size_t)width * 3);
        }

        image.width = (int)width;
        image.height = (int)height;
        image.format = IMAGE_FORMAT_RGB888;
        image.size = (int)(pixels * 3);
        image.virt_addr = rgb;
        snprintf(path, sizeof(path), "%s/depth_%02d.jpg", output_dir, view);
        if (write_image(path, &image) != 0) {
            printf("write depth heatmap failed path=%s\n", path);
            goto out;
        }
        printf("saved depth heatmap: %s\n", path);
    }
    {
        image_buffer_t image = {0};
        char path[PATH_MAX];

        image.width = (int)montage_width;
        image.height = (int)montage_height;
        image.format = IMAGE_FORMAT_RGB888;
        image.size = (int)((size_t)montage_width * montage_height * 3);
        image.virt_addr = montage;
        snprintf(path, sizeof(path), "%s/depth_montage.jpg", output_dir);
        if (write_image(path, &image) != 0) {
            printf("write depth montage failed path=%s\n", path);
            goto out;
        }
        printf("saved depth montage: %s layout=%ux%u canvas=%ux%u\n", path, montage_columns,
               montage_rows, montage_width, montage_height);
    }
    ret = 0;

out:
    free(valid_values);
    free(rgb);
    free(montage);
    return ret;
}

// DA3 Local -> Global -> Head pipeline.
int main(int argc, char** argv)
{
    const char* depth_output_dir;
    da3_model_ctx_t local = {0};
    da3_model_ctx_t global = {0};
    da3_model_ctx_t head = {0};
    da3_model_ctx_t* models[] = {&local, &global, &head};
    da3_internal_mem_pool_t internal_pool = {0};
    da3_run_timing_t local_timing = {0};
    da3_run_timing_t global_timing = {0};
    da3_run_timing_t head_timing = {0};
    char** image_paths = NULL;
    int image_line_count = 0;
    int image_count = 0;
    int views = 0;
    uint8_t* local_input = NULL;
    uint8_t* depth_valid_masks = NULL;
    float* local_output = NULL;
    float* local_tokens = NULL;
    void** global_features = NULL;
    size_t* global_feature_bytes = NULL;
    uint32_t feature_count = 0;
    da3_host_input_t* head_inputs = NULL;
    float* depth = NULL;
    float* confidence = NULL;
    size_t local_input_elems = 0;
    size_t local_output_elems = 0;
    size_t global_input_elems = 0;
    size_t head_input_elems = 0;
    size_t output_elems = 0;
    uint32_t image_height = 0;
    uint32_t image_width = 0;
    uint32_t image_channels = 0;
    uint32_t local_tokens_per_view = 0;
    uint32_t local_hidden = 0;
    uint32_t feature_tokens = 0;
    uint32_t feature_hidden = 0;
    double total_start;
    double stage_start;
    int ret = -1;
    int i;

    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    if (argc != 6 && argc != 7) {
        printf("%s <local.rknn> <global.rknn> <head.rknn> <image_list.txt> <views> [output_dir]\n",
               argv[0]);
        return -1;
    }
    depth_output_dir = argc == 7 ? argv[6] : "depth_vis";
    {
        char* end = NULL;
        long parsed_views = strtol(argv[5], &end, 10);
        if (end == argv[5] || *end != '\0' || parsed_views <= 0 || parsed_views > INT_MAX) {
            printf("invalid views: %s\n", argv[5]);
            return -1;
        }
        views = (int)parsed_views;
    }

    image_paths = read_lines_from_file(argv[4], &image_line_count);
    image_count = image_line_count;
    while (image_paths != NULL && image_count > 0 &&
           (image_paths[image_count - 1] == NULL || image_paths[image_count - 1][0] == '\0')) {
        --image_count;
    }
    if (image_paths == NULL || image_count < views) {
        printf("image list has %d entries, but views=%d, path=%s\n", image_count, views, argv[4]);
        goto cleanup;
    }
    for (i = 0; i < views; ++i) {
        if (image_paths[i] == NULL || image_paths[i][0] == '\0') {
            printf("image_list contains an empty path at line %d\n", i + 1);
            goto cleanup;
        }
    }

    total_start = get_time_ms();

    stage_start = get_time_ms();
    ret = da3_model_init(&local, argv[1], 1);
    if (ret != RKNN3_SUCCESS)
        goto cleanup;
    printf("[time] local init %.3f ms\n", get_time_ms() - stage_start);

    stage_start = get_time_ms();
    ret = da3_model_init(&global, argv[2], 1);
    if (ret != RKNN3_SUCCESS)
        goto cleanup;
    printf("[time] global init %.3f ms\n", get_time_ms() - stage_start);

    stage_start = get_time_ms();
    ret = da3_model_init(&head, argv[3], 1);
    if (ret != RKNN3_SUCCESS)
        goto cleanup;
    printf("[time] head init %.3f ms\n", get_time_ms() - stage_start);

    stage_start = get_time_ms();
    ret = da3_init_internal_mem_share(models, 3, &internal_pool);
    if (ret != RKNN3_SUCCESS)
        goto cleanup;
    printf("[time] shared internal memory init %.3f ms\n", get_time_ms() - stage_start);

    if (local.io_num.n_input != 1 || local.io_num.n_output != 1 ||
        local.input_attrs[0].n_dims != 4 || local.input_attrs[0].shape[0] != 1 ||
        local.input_attrs[0].layout != RKNN3_TENSOR_NHWC ||
        local.input_attrs[0].dtype != RKNN3_TENSOR_UINT8 || local.output_attrs[0].n_dims != 3 ||
        local.output_attrs[0].shape[0] != 1) {
        printf("local model must use uint8 NHWC input [1,H,W,C] and output [1,L,D]\n");
        ret = -1;
        goto cleanup;
    }
    image_height = local.input_attrs[0].shape[1];
    image_width = local.input_attrs[0].shape[2];
    image_channels = local.input_attrs[0].shape[3];
    if (image_channels != 3) {
        printf("local input channels=%u, but RGB888 preprocessing requires 3 channels\n",
               image_channels);
        ret = -1;
        goto cleanup;
    }
    local_tokens_per_view = local.output_attrs[0].shape[1];
    local_hidden = local.output_attrs[0].shape[2];
    local_input_elems = local.input_attrs[0].n_elems;
    local_output_elems = local.output_attrs[0].n_elems;
    global_input_elems = (size_t)views * local_output_elems;
    local_input = (uint8_t*)malloc(local_input_elems);
    depth_valid_masks = (uint8_t*)malloc((size_t)views * image_height * image_width);
    local_output = (float*)malloc(local_output_elems * sizeof(float));
    local_tokens = (float*)malloc(global_input_elems * sizeof(float));
    if (local_input == NULL || depth_valid_masks == NULL || local_output == NULL ||
        local_tokens == NULL) {
        printf("local host buffer allocation failed\n");
        ret = -1;
        goto cleanup;
    }
    printf("pipeline config: views=%d image=%ux%ux%u local_tokens=%u hidden=%u\n", views,
           image_channels, image_height, image_width, local_tokens_per_view, local_hidden);
    for (i = 0; i < views; ++i) {
        da3_host_input_t input;
        double preprocess_start = get_time_ms();
        ret =
            preprocess_image_rgb(image_paths[i], local_input, (int)image_height, (int)image_width);
        if (ret != 0)
            goto cleanup;
        build_depth_valid_mask(local_input, image_height, image_width,
                               depth_valid_masks + (size_t)i * image_height * image_width);
        printf("[time] local view %d preprocess %.3f ms\n", i + 1,
               get_time_ms() - preprocess_start);

        input.data = local_input;
        input.dtype = RKNN3_TENSOR_UINT8;
        input.n_elems = local_input_elems;
        ret = da3_model_run(&local, &input, 1, &local_timing);
        if (ret != RKNN3_SUCCESS)
            goto cleanup;
        ret = copy_output_to_fp32(&local, 0, local_output, local_output_elems);
        if (ret != RKNN3_SUCCESS)
            goto cleanup;
        memcpy(local_tokens + (size_t)i * local_output_elems, local_output,
               local_output_elems * sizeof(float));
        printf("local view %d/%d done\n", i + 1, views);
    }
    printf("[time] local detail copy %.3f ms sync_to %.3f ms execute %.3f ms sync_from %.3f ms\n",
           local_timing.copy_ms, local_timing.sync_to_ms, local_timing.execute_ms,
           local_timing.sync_from_ms);
    print_tensor_summary("local_tokens", local_tokens, global_input_elems);

    if (global.io_num.n_input != 1 || global.io_num.n_output == 0 ||
        global.input_attrs[0].n_dims != 4 || global.input_attrs[0].shape[0] != (uint32_t)views ||
        global.input_attrs[0].shape[1] != local_tokens_per_view ||
        global.input_attrs[0].shape[2] != 1 || global.input_attrs[0].shape[3] != local_hidden ||
        global.input_attrs[0].n_elems != global_input_elems) {
        printf("global input must match local outputs as [V,L,1,D], V=%d L=%u D=%u\n", views,
               local_tokens_per_view, local_hidden);
        ret = -1;
        goto cleanup;
    }
    if (global.output_attrs[0].n_dims != 4 || global.output_attrs[0].shape[0] != (uint32_t)views ||
        global.output_attrs[0].shape[2] != 1) {
        printf("global outputs must use [V,F,1,G]\n");
        ret = -1;
        goto cleanup;
    }
    feature_tokens = global.output_attrs[0].shape[1];
    feature_hidden = global.output_attrs[0].shape[3];
    head_input_elems = global.output_attrs[0].n_elems;
    feature_count = global.io_num.n_output;
    global_features = (void**)calloc(feature_count, sizeof(*global_features));
    global_feature_bytes = (size_t*)calloc(feature_count, sizeof(*global_feature_bytes));
    if (global_features == NULL || global_feature_bytes == NULL) {
        printf("global feature array allocation failed\n");
        ret = -1;
        goto cleanup;
    }
    for (i = 0; i < (int)feature_count; ++i) {
        const uint32_t expected_shape[] = {(uint32_t)views, feature_tokens, 1, feature_hidden};
        if (!check_tensor_shape(&global.output_attrs[i], expected_shape, 4) ||
            global.output_attrs[i].dtype != RKNN3_TENSOR_FLOAT16) {
            printf("global output[%d] shape or dtype mismatch\n", i);
            ret = -1;
            goto cleanup;
        }
    }
    {
        da3_host_input_t input = {local_tokens, RKNN3_TENSOR_FLOAT32, global_input_elems};
        stage_start = get_time_ms();
        ret = da3_model_run(&global, &input, 1, &global_timing);
        if (ret != RKNN3_SUCCESS)
            goto cleanup;
        printf("[time] global run %.3f ms\n", get_time_ms() - stage_start);
    }
    printf("[time] global detail copy %.3f ms sync_to %.3f ms execute %.3f ms sync_from %.3f ms\n",
           global_timing.copy_ms, global_timing.sync_to_ms, global_timing.execute_ms,
           global_timing.sync_from_ms);
    for (i = 0; i < (int)feature_count; ++i) {
        ret = copy_output_to_host(&global, (uint32_t)i, &global_features[i],
                                  &global_feature_bytes[i]);
        if (ret != RKNN3_SUCCESS || global_feature_bytes[i] != head_input_elems * sizeof(float16)) {
            printf("global output[%d] byte size mismatch got=%zu expected=%zu\n", i,
                   global_feature_bytes[i], head_input_elems * sizeof(float16));
            ret = -1;
            goto cleanup;
        }
    }
    free(local_tokens);
    local_tokens = NULL;

    if (head.io_num.n_input != feature_count || head.io_num.n_output != 2) {
        printf("head input count=%u must match global output count=%u; output count=%u must "
               "provide depth/conf\n",
               head.io_num.n_input, feature_count, head.io_num.n_output);
        ret = -1;
        goto cleanup;
    }
    for (i = 0; i < (int)head.io_num.n_input; ++i) {
        const uint32_t expected_shape[] = {(uint32_t)views, feature_tokens, feature_hidden};
        if (!check_tensor_shape(&head.input_attrs[i], expected_shape, 3)) {
            printf("head input[%d] must match global output with singleton axis removed\n", i);
            ret = -1;
            goto cleanup;
        }
    }
    if (head.output_attrs[0].n_dims != 4 || head.output_attrs[0].shape[0] != 1 ||
        head.output_attrs[0].shape[1] != (uint32_t)views ||
        head.output_attrs[0].shape[2] != image_height ||
        head.output_attrs[0].shape[3] != image_width) {
        printf("head output must use [1,V,H,W] matching the image input\n");
        ret = -1;
        goto cleanup;
    }
    output_elems = head.output_attrs[0].n_elems;
    for (i = 0; i < (int)head.io_num.n_output; ++i) {
        const uint32_t expected_shape[] = {1, (uint32_t)views, image_height, image_width};
        if (!check_tensor_shape(&head.output_attrs[i], expected_shape, 4)) {
            printf("head output[%d] shape mismatch\n", i);
            ret = -1;
            goto cleanup;
        }
    }
    depth = (float*)malloc(output_elems * sizeof(float));
    confidence = (float*)malloc(output_elems * sizeof(float));
    if (depth == NULL || confidence == NULL) {
        printf("head output host buffer allocation failed\n");
        ret = -1;
        goto cleanup;
    }
    head_inputs = (da3_host_input_t*)calloc(head.io_num.n_input, sizeof(*head_inputs));
    if (head_inputs == NULL) {
        printf("head input descriptor allocation failed\n");
        ret = -1;
        goto cleanup;
    }
    for (i = 0; i < (int)head.io_num.n_input; ++i) {
        head_inputs[i].data = global_features[i];
        head_inputs[i].dtype = RKNN3_TENSOR_FLOAT16;
        head_inputs[i].n_elems = head_input_elems;
    }
    stage_start = get_time_ms();
    ret = da3_model_run(&head, head_inputs, head.io_num.n_input, &head_timing);
    if (ret != RKNN3_SUCCESS)
        goto cleanup;
    printf("[time] head run %.3f ms\n", get_time_ms() - stage_start);
    printf("[time] head detail copy %.3f ms sync_to %.3f ms execute %.3f ms sync_from %.3f ms\n",
           head_timing.copy_ms, head_timing.sync_to_ms, head_timing.execute_ms,
           head_timing.sync_from_ms);
    ret = copy_output_to_fp32(&head, 0, depth, output_elems);
    if (ret != RKNN3_SUCCESS)
        goto cleanup;
    ret = copy_output_to_fp32(&head, 1, confidence, output_elems);
    if (ret != RKNN3_SUCCESS)
        goto cleanup;
    print_tensor_summary("depth", depth, output_elems);
    print_tensor_summary("confidence", confidence, output_elems);
    ret = save_depth_heatmaps(depth_output_dir, depth, depth_valid_masks, views, image_height,
                              image_width);
    if (ret != 0)
        goto cleanup;
    printf("depth output directory: %s\n", depth_output_dir);
    printf("[time] total %.3f ms\n", get_time_ms() - total_start);
    printf("DA3-BASE split RKNN C pipeline finished.\n");
    ret = 0;

cleanup:
    da3_model_release(&head);
    da3_model_release(&global);
    da3_internal_mem_pool_release(&internal_pool);
    da3_model_release(&local);
    if (image_paths != NULL)
        free_lines(image_paths, image_line_count);
    free(local_input);
    free(depth_valid_masks);
    free(local_output);
    free(local_tokens);
    if (global_features != NULL) {
        for (i = 0; i < (int)feature_count; ++i)
            free(global_features[i]);
    }
    free(global_features);
    free(global_feature_bytes);
    free(head_inputs);
    free(depth);
    free(confidence);
    return ret;
}

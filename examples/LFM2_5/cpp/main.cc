// Copyright (c) 2026 by Rockchip Electronics Co., Ltd. All Rights Reserved.
// Licensed under the Apache License, Version 2.0

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <errno.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>

#include "rknn_lfm2_5_llm.h"
#include "time_utils.h"

// -----------------------------------------------------------------------
// Sampling params (LFM2.5 argmax, C++11 compatible)
// -----------------------------------------------------------------------

static rknn3_sampling_params make_sampling_params()
{
    rknn3_sampling_params p;
    memset(&p, 0, sizeof(p));
    p.top_k = 1;
    p.top_p = 1.0f;
    p.temperature = 1.0f;
    p.repeat_penalty = 1.0f;
    p.frequency_penalty = 0.0f;
    p.presence_penalty = 0.0f;
    return p;
}

static const rknn3_sampling_params SAMPLE_PARAMS = make_sampling_params();

struct embedding_info
{
    int fd;
    float16 *embedding_data;
    int embedding_dim;
    int vocab_size;
};

// Encapsulates per-inference state passed to callbacks via userdata.
// Avoids global variables so the code is reentrant and thread-safe.
struct InferState
{
    Tokenizer *tokenizer;
    struct embedding_info *embed_info;
    int64_t first_token_time;
    bool first_decode;
};

// -----------------------------------------------------------------------
// Callbacks
// -----------------------------------------------------------------------

static int result_callback(void *userdata, RKLLMResult *result, LLMCallState state)
{
    struct InferState *st = (struct InferState *)userdata;
    Tokenizer *tokenizer = st->tokenizer;

    if (state == RKLLM_RUN_ERROR)
    {
        printf("\n\nError occurred during inference\n");
        return 0;
    }
    else if (state == RKLLM_RUN_FINISH)
    {
        printf("\n\n--------------------Finished-------------------- \n");
        return 0;
    }
    else if (state == RKLLM_RUN_WAITING)
    {
        printf("\n\nWaiting for UTF-8 encoded character\n");
        return 0;
    }
    else if (state == RKLLM_RUN_MAX_NEW_TOKEN_REACHED)
    {
        printf("\n\n--------------Max new token reached------------- \n");
        return 0;
    }
    else if (state == RKLLM_RUN_STOP)
    {
        printf("\n\n-----------------------Stop--------------------- \n");
        return 0;
    }
    else if (state == RKLLM_RUN_NORMAL)
    {
        // Get token text
        std::string piece;
        if (result->num_tokens == 1)
        {
            piece = tokenizer->TokenToPiece(result->token_ids[0]);
        }
        else
        {
            piece = tokenizer->Decode(result->token_ids, result->num_tokens);
        }

        // Print token text
        printf("%s", piece.c_str());

        if (st->first_decode)
        {
            st->first_token_time = getCurrentTimeUs();
            st->first_decode = false;
        }
        fflush(stdout);
    }
    return 0;
}

static int embed_callback(void *userdata, int32_t *tokens, uint64_t num_tokens,
                          void *embed, uint64_t len)
{
    struct InferState *st = (struct InferState *)userdata;
    struct embedding_info *embed_info = st->embed_info;

    // Overflow-safe computation of expected buffer size:
    //   expected = num_tokens * embedding_dim * sizeof(float16)
    uint64_t dim = static_cast<uint64_t>(embed_info->embedding_dim);
    uint64_t elem_size = sizeof(float16);
    if (dim == 0 || elem_size == 0 || num_tokens == 0)
    {
        printf("embed_callback: invalid params (num_tokens=%llu, dim=%llu)\n",
               static_cast<unsigned long long>(num_tokens),
               static_cast<unsigned long long>(dim));
        return -1;
    }
    if (num_tokens > UINT64_MAX / dim)
    {
        printf("embed_callback: num_tokens * dim overflow\n");
        return -1;
    }
    uint64_t row_bytes = num_tokens * dim;
    if (row_bytes > UINT64_MAX / elem_size)
    {
        printf("embed_callback: row_bytes * elem_size overflow\n");
        return -1;
    }
    uint64_t expected = row_bytes * elem_size;
    if (len != expected)
    {
        printf("invalid embed buffer (len=%llu, expected=%llu)\n",
               static_cast<unsigned long long>(len),
               static_cast<unsigned long long>(expected));
        return -1;
    }

    for (uint64_t n = 0; n < num_tokens; n++)
    {
        if (tokens[n] < 0 || tokens[n] >= embed_info->vocab_size)
        {
            printf("embed_callback: token id %d out of range [0, %d)\n", tokens[n], embed_info->vocab_size);
            return -1;
        }
        memcpy((unsigned char *)embed + n * embed_info->embedding_dim * sizeof(float16),
               embed_info->embedding_data + tokens[n] * embed_info->embedding_dim,
               embed_info->embedding_dim * sizeof(float16));
    }

    return 0;
}

static int tokenizer_callback(void *userdata, const char *text, int32_t text_len,
                               int32_t *tokens, int32_t n_tokens_max)
{
    int n_tokens = 0;
    struct InferState *st = (struct InferState *)userdata;
    n_tokens = st->tokenizer->Tokenize(text, text_len, tokens, n_tokens_max);

    if (n_tokens <= 0)
    {
        printf("tokenizer failed for %s\n", text);
        return n_tokens;
    }

    return n_tokens;
}

// -----------------------------------------------------------------------
// Performance
// -----------------------------------------------------------------------

void printf_perf(rknn_perf_metrics_t *p, int64_t first_token_time)
{
    printf("\n--------------------------------------------------------------------------------------\n");
    printf(" %-12s  %-15s  %-8s  %-23s  %-23s\n",
           "Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second");
    printf("--------------------------------------------------------------------------------------\n");

    float ttft_us = (float)(first_token_time - p->llm_start_time);
    int prefill_n_tokens = p->n_prefill_tokens;
    float prefill_ms = ttft_us / 1000.0;
    float prefill_tpt = prefill_n_tokens == 0 ? 0.0f : prefill_ms / prefill_n_tokens;
    float prefill_tps = prefill_n_tokens == 0 ? 0.0f : 1e3f / prefill_ms * prefill_n_tokens;
    printf(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f\n",
           "Prefill", prefill_ms, prefill_n_tokens, prefill_tpt, prefill_tps);

    float decode_time_us = (float)(p->llm_end_time - first_token_time);
    float decode_ms = decode_time_us / 1000.0;
    int decode_n_tokens = p->n_decode_tokens;
    float decode_tpt = decode_n_tokens == 0 ? 0.0f : decode_ms / decode_n_tokens;
    float decode_tps = decode_n_tokens == 0 ? 0.0f : 1e3f / decode_ms * decode_n_tokens;
    printf(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f\n",
           "Generate", decode_ms, decode_n_tokens, decode_tpt, decode_tps);

    printf("--------------------------------------------------------------------------------------\n");
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------

int main(int argc, char **argv)
{
    if (argc != 7 && argc != 8)
    {
        printf("Usage:\n");
        printf("  %s <model_path> <weight_path> <tokenizer_path> <embedding_path> <core_mask> <prompt> [max_new_tokens]\n", argv[0]);
        printf("\n");
        printf("max_new_tokens:\n");
        printf("  default: %d\n", LFM2_DEFAULT_NEW_TOKENS);
        printf("  range  : 1-%d\n", LFM2_MAX_NEW_TOKENS);
        return -1;
    }

    const char *model_path = argv[1];
    const char *weight_path = argv[2];
    const char *tokenizer_path = argv[3];
    const char *embedding_path = argv[4];
    char *endptr = NULL;
    errno = 0;
    uint32_t core_mask = strtoul(argv[5], &endptr, 0);
    if (errno != 0 || endptr == argv[5] || *endptr != '\0' || core_mask == 0)
    {
        printf("Invalid core_mask: %s\n", argv[5]);
        return -1;
    }
    const char *prompt = argv[6];

    // Parse max_new_tokens (optional, strict integer validation)
    int max_new_tokens = LFM2_DEFAULT_NEW_TOKENS;
    if (argc >= 8)
    {
        char *end2 = NULL;
        errno = 0;
        long val = strtol(argv[7], &end2, 0);
        if (errno != 0 || end2 == argv[7] || *end2 != '\0')
        {
            printf("Invalid max_new_tokens: %s (not a number)\n", argv[7]);
            return -1;
        }
        if (val < 1)
        {
            printf("Invalid max_new_tokens: %s (must be >= 1)\n", argv[7]);
            return -1;
        }
        if (val > LFM2_MAX_NEW_TOKENS)
        {
            printf("Invalid max_new_tokens: %s (must be <= %d)\n", argv[7], LFM2_MAX_NEW_TOKENS);
            return -1;
        }
        max_new_tokens = static_cast<int>(val);
    }

    printf("model           : %s\n", model_path);
    printf("weight          : %s\n", weight_path);
    printf("tokenizer       : %s\n", tokenizer_path);
    printf("embedding       : %s\n", embedding_path);
    printf("core mask       : 0x%x\n", core_mask);
    printf("Max context len : %d\n", LFM2_MAX_CONTEXT_LEN);
    printf("Max new tokens  : %d\n", LFM2_MAX_NEW_TOKENS);
    printf("Default new tokens: %d\n", LFM2_DEFAULT_NEW_TOKENS);
    printf("Requested tokens   : %d\n", max_new_tokens);
    printf("prompt          : %s\n", prompt);
    printf("Context and chat-template limits are managed by RKNN3 Session.\n");

    int ret;
    int infer_ret = 0;
    rknn_perf_metrics_t perf;

    // RKNN Context
    rknn_lfm2_llm_context rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_lfm2_llm_context));

    // Tokenizer
    Tokenizer *tokenizer = NULL;
    VocabInfo vocab_info;

    // Embedding
    struct embedding_info embedding_info;
    struct stat emb_st = {0};
    memset(&embedding_info, 0x00, sizeof(embedding_info));
    embedding_info.fd = -1;

    // Inference state (passed to callbacks via userdata)
    struct InferState infer_state;
    memset(&infer_state, 0, sizeof(infer_state));
    infer_state.tokenizer = NULL;
    infer_state.embed_info = &embedding_info;
    infer_state.first_token_time = 0;
    infer_state.first_decode = true;

    // LLM Param
    int n_params = 1;
    rknn3_llm_param params;
    memset(&params, 0, sizeof(rknn3_llm_param));

    // LLM Input Tensor
    int n_inputs = 1;
    rknn3_llm_tensor tensor;
    memset(&tensor, 0, sizeof(rknn3_llm_tensor));

    rknn3_llm_input input;
    memset(&input, 0, sizeof(rknn3_llm_input));

    // Callback
    RKLLMCallback callback;
    memset(&callback, 0, sizeof(RKLLMCallback));

    // Check files
    const char *files[] = {model_path, weight_path, tokenizer_path, embedding_path};
    for (int i = 0; i < 4; i++)
    {
        struct stat st;
        if (stat(files[i], &st) != 0)
        {
            printf("[ERROR] Not found: %s\n", files[i]);
            goto out;
        }
    }

    // Load Tokenizer
    tokenizer = new Tokenizer(TOKENIZER_BACKEND_LLAMA, tokenizer_path);
    if (!tokenizer)
    {
        printf("load tokenizer failed! tokenizer_path=%s\n", tokenizer_path);
        goto out;
    }
    infer_state.tokenizer = tokenizer;

    tokenizer->GetVocabInfo(&vocab_info);
    if (vocab_info.vocab_size <= 0)
    {
        printf("Invalid vocab size: %d\n", vocab_info.vocab_size);
        goto out;
    }

    // Read Embedding
    embedding_info.fd = open(embedding_path, O_RDONLY);
    if (embedding_info.fd == -1)
    {
        printf("Failed to open embedding file: %s\n", embedding_path);
        goto out;
    }

    if (fstat(embedding_info.fd, &emb_st) == -1)
    {
        printf("Failed to get embedding file size\n");
        goto out;
    }

    embedding_info.embedding_data = (float16 *)mmap(NULL, emb_st.st_size, PROT_READ, MAP_PRIVATE, embedding_info.fd, 0);
    if (embedding_info.embedding_data == MAP_FAILED)
    {
        printf("Failed to mmap embedding file\n");
        goto out;
    }

    embedding_info.vocab_size = vocab_info.vocab_size;
    if (emb_st.st_size % (vocab_info.vocab_size * sizeof(float16)) != 0)
    {
        printf("Embedding file size mismatch: st_size=%ld, vocab_size=%d, sizeof(float16)=%zu\n",
               emb_st.st_size, vocab_info.vocab_size, sizeof(float16));
        goto out;
    }
    embedding_info.embedding_dim = emb_st.st_size / (vocab_info.vocab_size * sizeof(float16));

    // Set LLM parameters
    params.logits_name = "output";
    params.max_context_len = LFM2_MAX_CONTEXT_LEN;
    params.sampling_param = SAMPLE_PARAMS;
    params.vocab_info.vocab_size = vocab_info.vocab_size;
    params.vocab_info.n_special_eos_id = vocab_info.n_special_eos_id;
    params.vocab_info.n_special_bos_id = vocab_info.n_special_bos_id;
    memcpy(params.vocab_info.special_eos_id, vocab_info.special_eos_id, sizeof(vocab_info.special_eos_id));
    memcpy(params.vocab_info.special_bos_id, vocab_info.special_bos_id, sizeof(vocab_info.special_bos_id));
    params.vocab_info.linefeed_id = vocab_info.linefeed_id;
    params.vocab_info.ignore_eos_token = 0;

    // LLM Callback
    callback.result_callback = result_callback;
    callback.result_userdata = &infer_state;
    callback.tokenizer_callback = tokenizer_callback;
    callback.tokenizer_userdata = &infer_state;
    callback.embed_callback = embed_callback;
    callback.embed_userdata = &infer_state;

    printf("\n--> init model\n");
    ret = init_lfm2_llm(&rknn_app_ctx, model_path, weight_path, &params, n_params, callback, core_mask);
    if (ret != 0)
    {
        printf("init_lfm2_llm fail! ret=%d model_path=%s weight_path=%s\n", ret, model_path, weight_path);
        infer_ret = ret;
        goto out;
    }

    // Clear cache
    ret = clear_lfm2_cache(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("clear_lfm2_cache fail! ret=%d\n", ret);
        infer_ret = ret;
        goto out;
    }

    // LLM Input
    tensor.name = NULL;
    tensor.prompt = prompt;
    tensor.embed = NULL;
    tensor.tokens = NULL;
    tensor.n_tokens = 0;
    tensor.enable_thinking = false;

    input.role = "user";
    input.input_type = RKNN3_LLM_INPUT_PROMPT;
    input.llm_input = tensor;

    printf("--> inference\n\n");
    printf("Response:\n");
    infer_state.first_token_time = getCurrentTimeUs();
    infer_ret = inference_lfm2_llm(&rknn_app_ctx, &input, n_inputs,
                                    static_cast<uint32_t>(max_new_tokens), false, &perf);
    if (infer_ret != 0)
    {
        printf("\n[ERROR] inference ret=%d\n", infer_ret);
        goto out;
    }

    printf("\n");
    printf_perf(&perf, infer_state.first_token_time);

out:
    ret = release_lfm2_llm(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release lfm2 llm fail! ret=%d\n", ret);
    }

    if (embedding_info.fd != -1)
    {
        if (embedding_info.embedding_data != MAP_FAILED && embedding_info.embedding_data != NULL)
        {
            munmap((void *)embedding_info.embedding_data, emb_st.st_size);
            embedding_info.embedding_data = NULL;
        }
        close(embedding_info.fd);
        embedding_info.fd = -1;
    }

    if (tokenizer != NULL)
    {
        delete tokenizer;
        tokenizer = NULL;
    }

    printf("done\n");

    return infer_ret;
}

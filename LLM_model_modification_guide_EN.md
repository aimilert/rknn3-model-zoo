# LLM Model Adaptation Guide

For LLM models, the coprocessor mainly accelerates the main neural network computation and sampling post-processing. Tokenizer handling, chat-template assembly, and multimodal handoff logic still run on the host (CPU/NPU).

The coprocessor does not impose a mandatory I/O format by itself, but for better performance the RKNN3-Toolkit LLM inference pipeline has clear requirements on exported ONNX inputs, outputs, and configuration files. This document reflects the current implementation, with primary references to `examples/Qwen3/python/export_llm.py` and `py_utils/export_llm_helper.py`.

## Export Scripts and Generated Artifacts

Using `examples/Qwen3/python/export_llm.py` as an example, LLM export typically produces:

- **`*.onnx`**: Main LLM ONNX model (exported by `causal_llm_to_onnx()`).
- **`*.config.pkl`**: On-device configuration (system/prompt fields, `chat_template`, `vocab_size`, `hidden_size`, `hf_config_json`, and optional quantization fields `q_params`).
- **`*.tokenizer.gguf`**: Tokenizer file (exported by `export_tokenizer()`).
- **`*.embed.bin`**: Embedding weights in fp16 (from `model.model.embed_tokens.weight`, exported by `export_embed_weight()`).

Notes:

- In `export_llm.py`, `--quant` triggers an external GRQ quantization path; on success, the export path switches to `grq_model_path`.

## ONNX Input/Output Constraints (Current Implementation)

### 1) Standard causal LLM (e.g. Qwen3)

Required inputs:

- **`input_ids`**: `[1, sequence]`, `int64`
- **`attention_mask`**: `[1, sequence]`, `float32`
- **`position_ids`**: `[1, sequence]`, `int64`

Optional inputs (appended automatically according to the model’s `forward` signature):

- **`num_logits_to_keep`**: `[1]`, `int32`. Inserted when `forward` includes `logits_to_keep` or `num_logits_to_keep`; the default value is `-1` (keep logits for the last token only, reducing redundant compute and memory).

Outputs:

- **Single output `output`** (logits).

## KV Cache Constraints

- RKNN3-Toolkit builds and manages KV cache from Attention ops (including performance and quantization-related optimizations). Users do not need explicit KV cache inputs or outputs in ONNX.
- Before export, set `use_cache=False` in config (the example uses `update_config(config, ['use_cache'], False)`) so the exported ONNX has no KV cache I/O.

## Shape and Export Behavior

- At deployment time, multiple static sequence lengths are recommended (e.g. prefill length `N` and decode length `1`).
- Inside `causal_llm_to_onnx()`, the defaults are:
  - `args.prompt_size = 64`
  - `args.dynamic_shape = True`
  so the ONNX marks the sequence dimension with the dynamic axis name `sequence`.
- On device, pass `seq_len` to `rknn.load_llm` (e.g. `[1, 128]`) to select the static length combinations used at runtime.

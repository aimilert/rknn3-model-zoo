# Qwen3-VL LoRA Model Deployment Guide

## 1. Environment Requirements

Please refer to `requirements.txt` for the Python environment.

## 2. Model Pruning Strategy

To support larger context lengths, appropriate pruning is required when deploying large-scale multimodal models.

### 2.1 Vision Model Pruning

Some operators are offloaded to the CPU of the host device (e.g., RK3588).

### 2.2 LLM Model Pruning

The LLM Head is separated and runs independently on the host device, reducing memory usage on the coprocessor. (Optional)

### 2.3 Full Model Mode (No Pruning)

Devices with larger memory (e.g., RK1828) can use the full model directly. Add the `--no_prune_mode` parameter when exporting the Vision model:

```bash
python export_rknn.py --no_prune_mode
```

## 3. Supported Models

Currently supports Qwen3-VL 2B, 4B, and other variants. Specify the corresponding model path when exporting.

### 3.1 Base Model Download

Example with **Qwen3-VL-4B**:

| Model | Download Link |
|-------|---------------|
| Qwen3-VL-4B-Instruct | [HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) / [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct) |

### 3.2 LoRA Model Download

This demo supports loading LoRA weights. The following are community-provided LoRA model examples:

| Model | Download Link |
|-------|---------------|
| qwen3-vl-4b-ui-confidence-lora | [HuggingFace](https://huggingface.co/bobbyzhong/qwen3-vl-4b-ui-confidence-lora) |

### 3.3 Model Export Commands

Example with **Qwen3-VL-4B**:

```bash
# Export ONNX model
python export_llm.py \
    --model_path Qwen/Qwen3-VL-4B-Instruct \
    --export_llm_path Qwen3-VL-4B-llm.onnx \
    --modelscope

# Export RKNN model (without LoRA)
python export_rknn.py \
    --onnx_path Qwen3-VL-4B-llm.onnx \
    --config Qwen3-VL-4B-llm.config.pkl \
    --rknn_path Qwen3-VL-4B-llm.rknn

# Export RKNN model (with LoRA)
python export_rknn.py \
    --onnx_path Qwen3-VL-4B-llm.onnx \
    --config Qwen3-VL-4B-llm.config.pkl \
    --rknn_path Qwen3-VL-4B-llm-lora.rknn \
    --lora_path /path/to/lora/adapter_model.safetensors \
    --lora_config_path /path/to/lora/adapter_config.json
```

> ⚠️ **Note**:
> - If the model structure and quantization parameters are unchanged and you only need to re-export the `.rknn` file, use `--rebuild` for a quick rebuild. It depends on intermediate artifacts generated during the previous `build` phase in the `./tmp` directory.
> - LoRA weights (e.g., `adapter_model.safetensors`) do **not** need to be converted to ONNX separately — simply pass the weight path via `--lora_path` and the config path via `--lora_config_path` in `export_rknn.py`, and the script will internally call `rknn.load_lora()` to load the `.safetensors` file directly, with no intermediate format conversion required.
> - When exporting an RKNN model with LoRA, an additional `.lora_weight` file is generated alongside the `.rknn` file. Both files are required for C++ inference (`.rknn` as the model path, `.lora_weight` as the LoRA weight path).
> - Make sure `--lora_path` points to the correct LoRA weight file (e.g., `.safetensors`) and `--lora_config_path` points to the corresponding config file (e.g., `adapter_config.json`).

## 4. Vision Model Resolution Adjustment

Use `--img_h` and `--img_w` parameters to adjust input resolution (must be a multiple of 32):

```bash
# Export Vision ONNX model (resolution specified via --img_h/--img_w)
python export_vision.py \
    --model_path Qwen/Qwen3-VL-4B-Instruct \
    --export_vision_path Qwen3-VL-4B-vision.onnx \
    --img_h 384 --img_w 384 \
    --modelscope

# Export Vision RKNN model (resolution auto-read from vision_config.json)
python export_rknn.py \
    --onnx_path Qwen3-VL-4B-vision.onnx \
    --rknn_path Qwen3-VL-4B-vision.rknn
```

> ⚠️ **Note**:
> - Higher resolution increases memory usage and affects the maximum LLM context length
> - Some resolutions may be incompatible with the RKNN inference framework; contact the RKNPU team if errors occur
> - Vision RKNN export resolution is controlled by `vision_config.json`; no need to pass `--img_h`/`--img_w` again

## 5. C++ Deployment

The C++ inference code automatically detects the model format and is compatible with both pruned and full models without code modification.

When running the demo, you **must** pass `model_width` and `model_height` via command line so they match the Vision export resolution (must be multiples of 32).

### 5.1 Base and LoRA Dual-Path Inference

The demo creates two LLM sessions at initialization:

| Session   | Description |
|-----------|-------------|
| **Base**  | Loads only the base model weights for standard inference. |
| **LoRA**  | Can load LoRA weights on top of the base model (when a LoRA path is provided and the SDK supports it) for LoRA-based inference. |

**Inference flow**: For the same image and prompt, Vision runs once to produce vision embeddings; then **Base model inference** and **LoRA model inference** run in sequence, each printing its own generated text and performance (Prefill/Generate, Vision latency, etc.).

### 5.2 Command-Line Arguments and Usage

Usage:

```text
./rknn_qwen3_vl_demo <vision_model_path> <vision_weight_path> <llm_model_path> <llm_weight_path> \
    <tokenizer_path> <embedding_path> <vision_core_mask> <llm_core_mask> <image_path> <prompt> \
    <model_width> <model_height> <max_context_len1> <max_context_len2> [llm_lora_weight_path]
```

| Arg count | Meaning |
|-----------|--------|
| **15**    | 14 required args (including `model_width`, `model_height`, `max_context_len1`, `max_context_len2`). |
| **16**    | Same as 15 plus `llm_lora_weight_path` for LoRA weights. |

New context-length arguments:

| Argument | Description |
|----------|-------------|
| `max_context_len1` | Maximum context length for the **Base session** (`session_base params.max_context_len`). |
| `max_context_len2` | Maximum context length for the **LoRA session** (`session_lora params.max_context_len`). |

> Both arguments are required and must be greater than 0. You can set them independently based on device memory, e.g., Base = 2048 and LoRA = 3072.

**Example (15 args; resolution must match Vision export, e.g., 384x384)**:

```bash
./rknn_qwen3_vl_demo ... 384 384 2048 3072
```

**Compare Base and LoRA (16 args, with LoRA weight path)**:

```bash
./rknn_qwen3_vl_demo \
    ./model/vision.rknn ./model/vision.weight \
    ./model/llm.rknn ./model/llm.weight \
    ./model/tokenizer.gguf ./model/embed.bin \
    0x3 0x3 \
    ./model/demo.jpg "Describe this image." \
    384 384 \
    2048 3072 \
    ./model/llm_lora.weight
```

The program prints **Base model** output and metrics first, then **LoRA model** output and metrics.

### 5.3 LoRA API Usage

The following describes the RKNN3 LoRA APIs and how this demo uses them.

#### API List and Call Order

| API | Description |
|-----|-------------|
| `rknn3_lora_init(context, lora_weight_path)` | Initialize LoRA from file; must be called before any load/enable. |
| `rknn3_lora_init_from_data(context, weight_data, weight_size)` | Initialize LoRA from memory buffer for direct loading from memory. |
| `rknn3_query(context, RKNN3_QUERY_LORA_NUM, &n_lora, sizeof(n_lora))` | Query the number of LoRAs available in the context. |
| `rknn3_query(context, RKNN3_QUERY_LORA_INFO, lora_list, sizeof(lora_list))` | Query the list of LoRAs available in the context, including lora_name and scale. |
| `rknn3_lora_load(context, lora)` | Load the given LoRA adapter into the context; different sessions under the same context share the same LoRA weights. |
| `rknn3_lora_enable(context, lora)` | Enable a loaded LoRA at the context level. |
| `rknn3_lora_disable(context, lora)` | Disable an enabled LoRA at the context level. |
| `rknn3_lora_unload(context, lora)` | Unload a loaded LoRA; this will automatically disable all enabled LoRAs. |
| `rknn3_session_enable_lora(session, lora)` | Enable LoRA for a specific session; this will automatically clear all KV caches. |
| `rknn3_session_disable_lora(session, lora)` | Disable LoRA for a specific session; this will automatically clear all KV caches. |

**Recommended order**: `lora_init` → `query(RKNN3_QUERY_LORA_NUM)` → `query(RKNN3_QUERY_LORA_INFO)` → `lora_load` → `session_enable_lora` → `lora_unload`.

#### Where this demo uses them

LoRA setup is done in `cpp/llm/rknn_qwen3_vl_llm.cc` inside `setup_context_lora()`, which is called from `init_qwen3_vl_llm()` when `lora_weight_path` is non-null:

1. `rknn3_lora_init(ctx, lora_weight_path)` — set LoRA weight path (file or directory).
2. `rknn3_query(ctx, RKNN3_QUERY_LORA_NUM, &n_lora, sizeof(n_lora))` — get LoRA count.
3. `rknn3_query(ctx, RKNN3_QUERY_LORA_INFO, lora_list, sizeof(lora_list))` — get LoRA list.
4. `rknn3_lora_load(ctx, &lora_list[0])` — load that LoRA into the context.
5. `rknn3_session_enable_lora(session_lora, &lora_list[0])` — enable it for the LoRA session.

Inference uses `inference_qwen3_vl_llm_base()` for the base session and `inference_qwen3_vl_llm_lora()` for the LoRA session.

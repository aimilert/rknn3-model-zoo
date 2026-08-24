# Qwen3-VL LoRA模型部署说明

## 1. 部署环境

python环境参考 `requirements.txt` 

## 2. 模型裁剪策略

为了支持更大的上下文长度，部署多模态模型时需进行适当裁剪。

### 2.1 Vision 模型裁剪

将部分算子迁移至 RK3588 等主控设备的 CPU 上运行。

### 2.2 LLM 模型裁剪

将 LLM Head 独立出来，在主控设备上单独运行，从而减少协处理器的内存占用。（可选）

### 2.3 完整模型模式（无裁剪）

RK1828 等内存较大的设备可直接使用完整模型。Vision 模型导出时添加 `--no_prune_mode` 参数即可关闭裁剪：

```bash
python export_rknn.py --no_prune_mode
```

## 3. 支持的模型

目前支持 Qwen3-VL 2B 和 4B 等模型。导出时请指定对应的模型路径。

### 3.1 基座模型下载

以 **Qwen3-VL-4B** 为例：

| 模型 | 下载链接 |
|------|----------|
| Qwen3-VL-4B-Instruct | [HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) / [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct) |

### 3.2 LoRA 模型下载

本demo支持加载LoRA权重。以下为社区提供的 LoRA 模型示例：

| 模型 | 下载链接 |
|------|----------|
| qwen3-vl-4b-ui-confidence-lora | [HuggingFace](https://huggingface.co/bobbyzhong/qwen3-vl-4b-ui-confidence-lora) |

### 3.3 模型导出命令

以 **Qwen3-VL-4B** 为例：

```bash
# 导出 ONNX 模型
python export_llm.py \
    --model_path Qwen/Qwen3-VL-4B-Instruct \
    --export_llm_path Qwen3-VL-4B-llm.onnx \
    --modelscope

# 导出 RKNN 模型（不含 LoRA）
python export_rknn.py \
    --onnx_path Qwen3-VL-4B-llm.onnx \
    --config Qwen3-VL-4B-llm.config.pkl \
    --rknn_path Qwen3-VL-4B-llm.rknn

# 导出 RKNN 模型（含 LoRA）
python export_rknn.py \
    --onnx_path Qwen3-VL-4B-llm.onnx \
    --config Qwen3-VL-4B-llm.config.pkl \
    --rknn_path Qwen3-VL-4B-llm-lora.rknn \
    --lora_path /path/to/lora/adapter_model.safetensors \
    --lora_config_path /path/to/lora/adapter_config.json
```

> ⚠️ **注意**：
> - LoRA 权重（如 `adapter_model.safetensors`）**无需**单独转换为 ONNX 模型，只需在 `export_rknn.py` 中通过 `--lora_path` 传入权重路径、`--lora_config_path` 传入配置文件路径，脚本内部会调用 `rknn.load_lora()` 直接加载 `.safetensors` 文件，无需中间格式转换。
> - 导出含 LoRA 的 RKNN 模型时，除 `.rknn` 文件外还会额外生成一个 `.lora_weight` 文件。C++ 推理时需同时提供这两个文件（`.rknn` 作为模型路径，`.lora_weight` 作为 LoRA 权重路径）。
> - 导出时需确保 `--lora_path` 指向正确的 LoRA 权重文件（如 `.safetensors`），`--lora_config_path` 指向对应的配置文件（如 `adapter_config.json`）。
> - 如果模型结构和量化参数未变，仅需重新导出 `.rknn` 文件，可使用 `--rebuild` 快速重建。重建依赖上次 `build` 阶段在 `./tmp` 目录下生成的中间产物。

## 4. Vision 模型分辨率调整

可通过 `--img_h` 和 `--img_w` 参数调整输入分辨率（必须为 32 的倍数）：

```bash
# 导出 Vision ONNX 模型（分辨率通过 --img_h/--img_w 指定）
python export_vision.py \
    --model_path Qwen/Qwen3-VL-4B-Instruct \
    --export_vision_path Qwen3-VL-4B-vision.onnx \
    --img_h 384 --img_w 384 \
    --modelscope

# 导出 Vision RKNN 模型（分辨率从 vision_config.json 自动读取）
python export_rknn.py \
    --onnx_path Qwen3-VL-4B-vision.onnx \
    --rknn_path Qwen3-VL-4B-vision.rknn
```

> ⚠️ **注意**：
> - 分辨率越大，内存占用越高，会影响 LLM 的最大上下文长度
> - 部分分辨率可能与 RKNN 推理框架不兼容，如遇报错请联系 RKNPU 团队
> - Vision RKNN 导出时分辨率由 `vision_config.json` 控制，无需再次传入 `--img_h`/`--img_w`

## 5. C++ 部署说明

C++ 推理代码已实现模型格式自动识别，无需修改代码即可兼容裁剪版与完整版模型。运行 demo 时**必须**通过命令行传入与 Vision 模型导出分辨率一致的 `model_width`、`model_height`（须为 32 的倍数）。

### 5.1 Base 与 LoRA 双路推理说明

本示例在初始化时同时创建两个 LLM session：

| Session       | 说明 |
|---------------|------|
| **Base**      | 仅加载基座模型权重，用于标准推理。 |
| **LoRA**      | 在基座基础上可加载 LoRA 权重（需传入 LoRA 路径且 SDK 支持），用于带 LoRA 的推理。 |

**推理流程**：对同一张图片与同一 prompt，先跑一次 Vision 得到vision embedding，再依次执行 **Base 模型推理** 和 **LoRA 模型推理**，分别输出两段生成结果与性能（Prefill/Generate、Vision 耗时等）。


### 5.2 命令行参数与用法

可执行文件用法：

```text
./rknn_qwen3_vl_demo <vision_model_path> <vision_weight_path> <llm_model_path> <llm_weight_path> \
    <tokenizer_path> <embedding_path> <vision_core_mask> <llm_core_mask> <image_path> <prompt> \
    <model_width> <model_height> <max_context_len1> <max_context_len2> [llm_lora_weight_path]
```

| 参数个数 | 含义 |
|----------|------|
| **15**   | 必选 14 个参数（含 `model_width`、`model_height`、`max_context_len1`、`max_context_len2`）。 |
| **16**   | 在 15 参数基础上增加 `llm_lora_weight_path`，用于 LoRA 权重路径。 |

新增上下文长度参数说明：

| 参数 | 说明 |
|------|------|
| `max_context_len1` | **Base session** 的最大上下文长度（`session_base params.max_context_len`）。 |
| `max_context_len2` | **LoRA session** 的最大上下文长度（`session_lora params.max_context_len`）。 |

> 两个参数均为必填，且必须大于 0。可根据设备内存情况分别设置，例如 Base 设为 2048、LoRA 设为 3072。

**示例（15 参数，分辨率需与导出 Vision 时一致，如 384×384）**：

```bash
./rknn_qwen3_vl_demo ... 384 384 2048 3072
```

**同时对比 Base 与 LoRA（16 参数，传入 LoRA 权重）**：

```bash
./rknn_qwen3_vl_demo \
    ./model/vision.rknn ./model/vision.weight \
    ./model/llm.rknn ./model/llm.weight \
    ./model/tokenizer.gguf ./model/embed.bin \
    0x3 0x3 \
    ./model/demo.jpg "描述这张图片" \
    384 384 \
    2048 3072 \
    ./model/llm_lora.weight
```

程序会先输出 **Base model** 的结果与性能，再输出 **LoRA model** 的结果与性能。

### 5.3 LoRA 相关 API 使用说明

以下为 RKNN3 LoRA 接口及在本 demo 中的使用方式。

#### 接口列表与调用顺序

| 接口 | 说明 |
|------|------|
| `rknn3_lora_init(context, lora_weight_path)` | 从文件初始化 LoRA 环境，必须在 load/enable 之前调用。 |
| `rknn3_lora_init_from_data(context, weight_data, weight_size)` | 从内存数据初始化 LoRA 环境，用于直接从内存加载 LoRA 权重。 |
| `rknn3_query(context, RKNN3_QUERY_LORA_NUM, &n_lora, sizeof(n_lora))` | 查询当前 context 中可用的 LoRA 数量。 |
| `rknn3_query(context, RKNN3_QUERY_LORA_INFO, lora_list, sizeof(lora_list))` | 查询当前 context 中可用的 LoRA 信息列表，包括lora_name和scale |
| `rknn3_lora_load(context, lora)` | 将指定 LoRA 适配器加载进 context，相同context的不同session共享同一个LoRA权重。 |
| `rknn3_lora_enable(context, lora)` | 在 context 级别启用已加载的 LoRA。 |
| `rknn3_lora_disable(context, lora)` | 在 context 级别关闭已启用的 LoRA。 |
| `rknn3_lora_unload(context, lora)` | 卸载已加载的 LoRA，会自动disable所有启用的LoRA |
| `rknn3_session_enable_lora(session, lora)` | 为指定 session 启用 LoRA，调用后会自动清空所有kvcache |
| `rknn3_session_disable_lora(session, lora)` | 为指定 session 关闭 LoRA，调用后会自动清空所有kvcache |

**推荐顺序**：`lora_init` → `query(RKNN3_QUERY_LORA_NUM)` → `query(RKNN3_QUERY_LORA_INFO)` → `lora_load` → `session_enable_lora` → `lora_unload`。

#### 本 demo 中的实现位置

LoRA 初始化集中在 `cpp/llm/rknn_qwen3_vl_llm.cc` 的 `setup_context_lora()` 中（由 `init_qwen3_vl_llm()` 在传入 `lora_weight_path` 时调用）：

1. `rknn3_lora_init(ctx, lora_weight_path)` — 指定 LoRA 权重路径（文件或目录）。
2. `rknn3_query(ctx, RKNN3_QUERY_LORA_NUM, &n_lora, sizeof(n_lora))` — 获取 LoRA 数量。
3. `rknn3_query(ctx, RKNN3_QUERY_LORA_INFO, lora_list, sizeof(lora_list))` — 获取 LoRA 列表。
4. `rknn3_lora_load(ctx, &lora_list[0])` — 在 context 上加载该 LoRA。
5. `rknn3_session_enable_lora(session_lora, &lora_list[0])` — 为 LoRA session 启用该 LoRA。

推理时通过 `inference_qwen3_vl_llm_base()` 使用 Base session，通过 `inference_qwen3_vl_llm_lora()` 使用已加载 LoRA 的 session。

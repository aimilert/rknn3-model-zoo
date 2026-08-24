# Qwen3.5-VL 模型部署说明

## 1. 环境要求

Qwen3.5-VL 模型导出依赖固定版本环境。请优先使用如下版本，避免因 transformers、ONNX 或
ONNXRuntime 版本差异导致模型结构、动态维度或算子导出不一致。

```bash
torch                             2.7.0
transformers                      5.3.0
onnx                              1.18.0
onnxruntime                       1.22.1
```

> ⚠️ **注意**：
> 
> - 若需要从 ModelScope 下载模型，可在导出 ONNX 时增加 `--modelscope` 参数。
> - Qwen3.5-VL 是多模态模型，包含 Vision Encoder 和 LLM 两部分，需要分别导出。

## 2. 支持的模型

目前支持 Qwen3.5-VL 系列多模态模型，以 **Qwen3.5-0.8B** 为例：

### 2.1 导出 Vision ONNX 模型

```bash
# 进入脚本目录
cd examples/Qwen3_5_VL/python/vision

# 生成 vision 模型量化校准数据
python make_calidata.py --model_path Qwen/Qwen3.5-0.8B

# 导出 Vision ONNX 模型
python export_vision.py --quant \
    --model_path Qwen/Qwen3.5-0.8B \
    --export_vision_path ../../model/vision/Qwen3.5-0.8B-vision.onnx \
    --img_h 384 \
    --img_w 384
```

`make_calidata.py` 参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | HuggingFace 模型路径或名称 | `Qwen/Qwen3.5` |
| `--datapath` | 校准数据集 JSON 路径（含图片路径和文本输入） | `../../../../datasets/MMBench/llm/dataset.json` |
| `--export_datapath` | 生成的量化校准数据输出路径 | `./quant_data/model_inputs.json` |

> ⚠️ `make_calidata.py` 会加载完整模型并捕获模块输入，需要较大内存。生成的校准数据用于 `export_vision.py --quant` 时的 GRQ 量化。

参数说明：
- `--img_h` / `--img_w`：输入图像尺寸，必须为 32 的倍数，默认 384。

### 2.2 导出 Vision RKNN 模型

```bash
python export_rknn.py \
    --onnx_path ../../model/vision/Qwen3.5-0.8B-vision.onnx \
    --rknn_path ../../model/vision/Qwen3.5-0.8B-vision.rknn \
    --platform rk1820
```

Vision RKNN 转换支持两种模式：

| 模式 | 说明 |
|------|------|
| **Prune 模式**（默认） | 裁剪 Vision 模型，放到CPU前处理计算，适用于推理部署，可减少模型内存。 |
| **完整模式** | 保留完整 Vision 模型，输入为原始图像 `[1, 3, H, W]`。通过 `--no_prune_mode` 启用。 |

### 2.3 导出 LLM ONNX 模型

```bash
cd examples/Qwen3_5_VL/python/llm

# 生成 LLM 模型量化校准数据
python make_calidata.py --model_path Qwen/Qwen3.5-0.8B

# 导出 LLM ONNX 模型
python export_llm.py --quant \
    --model_path Qwen/Qwen3.5-0.8B \
    --export_llm_path ../../model/llm/Qwen3.5-0.8B.onnx
```

`make_calidata.py` 参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | HuggingFace 模型路径或名称 | `Qwen/Qwen3.5` |
| `--datapath` | 校准数据集 JSON 路径（含图片路径和文本输入） | `../../../../datasets/MMBench/llm/dataset.json` |
| `--export_datapath` | 生成的量化校准数据输出路径 | `./quant_data/model_inputs.json` |

> ⚠️ `make_calidata.py` 会加载完整模型并捕获模块输入，需要较大内存。生成的校准数据用于 `export_llm.py --quant` 时的 GRQ 量化。

执行后会在指定目录下同步生成：
```bash
Qwen3.5-0.8B.onnx
Qwen3.5-0.8B.config.pkl
Qwen3.5-0.8B.tokenizer.gguf
Qwen3.5-0.8B.embed.bin
```

### 2.4 导出 LLM RKNN 模型

```bash
# 导出 LLM RKNN 模型
python export_rknn.py \
    --onnx_path ../../model/llm/Qwen3.5-0.8B.onnx \
    --config ../../model/llm/Qwen3.5-0.8B.config.pkl \
    --rknn_path ../../model/llm/Qwen3.5-0.8B.rknn \
    --platform rk1820

# 如需重新导出（仅修改 profile_mode 或 kvcache 相关参数），可使用 --rebuild 跳过 ONNX 加载和图优化等步骤，加速模型导出：
python export_rknn.py \
    --onnx_path ../../model/llm/Qwen3.5-0.8B.onnx \
    --config ../../model/llm/Qwen3.5-0.8B.config.pkl \
    --rknn_path ../../model/llm/Qwen3.5-0.8B.rknn \
    --platform rk1820 \
    --rebuild
```

> ⚠️ **`--rebuild` 说明**：当前仅支持重置 `profile_mode` 和 `llm_config` 中 kvcache 相关参数（如 `kvcache_buffer_len`或者`max_position_embeddings` 等）。其他参数变更需走完整导出流程。

## 3. RKNN LLM 量化配置

Qwen3.5-VL 的 LLM RKNN 转换默认使用 W4A16 量化，并采用 group32 量化方式。
转换脚本中默认目标平台为 `rk1820`，同时会根据 ONNX 输入信息自动生成动态输入配置。

与纯文本 Qwen3.5 不同，Qwen3.5-VL 的 LLM 需要配置 **MRoPE（Multi-Resolution RoPE）** 参数，
以支持视觉位置编码：

```python
llm_config = DEFAULT_RKNN_LLM_CONFIG.copy()
llm_config['attention_config'][0]['kvcache_buffer_len'] = 8 * 1024
llm_config['attention_config'][0]['max_position_embeddings'] = 8 * 1024
llm_config['attention_config'][0]['mrope_type'] = 'Qwen3.5'
llm_config['attention_config'][0]['mrope_section'] = [11, 11, 10]
llm_config['attention_config'][0]['mrope_new_id_name'] = 'mrope_id_input'

rknn.config(
    target_platform='rk1820',
    dynamic_input=dynamic_shapes,
    quantized_dtype='w4a16',
    quantized_algorithm='normal',
    quantized_method='group32',
    llm_config=llm_config,
    input_initial_value=input_initial_value,
    cvt_conv_streaming=cvt_conv_streaming,
)
```

> ⚠️ **注意**：
> - `mrope_section` 需要根据huggingface模型的 `config.json` 配置
> - `kvcache_buffer_len` 和 `max_position_embeddings` 默认设为 8K，请根据实际需求调整。
> - 目前模型不支持外部和内部grq。

## 4. KV Cache INT4 量化

在大规模语言模型推理过程中，KV Cache 用于存储历史的注意力键值，以避免重复计算。
随着序列长度增长，KV Cache 的内存占用会快速增加。为了减少 KV Cache 的存储带宽与
内存访问开销，可采用量化方式将其从 FP16/FP32 转换为 INT8 或更低位宽表示。

若需支持更长的上下文长度并进一步压缩 KV Cache 内存，可启用 `Int4_to_F16` 模式：

```python
llm_config['attention_config'][0]['kvcache_dtype'] = 'Int4_to_F16'
```

> ⚠️ **注意**：
> - `Int4_to_F16` 适用于更长上下文场景，但可能带来一定精度损失。
> - 修改 KV Cache 配置后，需要重新执行 RKNN 转换。

## 5. C++ 部署说明

### 5.1 编译

```bash
cd rknn3_model_zoo/

# 设置交叉编译工具链
export GCC_COMPILER=/path/to/aarch64-linux-gnu

# 编译
./build-linux.sh -t rk3588 -a aarch64 -d Qwen3_5_VL
```

编译完成后，文件生成在 `install/rk3588_linux_aarch64/rknn_Qwen3_5_VL_demo/` 目录。

### 5.2 所需文件

```bash
<vision_model_path>   ../model/vision/Qwen3.5-0.8B-vision.rknn
<vision_weight_path>  ../model/vision/Qwen3.5-0.8B-vision.weight
<llm_model_path>      ../model/llm/Qwen3.5-0.8B.rknn
<llm_weight_path>     ../model/llm/Qwen3.5-0.8B.weight
<tokenizer_path>      ../model/llm/Qwen3.5-0.8B.tokenizer.gguf
<embedding_path>      ../model/llm/Qwen3.5-0.8B.embed.bin
```

其中 `tokenizer.gguf` 和 `embed.bin` 由 `export_llm.py` 自动导出，部署时需与 RKNN 模型
保持同一模型版本。`weight_path` 为 RKNN 运行时使用的权重文件。

### 5.3 命令行参数

```bash
./rknn_qwen3_5_vl_demo \
    <vision_model_path> <vision_weight_path> \
    <llm_model_path> <llm_weight_path> \
    <tokenizer_path> <embedding_path> \
    <vision_core_mask> <llm_core_mask> \
    <image_path> <prompt> \
    [model_width] [model_height]
```

参数说明：

| 参数 | 说明 |
|------|------|
| `vision_model_path` | Vision RKNN 模型路径 |
| `vision_weight_path` | Vision RKNN 权重文件路径 |
| `llm_model_path` | LLM RKNN 模型路径 |
| `llm_weight_path` | LLM RKNN 权重文件路径 |
| `tokenizer_path` | tokenizer 文件路径 |
| `embedding_path` | embedding 权重文件路径 |
| `vision_core_mask` | Vision NPU 核心掩码，按 16 进制填写，如 `0x1`、`0x2`、`0xff` |
| `llm_core_mask` | LLM NPU 核心掩码，按 16 进制填写 |
| `image_path` | 输入图片路径 |
| `prompt` | 用户输入文本 |
| `model_width` | （可选）Vision 模型输入宽度，Prune 模式下需手动指定 |
| `model_height` | （可选）Vision 模型输入高度，Prune 模式下需手动指定 |

### 5.4 运行示例

以 **Qwen3.5-0.8B** 为例：

```bash
./rknn_qwen3_5_vl_demo \
    model/Qwen3.5-0.8B-vision.rknn \
    model/Qwen3.5-0.8B-vision.weight \
    model/Qwen3.5-0.8B.rknn \
    model/Qwen3.5-0.8B.weight \
    model/Qwen3.5-0.8B.tokenizer.gguf \
    model/Qwen3.5-0.8B.embed.bin \
    0xff 0xff \
    model/demo.jpg \
    "描述这张图片的内容"
```

> ⚠️ **注意**：
>
> - `prompt` 中如果包含空格，需要使用英文双引号包起来。
> - `core_mask` 通过 `strtoul(argv[n], nullptr, 16)` 解析，建议按 16 进制格式填写。
> - Vision 和 LLM 的 core_mask 需要与各自模型的 NPU 核心数匹配，否则初始化会报错。
> - demo 默认使用 `top_k=1`、`top_p=0.9`、`temperature=1.0`、`repeat_penalty=1.0`、
>   `frequency_penalty=0.0`、`presence_penalty=1.5`。
> - demo 中 `enable_thinking=false`，如需开启 thinking 模式，请在 C++ 代码中修改对应字段。
> - Prune 模式下，若未通过命令行指定 `model_width` / `model_height`，将使用代码中默认的
>   `MODEL_WIDTH=384` / `MODEL_HEIGHT=384`。请确保该值与导出 Vision ONNX 时的 `--img_h` / `--img_w` 一致。

### 5.4 架构说明

Qwen3.5-VL 的 C++ demo 采用 Vision + LLM 双模型架构：

1. **Vision Encoder**：加载 Vision RKNN 模型，对输入图像进行编码，输出 image embeddings。
2. **LLM Decoder**：加载 LLM RKNN 模型，接收 image embeddings 和文本 prompt，进行多模态推理。
3. **Internal Memory 共享**：Vision 和 LLM 模型共享 NPU 内部内存，减少内存占用。

推理流程：
```
输入图像 → Vision RKNN → Image Embeddings
                              ↓
用户 Prompt（含 <image> 标记） → LLM RKNN → 文本输出
```

Prompt 中会自动添加 `<image>` 前缀标记，以及 `<|vision_start|>` / `<|vision_end|>` /
`<|image_pad|>` 等视觉特殊 token。


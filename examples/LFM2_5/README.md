# LFM2.5-1.2B-Instruct 模型部署说明

模型地址：[LiquidAI/LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct)

## 模型简介

LFM2.5-1.2B-Instruct 是 LiquidAI 推出的 1.2B 参数指令微调语言模型。该模型采用混合架构，包含 6 层 Full Attention 和 10 层 LIV（Liquid Intelligence Vector）短卷积层，通过 RKNN3 的 `cvt_conv_streaming` 技术实现板端流式卷积推理。

requirements.txt位于python文件夹下，按照说明安装完toolkit后请执行 `pip install -r requirements.txt`

## 1. 导出 RKNN 模型

### 1.1 导出 LLM ONNX 模型

```bash
cd examples/LFM2_5/python

# 从 HuggingFace 下载并导出
python export_onnx.py \
    --model_path LiquidAI/LFM2.5-1.2B-Instruct \
    --output_dir ../model \
    --dtype fp16 

# 如需从 ModelScope 下载模型，增加 --modelscope 参数
pip install modelscope

python export_onnx.py \
    --model_path LiquidAI/LFM2.5-1.2B-Instruct \
    --output_dir ../model \
    --dtype fp16 \
    --modelscope
```

执行 `export_onnx.py` 后，会在 `../model/` 目录下生成以下文件：

```
LFM2.5-1.2B-Instruct.onnx                    # 标准 ONNX 模型
LFM2.5-1.2B-Instruct.onnx.data               # ONNX 外部权重
LFM2.5-1.2B-Instruct-convstream.onnx         # Conv Streaming ONNX（Conv pads 改为 [2,0]）
LFM2.5-1.2B-Instruct.config.pkl              # LLM 配置
LFM2.5-1.2B-Instruct.tokenizer.gguf          # GGUF tokenizer
LFM2.5-1.2B-Instruct.embed.bin               # FP16 embedding 权重
```

> Conv Streaming 默认开启，会将 10 个 LFM2 ShortConv 的 padding 从 `[2,2]` 改为 `[2,0]`，以适配 RKNN3 的 `cvt_conv_streaming` 要求。

### 1.2 导出 LLM RKNN 模型

```bash
python export_rknn.py \
    --seq_len 128 \
    --target_platform rk1820
```

RKNN 转换通过 `cvt_conv_streaming` 将 10 个 LFM2 ShortConv 转换为流式卷积：

转换完成后生成：

```
LFM2.5-1.2B-Instruct-convstream.rknn         # RKNN 模型
LFM2.5-1.2B-Instruct-convstream.weight        # RKNN 权重
```

> ⚠️ Conv Streaming Cache 由 RKNN3 Runtime 内部自动管理，无需手动维护。

## 2. C++ 部署说明

C++ 推理侧需使用 RKNN 模型、权重文件、tokenizer 文件和 embedding 权重文件进行部署。

### 2.1 编译

```bash
cd rknn3_model_zoo/

# 设置交叉编译工具链
export GCC_COMPILER=/path/to/aarch64-linux-gnu

# 编译
./build-linux.sh -t rk3588 -a aarch64 -b Release -d LFM2.5-1.2B-Instruct
```

编译完成后，文件生成在 `install/rk3588_linux_aarch64/rknn_lfm2_5_demo/` 目录。

### 2.2 运行示例

```bash
./rknn_lfm2_5_demo \
    <model_path> <weight_path> \
    <tokenizer_path> <embedding_path> \
    <core_mask> <prompt> [max_new_tokens]
```

以 **LFM2.5-1.2B-Instruct** 为例：

```bash
./rknn_lfm2_5_demo \
    model/LFM2.5-1.2B-Instruct-convstream.rknn \
    model/LFM2.5-1.2B-Instruct-convstream.weight \
    model/LFM2.5-1.2B-Instruct.tokenizer.gguf \
    model/LFM2.5-1.2B-Instruct.embed.bin \
    0xff \
    "你是谁？" \
    1024
```

参数说明：

| 参数 | 说明 |
|---|---|
| model_path | LLM RKNN 模型路径 |
| weight_path | LLM RKNN 权重文件路径 |
| tokenizer_path | tokenizer 文件路径（.tokenizer.gguf） |
| embedding_path | embedding 权重文件路径（.embed.bin） |
| core_mask | NPU 核心掩码，按 16 进制填写，例如 0x1、0xff |
| prompt | 用户输入文本，包含空格时需用英文双引号包裹 |
| max_new_tokens | 最大生成 token 数（可选，默认 1024，范围 1-2048） |

> ⚠️ demo 默认使用 `top_k=1`、`top_p=1.0`、`temperature=1.0`、`repeat_penalty=1.0`（纯 argmax 采样）。

## 3. Conv Streaming 技术说明

LFM2.5 的 10 个 LIV 层使用短卷积（kernel_size=3, depthwise），在自回归生成中需要维护 Conv State。

- **ONNX 导出**：标准 4 输入 ONNX，将 Conv padding 从 `[2,2]` 改为 `[2,0]`
- **RKNN 转换**：使用 `cvt_conv_streaming` 将 10 个 Conv 节点转换为流式卷积，Runtime 内部自动维护 Conv Cache；KV Cache buffer 长度 4096，max_position_embeddings 8192
- **Session 推理**：`RKNN3_LLM_INPUT_PROMPT` 高级 API 自动处理 Prefill/Decode 调度、Conv Streaming Cache 和 Attention KV Cache

Conv_streaming_Tc、Conv Streaming Cache 重置、Attention KV Cache、Chat Template 均由 Session 自动管理，无需手动处理。

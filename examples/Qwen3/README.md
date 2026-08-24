# Qwen3 模型部署说明

模型地址：[Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)

## 1. 导出 RKNN 模型

### 1.1 导出 LLM ONNX 模型

```bash
cd examples/Qwen3/python

# 默认导出 Qwen3-1.7B
python export_llm.py \
    --model_path Qwen/Qwen3-1.7B \
    --export_llm_path ../model/llm/Qwen3-1.7B.onnx

# 如需从 ModelScope 下载模型，增加 --modelscope 参数
python export_llm.py --modelscope
```

执行 `export_llm.py` 后，会在 ONNX 同目录下同步生成以下文件：

```
Qwen3-1.7B.onnx
Qwen3-1.7B.config.pkl
Qwen3-1.7B.tokenizer.gguf
Qwen3-1.7B.embed.bin
```

> `--quant` 参数可启用 GRQ 量化（需要 CUDA 环境）。

### 1.2 导出 LLM RKNN 模型

```bash
python export_rknn.py \
    --onnx_path ../model/llm/Qwen3-1.7B.onnx \
    --config ../model/llm/Qwen3-1.7B.config.pkl \
    --rknn_path ../model/llm/Qwen3-1.7B.rknn \
    --platform rk1820

# 如需重新导出（仅修改 profile_mode 或 kvcache 相关参数），可使用 --rebuild 跳过 ONNX 加载和图优化等步骤，加速模型导出：
python export_rknn.py \
    --onnx_path ../model/llm/Qwen3-1.7B.onnx \
    --config ../model/llm/Qwen3-1.7B.config.pkl \
    --rknn_path ../model/llm/Qwen3-1.7B.rknn \
    --platform rk1820 \
    --rebuild
```

> ⚠️ **`--rebuild` 说明**：当前仅支持重置 `profile_mode` 和 `llm_config` 中 kvcache 相关参数（如 `kvcache_buffer_len`或者`max_position_embeddings` 等）。其他参数变更需走完整导出流程。

RKNN 转换默认使用 W4A16 量化，采用 normal + group32 方式：

```python
rknn.config(target_platform=args.platform,
            quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32')
```

> ⚠️ `--platform` 默认为 `rk1820`，可通过参数修改。

## 2. C++ 部署说明

### 2.1 编译

```bash
cd rknn3_model_zoo/

# 设置交叉编译工具链
export GCC_COMPILER=/path/to/aarch64-linux-gnu

# 编译
./build-linux.sh -t rk3588 -a aarch64 -d Qwen3
```

编译完成后，文件生成在 `install/rk3588_linux_aarch64/rknn_Qwen3_demo/` 目录。

### 2.2 运行示例

```bash
./rknn_qwen3_demo \
    <model_path> <weight_path> \
    <tokenizer_path> <embedding_path> \
    <core_mask> <prompt>
```

以 **Qwen3-1.7B** 为例：

```bash
./rknn_qwen3_demo \
    model/Qwen3-1.7B.rknn \
    model/Qwen3-1.7B.weight \
    model/Qwen3-1.7B.tokenizer.gguf \
    model/Qwen3-1.7B.embed.bin \
    0xff \
    "解释相对论"
```

参数说明：

| 参数            | 说明                                          |
| --------------- | --------------------------------------------- |
| model_path      | LLM RKNN 模型路径                             |
| weight_path     | LLM RKNN 权重文件路径                         |
| tokenizer_path  | tokenizer 文件路径（.tokenizer.gguf）         |
| embedding_path  | embedding 权重文件路径（.embed.bin）          |
| core_mask       | NPU 核心掩码，按 16 进制填写，例如 0x1、0xff  |
| prompt          | 用户输入文本，包含空格时需用英文双引号包裹    |

> ⚠️ demo 默认使用 `top_k=1`、`top_p=0.9`、`temperature=1.0`、`repeat_penalty=1.2`。
> demo 中 `enable_thinking=false`，如需开启 thinking 模式，请在 C++ 代码中修改对应字段。

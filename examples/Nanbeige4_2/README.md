# Nanbeige4.2-3B 模型部署说明

模型地址：[Nanbeige/Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B)

Nanbeige4.2-3B 是南北阁（BOSS直聘）研发的 3B 参数量大语言模型，采用 Looped Transformer 架构（`num_loops=2`，22 层权重复用两次），具备较强的推理与智能体能力。

> ⚠️ 该模型使用 `trust_remote_code=True` 加载，需要模型目录中包含 `configuration_nanbeige.py` 和 `modeling_nanbeige.py`。

## 1. 导出 RKNN 模型

### 1.1 导出 LLM ONNX 模型

```bash
cd examples/Nanbeige4_2/python

# 安装依赖
pip install -r ../requirements.txt


# 使用本地模型路径导出
python export_llm.py \
    --model_path "Nanbeige/Nanbeige4.2-3B" \
    --export_llm_path ../model/llm/Nanbeige4.2-3B.onnx

# 如需从 ModelScope 下载模型，增加 --modelscope 参数
python export_llm.py --modelscope --model_path Nanbeige/Nanbeige4.2-3B
```

执行 `export_llm.py` 后，会在 ONNX 同目录下同步生成以下文件：

```
Nanbeige4.2-3B.onnx
Nanbeige4.2-3B.config.pkl
Nanbeige4.2-3B.tokenizer.gguf
Nanbeige4.2-3B.embed.bin
```

同时会基于 `datasets/CMMLU/dataset.json` 生成量化数据集 `datasets/CMMLU/dataset.txt`。

> `--quant` 参数可启用 GRQ 量化（需要 CUDA 环境）。

### 1.2 导出 LLM RKNN 模型

```bash
python export_rknn.py \
    --onnx_path ../model/llm/Nanbeige4.2-3B.onnx \
    --config ../model/llm/Nanbeige4.2-3B.config.pkl \
    --rknn_path ../model/llm/Nanbeige4.2-3B.rknn \
    --platform rk1820
```

RKNN 转换默认使用 W4A16 量化，采用 normal + group32 方式：

```python
rknn.config(target_platform=args.platform,
            quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32')
```

> ⚠️ `--platform` 为必填参数，目前可选 `rk1820`。

## 2. C++ 部署说明

C++ 推理侧需使用 RKNN 模型、权重文件、tokenizer 文件和 embedding 权重文件进行部署。

### 2.1 编译

```bash
cd rknn3_model_zoo/

# 设置交叉编译工具链
export GCC_COMPILER=/path/to/aarch64-linux-gnu

# 编译
./build-linux.sh -t rk3588 -a aarch64 -d Nanbeige4_2
```

编译完成后，文件生成在 `install/rk3588_linux_aarch64/rknn_Nanbeige4_2_demo/` 目录。

### 2.2 运行示例

```bash
./rknn_nanbeige4_2_demo \
    <model_path> <weight_path> \
    <tokenizer_path> <embedding_path> \
    <core_mask> <prompt>
```

以 **Nanbeige4.2-3B** 为例：

```bash
./rknn_nanbeige4_2_demo \
    model/Nanbeige4.2-3B.rknn \
    model/Nanbeige4.2-3B.weight \
    model/Nanbeige4.2-3B.tokenizer.gguf \
    model/Nanbeige4.2-3B.embed.bin \
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

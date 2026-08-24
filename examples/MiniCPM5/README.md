# MiniCPM5 模型部署说明

模型地址：[openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B)

## 1. 部署环境

> ⚠️ 需安装最新版 rknn3-toolkit（当前示例基于rknn3-toolkit-1.0.4）
```bash
numpy==2.5.1
py_utils==0.1.1
rknn3_toolkit>=1.0.4
torch==2.7.0
tqdm==4.68.4
transformers==4.51.3
```

## 2. 导出 RKNN 模型

### 2.1 导出 LLM ONNX 模型

```bash
cd examples/MiniCPM5/python

# 默认导出 MiniCPM5-1B
python export_llm.py \
    --model_path openbmb/MiniCPM5-1B \
    --export_llm_path ../model/llm/MiniCPM5-1B.onnx

# 如需从 ModelScope 下载模型，增加 --modelscope 参数
python export_llm.py --modelscope
```

执行 `export_llm.py` 后，会在 ONNX 同目录下同步生成以下文件：

```
MiniCPM5-1B.onnx
MiniCPM5-1B.config.pkl
MiniCPM5-1B.tokenizer.gguf
MiniCPM5-1B.embed.bin
```

> `--quant` 参数可启用 GRQ 量化（需要 CUDA 环境）。

### 2.2 导出 LLM RKNN 模型

```bash
python export_rknn.py\
    --onnx_path ../model/llm/MiniCPM5-1B.onnx\
    --config ../model/llm/MiniCPM5-1B.config.pkl\
    --rknn_path ../model/llm/MiniCPM5-1B.rknn\
    --platform rk1820
```

RKNN 转换默认使用 W4A16 量化，采用 normal + group32 方式：

```python
rknn.config(target_platform=args.platform,
            quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32')
```

> ⚠️ `--platform` 默认为 `rk1820`，可通过参数修改。

## 3. C++ 部署说明

### 3.1 编译

```bash
cd rknn3_model_zoo/

# 设置交叉编译工具链
export GCC_COMPILER=/path/to/aarch64-linux-gnu

# 编译
./build-linux.sh -t rk3588 -a aarch64 -d MiniCPM5
```

编译完成后，文件生成在 `install/rk3588_linux_aarch64/rknn_MiniCPM5_demo/` 目录。

### 3.2 部署到开发板

```bash
# 推送 demo 目录
adb push rknn_MiniCPM5_demo /data/

# 推送运行库
adb push rknn_MiniCPM5_demo/lib/* /usr/lib/
```

### 3.3 Run the Example

```bash
./rknn_minicpm5_demo\
    <model_path> <weight_path>\
    <tokenizer_path> <embedding_path>\
    <core_mask> <prompt>\
```

以 **MiniCPM5-1B** 为例：

```bash
export LD_LIBRARY_PATH=./lib

./rknn_minicpm5_demo\
    model/MiniCPM5-1B.rknn\
    model/MiniCPM5-1B.weight\
    model/MiniCPM5-1B.tokenizer.gguf\
    model/MiniCPM5-1B.embed.bin\
    0xff\
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

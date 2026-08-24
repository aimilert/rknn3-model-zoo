# MiniCPM5 Model Deployment Instructions

Model Address: [openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B)

## 1. Deployment Environment

> ⚠️ Requires installing the latest version of rknn3-toolkit(For the current example, based on rknn3-toolkit-1.0.4)
```bash
numpy==2.5.1
py_utils==0.1.1
rknn3_toolkit>=1.0.4
torch==2.7.0
tqdm==4.68.4
transformers==4.51.3
```

## 2. Export RKNN Model

### 2.1 Export LLM ONNX Model

```bash
cd examples/MiniCPM5/python

# Default export MiniCPM5-1B
python export_llm.py \
    --model_path openbmb/MiniCPM5-1B \
    --export_llm_path ../model/llm/MiniCPM5-1B.onnx

# To download models from ModelScope, add --modelscope
python export_llm.py --modelscope
```

After executing 'export_llm. py', the following files will be generated synchronously in the ONNX directory:

```
MiniCPM5-1B.onnx
MiniCPM5-1B.config.pkl
MiniCPM5-1B.tokenizer.gguf
MiniCPM5-1B.embed.bin
```

> '--quant' parameter enables GRQ quantization (requires CUDA environment).

### 2.2 Export LLM RKNN Model


```bash
python export_rknn.py\
    --onnx_path ../model/llm/MiniCPM5-1B.onnx\
    --config ../model/llm/MiniCPM5-1B.config.pkl\
    --rknn_path ../model/llm/MiniCPM5-1B.rknn\
    --platform rk1820
```

RKNN conversion defaults to W4A16 quantization, using the normal+group32 method:

```python
rknn.config(target_platform=args.platform,
            quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32')
```

> ⚠️ --Platform defaults to 'rk1820' and can be modified through parameters.

## 3. C++ Deployment

### 3.1 Compilation

```bash
cd rknn3_model_zoo/

# Set cross-compilation toolchain
export GCC_COMPILER=/path/to/aarch64-linux-gnu

# Compile
./build-linux.sh -t rk3588 -a aarch64 -d MiniCPM5
```

After compilation, the files are generated in the `install/rk3588_linux_aarch64/rknn_MiniCPM5_demo/` directory:

### 3.2 Deploy to Development Board

```bash
# Push the demo directory
adb push rknn_MiniCPM5_demo /data/

# Push runtime libraries
adb push rknn_MiniCPM5_demo/lib/* /usr/lib/
```

### 3.3 Run the Example

```bash
./rknn_minicpm5_demo\
    <model_path> <weight_path>\
    <tokenizer_path> <embedding_path>\
    <core_mask> <prompt>
```

Taking MiniCPM5-1B as an example:

```bash
export LD_LIBRARY_PATH=./lib

./rknn_minicpm5_demo\
    model/MiniCPM5-1B.rknn\
    model/MiniCPM5-1B.weight\
    model/MiniCPM5-1B.tokenizer.gguf\
    model/MiniCPM5-1B.embed.bin\
    0xff\
    "Explain the theory of relativity"
```

Parameter description:

| Parameter            | Description                                          |
| --------------- | --------------------------------------------- |
| model_path      | LLM RKNN model path                             |
| weight_path     | LLM RKNN weight file path                         |
| tokenizer_path  | tokenizer file path（.tokenizer.gguf）         |
| embedding_path  | embedding weight file path（.embed.bin）          |
| core_mask       | NPU core mask, filled in hexadecimal format, such as 0x1, 0xff  |
| prompt          | The user inputs text, when including spaces, use double quotation marks in English to wrap around    |

> ⚠️ The demo defaults to using `top_k=1`, `top_p=0.9`, `temperature=1.0`, `repeat_penalty=1.2`.
> demo 中 `enable_thinking=false`, To enable the thinking mode, please modify the corresponding field in the C++ code.
# Janus-Pro 模型部署说明

模型地址：[deepseek-ai/Janus-Pro-1B](https://huggingface.co/deepseek-ai/Janus-Pro-1B)

Janus-Pro 是 DeepSeek 提出的统一多模态理解与生成模型。本示例部署其多模态理解（视觉问答）能力。

## 1. 导出 RKNN 模型

### 1.1 Vision 模型

```bash
cd examples/Janus_Pro/python/vision

# 导出 ONNX 模型
python export_vision.py \
    --model_path deepseek-ai/Janus-Pro-1B \
    --export_vision_path ../../model/vision/Janus-Pro-1B-vision.onnx \
    --modelscope

# 导出 RKNN 模型
python export_rknn.py
```

### 1.2 LLM 模型

```bash
cd examples/Janus_Pro/python/llm

# 导出 ONNX 模型
python export_llm.py \
    --model_path deepseek-ai/Janus-Pro-1B \
    --export_llm_path ../../model/llm/Janus-Pro-1B-llm.onnx \
    --modelscope

# 导出 RKNN 模型
python export_rknn.py
```

> 导出 LLM ONNX 时会同步生成 `.config.pkl`、`.tokenizer.gguf`、`.embed.bin` 文件，并基于
> `datasets/MMBench/llm/dataset.json` 生成量化数据集。

LLM RKNN 转换默认使用 W4A16 量化，采用 GRQ + group32 方式：

```python
rknn.config(target_platform=args.platform,
            quantized_dtype='w4a16', quantized_algorithm='grq', quantized_method='group32')
```

## 2. C++ 部署说明

### 2.1 编译

```bash
cd rknn3_model_zoo/

# 设置交叉编译工具链
export GCC_COMPILER=/path/to/aarch64-linux-gnu

# 编译
./build-linux.sh -t rk3588 -a aarch64 -d Janus_Pro
```

编译完成后，文件生成在 `install/rk3588_linux_aarch64/rknn_Janus_Pro_demo/` 目录。

### 2.2 运行示例

```bash
./rknn_janus_demo \
    <vision_model_path> <vision_weight_path> \
    <llm_model_path> <llm_weight_path> \
    <tokenizer_path> <embedding_path> \
    <vision_core_mask> <llm_core_mask> \
    <image_path> <prompt>
```

示例：

```bash
./rknn_janus_demo \
    model/Janus-Pro-1B-vision.rknn model/Janus-Pro-1B-vision.weight \
    model/Janus-Pro-1B-llm.rknn model/Janus-Pro-1B-llm.weight \
    model/Janus-Pro-1B-llm.tokenizer.gguf model/Janus-Pro-1B-llm.embed.bin \
    0xff 0xff \
    demo.jpg "Briefly describe this image?"
```

参数说明：

| 参数               | 说明                                              |
| ------------------ | ------------------------------------------------- |
| vision_model_path  | Vision RKNN 模型路径                              |
| vision_weight_path | Vision RKNN 权重文件路径                          |
| llm_model_path     | LLM RKNN 模型路径                                 |
| llm_weight_path    | LLM RKNN 权重文件路径                             |
| tokenizer_path     | tokenizer 文件路径（.tokenizer.gguf）             |
| embedding_path     | embedding 权重文件路径（.embed.bin）              |
| vision_core_mask   | Vision 模型 NPU 核心掩码，16 进制，例如 0xff      |
| llm_core_mask      | LLM 模型 NPU 核心掩码，16 进制，例如 0xff         |
| image_path         | 输入图像路径                                      |
| prompt             | 用户输入文本，包含空格时需用英文双引号包裹        |

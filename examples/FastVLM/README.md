# FastVLM 模型部署说明

模型地址：[apple/llava-fastvithd_1.5b_stage3](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_1.5b_stage3.zip)

## 1. 导出 RKNN 模型

### 1.1 Vision 模型

```bash
cd examples/FastVLM/python/vision

# 导出 ONNX 模型（默认输入尺寸 512）
python export_vision.py \
    --model_path ../../llava-fastvithd_1.5b_stage3 \
    --export_vision_path ../../model/vision/FastVLM-vision.onnx

# 导出 RKNN 模型
python export_rknn.py
```

### 1.2 LLM 模型

```bash
cd examples/FastVLM/python/llm

# 导出 ONNX 模型
python export_llm.py \
    --model_path ../../llava-fastvithd_1.5b_stage3 \
    --export_llm_path ../../model/llm/FastVLM-llm.onnx

# 导出 RKNN 模型
python export_rknn.py
```

> 导出 LLM ONNX 时会同步生成 `.config.pkl`、`.tokenizer.gguf`、`.embed.bin` 文件，并基于
> `datasets/MMBench/llm/dataset.json` 生成量化数据集。

LLM RKNN 转换默认使用 W4A16 量化，采用 normal + group32 方式：

```python
rknn.config(target_platform=args.platform, core_num=args.core_num,
            quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32')
```

## 2. C++ 部署说明

### 2.1 编译

```bash
cd rknn3_model_zoo/

# 设置交叉编译工具链
export GCC_COMPILER=/path/to/aarch64-linux-gnu

# 编译
./build-linux.sh -t rk3588 -a aarch64 -d FastVLM
```

编译完成后，文件生成在 `install/rk3588_linux_aarch64/rknn_FastVLM_demo/` 目录。

### 2.2 运行示例

```bash
./rknn_fastvlm_demo \
    <vision_model_path> <vision_weight_path> \
    <llm_model_path> <llm_weight_path> \
    <tokenizer_path> <embedding_path> \
    <vision_core_mask> <llm_core_mask> \
    <image_path> <prompt>
```

示例：

```bash
./rknn_fastvlm_demo \
    model/FastVLM-vision.rknn model/FastVLM-vision.weight \
    model/FastVLM-llm.rknn model/FastVLM-llm.weight \
    model/FastVLM-llm.tokenizer.gguf model/FastVLM-llm.embed.bin \
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

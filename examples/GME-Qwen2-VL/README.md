# GME-Qwen2-VL 模型部署说明

模型地址：[iic/gme-Qwen2-VL-2B-Instruct](https://www.modelscope.cn/models/iic/gme-Qwen2-VL-2B-Instruct)

GME-Qwen2-VL 是基于 Qwen2-VL 的通用多模态嵌入模型，可用于图文检索等任务。

## 1. 导出 RKNN 模型

### 1.1 Vision 模型

```bash
cd examples/GME-Qwen2-VL/python/vision

# 生成 vision 模型量化校准数据
python make_calidata.py --model_path iic/gme-Qwen2-VL-2B-Instruct

# 导出 ONNX 模型（默认输入尺寸 448x448，须为 28 的倍数）
python export_vision.py \
    --model_path iic/gme-Qwen2-VL-2B-Instruct \
    --export_vision_path ../../model/vision/GmeQwen2VL-vision.onnx \
    --modelscope

# 导出 RKNN 模型
python export_rknn.py \
    --onnx_path ../../model/vision/GmeQwen2VL-vision.onnx \
    --rknn_path ../../model/vision/GmeQwen2VL-vision.rknn \
    --platform rk1820
```

Vision RKNN 转换支持两种模式：

| 模式 | 说明 |
|------|------|
| **Prune 模式**（默认） | 裁剪 Vision 模型，放到 CPU 前处理计算，适用于推理部署，可减少模型内存。 |
| **完整模式** | 保留完整 Vision 模型，输入为原始图像 `[1, 3, H, W]`。通过 `--no_prune_mode` 启用。 |

`make_calidata.py` 参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | HuggingFace/ModelScope 模型路径或名称 | `iic/gme-Qwen2-VL-2B-Instruct` |
| `--datapath` | 校准数据集 JSON 路径（含图片路径和文本输入） | `../../../../datasets/MMBench/llm/dataset.json` |
| `--export_datapath` | 生成的量化校准数据输出路径 | `./quant_data/model_inputs.json` |

> ⚠️ `make_calidata.py` 会加载完整模型并捕获模块输入，需要较大内存。生成的校准数据用于 `export_vision.py --quant` 时的 GRQ 量化。

### 1.2 LLM 模型

```bash
cd examples/GME-Qwen2-VL/python/llm

# 生成 LLM 模型量化校准数据
python make_calidata.py --model_path iic/gme-Qwen2-VL-2B-Instruct

# 导出 ONNX 模型
python export_llm.py \
    --model_path iic/gme-Qwen2-VL-2B-Instruct \
    --export_llm_path ../../model/llm/GmeQwen2VL-llm.onnx \
    --quant \
    --modelscope

# 导出 RKNN 模型
python export_rknn.py \
    --onnx_path ../../model/llm/GmeQwen2VL-llm.onnx \
    --config ../../model/llm/GmeQwen2VL-llm.config.pkl \
    --rknn_path ../../model/llm/GmeQwen2VL-llm.rknn \
    --platform rk1820

# 如需重新导出（仅修改 profile_mode 或 kvcache 相关参数），可使用 --rebuild 跳过 ONNX 加载和图优化等步骤，加速模型导出：
python export_rknn.py \
    --onnx_path ../../model/llm/GmeQwen2VL-llm.onnx \
    --config ../../model/llm/GmeQwen2VL-llm.config.pkl \
    --rknn_path ../../model/llm/GmeQwen2VL-llm.rknn \
    --platform rk1820 \
    --rebuild
```

> ⚠️ **`--rebuild` 说明**：当前仅支持重置 `profile_mode` 和 `llm_config` 中 kvcache 相关参数（如 `kvcache_buffer_len`或者`max_position_embeddings` 等）。其他参数变更需走完整导出流程。

> 导出 LLM ONNX 时会同步生成 `.config.pkl`、`.tokenizer.gguf`、`.embed.bin` 文件。

`make_calidata.py` 参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | HuggingFace/ModelScope 模型路径或名称 | `iic/gme-Qwen2-VL-2B-Instruct` |
| `--datapath` | 校准数据集 JSON 路径（含图片路径和文本输入） | `../../../../datasets/MMBench/llm/dataset.json` |
| `--export_datapath` | 生成的量化校准数据输出路径 | `./quant_data/model_inputs.json` |

> ⚠️ `make_calidata.py` 会加载完整模型并捕获模块输入，需要较大内存。生成的校准数据用于 `export_llm.py --quant` 时的 GRQ 量化。

LLM RKNN 转换默认使用 W6A16 量化，采用 normal + group32 方式：

```python
rknn.config(target_platform=args.platform,
            quantized_dtype='w6a16', quantized_algorithm='normal', quantized_method='group32',
            llm_config=llm_config)
```

## 2. C++ 部署说明

### 2.1 编译

```bash
cd rknn3_model_zoo/

# 设置交叉编译工具链
export GCC_COMPILER=/path/to/aarch64-linux-gnu

# 编译
./build-linux.sh -t rk3588 -a aarch64 -d GME-Qwen2-VL
```

编译完成后，文件生成在 `install/rk3588_linux_aarch64/rknn_GME-Qwen2-VL_demo/` 目录。

### 2.2 运行示例

```bash
./rknn_gme_qwen_vl_demo \
    <vision_model_path> <vision_weight_path> \
    <llm_model_path> <llm_weight_path> \
    <tokenizer_path> <embedding_path> \
    <vision_core_mask> <llm_core_mask> \
    <image_path> <prompt>
```

示例：

```bash
./rknn_gme_qwen_vl_demo \
    model/GmeQwen2VL-vision.rknn model/GmeQwen2VL-vision.weight \
    model/GmeQwen2VL-llm.rknn model/GmeQwen2VL-llm.weight \
    model/GmeQwen2VL-llm.tokenizer.gguf model/GmeQwen2VL-llm.embed.bin \
    0xff 0xff \
    demo.jpg "Describe this image."
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

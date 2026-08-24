# UI-TARS 模型部署说明

模型地址：[ByteDance-Seed/UI-TARS-2B-SFT](https://huggingface.co/ByteDance-Seed/UI-TARS-2B-SFT)

UI-TARS 是字节跳动开源的 GUI Agent 视觉语言模型，基于 Qwen2-VL，可用于界面操作等 Agent 任务。

## 1. 导出 RKNN 模型

### 1.1 Vision 模型


```bash
cd examples/UI_TARS/python/vision

# 生成校准数据集：
python3 make_calidata.py --model_path ByteDance-Seed/UI-TARS-2B-SFT --modelscope

# 导出 ONNX 模型（默认输入尺寸 448x448，须为 28 的倍数）
python export_vision.py \
    --model_path ByteDance-Seed/UI-TARS-2B-SFT \
    --export_vision_path ../../model/vision/UI-TARS-2B-SFT-vision.onnx \
    --modelscope

# 导出 RKNN 模型
python export_rknn.py
```


### 1.2 LLM 模型

```bash
cd examples/UI_TARS/python/llm

# 导出 ONNX 模型
python export_llm.py \
    --model_path ByteDance-Seed/UI-TARS-2B-SFT \
    --export_llm_path ../../model/llm/UI-TARS-2B-SFT-llm.onnx \

# 导出 ONNX 模型
python export_llm.py \
    --model_path ByteDance-Seed/UI-TARS-2B-SFT \
    --export_llm_path ../../model/llm/UI-TARS-2B-SFT-llm.onnx \
    --modelscope

# 导出 RKNN 模型
python export_rknn.py
```

> 导出 LLM ONNX 时会同步生成 `.config.pkl`、`.tokenizer.gguf`、`.embed.bin` 文件。

LLM RKNN 转换默认使用 W4A16 量化，采用外部GRQ + group32 方式：

```python
rknn.config(target_platform=args.platform,
            quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32',
            llm_config=llm_config)
```

## 2. C++ 部署说明

### 2.1 编译

```bash
cd rknn3_model_zoo/

# 设置交叉编译工具链
export GCC_COMPILER=/path/to/aarch64-linux-gnu

# 编译
./build-linux.sh -t rk3588 -a aarch64 -d UI_TARS
```

编译完成后，文件生成在 `install/rk3588_linux_aarch64/rknn_UI_TARS_demo/` 目录。

### 2.2 运行示例

```bash
./rknn_ui_tars_vl_demo \
    <vision_model_path> <vision_weight_path> \
    <llm_model_path> <llm_weight_path> \
    <tokenizer_path> <embedding_path> \
    <vision_core_mask> <llm_core_mask> \
    <image_path> <prompt>
```

示例：

```bash
adb shell
cd /data/rknn_UI_TARS_demo

./rknn_ui_tars_vl_demo \
    model/UI-TARS-2B-SFT-vision.rknn model/UI-TARS-2B-SFT-vision.weight \
    model/UI-TARS-2B-SFT-llm.rknn model/UI-TARS-2B-SFT-llm.weight \
    model/UI-TARS-2B-SFT-llm.tokenizer.gguf model/UI-TARS-2B-SFT-llm.embed.bin \
    0xff 0xff \
    model/demo.jpg "What should I do next on this screen?"
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


输出示例：

```
"To proceed with the task, you should interact with the text box labeled '1. 简介 ======' at the top of the screen. This text box likely contains an introduction or summary of the model's capabilities and features. Clicking on this text box will allow you to view or edit the content."
```

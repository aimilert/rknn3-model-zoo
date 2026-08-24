# SmolVLM2 模型部署说明

模型地址：[HuggingFaceTB/SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)

## 1. 导出 RKNN 模型
### 1.1 安装依赖包
```bash
cd python
pip install -r requirements.txt
```

### 1.2 导出 Vision 模型

```bash
cd python/vision

# 导出 ONNX 模型
python export_vision.py --modelscope

# 导出 RKNN 模型
python export_rknn.py
```

### 1.3 导出 LLM 模型

```bash
cd python/llm

# 导出 ONNX 模型
python export_llm.py --modelscope

# 导出 RKNN 模型
python export_rknn.py
```

> 导出完成后，模型文件将生成在 `model` 目录下。

## 2. C++ 板端部署

### 2.1 编译

```bash
cd rknn3_model_zoo/

# 设置交叉编译工具链
export GCC_COMPILER=/path/to/aarch64-linux-gnu

# 编译
./build-linux.sh -t rk3588 -a aarch64 -d SmolVLM2
```

编译完成后，文件生成在 `install/rk3588_linux_aarch64/rknn_SmolVLM2_demo/` 目录：

```
rknn_SmolVLM2_demo/
├── lib/
│   ├── librga.so
│   └── librknn3_api.so
├── model/
│   ├── demo.jpg
│   ├── SmolVLM2-500M-llm.rknn
│   ├── SmolVLM2-500M-llm.weight
│   ├── SmolVLM2-500M-llm.embed.bin
│   ├── SmolVLM2-500M-llm.tokenizer.gguf
│   ├── SmolVLM2-500M-vision.rknn
│   └── SmolVLM2-500M-vision.weight
└── rknn_smol_vl_demo
```

### 2.2 部署到开发板

```bash
# 推送 demo 目录
adb push rknn_SmolVLM2_demo /data/
```

### 2.3 运行示例

```bash
adb shell
cd /data/rknn_SmolVLM2_demo
```
运行命令如下：

```bash
./rknn_smol_vl_demo model/SmolVLM2-500M-vision.rknn model/SmolVLM2-500M-vision.weight model/SmolVLM2-500M-llm.rknn model/SmolVLM2-500M-llm.weight model/SmolVLM2-500M-llm.tokenizer.gguf model/SmolVLM2-500M-llm.embed.bin 0xff 0xff model/demo.jpg "Briefly describe this image?"
```

输出示例：
```
 A spaceman is drinking a beer on the moon.
```
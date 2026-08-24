# SmolVLM 模型部署说明

模型地址：[HuggingFaceTB/SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)

## 1. 导出 RKNN 模型

### 1.1 Vision 模型

```bash
cd python/vision

# 导出 ONNX 模型
python export_vision.py --modelscope

# 导出 RKNN 模型
python export_rknn.py
```

### 1.2 LLM 模型

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
./build-linux.sh -t rk3588 -a aarch64 -d SmolVLM
```

编译完成后，文件生成在 `install/rk3588_linux_aarch64/rknn_SmolVLM_demo/` 目录：

```
rknn_SmolVLM_demo/
├── lib/
│   ├── librga.so
│   └── librknn3_api.so
├── model/
│   ├── demo.jpg
│   ├── SmolVLM-500M-llm.rknn
│   ├── SmolVLM-500M-llm.weight
│   ├── SmolVLM-500M-llm.embed.bin
│   ├── SmolVLM-500M-llm.tokenizer.gguf
│   ├── SmolVLM-500M-vision.rknn
│   └── SmolVLM-500M-vision.weight
└── rknn_smol_vl_demo
```

### 2.2 部署到开发板

```bash
# 推送 demo 目录
adb push rknn_SmolVLM_demo /data/

# 推送运行库
adb push rknn_SmolVLM_demo/lib/* /usr/lib/
```

### 2.3 运行示例

```bash
adb shell
cd /data/rknn_SmolVLM_demo
```
运行命令如下：

```bash
./rknn_smol_vl_demo model/SmolVLM-500M-vision.rknn model/SmolVLM-500M-vision.weight model/SmolVLM-500M-llm.rknn model/SmolVLM-500M-llm.weight model/SmolVLM-500M-llm.tokenizer.gguf model/SmolVLM-500M-llm.embed.bin 0xff 0xff model/demo.jpg "Briefly describe this image?"
```

输出示例：
```
 A spaceman is drinking a beer on the moon.
```
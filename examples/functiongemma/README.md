# google/functiongemma-270m-it 模型部署说明

模型地址：[google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it)

## 1.安装依赖库
本仓库根目录下的 `requirements.txt`中的pytorch/onnx即可导出onnx

## 2.导出onnx模型

```shell
cd python
python export_llm.py --modelscope
```

## 3.转换rknn模型

```shell
cd python
python export_rknn.py
```

可通过 `--platform` 指定目标平台（默认 `rk1820`，可选 `rk1820`/`rk3572`）：

```shell
python export_rknn.py --platform rk3572
```

## 4. C++ 板端部署

```bash
cd rknn3_model_zoo/

# 设置交叉编译工具链
export GCC_COMPILER=/path/to/aarch64-linux-gnu

# 编译
./build-linux.sh -t rk3588 -a aarch64 -d functiongemma
```

编译完成后，文件生成在 `install/rk3588_linux_aarch64/rknn_functiongemma_demo/` 目录：

```
rknn_functiongemma_demo/
├── lib
│   ├── librga.so
│   └── librknn3_api.so
├── model
│   ├── functiongemma-270m-it.embed.bin
│   ├── functiongemma-270m-it.rknn
│   ├── functiongemma-270m-it.weight
│   ├── functiongemma-270m-it.safetensors
│   └── functiongemma-270m-it.tokenizer.gguf
└── rknn_functiongemma_demo
```

### 3.2 部署到开发板

```bash
# 推送 demo 目录
adb push install/rk3588_linux_aarch64/rknn_functiongemma_demo/ /data
```

### 3.3 运行示例

```bash
adb shell
cd /data/rknn_functiongemma_demo

./rknn_functiongemma_demo model/functiongemma-270m-it.rknn model/functiongemma-270m-it.weight 0xff model/functiongemma-270m-it.tokenizer.gguf model/functiongemma-270m-it.embed.bin 1024 128 model/functiongemma-270m-it.safetensors "who are you?"
```

输出示例：
```
I am a large language model, a type of artificial intelligence. I can only interact with the capabilities of my assistant and I cannot "be" or "be" individuals. I cannot form opinions or beliefs about people. My capabilities are limited to processing information using the tools I have been programmed with.
-----------------------------------------------------------------------------------------
 Stage      | Total Time (ms)  | Tokens   | Time per Token (ms)  | Tokens per Second    
-----------------------------------------------------------------------------------------
 Prefill    | 18.39            | 13       | 1.41                 | 707.06               
 Generate   | 305.79           | 60       | 5.10                 | 196.22               
-----------------------------------------------------------------------------------------
```

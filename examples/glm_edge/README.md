# glm-edge-1.5b-chat 模型部署说明

模型地址：[glm-edge-1.5b-chat](https://modelscope.cn/models/ZhipuAI/glm-edge-1.5b-chat)

## 1.安装依赖库
本仓库根目录下的 `requirements.txt`中的pytorch/onnx无法导出该模型，请使用本目录下的 `requirements.txt`安装依赖库，以下命令安装特定版本：

```shell
cd python
pip install -r requirements.txt
```

## 2.导出onnx模型

```shell
cd python
python export_llm.py --quant --modelscope
```

## 3.转换rknn模型

```shell
cd python
python export_rknn.py
```

## 4. C++ 板端部署

```bash
cd rknn3_model_zoo/

# 设置交叉编译工具链
export GCC_COMPILER=/path/to/aarch64-linux-gnu

# 编译
./build-linux.sh -t rk3588 -a aarch64 -d glm_edge
```

编译完成后，文件生成在 `install/rk3588_linux_aarch64/rknn_glm_edge_demo/` 目录：

```
rknn_glm_edge_demo/
├── lib
│   ├── librga.so
│   └── librknn3_api.so
├── model
│   ├── glm-edge-1.5b-chat.embed.bin
│   ├── glm-edge-1.5b-chat.rknn
│   ├── glm-edge-1.5b-chat.tokenizer.gguf
│   └── glm-edge-1.5b-chat.weight
└── rknn_glm_demo
```

### 3.2 部署到开发板

```bash
# 推送 demo 目录
adb push install/rk3588_linux_aarch64/rknn_glm_edge_demo /data/
```

### 3.3 运行示例

```bash
adb shell
cd /data/rknn_glm_edge_demo

./rknn_glm_demo ./model/glm-edge-1.5b-chat.rknn model/glm-edge-1.5b-chat.weight model/glm-edge-1.5b-chat.tokenizer.gguf model/glm-edge-1.5b-chat.embed.bin 0xff "who are you"
```

输出示例：
```
I am GLM-1.5, a language model trained by ZhiPu AI Company.<|im_end|>
```

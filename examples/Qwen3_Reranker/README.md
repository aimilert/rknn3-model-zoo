# Qwen3-Reranker 系列模型部署说明

- [Qwen3-Reranker 系列模型部署说明](#qwen3-reranker-系列模型部署说明)
  - [模型转换](#模型转换)
    - [导出 ONNX 模型](#导出-onnx-模型)
    - [导出 RKNN 模型](#导出-rknn-模型)
  - [板端部署](#板端部署)
    - [Linux 平台使用示例](#linux-平台使用示例)
      - [编译](#编译)
      - [推送到板端](#推送到板端)
      - [运行](#运行)
    - [Android 平台使用示例](#android-平台使用示例)
      - [编译](#编译-1)
      - [推送到板端](#推送到板端-1)
      - [运行](#运行-1)


## 模型转换

### 导出 ONNX 模型

以 Qwen3-Reranker-0.6B 为例，导出 ONNX 模型参考命令：

```bash
python export_llm.py \
    --model_path Qwen/Qwen3-Reranker-0.6B \
    --export_llm_path ../model/llm/Qwen3-Reranker-0.6B.onnx \
    --quant \
    --modelscope
```

**注意：** Qwen3-Reranker 系列模型与 Qwen3 系列模型用途不同：

- Qwen3-Reranker 系列模型的输出结果为 *reranker* 分数，用于语意排序。
- Qwen3 系列模型的输出是通过自回归生成的文本，主要用于文本生成、对话等自然语言生成场景。

因此，在导出 Qwen3-Reranker 的 ONNX 模型时，需通过配置文件明确指定模型为 reranker 任务，具体可参考 `export_llm.py` 中的如下示例代码：

```python 
# Export LLM configuration 
# 0: the generation task.
# 1: the Embedding task.
# 2: the reranker task.
user_config = {"task_type": 2}
export_llm_config(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.config.pkl', None, None, user_config)
```
> 如未显式指定 task_type，默认值为 0（generation 任务）

### 导出 RKNN 模型

以 Qwen3-Reranker-0.6B 为例，导出 RKNN 模型参考命令：

```bash
python export_rknn.py \
    --onnx_path ../model/llm/Qwen3-Reranker-0.6B.onnx \
    --config ../model/llm/Qwen3-Reranker-0.6B.config.pkl \
    --rknn_path ../model/llm/Qwen3-Reranker-0.6B.rknn \
    --platform rk1820
```
用户可在 `rknn.config` 中设置 `max_ctx_len` 参数，以调整模型支持的最大输入 token 长度（即最大上下文长度）。



## 板端部署

为支持动态输入长度，Qwen3-Reranker 系列模型采用 session API 部署，与 Qwen3 系列模型用法类似。但主要区别在于 reranker 结果是通过 output_callback 接口返回。具体实现可参考 `Qwen3_Reranker/cpp/main.cc` 示例，核心流程如下：

1. 查询输出的属性信息
2. 指定需要获取的输出 index，并创建对应的 output tensor，同时绑定 output_callback
3. 通过 output_callback 返回的 output tensor 获取最终的 reranker 分数结果

**输入格式说明：** Qwen3-Reranker 使用特定的聊天模板对输入进行格式化，模板格式如下：

```
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {instruction}
<Query>: {query}
<Document>: {document}
<|im_end|>
<|im_start|>assistant
<think>

```

其中 `instruction` 默认为 `"Given a web search query, retrieve relevant passages that answer the query"`。C++ demo 会自动将输入的 `query` 和 `document` 参数按此模板格式化后送入模型推理。

### Linux 平台使用示例

#### 编译

```sh
# 请先指定编译器路径
(optional)export GCC_COMPILER=<GCC_COMPILER_PATH>

./build-linux.sh -t <target> -a <arch> -d <build_demo_name>

# 例如
./build-linux.sh -t rk3588 -a aarch64 -d Qwen3_Reranker
```

#### 推送到板端

```sh
adb push install/<target>_linux_<arch>/rknn_Qwen3_Reranker_demo /data/
```

#### 运行

```sh
adb shell
cd /data/rknn_Qwen3_Reranker_demo

export LD_LIBRARY_PATH=./lib

# Usage: ./rknn_qwen3_reranker_demo <model_path> <weight_path> <tokenizer_path> <reranker_path> <core_mask> <query> <document>
./rknn_qwen3_reranker_demo model/Qwen3-Reranker-0.6B.rknn model/Qwen3-Reranker-0.6B.weight model/Qwen3-Reranker-0.6B.tokenizer.gguf model/Qwen3-Reranker-0.6B.embed.bin 0xff '"What is deep learning?"' '"Deep learning is a subset of machine learning that uses neural networks with multiple layers."'
```
参数说明:
- `model_path`: rknn 文件路径
- `weight_path`: weight 文件路径
- `tokenizer_path`: tokenizer.gguf 文件路径
- `reranker_path`: reranker.bin 文件路径
- `core_mask`: 目前有 8 个核，对应 8 bit 数，使用哪一个核，就将哪一位置 1，例如使用核 0 和核 1，就将第 0 位和第 1 位置 1，得到的二进制数是0b11，对应的十六进制数是 0x3，core_mask 设置成 0x3
- `query`: 查询文本
- `document`: 待排序的文档文本

### Android 平台使用示例

#### 编译

```sh
# 请先指定编译器路径
(optional)export ANDROID_NDK_PATH=<ANDROID_NDK_PATH>

./build-android.sh -t <target> -a <arch> -d <build_demo_name>

# 例如
./build-android.sh -t rk3588 -a arm64-v8a -d Qwen3_Reranker
```

#### 推送到板端

```sh
adb root
adb remount
adb push install/<target>_android_<arch>/rknn_Qwen3_Reranker_demo /data/
```

#### 运行

```sh
adb shell
cd /data/rknn_Qwen3_Reranker_demo

export LD_LIBRARY_PATH=./lib

# Usage: ./rknn_qwen3_reranker_demo <model_path> <weight_path> <tokenizer_path> <reranker_path> <core_mask> <query> <document>
./rknn_qwen3_reranker_demo model/Qwen3-Reranker-0.6B.rknn model/Qwen3-Reranker-0.6B.weight model/Qwen3-Reranker-0.6B.tokenizer.gguf model/Qwen3-Reranker-0.6B.embed.bin 0xff '"What is deep learning?"' '"Deep learning is a subset of machine learning that uses neural networks with multiple layers."'
```
参数说明:
- `model_path`: rknn 文件路径
- `weight_path`: weight 文件路径
- `tokenizer_path`: tokenizer.gguf 文件路径
- `reranker_path`: reranker.bin 文件路径
- `core_mask`: 目前有 8 个核，对应 8 bit 数，使用哪一个核，就将哪一位置 1，例如使用核 0 和核 1，就将第 0 位和第 1 位置 1，得到的二进制数是0b11，对应的十六进制数是 0x3，core_mask 设置成 0x3
- `query`: 查询文本
- `document`: 待排序的文档文本

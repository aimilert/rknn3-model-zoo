# Qwen3-ASR 模型部署说明

Qwen3-ASR 系列包括 Qwen3-ASR-1.7B 和 Qwen3-ASR-0.6B，支持 52 种语言和方言的语言识别与语音识别（ASR）。两者均利用大规模语音训练数据以及其基础模型 Qwen3-Omni 强大的音频理解能力。

RKNN3部署分为Audio Encoder和LLM两个部分，分别导出模型。

## 1. 项目结构

```
Qwen3_ASR/
├── cpp                                  # C++ 推理代码
│   ├── audio
│   │   ├── rknn_qwen3_asr_audio.cc
│   │   └── rknn_qwen3_asr_audio.h
│   ├── CMakeLists.txt
│   ├── llm
│   │   ├── rknn_qwen3_asr_llm.cc
│   │   └── rknn_qwen3_asr_llm.h
│   ├── main.cc
│   ├── online_main.cc
│   ├── qwen3_asr.cc
│   └── qwen3_asr.h
├── data
│   └── audio
│       ├── asr_en.wav
│       └── mel_128_filters.txt
├── python
│   ├── audio                            # Audio Encoder 导出
│   │   ├── offline
│   │   │   ├── export_audio_onnx.py
│   │   │   └── export_audio_rknn.py
│   │   └── online
│   │       ├── export_audio_onnx.py
│   │       └── export_audio_rknn.py
│   └── llm                              # LLM 模型导出
│       ├── export_llm.py
│       └── export_rknn.py
└── README.md
```

## 2. ONNX 模型导出

### 模型导出环境

需要在同一个环境中安装rknn3和Qwen3-ASR的环境。首先安装rknn3环境，安装完成后安装Qwen3-ASR环境。

#### RKNN3

根据发布包直接安装即可。

#### Qwen3ASR

按Qwen3-ASR官方代码，从源码安装环境，便于修改。官方环境安装命令如下：

```bash
git clone https://github.com/QwenLM/Qwen3-ASR.git
cd Qwen3-ASR
pip install -e .
```

#### 常见问题

导出llm onnx模型时出现报错，通常是torch版本过低，已验证通过`torch==2.10.0`

### 2.1 导出 Audio Encoder 模型

将 Qwen3-ASR 模型的音频编码器导出为 ONNX 格式：

```bash
# 非流式
cd python/audio/offline
python export_audio_onnx.py \
    --model_path Qwen/Qwen3-ASR-0.6B \
    --export_encoder_path ../../models/encoder.onnx \
    --modelscope

# 流式
cd python/audio/online
python export_audio_onnx.py \
    --model_path Qwen/Qwen3-ASR-0.6B \
    --export_encoder_path ../../models/encoder_online.onnx \
    --modelscope
```

#### 参数说明

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--model_path` | str | 模型路径或 HuggingFace 名称 | `Qwen/Qwen3-ASR-0.6B` |
| `--export_encoder_path` | str | 输出 ONNX 路径 | `../../../models/encoder.onnx` |
| `--modelscope` | bool | 从 ModelScope 下载（推荐国内用户） | `False` |

### 2.2 生成量化校准数据

在启用 `--quant`（GRQ 量化）导出 LLM ONNX 之前，需要先准备量化校准数据。
`make_calidata.py` 会加载 Qwen3-ASR 模型，对一批音频执行语音识别，并捕获进入 thinker
模型的真实输入，pickle 落盘后生成校准数据索引文件，供 `export_llm.py` 的 `--cali_dataset`
使用。

```bash
cd python/llm
python make_calidata.py
```

执行后会在当前目录下生成：

```
quant_data/
├── model_input.json          # 校准数据索引（供 export_llm.py --cali_dataset 使用）
└── model_inputs/
    ├── sample_0              # 每条音频捕获的输入（pickle）
    ├── sample_1
    └── ...
```

脚本逻辑要点：

- 音频清单从 `../../../../datasets/ASR/audio_data.txt` 读取，每行为 `datasets/ASR/` 目录下
  的一个音频文件名；可按需替换为你自己的音频清单与音频文件。
- 对每条音频调用 `model.transcribe(...)`，同时通过 `capture_module_input` 钩住
  `model.model.thinker.model` 捕获其输入张量。
- 索引文件 `model_input.json` 的路径与 `export_llm.py` 的默认 `--cali_dataset`
  （`quant_data/model_input.json`）一致，无需额外传参。

> ⚠️ **注意**：
> - `make_calidata.py` 中的模型路径为脚本内硬编码（如 `/data/public/Qwen3-ASR-0.6B`），
>   执行前请修改为本地实际的模型路径。
> - 仅当 LLM 导出启用 `--quant` 时才需要执行本步骤；导出 float（非量化）模型时可跳过。

### 2.3 导出 LLM 模型

将 Qwen3-ASR 模型的 LLM 部分导出为 ONNX 格式：

```bash
cd python/llm
python export_llm.py \
    --model_path Qwen/Qwen3-ASR-0.6B \
    --export_llm_path llm.onnx \
    --modelscope \
    --quant
```

#### 参数说明

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--model_path` | str | 模型路径或 HuggingFace 名称 | `Qwen/Qwen3-ASR-0.6B` |
| `--export_llm_path` | str | 输出 ONNX 路径 | `llm.onnx` |
| `--modelscope` | bool | 从 ModelScope 下载 | `False` |

## 3. RKNN 模型导出

### 3.1 导出 Audio Encoder RKNN 模型

```bash
# 非流式
cd python/audio/offline
python export_audio_rknn.py \
    --onnx_path ../../../models/encoder.onnx \
    --rknn_path ../../../models/encoder.rknn

# 流式
cd python/audio/online
python export_audio_rknn.py \
    --onnx_path ../../../models/encoder_online.onnx \
    --rknn_path ../../../models/encoder_online.rknn
```

#### 参数说明

非流式与流式脚本参数一致，仅默认路径不同（流式默认为 `encoder_online.onnx` / `encoder_online.rknn`）：

| 参数 | 类型 | 说明 | 非流式默认值 | 流式默认值 |
|------|------|------|--------|--------|
| `--onnx_path` | str | 输入 ONNX 模型路径 | `../../../models/encoder.onnx` | `../../../models/encoder_online.onnx` |
| `--rknn_path` | str | 输出 RKNN 模型路径 | `../../../models/encoder.rknn` | `../../../models/encoder_online.rknn` |
| `--platform` | str | 目标平台 | `rk1820` | `rk1820` |
| `--do_quant` | bool | 开启量化构建 | `False` | `False` |

### 3.2 导出 LLM RKNN 模型

```bash
cd python/llm
python export_rknn.py \
    --onnx_path llm.onnx \
    --config llm.config.pkl \
    --rknn_path ../../models/llm.rknn
```

Qwen3-ASR最长可支持20分钟音频，代码默认导出的模型上下文为24k, 可支持约20分钟的音频转录。    
若需要降低182x内存占用，可缩短上下文长度，即缩短支持的转录音频时长。需要修改`python/llm/export_rknn.py`中的配置，找到下面两行：
```
    my_config['attention_config'][0]['kvcache_buffer_len'] = 1024*24
    my_config['attention_config'][0]['max_position_embeddings'] = 1024*24
```
将`1024*24`改小重新导出rknn即可。

#### 参数说明

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--onnx_path` | str | 输入 ONNX 模型路径 | `llm.onnx` |
| `--config` | str | 模型 Config 路径（LLM 必需） | `llm.config.pkl` |
| `--rknn_path` | str | 输出 RKNN 模型路径 | `../../models/llm.rknn` |

> **注意**：导出使用权重分离模式，会生成 `.rknn` 和 `.weight` 两个文件。

## 4. C++ 部署

### 4.1 编译

```bash
cd ../..
./build-linux.sh -t rk3588 -a aarch64 -d Qwen3_ASR -b Release
```

### 4.2 推送到开发板

```bash
adb push install/rk3588_linux_aarch64/rknn_Qwen3_ASR_demo/ /data/
```

由于模型体积较大，为了避免每次修改代码后，编译和推到开发板上耗时过长，目前编译脚本不会将model拷贝到install目录，需要手动将导出的模型推到开发板上。

```bash
adb push models/ /data/rknn_Qwen3_ASR_demo/model
```

### 4.3 运行

```bash
adb shell
cd /data/rknn_Qwen3_ASR_demo

export LD_LIBRARY_PATH=./lib

# 非流式
./rknn_qwen3_asr_demo \
    model/encoder.rknn \
    model/encoder.weight \
    model/llm.rknn \
    model/llm.weight \
    model/llm.tokenizer.gguf \
    model/llm.embed.bin \
    0xff \
    0xff \
    asr_en.wav

# 流式
./rknn_qwen3_asr_demo_online \
    model/encoder_online.rknn \
    model/encoder_online.weight \
    model/llm.rknn \
    model/llm.weight \
    model/llm.tokenizer.gguf \
    model/llm.embed.bin \
    0xff \
    0xff \
    asr_en.wav

# 流式：输出实时刷新，可以更直观的看到ASR流式推理输出的过程，仅支持单行输出文本，请使用短音频测试
./rknn_qwen3_asr_demo_online \
    model/encoder_online.rknn \
    model/encoder_online.weight \
    model/llm.rknn \
    model/llm.weight \
    model/llm.tokenizer.gguf \
    model/llm.embed.bin \
    0xff \
    0xff \
    asr_en.wav \
    -s
```

应该能看到下面的输出：

```
# 非流式
language res: language English
text res:
Oh yeah, yeah. He wasn't even that big when I started listening to him, but and his solo music didn't do overly well, but he did very well when he started writing for other people.

# 流式
================ Final Commit Result ================
Uh huh. Oh yeah, yeah he wasn't even that big when I started listening to him, but and his solo music didn't do overly well, but he did very well when he started writing for other people.

```

#### 参数说明

| 参数 | 说明 |
|------|------|
| `model/encoder.rknn` | Audio Encoder RKNN 模型 |
| `model/encoder.weight` | Audio Encoder 权重文件 |
| `model/llm.rknn` | LLM RKNN 模型 |
| `model/llm.weight` | LLM 权重文件 |
| `model/llm.tokenizer.gguf` | Tokenizer 文件 |
| `model/llm.embed.bin` | Embed 文件 |
| `0xff` | Audio Encoder 使用的 182x核心数（0xff 表示8核） |
| `0xff`                     | LLM 使用的 182x核心数（0xff 表示8核）           |
| `asr_en.wav` | 输入音频文件 |

## 5. 支持的模型

目前支持 Qwen3-ASR-0.6B、Qwen3-ASR-1.7B 等模型。导出时请指定对应的模型路径。

## 6. 常见问题

### ONNX 文件路径问题

使用 PyTorch ≥ 2.9.0 导出的模型会生成 `xxx.onnx` 和 `xxx.onnx.data` 两个文件。执行 `rknn.load_llm` 时，必须确保这两个文件在同一目录下。





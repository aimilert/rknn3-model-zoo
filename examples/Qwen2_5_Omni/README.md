# Qwen2.5-Omni 模型部署说明

模型地址：[Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B)

Qwen2.5-Omni 是端到端多模态大模型，支持图像、音频、文本的统一理解。RKNN3 部署将模型拆分为 Vision、Audio、LLM 三个部分分别导出，再在板端联合推理。

## 1. 部署环境

本仓库 `requirements.txt` 中的 `transformers==4.51.3` 无法导出该模型，请使用以下命令安装特定版本：

```bash
pip install git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview
```
 若需进行外部grq量化，请在环境中安装数据集依赖
 ```bash
pip install qwen-omni-utils
 ```

> ⚠️ 还需安装最新版 rknn3-toolkit

由于 Qwen2.5-Omni 与 RKNN3 部分依赖包可能存在冲突，建议分别为 ONNX 导出和 RKNN 转换搭建独立的运行环境，避免潜在问题。

> ⚠️ **关于 CUDA**：仅当启用 `--quant`（GRQ 量化）导出 Vision/LLM 的 ONNX 模型时需要 NVIDIA GPU；导出 float（非量化）模型、生成校准数据、RKNN 转换以及板端推理均不需要 CUDA。

## 2. 项目结构

```
Qwen2_5_Omni/
├── cpp/                          # C++ 推理代码
│   ├── main.cc                   # 程序入口，多模态输入组装与调度
│   ├── qwen2_5_omni.cc/.h        # Omni 模型调度逻辑
│   ├── vision/                   # Vision 推理封装
│   ├── audio/                    # Audio 推理封装
│   ├── llm/                      # LLM 推理封装
│   └── CMakeLists.txt
├── data/
│   ├── vision/demo.jpg           # 示例图片
│   └── audio/
│       ├── demo.wav              # 示例音频
│       └── mel_128_filters.txt   # Mel 滤波器组系数
├── python/
│   ├── vision/                   # Vision ONNX/RKNN 导出 + 校准数据生成
│   │   ├── export_vision.py
│   │   ├── export_rknn.py
│   │   └── make_calidata.py
│   ├── audio/                    # Audio ONNX/RKNN 导出
│   │   ├── export_audio.py
│   │   └── export_rknn.py
│   └── llm/                      # LLM ONNX/RKNN 导出 + 校准数据生成
│       ├── export_llm.py
│       ├── export_rknn.py
│       └── make_calidata.py
└── README.md
```

## 3. ONNX 模型导出

### 3.1 导出 Audio 模型

Audio 模型不支持量化，只有一条导出路径。

```bash
cd python/audio

# 导出 ONNX 模型
python export_audio.py --modelscope
```

#### 参数说明

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--model_path` | str | 模型路径或名称 | `Qwen/Qwen2.5-Omni-3B` |
| `--export_audio_path` | str | 输出 ONNX 路径 | `../../model/audio/Qwen2.5-Omni-3B-audio.onnx` |
| `--modelscope` | bool | 从 ModelScope 下载（推荐国内用户） | `False` |

### 3.2 生成量化校准数据

仅当 Vision/LLM 启用 `--quant`（GRQ 量化）导出时需要执行本步骤；导出 float（非量化）模型时可跳过。

校准数据生成脚本在 CPU 上运行（不需要 CUDA），它会加载浮点模型对一批图文数据前向，通过 hook 捕获被量化子模块的真实输入，pickle 落盘后生成索引文件，供 `export_*.py` 的 `--cali_dataset` 使用。

#### LLM 校准数据

`python/llm/make_calidata.py` 捕获进入 `model.thinker.model` 的输入，模型加载类型为 `bfloat16`（与 `export_llm.py` 一致）。

```bash
cd python/llm
python make_calidata.py
```

执行后会在当前目录下生成：

```
quant_data/
├── model_inputs.json          # 校准数据索引（供 export_llm.py --cali_dataset 使用）
└── model_inputs/
    ├── sample_0               # 每条样本捕获的输入（pickle）
    ├── sample_1
    └── ...
```

#### Vision 校准数据

`python/vision/make_calidata.py` 捕获进入 `model.thinker.visual` 的输入，模型加载类型为 `float32`（与 `export_vision.py` 一致）。

```bash
cd python/vision
python make_calidata.py
```

生成的目录结构与 LLM 一致，索引路径与 `export_vision.py` 的默认 `--cali_dataset`（`./quant_data/model_inputs.json`）对齐，无需额外传参。

#### 参数说明

两个 `make_calidata.py` 参数一致：

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--model_path` | str | 模型路径或名称 | `Qwen/Qwen2.5-Omni-3B` |
| `--datapath` | str | 原始图文标注数据路径 | `../../../../datasets/MMBench/llm/dataset.json` |
| `--export_datapath` | str | 输出校准数据索引路径 | `./quant_data/model_inputs.json` |

> ⚠️ **注意**：
> - `make_calidata.py` 中的模型加载 `torch_dtype` 必须与对应 `export_*.py` 一致，否则 GRQ 量化会报错。
> - 校准数据生成在 CPU 上运行；但后续 GRQ 量化（`--quant`）需要 CUDA。

### 3.3 导出 Vision 模型

支持 float 与 GRQ 量化两种导出方式。

**方式一：导出 float 模型（默认，无需 CUDA）**

```bash
cd python/vision
python export_vision.py --modelscope
```

**方式二：导出 GRQ 量化模型（需 CUDA，需先执行 3.2 生成校准数据）**

```bash
cd python/vision
python export_vision.py --modelscope --quant
```

#### 参数说明

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--model_path` | str | 模型路径或名称 | `Qwen/Qwen2.5-Omni-3B` |
| `--export_vision_path` | str | 输出 ONNX 路径 | `../../model/vision/Qwen2.5-Omni-3B-vision.onnx` |
| `--modelscope` | bool | 从 ModelScope 下载 | `False` |
| `--img_size` | int | 输入图像尺寸（须为 28 的倍数） | `392` |
| `--quant` | bool | 启用 GRQ 量化（需 CUDA） | `False` |
| `--cali_dataset` | str | GRQ 量化校准数据路径 | `./quant_data/model_inputs.json` |

### 3.4 导出 LLM 模型

支持 float 与 GRQ 量化两种导出方式。

**方式一：导出 float 模型（默认，无需 CUDA）**

```bash
cd python/llm
python export_llm.py --modelscope
```

**方式二：导出 GRQ 量化模型（需 CUDA，需先执行 3.2 生成校准数据）**

```bash
cd python/llm
python export_llm.py --modelscope --quant
```

#### 参数说明

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--model_path` | str | 模型路径或名称 | `Qwen/Qwen2.5-Omni-3B` |
| `--export_llm_path` | str | 输出 ONNX 路径 | `../../model/llm/Qwen2.5-Omni-3B-llm.onnx` |
| `--modelscope` | bool | 从 ModelScope 下载 | `False` |
| `--quant` | bool | 启用 GRQ 量化（需 CUDA） | `False` |
| `--cali_dataset` | str | GRQ 量化校准数据路径 | `./quant_data/model_inputs.json` |

> **注意**：LLM 导出除 ONNX 外，还会生成配套的 `.config.pkl`、`.tokenizer.gguf`、`.embed.bin` 等文件，均位于 `--export_llm_path` 同目录下。

## 4. RKNN 模型导出

ONNX 导出完成后，分别将三个子模型转换为 RKNN 格式。

### 4.1 导出 Vision RKNN 模型

```bash
cd python/vision
python export_rknn.py
```

#### 参数说明

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--onnx_path` | str | 输入 ONNX 路径 | `../../model/vision/Qwen2.5-Omni-3B-vision.onnx` |
| `--rknn_path` | str | 输出 RKNN 路径 | `../../model/vision/Qwen2.5-Omni-3B-vision.rknn` |
| `--platform` | str | 目标平台 | `rk1820` |
| `--dataset_path` | str | RKNN 量化数据集路径 | `None` |
| `--core_num` | int | NPU 核心数（1-8） | `8` |

> Vision 的 RKNN 转换使用 `w4a16 / normal / group32` 量化方案，采用权重分离模式生成 `.rknn` + `.weight`。

### 4.2 导出 Audio RKNN 模型

```bash
cd python/audio
python export_rknn.py
```

#### 参数说明

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--onnx_path` | str | 输入 ONNX 路径 | `../../model/audio/Qwen2.5-Omni-3B-audio.onnx` |
| `--platform` | str | 目标平台 | `rk1820` |
| `--rknn_path` | str | 输出 RKNN 路径 | `../../model/audio/Qwen2.5-Omni-3B-audio.rknn` |
| `--core_num` | int | NPU 核心数（1-8） | `8` |

> Audio 模型不做量化（`do_quantization=False`），支持动态输入形状，采用权重分离模式生成 `.rknn` + `.weight`。

### 4.3 导出 LLM RKNN 模型

```bash
cd python/llm
python export_rknn.py
```

#### 参数说明

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--onnx_path` | str | 输入 ONNX 路径 | `../../model/llm/Qwen2.5-Omni-3B-llm.onnx` |
| `--config` | str | 模型 Config 路径（LLM 必需） | `../../model/llm/Qwen2.5-Omni-3B-llm.config.pkl` |
| `--rknn_path` | str | 输出 RKNN 路径 | `../../model/llm/Qwen2.5-Omni-3B-llm.rknn` |
| `--platform` | str | 目标平台 | `rk1820` |
| `--dataset_path` | str | RKNN 量化数据集路径 | `None` |

> LLM 的 RKNN 转换使用 `w4a16 / grq / group32` 量化方案，采用权重分离模式生成 `.rknn` + `.weight`。

> 导出完成后，模型文件将生成在 `model` 目录下。

## 5. C++ 板端部署

### 5.1 编译

```bash
cd rknn3_model_zoo/

# 设置交叉编译工具链
export GCC_COMPILER=/path/to/aarch64-linux-gnu

# 编译
./build-linux.sh -t rk3588 -a aarch64 -d Qwen2_5_Omni
```

`-b` 参数默认为 `Release`，无需显式指定；如需 Debug 构建可加 `-b Debug`。

编译完成后，文件生成在 `install/rk3588_linux_aarch64/rknn_Qwen2_5_Omni_demo/` 目录：

```
rknn_Qwen2_5_Omni_demo/
├── lib/
│   ├── librga.so
│   └── librknn3_api.so
├── model/
│   ├── Qwen2.5-Omni-3B-audio.rknn
│   ├── Qwen2.5-Omni-3B-audio.weight
│   ├── Qwen2.5-Omni-3B-llm.rknn
│   ├── Qwen2.5-Omni-3B-llm.weight
│   ├── Qwen2.5-Omni-3B-llm.embed.bin
│   ├── Qwen2.5-Omni-3B-llm.tokenizer.gguf
│   ├── Qwen2.5-Omni-3B-vision.rknn
│   └── Qwen2.5-Omni-3B-vision.weight
├── mel_128_filters.txt
├── demo.jpg
├── demo.wav
└── rknn_qwen2_5_omni_demo
```

### 5.2 部署到开发板

```bash
# 推送 demo 目录
adb push install/rk3588_linux_aarch64/rknn_Qwen2_5_Omni_demo /data/

# 推送运行库
adb push install/rk3588_linux_aarch64/rknn_Qwen2_5_Omni_demo/lib/* /usr/lib/
```

### 5.3 运行示例

```bash
adb shell
cd /data/rknn_Qwen2_5_Omni_demo
```

#### 参数说明

| 参数 | 说明 |
|------|------|
| `vision_model_path` | Vision RKNN 模型路径 |
| `vision_weight_path` | Vision 权重文件路径 |
| `audio_model_path` | Audio RKNN 模型路径 |
| `audio_weight_path` | Audio 权重文件路径 |
| `llm_model_path` | LLM RKNN 模型路径 |
| `llm_weight_path` | LLM 权重文件路径 |
| `tokenizer_path` | Tokenizer 文件路径（`.tokenizer.gguf`） |
| `embedding_path` | Embedding 文件路径（`.embed.bin`） |
| `vision_core_mask` | Vision NPU 核心掩码（16 进制，`0xff` 表示 8 核） |
| `audio_core_mask` | Audio NPU 核心掩码（16 进制） |
| `llm_core_mask` | LLM NPU 核心掩码（16 进制） |
| `image_path` | 输入图片路径 |
| `audio_path` | 输入音频路径 |
| `prompt` | 提示词（用 `<image>`/`<audio>` 标记启用对应模态） |

#### 示例 1：Vision + LLM（图像理解）

```bash
./rknn_qwen2_5_omni_demo \
    model/Qwen2.5-Omni-3B-vision.rknn model/Qwen2.5-Omni-3B-vision.weight \
    model/Qwen2.5-Omni-3B-audio.rknn model/Qwen2.5-Omni-3B-audio.weight \
    model/Qwen2.5-Omni-3B-llm.rknn model/Qwen2.5-Omni-3B-llm.weight \
    model/Qwen2.5-Omni-3B-llm.tokenizer.gguf model/Qwen2.5-Omni-3B-llm.embed.bin \
    0xff 0xff 0xff \
    demo.jpg demo.wav \
    "<image>描述下这张图："
```

输出示例：
```
这张图展示了一位宇航员在月球表面的场景。背景中可以看到地球，天空中有星星和云彩。宇航员穿着白色太空服，手里拿着一瓶绿色啤酒瓶，并且旁边有一个装有其他物品的小箱子。整个画面充满了科幻感和幽默感。
```

#### 示例 2：Audio + LLM（语音转文本）

```bash
./rknn_qwen2_5_omni_demo \
    model/Qwen2.5-Omni-3B-vision.rknn model/Qwen2.5-Omni-3B-vision.weight \
    model/Qwen2.5-Omni-3B-audio.rknn model/Qwen2.5-Omni-3B-audio.weight \
    model/Qwen2.5-Omni-3B-llm.rknn model/Qwen2.5-Omni-3B-llm.weight \
    model/Qwen2.5-Omni-3B-llm.tokenizer.gguf model/Qwen2.5-Omni-3B-llm.embed.bin \
    0xff 0xff 0xff \
    demo.jpg demo.wav \
    "<audio>将这段语音转为文本."
```

输出示例：
```
图片里是什么？
```

#### 示例 3：Vision + Audio + LLM（多模态理解）

```bash
./rknn_qwen2_5_omni_demo \
    model/Qwen2.5-Omni-3B-vision.rknn model/Qwen2.5-Omni-3B-vision.weight \
    model/Qwen2.5-Omni-3B-audio.rknn model/Qwen2.5-Omni-3B-audio.weight \
    model/Qwen2.5-Omni-3B-llm.rknn model/Qwen2.5-Omni-3B-llm.weight \
    model/Qwen2.5-Omni-3B-llm.tokenizer.gguf model/Qwen2.5-Omni-3B-llm.embed.bin \
    0xff 0xff 0xff \
    demo.jpg demo.wav \
    "<image><audio>"
```

输出示例：
```
图片里是一个宇航员在月球上拿着一瓶啤酒的场景。背景是地球和星空，看起来像是一个幽默或创意的艺术作品。
```

## 6. 支持的模型

目前支持 Qwen2.5-Omni-3B 模型。导出时请通过 `--model_path` 指定对应的模型路径或 HuggingFace 名称。

## 7. 常见问题

### ONNX 文件路径问题

使用 PyTorch ≥ 2.9.0 导出的模型会生成 `xxx.onnx` 和 `xxx.onnx.data` 两个文件。执行 `rknn.load_llm` 时，必须确保这两个文件在同一目录下。

### GRQ 量化需 CUDA

`--quant` 启用 GRQ 量化时必须有 NVIDIA GPU。CPU 环境下量化分支会被静默跳过（不报错），导出的实际是未量化的 float 模型。如需量化导出，请确认在带 GPU 的机器上执行。

### 校准数据 dtype 一致性

`make_calidata.py` 中模型加载的 `torch_dtype` 必须与对应 `export_*.py` 中的加载类型一致（LLM 均为 `bfloat16`，Vision 均为 `float32`），否则捕获的输入数值精度与量化阶段不匹配，GRQ 量化会报错。

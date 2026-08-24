# Qwen3-TTS 部署说明

- [Qwen3-TTS 部署说明](#qwen3-tts-部署说明)
  - [环境说明](#环境说明)
  - [目录说明](#目录说明)
  - [模型转换](#模型转换)
    - [1. 导出基础 Embeds](#1-导出基础-embeds)
    - [2. 导出 text_projector](#2-导出-text_projector)
    - [3. 导出 code_predictor](#3-导出-code_predictor)
    - [4. 导出 talker](#4-导出-talker)
    - [5. 导出 speech_decoder](#5-导出-speech_decoder)
  - [量化数据集准备](#量化数据集准备)
  - [板端部署](#板端部署)
    - [Linux 平台编译](#linux-平台编译)
    - [推送到板端](#推送到板端)
    - [运行](#运行)
  - [补充说明](#补充说明)


## 环境说明

建议优先使用 **Qwen3-TTS 官方仓库**推荐的 Python 环境完成 ONNX 导出。

如果 `rknn3-toolkit` 无法安装在同一环境中，可以另外单独创建一个 Python 环境用于 RKNN 转换。


## 目录说明

当前 `examples/Qwen3_TTS` 主要包含以下内容：

- `python/`
  - `embeds/`：导出 `talker_text_embed.fp16.bin`、`talker_input_embed.fp16.bin`、`codec_embed.fp16.bin`、`tokenizer.json`
  - `text_projector/`：导出 `text_projection.onnx / .rknn`
  - `code_predictor/`：导出 `code_predictor.onnx / .rknn`
  - `talker/`：导出 `talker.onnx / .rknn`，并支持准备 talker 量化数据集
  - `speech_decoder/`：导出 `speech_decoder.onnx / .rknn`
  - `Qwen3_TTS/`：本地 vendored 的最小 Python 包，避免依赖外部仓库路径
- `models/`
  - `embeds/`
  - `text_projector/`
  - `code_predictor/`
  - `talker/`
  - `speech_decoder/`
- `cpp/`
  - `main.cc`
  - `talker.cc/.h`
  - `speech_decoder.cc/.h`


## 模块关系

下面给出当前 `Qwen3_TTS` example 中几个核心模块的大致关系：

```text
                 +----------------------+
                 |      embeds/         |
                 |----------------------|
文本 token ----->| talker_text_embed    |
控制 token ----->| talker_input_embed   |
codec token ---->| codec_embed          |
                 | tokenizer.json       |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 |   text_projector/    |
                 |----------------------|
文本 embedding --> 投影到 talker hidden   |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 |      talker/         |
                 |----------------------|
                 | 负责主 talker 自回归   |
                 | 输出第 1 路 codec 和   |
                 | 当前 step 的 hidden   |
                 +----------+-----------+
                            |
                  +---------+---------+
                  |                   |
                  v                   v
      +--------------------+   +----------------------+
      |  code_predictor/   |   |  speech_decoder/     |
      |--------------------|   |----------------------|
      | 预测剩余 15 路codec |   | 16 路 codec -> wav   |
      +--------------------+   +----------------------+
                  |
                  | 预测得到的其余 codec embedding
                  | 会回送给 talker 参与下一步生成
                  v
               talker
```

更具体的数据流可以理解为：

1. `embeds/`
   - 保存运行时需要的基础 embedding 和 tokenizer 文件。
   - `talker_text_embed.fp16.bin`、`talker_input_embed.fp16.bin`、`codec_embed.fp16.bin` 会在 C++ 侧被直接加载使用。

2. `text_projector/`
   - 负责把文本 token 对应的 text embedding 投影到 talker 所需的 hidden space。
   - 可以理解为 talker 的文本侧前处理模块。

3. `talker/`
   - 是主自回归模块。
   - 输入是已经拼好的 `inputs_embeds / attention_mask / position_ids`。
   - 每一步会输出：
     - 当前主 codec token 的 logits
     - `past_hidden`
   - 其中 `past_hidden` 会继续参与下一步推理，也会作为 `code_predictor` 的输入之一。

4. `code_predictor/`
   - 用于补齐除第一路之外的其余 codec token。
   - 当前实现里，通常是基于：
     - `past_hidden`
     - 当前 step 的主 codec embedding
     来预测剩余的 codec 分组。
   - 这些预测得到的 codec token / codec embedding 不只是最终输出的一部分，
     还会继续回送给 `talker`，参与下一步自回归生成。

5. `speech_decoder/`
   - 接收最终拼好的 16 路 codec 序列。
   - 按窗口方式把 codec 解码成最终 wav。

从整体流程上看，可以简化为：

```text
文本
  -> tokenizer
  -> embeds + text_projector
  -> talker
  -> code_predictor
  -> 拼成 16 路 codec
  -> speech_decoder
  -> 音频
```


## 模型转换

### 1. 导出基础 Embeds

```bash
cd examples/Qwen3_TTS/python/embeds

python export_embeds.py \
    --model_path /path/to/Qwen3-TTS-12Hz-1.7B-Base \
    --export_dir ../../models/embeds
```

如需更新板端默认 speaker 头文件，可额外执行：

```bash
python export_speaker_embed_hpp.py \
    --ref_audio /path/to/ref.wav \
    --speaker_name custom_speaker
```


### 2. 导出 text_projector

导出 ONNX：

```bash
cd examples/Qwen3_TTS/python/text_projector

python export_text_projector_onnx.py
```

导出 RKNN：

```bash
python export_text_projector_rknn.py
```


### 3. 导出 code_predictor

导出 ONNX：

```bash
cd examples/Qwen3_TTS/python/code_predictor

python export_code_predictor_onnx.py
```

导出 RKNN：

```bash
python export_code_predictor_rknn.py
```


### 4. 导出 talker

导出 ONNX （quantized_algorithm使用normal，并开启混合量化（比例为0.2））：

```bash
cd examples/Qwen3_TTS/python/talker

python export_talker_onnx.py --quant
```

导出 RKNN：

```bash
python export_talker_rknn.py
```


### 5. 导出 speech_decoder

导出 ONNX：

```bash
cd examples/Qwen3_TTS/python/speech_decoder

python export_speech_decoder_onnx.py
```

导出 RKNN：

```bash
python export_speech_decoder_rknn.py
```


## 板端部署

### Linux 平台编译

```bash
cd rknn3-model-zoo

./build-linux.sh -t rk3588 -a aarch64 -d Qwen3_TTS
```

编译完成后会生成：

- 可执行文件：`rknn_qwen3_tts_demo`
- 安装目录：`install/<target>_linux_<arch>/rknn_Qwen3_TTS_demo`

同时会把 `models/` 下的以下文件安装到板端目录的 `model/` 中：

- `code_predictor/*.rknn`
- `code_predictor/*.weight`
- `speech_decoder/*.rknn`
- `speech_decoder/*.weight`
- `talker/*.rknn`
- `talker/*.weight`
- `text_projector/*.rknn`
- `text_projector/*.weight`
- `embeds/*.bin`
- `embeds/tokenizer.json`


### 推送到板端

```bash
adb push install/rk3588_linux_aarch64/rknn_Qwen3_TTS_demo /data/
```


### 运行

```bash
adb shell
cd /data/rknn_Qwen3_TTS_demo

export LD_LIBRARY_PATH=./lib
./rknn_qwen3_tts_demo <model_dir> <ref_speaker> <output_dir> <text...>
```

**参数说明：**

```
Usage: rknn_qwen3_tts_demo <model_dir> <ref_speaker> <output_dir> <text...>
```

| 参数           | 说明                                                                 |
| -------------- | -------------------------------------------------------------------- |
| `model_dir`    | 模型目录，例如 `./model`，需包含 `code_predictor`、`speech_decoder`、`talker`、`text_projector`、`embeds` 等子目录 |
| `ref_speaker`  | 通过 export_speaker_embed_hpp.py 转换得到的 speaker_embed 的名称，即所传入的自定义的 speaker_name                          |
| `output_dir`   | 输出音频目录，输出文件为该目录下的 `output.wav`                      |
| `text...`      | 待合成的文本内容（可包含多段，从第 4 个参数起拼接）                  |

示例：

```bash
./rknn_qwen3_tts_demo ./model girl_base ./output "你好，这是一个测试。"
```


## 补充说明

1. `talker` ONNX 导出时，如果遇到：

```text
RuntimeError: unordered_map::at
```

请先按脚本提示修改 `transformers/masking_utils.py` 中的 `sdpa_mask` 选择逻辑。

2. 当前 `talker` 导出链中：

- 默认推理路径下，`past_hidden` 返回最后一帧 `hidden_states[:, -1:, :]`
- ONNX 导出时，会在导出 wrapper 中显式切到“返回完整 hidden_states”的模式

3. 当前 C++ demo 是简化实现，不包含完整 pipeline 框架，但：

- `talker`
- `speech_decoder`
- `text_projector`
- `code_predictor`

这几条主链已经可以独立配合运行。

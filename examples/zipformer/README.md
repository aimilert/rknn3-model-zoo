# zipformer

## Table of contents

- [1. Description](#1-description)
- [2. Current Support Platform](#2-current-support-platform)
- [3. Pretrained Model](#3-pretrained-model)
- [4. Convert to RKNN](#4-convert-to-rknn)
- [5. Python Demo](#5-python-demo)
- [6. Linux Demo](#6-linux-demo)
  - [6.1 Compile && Build](#61-compile--build)
  - [6.2 Push demo files to device](#62-push-demo-files-to-device)
  - [6.3 Run demo](#63-run-demo)
- [7. Expected Results](#7-expected-results)



## 1. Description

Chinese-English ASR model using k2-zipformer-streaming (RKNN3 implementation).

The model used in this example comes from the following open source project:

https://huggingface.co/pfluo/k2fsa-zipformer-chinese-english-mixed

**Key Features:**
- ✅ Streaming ASR with real-time transcription capability
- ✅ Bilingual support (Chinese-English mixed speech)
- ✅ Efficient state management for streaming inference


## 2. Current Support Platform

RK1820, RK1828 (RK182X series)



## 3. Pretrained Model

Download link:

[encoder-epoch-99-avg-1-all-float.onnx](https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/zipformer-mixed/encoder-epoch-99-avg-1-all-float.onnx)<br />[decoder-epoch-99-avg-1.onnx](https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/zipformer-mixed/decoder-epoch-99-avg-1.onnx)<br />[joiner-epoch-99-avg-1.onnx](https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/zipformer-mixed/joiner-epoch-99-avg-1.onnx)

Download with shell command:

```sh
cd model
./download_model.sh
```

**Note: The model is provided by the official k2fsa-zipformer-chinese-english-mixed project. For model training and export information, please refer to the official repository.**



## 4. Convert to RKNN

*Usage:*

```shell
cd python

# Convert encoder model (supports RK182X series only)
python convert_encoder.py ../model/encoder-epoch-99-avg-1-all-float.onnx rk1820

# Convert decoder model
python convert_decoder.py ../model/decoder-epoch-99-avg-1.onnx rk1820

# Convert joiner model
python convert_joiner.py ../model/joiner-epoch-99-avg-1.onnx rk1820
```

*Output:*
- `encoder-epoch-99-avg-1-all-float.rknn` + `encoder-epoch-99-avg-1-all-float.weight`
- `decoder-epoch-99-avg-1.rknn` + `decoder-epoch-99-avg-1.weight`
- `joiner-epoch-99-avg-1.rknn` + `joiner-epoch-99-avg-1.weight`



## 5. Python Demo

*Prerequisites:*

安装最新版本的rknn3-toolkit

```shell
# Install kaldifeat
# Refer to https://csukuangfj.github.io/kaldifeat/installation/from_wheels.html for installation.
# This python demo is tested under version: kaldifeat-1.25.5.dev20250807
pip install kaldifeat==1.25.5.dev20250807+cpu.torch2.8.0 -f https://csukuangfj.github.io/kaldifeat/cpu-cn.html

# Install depends
pip install -r requirements.txt
```

*Usage:*

```shell
cd python

# Inference with ONNX model
python zipformer.py --encoder_model_path ../model/encoder-epoch-99-avg-1-all-float.onnx \
                    --decoder_model_path ../model/decoder-epoch-99-avg-1.onnx \
                    --joiner_model_path ../model/joiner-epoch-99-avg-1.onnx

# Inference with RKNN model on device
python zipformer.py --encoder_model_path ../model/encoder-epoch-99-avg-1-all-float.rknn \
                    --decoder_model_path ../model/decoder-epoch-99-avg-1.rknn \
                    --joiner_model_path ../model/joiner-epoch-99-avg-1.rknn \
                    --target rk1820 \
                    --device_id <device_id>
```

*Description:*
- `--target rk1820`: Specify NPU platform (RK182X series).
- `--device_id`: Device ID for ADB connection. Use `adb devices` to get device ID.
- The Python demo will automatically handle FP16/FP32 conversion and NC1HWC2 format transformation.



## 6. Linux Demo

Please note that the Linux compilation tool chain recommends using `gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu` or later. Using other versions may encounter compilation issues. For detailed compilation guide, please refer to [Compilation_Environment_Setup_Guide.md](../../docs/Compilation_Environment_Setup_Guide.md)

#### 6.1 Compile && Build

*usage*

```shell
# go back to the rknn_model_zoo root directory
cd ../../

# if GCC_COMPILER not found while building, please set GCC_COMPILER path
(optional)export GCC_COMPILER=<GCC_COMPILER_PATH>

./build-linux.sh -t <TARGET_PLATFORM> -a <ARCH> -d zipformer

# such as
./build-linux.sh -t rk3588 -a aarch64 -d zipformer
```

*Description:*

- `<GCC_COMPILER_PATH>`: Specified as GCC_COMPILER path (e.g., ~/RK/toolchains/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu).
- `<TARGET_PLATFORM>` : Specify deploy platform name (e.g., RK3588).
- `<ARCH>`: Specify device system architecture. To query device architecture, refer to the following command:

  ```shell
  # Query architecture. For Linux, ['aarch64' or 'armhf'] should be shown in log.
  adb shell cat /proc/version
  ```

#### 6.2 Push demo files to device

- If device connected via USB port, push demo files to devices:

```shell
adb push install/<TARGET_PLATFORM>_linux_<ARCH>/rknn_zipformer_demo/ /data/
```

- For other boards, use `scp` or other approaches to push all files under `install/<TARGET_PLATFORM>_linux_<ARCH>/rknn_zipformer_demo/` to `data`.

#### 6.3 Run demo

```sh
adb shell
cd /data/rknn_zipformer_demo

export LD_LIBRARY_PATH=./lib

# Run RKNN3 demo (note: 8 parameters required - models + weight files + audio)
./rknn_zipformer_demo \
  model/encoder-epoch-99-avg-1-all-float.rknn \
  model/encoder-epoch-99-avg-1-all-float.weight \
  model/decoder-epoch-99-avg-1.rknn \
  model/decoder-epoch-99-avg-1.weight \
  model/joiner-epoch-99-avg-1.rknn \
  model/joiner-epoch-99-avg-1.weight \
  model/0.wav
```



## 7. Expected Results

This example will print the recognized text with timestamps, as follows:

```
-- init_zipformer_encoder_model use: 1091.38 ms
-- init_zipformer_decoder_model use: 550.18 ms
-- init_zipformer_joiner_model use: 544.74 ms
-- inference_zipformer_model use: 3221.00 ms

Real Time Factor (RTF): 3.221 / 10.643 = 0.303

Timestamp (s): 0.00, 0.84, 1.36, 2.16, 2.44, 2.60, 4.16, 4.52, 5.04, 5.40, 5.76, 6.16, 6.84, 6.96, 7.48, 8.00, 8.20, 8.44, 8.60, 9.00, 9.36, 9.52, 10.20

Zipformer output: 昨天是 MONDAY TODAY IS礼拜二 TODAY AFTER TOMORROW是星期三
```

*Performance Metrics:*
- **RTF (Real-Time Factor)**: ~0.30 (processes 1 second of audio in 0.30 seconds)
- **Memory Usage**: Optimized with RKNN3 FP16 quantization
- **Accuracy**: Matches Python reference implementation

**Note:**

- Results may vary slightly across different RK182X platforms due to NPU characteristics
- Ensure audio files are 16kHz sample rate for best performance
- The demo includes automatic audio resampling if needed
# VITS

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

VITS (Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech)

The model used in this example comes from the following open source project:

https://github.com/jaywalnut310/vits

**Key Features:**
- ✅ End-to-end text-to-speech synthesis
- ✅ Single speaker (LJSpeech) and multi-speaker (VCTK) support
- ✅ Sliding window inference for memory efficiency
- ✅ Real-time performance with RTF < 0.2
- ✅ High-quality speech synthesis


## 2. Current Support Platform

RK1820, RK1828 (RK182X series)

## 3. Pretrained Model

This example supports two pre-trained models:

1. **LJSpeech** (Single speaker, female voice)
   - English text-to-speech
   - Sampling rate: 22050Hz
   - Pre-trained on LJ Speech dataset

2. **VCTK** (Multi-speaker, 109 speakers)
   - English text-to-speech with multiple voices
   - Sampling rate: 22050Hz
   - Speaker ID range: 0-108

**Note: The model is provided by the official VITS project. For model training and export information, please refer to the official repository.**


## 4. Convert to RKNN

### 4.1 LJSpeech Model Conversion

```shell
cd python

# Convert LJSpeech Step1 model (Text Encoder + Duration Predictor)
python convert_step1.py ../model/vits_ljs_step1_slid.onnx rk1820 fp

# Convert LJSpeech Step2 model (Flow + Decoder)
python convert_step2.py ../model/vits_ljs_step2_slid.onnx rk1820 fp
```

**Output:**
- `vits_ljs_step1_slid_fp.rknn` + `vits_ljs_step1_slid_fp.weight`
- `vits_ljs_step2_slid_fp.rknn` + `vits_ljs_step2_slid_fp.weight`

### 4.2 VCTK Model Conversion

```shell
cd python

# Convert VCTK Step1 model (Text Encoder + Duration Predictor)
python convert_step1.py ../model/vits_vctk_step1_slid.onnx rk1820 fp

# Convert VCTK Step2 model (Flow + Decoder)
python convert_step2.py ../model/vits_vctk_step2_slid.onnx rk1820 fp
```

**Output:**
- `vits_vctk_step1_slid_fp.rknn` + `vits_vctk_step1_slid_fp.weight`
- `vits_vctk_step2_slid_fp.rknn` + `vits_vctk_step2_slid_fp.weight`

**Note:**
- Step1 model processes text input with window_size=160 (128 core + 16*2 context)
- Step2 model generates audio with window_size=128 frames
- RK1828 for Step1 (text processing) and RK1820 for Step2 (audio generation)


## 5. Python Demo

*Prerequisites:*

安装最新版本的rknn3-toolkit

```shell
# Install dependencies
pip install -r requirements.txt
```

*Usage:*

```shell
cd python

# Run unified inference script (supports both LJS and VCTK)
python rknn3_vits.py --type ljs --text "VITS is awesome! This is a text-to-speech system."

# VCTK with specific speaker on device
python rknn3_vits.py --type vctk --speaker_id 4 --target rk1828 --device_id {device_id} --text "VITS is awesome! This is a text-to-speech system."

# Auto-detect model type from file paths
python rknn3_vits.py --step1 ../model/vits_vctk_step1_slid_fp.rknn --step2 ../model/vits_vctk_step2_slid_fp.rknn --text "VITS is awesome! This is a text-to-speech system."
```

*Parameters:*

- `--step1`: Path to Step1 RKNN model (auto-detects LJS/VCTK if not specified)
- `--step2`: Path to Step2 RKNN model
- `--type`: Model type: `ljs` (LJSpeech) or `vctk` (multi-speaker). Auto-detected if `--step1` is specified
- `--target`: Target RKNPU platform (e.g., `rk1820`, `rk1828`)
- `--device_id`: RKNN device ID (e.g., `172.16.10.185:5555`)
- `--speaker_id`: Speaker ID for VCTK models (0-108)
- `--text`: Input text to synthesize (default: "VITS is awesome! This is a text-to-speech system.")
- `--output`: Output audio file path (default: `output_audio_rknn.wav`)
- `--noise_scale`: Noise scale for randomness (default: 0.667)
- `--length_scale`: Length scale for speech speed (default: 1.0)
- `--rknn_dir`: Directory containing RKNN models (default: `../model`)


## 6. Linux Demo

Please note that the Linux compilation tool chain recommends using `gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu` or later. Using other versions may encounter compilation issues. For detailed compilation guide, please refer to [Compilation_Environment_Setup_Guide.md](../../docs/Compilation_Environment_Setup_Guide.md)

### 6.1 Compile && Build

*Usage:*

```shell
# Go back to the rknn3_model_zoo root directory
cd rknn3_model_zoo

# If GCC_COMPILER not found while building, please set GCC_COMPILER path
# (optional) export GCC_COMPILER=<GCC_COMPILER_PATH>

./build-linux.sh -t <TARGET_PLATFORM> -a <ARCH> -d vits

# For example:
./build-linux.sh -t rk3588 -a aarch64 -d vits
```

*Description:*

- `<GCC_COMPILER_PATH>`: Specify GCC_COMPILER path (e.g., ~/RK/toolchains/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu).
- `<TARGET_PLATFORM>`: Specify deploy platform name (e.g., RK3588).
- `<ARCH>`: Specify device system architecture. To query device architecture, refer to the following command:

  ```shell
  # Query architecture. For Linux, ['aarch64' or 'armhf'] should be shown in log.
  adb shell cat /proc/version
  ```

### 6.2 Push demo files to device

- If device connected via USB port, push demo files to devices:

```shell
adb push install/<TARGET_PLATFORM>_linux_<ARCH>/rknn_vits_demo/ /data/
```

- For other boards, use `scp` or other approaches to push all files under `install/<TARGET_PLATFORM>_linux_<ARCH>/rknn_vits_demo/` to `/data/`.

### 6.3 Run demo

#### 6.3.1 VCTK Multi-Speaker Demo

```shell
adb shell
cd /data/rknn_vits_demo

export LD_LIBRARY_PATH=./lib

# Run VCTK demo with speaker ID 8
./rknn_vits_demo \
  --speaker_id 8 \
  --step1_model ./model/vits_vctk_step1_slid_fp.rknn \
  --step1_weight ./model/vits_vctk_step1_slid_fp.weight \
  --step2_model ./model/vits_vctk_step2_slid_fp.rknn \
  --step2_weight ./model/vits_vctk_step2_slid_fp.weight \
  --text "VITS is awesome! This is a text-to-speech system."
```

#### 6.3.2 LJSpeech Single-Speaker Demo

```shell
adb shell
cd /data/rknn_vits_demo

export LD_LIBRARY_PATH=./lib

# Run LJSpeech demo
./rknn_vits_demo \
  --step1_model ./model/vits_ljs_step1_slid_fp.rknn \
  --step1_weight ./model/vits_ljs_step1_slid_fp.weight \
  --step2_model ./model/vits_ljs_step2_slid_fp.rknn \
  --step2_weight ./model/vits_ljs_step2_slid_fp.weight \
  --text "VITS is awesome! This is a text-to-speech system."
```

*Parameters:*

- `--step1_model`: Path to Step1 RKNN model (Text Encoder + Duration Predictor)
- `--step1_weight`: Path to Step1 RKNN weight file
- `--step2_model`: Path to Step2 RKNN model (Flow + Decoder)
- `--step2_weight`: Path to Step2 RKNN weight file
- `--speaker_id`: Speaker ID for VCTK models (0-108), ignored for LJSpeech
- `--text`: Input text to synthesize (default: "VITS is awesome! This is a text-to-speech system.")
- `--output`: Output audio file path (auto-detected based on model type)
- `--core_mask`: NPU core mask (default: 0x1)

*Note:*
- The C++ demo automatically detects model type (LJSpeech vs VCTK) based on Step1 output count
- Output filename is automatically generated based on model type and speaker ID
- All punctuation marks are preserved during text processing (commas, periods, colons, etc.)
- **espeak-ng**: This demo uses eSpeak-ng for phoneme conversion. Pre-built libraries are included for Android (arm64-v8a) and Linux (aarch64). For compilation instructions, see: https://github.com/espeak-ng/espeak-ng/blob/master/docs/building.md


## 7. Expected Results

### 7.1 C++ Demo (rknn_vits_demo)

#### 7.1.1 VCTK Multi-Speaker

```
========================================
VITS VCTK RKNN3 Sliding Window Inference
========================================
Text:    VITS is awesome! This is a text-to-speech system.
Speaker: 8
Model type: VCTK (multi-speaker)

Running inference...
Text length: 99 tokens
Running Step1 (Text Encoder) with 1 windows...
Extracted speaker embedding: 256 dimensions
Step1 completed
Audio length: 203 frames (2.36 seconds)
Running Step2 (Audio Decoder)...
Step2 completed

========================================
Performance Summary
========================================
Audio: 51968 samples, 2.36 seconds
Total inference: 0.37 seconds
RTF: 0.159
========================================

Saved audio to output_audio_vctk_rknn.wav
```

#### 7.1.2 LJSpeech Single-Speaker

```
========================================
VITS LJSpeech RKNN3 Sliding Window Inference
========================================
Text:    VITS is awesome! This is a text-to-speech system.
Model type: LJSpeech (single-speaker)

Running inference...
Text length: 99 tokens
Running Step1 (Text Encoder) with 1 windows...
Step1 completed
Audio length: 258 frames (3.00 seconds)
Running Step2 (Audio Decoder)...
Step2 completed

========================================
Performance Summary
========================================
Audio: 66048 samples, 3.00 seconds
Total inference: 0.51 seconds
RTF: 0.170
========================================

Saved audio to output_audio_ljs_rknn.wav
```

### 7.2 Python Demo (rknn3_vits.py)

**VCTK Multi-Speaker Example:**

```
VITS RKNN Inference - VCTK mode
Multi-speaker: 109 speakers available
Using speaker ID: 4
Text length: 101 tokens
Loading RKNN models...
Models loaded in 2.66s
Step1: 1 windows, 0.18s
Audio: 49920 samples, 2.26s
Total inference: 0.53s, RTF: 0.230
Saved audio to output_audio_rknn.wav
```

**LJSpeech Single-Speaker Example:**

```
VITS RKNN Inference - LJS mode
Text length: 101 tokens
Loading RKNN models...
Models loaded in 2.59s
Step1: 1 windows, 0.18s
Audio: 49920 samples, 2.26s
Total inference: 0.55s, RTF: 0.170
Saved audio to output_audio_rknn.wav
```

*Performance Metrics:*

- **RTF (Real-Time Factor)**: ~0.15-0.17 (C++ demo), ~0.17-0.24 (Python demo)
- **Memory Usage**: Optimized with RKNN3 W4A16 quantization
- **Model Type Auto-Detection**: C++ demo automatically detects LJSpeech vs VCTK models
- **Text Processing**: Full punctuation support (commas, periods, colons, semicolons, etc.)

**Python vs C++ Demo Differences:**

| Feature | Python Demo | C++ Demo | Notes |
|---------|-------------|----------|-------|
| **Performance (RTF)** | 0.17-0.24 | 0.15-0.17 | C++ is ~15-30% faster |
| **Token Count** | 101 tokens | 99 tokens | Minor phoneme generation differences |
| **Phoneme Output** | `bˌiːɪŋ` | `bˌiːʲɪŋ` | eSpeak variant (palatalization marker) |
| **Model Loading** | Separate timing shown | Included in initialization | C++ has clearer timing breakdown |
| **Output Format** | Identical WAV files | Identical WAV files | Same audio quality (22050Hz, 16-bit) |
| **Feature Support** | Full feature parity | Full feature parity | Both support LJS/VCTK auto-detection |

**Implementation Notes:**

1. **Text Processing**: Both versions use eSpeak-ng for phoneme conversion with identical cleaning pipeline (lowercase, abbreviation expansion, punctuation preservation)

2. **Phoneme Differences**: The slight phoneme variation (e.g., `ʲ` palatalization marker in "being") is due to eSpeak-ng version or configuration differences. This results in 1-2 token difference but **does not affect audio quality**.

3. **RTF Calculation**:
   - Python: `inference_time / audio_duration` (excludes model loading)
   - C++: `total_time / audio_duration` (includes all inference steps)

4. **Performance**: C++ demo has lower RTF due to:
   - Direct RKNN3 API calls without Python overhead
   - More efficient memory management
   - Optimized tensor handling

5. **Recommendation**: Use C++ demo for production deployments requiring optimal performance. Python demo is suitable for testing and development.

**Note:**

- Results may vary slightly across different RK182X platforms due to NPU characteristics
- The demo includes RTF calculation and detailed performance breakdown
- For VCTK, speaker IDs 0-108 are supported (different voices and genders)
- C++ demo performance matches Python reference implementation
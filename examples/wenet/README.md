# conformer

## Table of contents

- [1. Description](#1-description)
- [2. Model Architecture & Streaming Pipeline](#2-model-architecture--streaming-pipeline)
- [3. Current Support Platform](#3-current-support-platform)
- [4. Pretrained Model](#4-pretrained-model)
- [5. Export ONNX Model](#5-export-onnx-model)
- [6. Modifying Static Model Shapes](#6-modifying-static-model-shapes)
- [7. Convert to RKNN](#7-convert-to-rknn)
- [8. Python Demo](#8-python-demo)
- [9. Linux Demo](#9-linux-demo)
  - [9.1 Compile && Build](#91-compile--build)
  - [9.2 Push demo files to device](#92-push-demo-files-to-device)
  - [9.3 Run demo](#93-run-demo)
- [10. Expected Results](#10-expected-results)



## 1. Description

This example deploys the **Wenet U2++ Conformer** for Chinese ASR. WeNet, a production-oriented
end-to-end speech recognition toolkit and U2++ is an unified streaming / non-streaming end-to-end model 
trained with **bidirectional attention rescoring**:

- The **encoder** runs in streaming mode: audio is sliced into fixed-size chunks and
  encoded one chunk at a time, while a cached left-context (attention cache + convolution
  cache) is carried across chunks to preserve long-range dependencies.
- Decoding uses a **two-pass** scheme — a fast **CTC** pass (CTC prefix beam search)
  generates N-best candidates, then an **attention rescoring** pass re-scores them with
  the attention decoder (forward + reverse directions) and returns the best hypothesis.

This model used in this example comes from the following open source project:

https://github.com/wenet-e2e/wenet

**Key Features:**

- Streaming ASR with real-time transcription capability
- CTC + Attention decoding with attention rescoring
- Efficient state management for streaming inference


## 2. Model Architecture & Streaming Pipeline

The example is split into three RKNN models — **encoder**, **ctc** and **decoder**
(attention rescoring) — that together form the streaming ASR pipeline:

```text
            wav (16kHz mono)
                 |
                 v
        +-------------------+
        |  fbank (80-dim)   |   log-mel filterbank; CMVN is built into the encoder
        +---------+---------+
                  |
                  v   (streaming, chunk by chunk)
        +-------------------+      att_cache / cnn_cache
        |     encoder       | <------------------------+  (fed back to next chunk)
        +---------+---------+                         |
                  |                                   |
                  v                                   |
        +-------------------+                         |
        |       ctc         |   per-frame softmax ->  |
        +---------+---------+   prefix beam search -> |
                  |               N-best candidates   |
                  v                                   |
        +-------------------+                         |
        |  attention        |   re-score N-best with  |
        |  rescoring (dec)  |   forward + reverse     |
        +---------+---------+   attention decoder     |
                  |
                  v
               text
```

### 2.1 encoder

The Conformer encoder, exported for **streaming** inference. It processes one
`decoding_window`-frame fbank chunk at a time and carries streaming state in two caches:

- **input**
  - `chunk`      — `(1, decoding_window, 80)`, the fbank chunk, e.g. `(1, 67, 80)`
  - `att_cache`  — `(num_blocks, head, required_cache_size, output_size/head*2)`, e.g. `(12, 4, 64, 128)`
  - `cnn_cache`  — `(num_blocks, 1, output_size, cnn_module_kernel-1)`, e.g. `(12, 1, 256, 7)`
- **output**
  - `enc_out`     — `(1, chunk_size, output_size)`, e.g. `(1, 16, 256)`
  - `r_att_cache` — same shape as `att_cache`, carried into the **next** chunk
  - `r_cnn_cache` — same shape as `cnn_cache`, carried into the **next** chunk

This cache feedback (`r_att_cache` / `r_cnn_cache` → `att_cache` / `cnn_cache` of the next
chunk, both zero-initialized for the first chunk) is what lets a fixed-shape model run
streaming inference while still seeing its left context.

### 2.2 ctc

A thin CTC projection on top of the encoder output:

- **input**  `hidden` — `(1, chunk_size, output_size)`, e.g. `(1, 16, 256)`
- **output** `probs`  — `(1, chunk_size, vocab_size)`, e.g. `(1, 16, 4233)`

(`LogSoftmax` is replaced with `Softmax` during export for NPU compatibility; the runtime
converts back to log-space.) Per-chunk CTC outputs are concatenated across chunks and
decoded with **CTC prefix beam search** to produce N-best candidates.

### 2.3 attention rescoring (decoder)

The attention decoder re-scores the N-best hypotheses using the full encoder output
(forward + reverse directions) and returns the best one:

- **input**
  - `hyps`        — `(batch_size, MAX_HYP_LEN)`, the N-best token ids, e.g. `(10, 13)`
  - `encoder_out` — `(1, ENCODER_OUT_LEN, output_size)`, e.g. `(1, 200, 256)`
- **output**
  - `score`   — `(batch_size, MAX_HYP_LEN, vocab_size)`, forward scores
  - `r_score` — same shape, reverse-direction scores

The final score of each hypothesis combines the forward/reverse attention scores with the
CTC score (`ctc_weight`) and `reverse_weight`, and the best hypothesis is selected.

### 2.4 Streaming parameters

Audio is sliced into overlapping chunks before being fed to the encoder. The key streaming
parameters (defined in [cpp/process.h](cpp/process.h) and [python/conformer.py](python/conformer.py))
are:

| parameter | symbol | value | meaning |
| --- | --- | --- | --- |
| `CHUNK_SIZE` / `chunk_size` | — | 16 | encoder output frames produced per chunk |
| `LEFT_CHUNKS` / `num_decoding_left_chunks` | — | 4 | number of left-context chunks retained in `att_cache` |
| `SUBSAMPLING_RATE` | — | 4 | fbank-frame reduction factor of the encoder subsampling layer |
| `RIGHT_CONTEXT` | — | 6 | look-ahead frames baked into each chunk |
| `DECODING_WINDOW` | — | 67 | fbank frames fed to the encoder per chunk = `(chunk_size-1)*4 + right_context + 1` |
| `STRIDE` | — | 64 | fbank-frame hop between chunks = `chunk_size * 4` |
| `required_cache_size` | — | 64 | `att_cache` time dimension = `chunk_size * num_decoding_left_chunks` |

So each chunk consumes **67** fbank frames, the encoder outputs **16** frames, and the next
chunk starts **64** frames later (3 frames of look-ahead overlap). For an audio with `T`
fbank frames the number of chunks is `max(1, (T - decoding_window) / stride + 1)`, and the
total encoder output length is `num_chunks * chunk_size`.



## 3. Current Support Platform

RK1820, RK1828 (RK182X series)



## 4. Pretrained Model

Please refer to the [wenet](https://github.com/wenet-e2e/wenet) project to train or download a pretrained Conformer model (U2++).

Required files:
- `train.yaml` — model configuration
- `final.pt` — model checkpoint
- `units.txt` — vocabulary mapping

Alternatively,  example models can be downloaded by following links:

[conformer](https://meta.zbox.filez.com/v/link/view/7876dff59d9b40518ba93d3239c4cc2b)

pwd: rknn




## 5. Export ONNX Model

*Prerequisites:*

```shell
pip install onnx==1.14.0 onnxruntime==1.15.0 onnx-simplifier
# Also requires wenet source code
cd <path_to_wenet>/wenet
pip install -r requirements.txt
```

*Usage:*

- Modifying the **WENET_PATH** to your wenet path berfore exporting model 

```shell
cd python/streaming

python3 export_encoder_static.py \
    --config_path ./20210601_u2++_conformer_exp_aishell/train.yaml \
    --checkpoint_path ./20210601_u2++_conformer_exp_aishell/final.pt \
    --output_dir onnx_export/
```

*Output:*
- `onnx_export/encoder_static.onnx` — Conformer encoder (static shape)
- `onnx_export/ctc.onnx` — CTC layer
- `onnx_export/decoder.onnx` — Attention decoder

*Description:*

- Uses `torch.jit.trace` to export encoder with fixed input shapes for RKNN compatibility
- Replaces `LogSoftmax` with `Softmax` in CTC and decoder for NPU compatibility
- The exported onnx model is for streaming mode. The non-streaming mode is required to export differently using chunk_size=-1, num_decoding_left_chunks=-1 and coresponding config file for non-steaming model
- **Note: The hyps_len, encoder_time and vocab_size for encoder and decoder is required to set up to the static shape because rknn3 only support static shape** (see [6. Modifying Static Model Shapes](#6-modifying-static-model-shapes))



## 6. Modifying Static Model Shapes

RKNN3 only supports **static** input/output shapes, so the shapes are set up into the ONNX
models at export time by [python/streaming/export_conformer_static.py](python/streaming/export_conformer_static.py).


### 6.1 Encoder decoding window — `chunk_size` / `num_decoding_left_chunks`

These two parameters (defaults of `export_encoder(...)`, `chunk_size=16`,
`num_decoding_left_chunks=4`) define the encoder's per-chunk input and cache shapes:

```python
decoding_window      = (chunk_size - 1) * subsampling_rate + right_context + 1   # -> input `chunk` width
required_cache_size  = chunk_size * num_decoding_left_chunks                      # -> `att_cache` time dim
att_cache_shape      = (num_blocks, head, required_cache_size, output_size // head * 2)
```

- Increasing `chunk_size` lowers latency (fewer, larger chunks) but raises per-chunk NPU
  compute and memory.
- Increasing `num_decoding_left_chunks` gives the encoder more left context (better
  accuracy) but enlarges `att_cache`.
- `subsampling_rate` / `right_context` / `output_size` / `num_blocks` / `head` /
  `cnn_module_kernel` come from the model's `train.yaml` (`encoder_conf`) and should
  normally **not** be changed — they must match the trained model.

After changing them, update `CHUNK_SIZE` / `LEFT_CHUNKS` in [cpp/process.h](cpp/process.h)
and the matching values in [python/conformer.py](python/conformer.py) so the runtime slices
chunks and allocates caches of the right size.

### 6.2 Decoder max candidate token length — `HYPS_LEN` / `MAX_HYP_LEN`

`HYPS_LEN` (default `13`) fixes the decoder input `hyps` dimension
`(batch_size, HYPS_LEN)`. It is the **longest N-best hypothesis length** the decoder can
score. It must be large enough for the longest utterance you expect (roughly one token per
Chinese character); if it is too small, long hypotheses get truncated.

Update `MAX_HYP_LEN` in both [python/conformer.py](python/conformer.py) and
[cpp/process.h](cpp/process.h) to the same value.

### 6.3 Encoder time — `ENCODER_TIME` / `ENCODER_OUT_LEN`

`ENCODER_TIME` (default `200`) fixes the decoder input `encoder_out` dimension
`(1, ENCODER_TIME, output_size)`. This is the **fixed encoder-output length** the decoder
consumes.

At runtime the per-chunk encoder outputs are concatenated and then either zero-padded (if
shorter) or **truncated with a warning** (if longer) to `ENCODER_OUT_LEN`:

```text
[WARN] encoder output 208 > fixed length 200, truncating
```

Because the actual encoder length is `num_chunks * chunk_size` and
`num_chunks ≈ (T_fbank - decoding_window) / stride + 1`, `ENCODER_OUT_LEN` bounds the
**longest audio** the model can transcribe without losing the tail. With the defaults
(`chunk_size=16`, `decoding_window=67`, `stride=64`), `ENCODER_OUT_LEN=200` supports
roughly 12 chunks (≈ 8 s of audio); for longer utterances raise `ENCODER_TIME` and
`ENCODER_OUT_LEN` together. Update `ENCODER_OUT_LEN` in [python/conformer.py](python/conformer.py)
and [cpp/process.h](cpp/process.h) to match.

> **Note:** `MAX_HYP_LEN` (token count) and `ENCODER_OUT_LEN` (audio length) are related —
> longer audio needs both a larger `ENCODER_OUT_LEN` *and* a larger `MAX_HYP_LEN` to hold
> the resulting text.



## 7. Convert to RKNN

###  Using Individual Conversion Scripts

*Usage:*

```shell
cd streaming

# Convert encoder model
python convert_encoder.py ../../model/encoder.onnx rk1820 fp

# Convert CTC model
python convert_ctc.py ../../model/ctc.onnx rk1820 fp

# Convert decoder model
python convert_decoder.py ../../model/decoder.onnx rk1820 fp
```

- `platform`: Choose from [rk1820, rk1828]
- `dtype`: i8 (int8), fp (float16 default)
- `output_rknn_path`: Optional custom output path (default: replace .onnx with .rknn)



## 8. Python Demo

*Prerequisites:*

安装最新的rknn3-toolkit, 并推荐安装下面的pytorch版本

```shell
# Install torch (better using 2.2.2)
pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

```

*Usage:*

```shell
cd python
# Convert to the rknn model
python conformer.py \
      --onnx_dir ori_onnx_models \
      --platform rk1820 \
      --convert_only
# Inference with RKNN model on device
python conformer.py \
          --platform rk1820 \
          --onnx_dir onnx_models/ \
          --rknn_dir ./rknn_models/ \
          --wav ../model/test_ch.wav \
          --units ../model/units.txt \
          --use_runtime
```

*Description:*
- `--platform`: Specify NPU platform (RK182X series).
- `--device_id`: Device ID for ADB connection. Use `adb devices` to get device ID.
- `--use_runtime`: running inference on RK182X devices.
- The Python demo will automatically handle FP16/FP32 conversion and NC1HWC2 format transformation.



## 9. Linux Demo

Please note that the Linux compilation tool chain recommends using `gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu` or later. Using other versions may encounter compilation issues. For detailed compilation guide, please refer to [Compilation_Environment_Setup_Guide.md](../../docs/Compilation_Environment_Setup_Guide.md)

#### 9.1 Compile && Build

*usage*

```shell
# go back to the rknn_model_zoo root directory
cd ../../

# if GCC_COMPILER not found while building, please set GCC_COMPILER path
(optional)export GCC_COMPILER=<GCC_COMPILER_PATH>

./build-linux.sh -t <TARGET_PLATFORM> -a <ARCH> -d wenet

# such as
./build-linux.sh -t rk3588 -a aarch64 -d wenet
```

*Description:*

- `<GCC_COMPILER_PATH>`: Specified as GCC_COMPILER path (e.g., ~/RK/toolchains/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu).
- `<TARGET_PLATFORM>` : Specify deploy platform name (e.g., RK3588).
- `<ARCH>`: Specify device system architecture. To query device architecture, refer to the following command:

  ```shell
  # Query architecture. For Linux, ['aarch64' or 'armhf'] should shown in log.
  adb shell cat /proc/version
  ```

#### 9.2 Push demo files to device

- If device connected via USB port, push demo files to devices:

```shell
adb push install/<TARGET_PLATFORM>_linux_<ARCH>/rknn_conformer_demo/ /data/
```

- For other boards, use `scp` or other approaches to push all files under `install/<TARGET_PLATFORM>_linux_<ARCH>/rknn_conformer_demo/` to `data`.

#### 9.3 Run demo

```sh
adb shell
cd /data/rknn_conformer_demo

export LD_LIBRARY_PATH=./lib

# Run demo (7 required parameters: 3 models + 3 weight files + audio, plus 3 optional core masks)
# specify core masks for each model (default: encoder 0xff, ctc 0xff, decoder 0x1)
./rknn_conformer_demo \
  model/encoder.rknn \
  model/encoder.weight \
  model/ctc.rknn \
  model/ctc.weight \
  model/decoder.rknn \
  model/decoder.weight \
  model/test_audio.wav \
  0xff 0xff 0x1
```



## 10. Expected Results

This example will print the recognized text, as follows:

```txt
init_encoder_model use: 521.669006 ms
model input num: 1, output num: 1
input tensors:
  index=0, name=hidden, n_dims=3, shape=[1, 16, 256], n_elems=4096, aligned_size=8192, fmt=UNDEFINED, type=FP16, qnt_type=NONE, core_id=0
output tensors:
  index=0, name=probs, n_dims=3, shape=[1, 16, 4233], n_elems=67728, aligned_size=135680, fmt=UNDEFINED, type=FP16, qnt_type=NONE, core_id=0
init_ctc_model use: 235.024002 ms
model input num: 2, output num: 2
input tensors:
  index=0, name=hyps, n_dims=2, shape=[10, 13], n_elems=130, aligned_size=1024, fmt=UNDEFINED, type=INT32, qnt_type=NONE, core_id=0
  index=1, name=encoder_out.3, n_dims=3, shape=[1, 200, 256], n_elems=51200, aligned_size=102400, fmt=UNDEFINED, type=FP16, qnt_type=NONE, core_id=0
output tensors:
  index=0, name=score, n_dims=3, shape=[10, 13, 4233], n_elems=550290, aligned_size=1100800, fmt=UNDEFINED, type=FP16, qnt_type=NONE, core_id=0
  index=1, name=r_score, n_dims=3, shape=[10, 13, 4233], n_elems=550290, aligned_size=1100800, fmt=UNDEFINED, type=FP16, qnt_type=NONE, core_id=0
init_decoder_model use: 377.338013 ms

[STREAMING] 500 fbank frames => ~7 chunks
  chunk   0: fbank [0,67), enc 16 frames, partial: (silence)
  chunk   1: fbank [64,131), enc 16 frames, partial: 我认为跑
  chunk   2: fbank [128,195), enc 16 frames, partial: 步最重要的
  chunk   3: fbank [192,259), enc 16 frames, partial: 就市
  chunk   4: fbank [256,323), enc 16 frames, partial: 给
  chunk   5: fbank [320,387), enc 16 frames, partial: 我带来了
  chunk   6: fbank [384,451), enc 16 frames, partial: 身体健
  chunk   7: fbank [448,515), enc 16 frames, partial: 康
[INFO] encoder output total length: 128
[INFO] CTC outputs exported to rt_ctc_out.bin (size=541824)
  CTC nbest[0]: score=-3.50  我认为跑不最重要的就市给我带来了身体健康
  CTC nbest[1]: score=-4.80  我认为跑不最重要的救市给我带来了身体健康
  CTC nbest[2]: score=-4.87  我认为跑步最重要的就事给我带来了身体健康
inference_conformer_model use: 492.673004 ms

Real Time Factor (RTF): 0.493 / 4.997 = 0.099

Conformer result: 我认为跑步最重要的就是给我带来了身体健康
```

*Performance Metrics:*

- **RTF (Real-Time Factor)**: Processes audio in near real-time
- **Memory Usage**: Optimized with RKNN3 FP16 quantization
- **Accuracy**: CTC prefix beam search + attention rescoring

**Note:**

- Results may vary slightly across different RK182X platforms due to NPU characteristics
- Ensure audio files are 16kHz sample rate for best performance
- The demo includes automatic audio resampling and channel conversion if needed

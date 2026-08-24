# PaddleOCR-VL Model Deployment Guide

## 1. Deployment Environment

PaddleOCR-VL has specific dependency requirements that are incompatible with the versions in this repository's `requirements.txt`. Please install the following dependency manually:

```bash
transformers == 4.55.0
```

> ⚠️ Using the default dependencies will cause model conversion to fail. Please install the version specified above.

## 2. Model Pruning Strategy

To support a larger context length, appropriate pruning can be applied when deploying the multimodal model.

### 2.1 Vision Model Pruning

Some operators are moved to the CPU of host devices such as RK3588.

### 2.2 MLP-AR Model Partitioning

Because the MLP-AR layer involves dynamic shape transformations, it is split into a separate subgraph. Shape manipulation operations run on the CPU.

## 3. Model Export Flow

```bash
cd examples/paddleocr_vl/python/llm

# Generate GRQ calibration data
python make_calidata.py --model_path PaddlePaddle/PaddleOCR-VL

# Export the LLM ONNX model
python export_llm.py \
    --model_path PaddlePaddle/PaddleOCR-VL \
    --quant

# Export the LLM RKNN model
python export_rknn.py \
    --onnx_path ../../model/llm/PaddleOCR-llm.onnx \
    --config ../../model/llm/PaddleOCR-llm.config.pkl \
    --rknn_path ../../model/llm/PaddleOCR-llm.rknn

cd ../vision

# Generate GRQ calibration data
python make_calidata.py --model_path PaddlePaddle/PaddleOCR-VL

# Export the Vision ONNX model and MLP-AR ONNX model
python export_vision.py \
    --model_path PaddlePaddle/PaddleOCR-VL \
    --quant

# Export the Vision RKNN model
python export_rknn.py \
    --onnx_path ../../model/vision/PaddleOCR-vision.onnx \
    --rknn_path ../../model/vision/PaddleOCR-vision.rknn \
    --mlpar_onnx_path ../../model/vision/PaddleOCR-vision-mlp_AR.onnx \
    --mlpar_rknn_path ../../model/vision/PaddleOCR-vision-mlp_AR.rknn
```

## 4. Generate Quantization Calibration Data

The LLM and Vision parts of PaddleOCR-VL use GRQ (W4A16) quantization by default during export, and quantization requires calibration data. The repository provides `make_calidata.py` scripts under `python/llm/` and `python/vision/`. They capture the real inputs entering the LLM trunk (`model.model`) and Vision trunk (`model.visual`), pickle the data to disk, and generate an index file for the corresponding export script's `--cali_dataset` option.

### 4.1 LLM Calibration Data

```bash
cd python/llm
python make_calidata.py \
    --model_path PaddlePaddle/PaddleOCR-VL \
    --datapath ../../../../datasets/OmniDocBench_ROI/llm/dataset.json \
    --export_datapath ./quant_data/model_inputs.json
```

After execution, `model_inputs.json` and `model_inputs/sample_N` calibration samples are generated under `./quant_data/`. This matches the default `--cali_dataset` path of `export_llm.py`.

### 4.2 Vision Calibration Data

```bash
cd python/vision
python make_calidata.py \
    --model_path PaddlePaddle/PaddleOCR-VL \
    --datapath ../../../../datasets/OmniDocBench_ROI/llm/dataset.json \
    --export_datapath ./quant_data/model_inputs.json
```

After execution, `model_inputs.json` and `model_inputs/sample_N` are generated under `./quant_data/`. This matches the default `--cali_dataset` path of `export_vision.py`.

### 4.3 Data Description

Both scripts follow the same logic:

1. Read the calibration sample list from the JSON specified by `--datapath`. Each sample contains fields such as `image`, `image_path`, and `input` (prompt).
2. Determine the maximum pixel count according to the `input` type (`spotting` uses `2048*28*28`, while other tasks use `1280*28*28`). For `spotting` images whose resolution is below 1500px, apply 2x LANCZOS upsampling.
3. Build multimodal inputs with `processor.apply_chat_template(...)`, call `model.generate(...)`, and capture the inputs of the LLM/Vision trunk through `capture_module_input`.
4. Pickle the captured tensors to `model_inputs/sample_N` and write the `model_inputs.json` index.

#### Parameters

| Parameter | Type | Description | Default |
|------|------|------|--------|
| `--model_path` | str | Model path or HuggingFace name | `PaddlePaddle/PaddleOCR-VL` |
| `--datapath` | str | Path to the calibration sample list JSON | `../../../../datasets/OmniDocBench_ROI/llm/dataset.json` |
| `--export_datapath` | str | Output index JSON path (`model_inputs/` is created in the same directory) | llm: `.../OmniDocBench_ROI/llm/model_inputs.json`<br>vision: `.../OmniDocBench_ROI/vision/model_inputs.json` |

> ⚠️ **Note**:
> - Calibration data has a large impact on GRQ quantization accuracy. It is recommended to use document images and prompts consistent with the actual deployment scenario. You can build your own dataset following the format of `datasets/OmniDocBench_ROI/llm/dataset.json`.
> - The recommended number of calibration samples is 20 to 128.
> - `make_calidata.py` must run in the same environment as export (`transformers==4.55.0`).
> - This step is only required when GRQ quantization is enabled. It can be skipped when exporting float models.

## 5. KV Cache INT4 Quantization

During large language model inference, the KV Cache (Key/Value Cache) stores historical attention keys and values to avoid repeated computation and improve inference speed. As the sequence length grows, KV Cache memory usage increases quickly. To reduce storage bandwidth and memory access overhead, quantization can convert KV Cache from FP16/FP32 to INT8 or lower bit widths. However, because KV Cache values change dynamically token by token, using a single quantization parameter for the whole KV segment can accumulate quantization error and affect inference accuracy. Group quantization is therefore commonly used to reduce accuracy loss.

Currently, RKNN LLM supports two KV Cache quantization modes:
Int8_to_F16 (default): store as INT8 and convert back to FP16 during computation;
Int4_to_F16 (for longer-context scenarios, with some accuracy loss): store as INT4 and convert back to FP16 during computation.
To support longer context lengths and further compress KV Cache memory, enabling Int4_to_F16 is recommended.

> ⚠️ **Note**: `rknn.config()` in `python/llm/export_rknn.py` does not currently enable KV Cache quantization. `max_ctx_len`, `max_position_embeddings`, and related parameters are commented out. The actual configuration is:

```python
rknn.config(target_platform='rk1820',
            quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32',
            # max_ctx_len=2048, max_position_embeddings=2048,
            )
```

To enable Int4_to_F16 for longer context support, uncomment the related parameters in `rknn.config()`, or refer to the RKNN3 LLM configuration documentation and set `kvcache_dtype='Int4_to_F16'` through `llm_config`. Re-run RKNN conversion after modifying the configuration.

## 6. Vision Model Resolution Adjustment

You can adjust the input resolution through `--img_h` and `--img_w` (both must be multiples of 28):

```bash
python export_vision.py --img_h 504 --img_w 504
```

> ⚠️ **Note**:
> - Higher resolution increases memory usage and affects the maximum LLM context length.
> - Some resolutions may be incompatible with the RKNN inference framework. If errors occur, contact the RKNPU team.

## 7. C++ Deployment

If you modify the Vision model resolution, update the parameters in `rknn_paddleocr_vl_vision.h` accordingly:

```cpp
#define MODEL_WIDTH  <your_width>
#define MODEL_HEIGHT <your_height>
```

### 1. Build the C++ Demo

C++ demo code location: `examples/paddleocr_vl/cpp/`

```bash
cd ../../../../
./build-linux.sh -t rk3588 -a aarch64 -d paddleocr_vl
```

After a successful build, the C++ demo is saved under `install/rk3588_linux_aarch64/rknn_paddleocr_vl_demo`, including model files, dependency libraries, and the demo executable. The directory structure is:

```bash
- install/rk3588_linux_aarch64/rknn_paddleocr_vl_demo
 - lib
   - librknn3_api.so
   - librga.so
 - model
   - PaddleOCR-vision.rknn
   - PaddleOCR-vision.weight
   - PaddleOCR-vision-mlp_AR.rknn
   - PaddleOCR-vision-mlp_AR.weight
   - position_embedding_model.bin
   - PaddleOCR-llm.rknn
   - PaddleOCR-llm.weight
   - PaddleOCR-llm.tokenizer.gguf
   - PaddleOCR-llm.embed.bin
   - test.png
 - rknn_paddleocr_vl_demo
```

### 2. Push the Demo to an RK3588 Board

```bash
adb push install/rk3588_linux_aarch64/rknn_paddleocr_vl_demo /userdata/
```

### 3. Run the C++ Demo

```bash
adb shell
cd /userdata/rknn_paddleocr_vl_demo
./rknn_paddleocr_vl_demo model/PaddleOCR-vision.rknn model/PaddleOCR-vision.weight model/position_embedding_model.bin model/PaddleOCR-llm.rknn model/PaddleOCR-llm.weight model/PaddleOCR-llm.tokenizer.gguf model/PaddleOCR-llm.embed.bin model/PaddleOCR-vision-mlp_AR.rknn model/PaddleOCR-vision-mlp_AR.weight 0xff 0xff 0xff model/test.png "table"
```

After a successful run, the recognition result and inference performance are printed as follows:

```bash
--> inference paddleocr_vl llm model
rknn_session_run
<fcel>考评项目<lcel><fcel>权重<fcel>考评内容<fcel>评分标准<fcel>评分方法<fcel>自评得分<nl><fcel>一、组织领导体系(120分)<fcel>学校安全工作目标责任制<fcel>20<fcel>建立健全创建平安校园组织领导工作机制，责任明确、措施落实<fcel>学校有创建平安校园工作组织领导机构，第一责任人到位，安全岗位职责明确的得20分<fcel>查看会议记录、学校文件、岗位安全职责分工等资料<ecel><nl><ucel><ucel><fcel>20<fcel>学校有创建平安校园工作年度目标，计划<fcel>每当年创建目标、计划的得20分；制定目标和计划重点不突出、针对性不强的扣10分；无创建目标和计划的不得分<fcel>查看台账资料<ecel><nl><ucel><ucel><fcel>30<fcel>定期研究分析校园安全稳定问题，提出针对性对策措施<fcel>每月不少于1次安全分析会议（校园平安情况排查分析研究部署）等20分；有针对突出措施的得10分；不落实落实扣15分；未答不得分<fcel>查看会议、工作记录<ecel><nl><ucel><fcel>各部门安全工作责任分解<fcel>30<fcel>学校签订岗位安全工作责任书<fcel>学校根据各部门特点制定相应的责任书，签订、落实并上交责任书张榜张榜30分；部门岗位责任书没有针对性的扣10分；不完全落实扣15分；未答不得分<fcel>查看责任书<ecel><nl><ucel><fcel>安全保卫人员配备<fcel>20<fcel>校园安全管理机构、安全保卫人员落实<fcel>有具体负责学校安全工作的职能科室（安管办）和专职的安全保卫人员得20分；兼职的得10分；没有配备的不得分<fcel>查看教职工工作资料等资料，实地考察<ecel><nl><fcel>二、制度建设体系(150分)<fcel>治安制度<fcel>25<fcel>有门卫、巡逻、实验室、重点部位场所、学生生活区安全保卫制度<fcel>有制度得25分；制度不全扣10分<fcel>查看制度<ecel><nl>

--------------------Finished-------------------- 

--------------------------------------------------------------------------------------
 Stage         Total Time (ms)  Tokens    Time per Token (ms)      Tokens per Second      
--------------------------------------------------------------------------------------
 Prefill       64.30            338       0.19                     5256.69                
 Generate      1743.32          418       4.17                     239.77                 
--------------------------------------------------------------------------------------
 Vision latency = 415.10 ms, FPS = 2.41
```

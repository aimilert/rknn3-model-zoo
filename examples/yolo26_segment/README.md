# YOLO26 Segment 示例说明

## ONNX 模型说明

本示例用于 YOLO26 Segment 实例分割模型，支持 COCO 80 类目标检测和实例分割 mask 输出。

### 1. 模型文件

当前示例提供 `model/download_model.sh` 用于下载 PyTorch 权重文件：

```bash
cd examples/yolo26_segment/model
chmod +x download_model.sh
./download_model.sh
```

执行后会在 `model/` 目录下得到：

```text
model/
├── yolo26n-seg.pt
├── yolo26s-seg.pt
└── yolo26m-seg.pt
```

`bus.jpg`、`coco_80_labels_list.txt` 和 `images.txt` 为示例运行与量化校准使用的配套文件，不由 `download_model.sh` 下载。

ONNX、RKNN 和 weight 文件需要按后续章节导出和转换生成，例如：

```text
yolo26n_seg.onnx
yolo26n_seg.rknn
yolo26n_seg.weight
```

### 2. 输出结构

`convert.py` 会从 YOLO26 Segment ONNX 中选择 3 个尺度的检测头输出，并保留分割原型图输出。

每个尺度包含：

- box head：边界框 DFL 回归输出
- score head：80 类类别分数输出
- mask coeff head：实例 mask 系数输出，通道数为 `32`

转换后的默认输出为 10 个 tensor。当前 `yolo26n_seg.rknn` 在板端的实际输出为 INT8 / NCHW：

| 输出顺序 | 输出名 | Shape | 说明 |
|----------|--------|-------|------|
| output0 | box_1 | `[1, 4, 80, 80]` | stride 8 DFL 边界框回归 |
| output1 | score_1_sigmoid | `[1, 80, 80, 80]` | stride 8 80 类分数 |
| output2 | seg_1 | `[1, 32, 80, 80]` | stride 8 mask 系数 |
| output3 | box_2 | `[1, 4, 40, 40]` | stride 16 DFL 边界框回归 |
| output4 | score_2_sigmoid | `[1, 80, 40, 40]` | stride 16 80 类分数 |
| output5 | seg_2 | `[1, 32, 40, 40]` | stride 16 mask 系数 |
| output6 | box_3 | `[1, 4, 20, 20]` | stride 32 DFL 边界框回归 |
| output7 | score_3_sigmoid | `[1, 80, 20, 20]` | stride 32 80 类分数 |
| output8 | seg_3 | `[1, 32, 20, 20]` | stride 32 mask 系数 |
| output9 | proto | `[1, 32, 160, 160]` | 32 通道 mask prototype |

Python 推理脚本同时兼容两种输出形式：

- ONNX 原始输出：通常为 `output0` 检测结果和 `output1` proto，共 2 个输出
- RKNN 转换输出：3 个尺度 raw heads 加 proto，共 10 个输出

### 3. 后处理方式

当前示例默认使用主控端后处理：

- C++ 后处理实现在 `cpp/postprocess.cc`
- Python 后处理实现在 `python/infer.py`
- 支持 W8A8、FP16、FP32 输出解析
- 支持 DFL 解码、score 解析、mask 系数解析、NMS、proto mask 生成、letterbox 坐标还原
- C++ INT8 mask 路径保留当前优化版：在 proto 分辨率生成 mask，再映射到输出图；mask 存储使用 bbox-local 尺寸，并复用后处理 scratch buffer，避免回到逐原图像素做 32 通道 dot product 的慢路径


## ONNX 模型导出

### 1. 安装 Python 依赖

```bash
cd examples/yolo26_segment/python
pip install -r requirements.txt
```

注意：

- `rknn3-toolkit` 不是 PyPI 包，需要手动安装 Rockchip 提供的 whl 包
- `export_onnx.py` 当前使用 `device='cuda'` 导出，导出环境需要可用 CUDA；如需 CPU 导出，请按实际环境调整脚本中的 `device` 参数

### 2. 导出 ONNX

以 `yolo26n-seg.pt` 为例：

```bash
cd examples/yolo26_segment/python
python3 export_onnx.py \
  --model_path ../model/yolo26n-seg.pt \
  --output ../model/yolo26n_seg.onnx \
  --img_size 640
```


导出脚本主要执行：

- Ultralytics YOLO ONNX 导出
- ONNX checker 校验
- onnxsim 简化

## 模型转换

### 1. RK1820 W8A8 转换

```bash
cd examples/yolo26_segment/python
python3 convert.py \
  --onnx ../model/yolo26n_seg.onnx \
  --platform rk1820 \
  --dtype w8a8 \
  --output ../model/yolo26n_seg.rknn \
  --dataset ../model/images.txt \
  --core-num 1
```

注意：

- `--dataset` 默认使用 `../model/images.txt`，用于 W8A8 量化校准
- `--output` 建议显式指定；`convert.py` 的默认输出名为 `../model/yolo26n.rknn`
- `--core-num` 仅在 `--platform rk1820` 时生效，支持 `1` 或 `8`，默认 `1`
- RK1820 模型转换时的 `--core-num` 需要和板端运行时的 `core_mask` 匹配：
  - `--core-num 1`：运行时使用 `0x1`
  - `--core-num 8`：运行时使用 `0xff`
- W8A8 转换时，`convert.py` 会对部分注意力和检测头子图配置 `subgraph_dtype_target=w16a16`，用于降低量化误差

### 2. 转换并执行板端推理测试

远程板端测试示例：

```bash
python3 convert.py \
  --onnx ../model/yolo26n_seg.onnx \
  --platform rk1820 \
  --dtype w8a8 \
  --output ../model/yolo26n_seg.rknn \
  --dataset ../model/images.txt \
  --device-id ip:port
```

RKNN3 平台精度分析示例：

```bash
python3 convert.py \
  --onnx ../model/yolo26n_seg.onnx \
  --platform rk1820 \
  --dtype w8a8 \
  --output ../model/yolo26n_seg.rknn \
  --dataset ../model/images.txt \
  --device-id ip:port \
  --accuracy-analysis
```

## C++ 示例编译和运行

### 1. 编译主程序

在项目根目录执行：

```bash
./build-linux.sh -t rk3588 -a aarch64 -d yolo26_segment -b Release
```

编译完成后，会在安装目录 `install/rk3588_linux_aarch64/rknn_yolo26_segment_demo/` 下生成运行包：

- `rknn_yolo26_segment_demo`：单张图片推理程序
- `rknn_yolo26_segment_dataset_eval`：COCO bbox 数据集批量测试程序
- `model/`：测试图片、标签文件以及已转换的 RKNN/weight 文件
- `lib/`：运行依赖库

### 2. 单张图片推理

将安装目录推送到板端后执行。当前板端测试目录使用 `/data/rknn_yolo26_segment_demo`：

```bash
adb push install/rk3588_linux_aarch64/rknn_yolo26_segment_demo /data/
adb shell
cd /data/rknn_yolo26_segment_demo
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH
```

通用命令格式：

```bash
./rknn_yolo26_segment_demo <model_path> <weight_path> <image_path> <core_mask> [postprocess_plugin_path]
```

参数说明：

- `model_path`：RKNN 模型路径
- `weight_path`：RKNN3 模型权重文件路径
- `image_path`：输入图片路径
- `core_mask`：NPU 核心掩码，十六进制格式
  - `0x1`：使用 NPU 核心 0
  - `0x2`：使用 NPU 核心 1
  - `0xff`：使用 8 核，需与 `--core-num 8` 转换出的 RK1820 模型匹配
- `postprocess_plugin_path`：可选，默认不需要传入

运行示例：

```bash
./rknn_yolo26_segment_demo model/yolo26n_seg.rknn model/yolo26n_seg.weight model/bus.jpg 0x1
```

程序会在当前目录生成：

- `out.png`：绘制检测框、类别和半透明实例分割 mask 后的图片

### bus.jpg 最新实测输出

测试命令：

```bash
adb -s device-id shell \
  "cd /data/rknn_yolo26_segment_demo && \
   export LD_LIBRARY_PATH=./lib:\$LD_LIBRARY_PATH && \
   ./rknn_yolo26_segment_demo model/yolo26n_seg.rknn model/yolo26n_seg.weight model/bus.jpg 0x1"
```

```text
person @ (209 241 284 506) 0.894
person @ (111 234 225 537) 0.876
bus @ (98 137 556 430) 0.853
person @ (476 241 558 525) 0.718
```

![YOLO26 Segment result](model/out.png)

## Python 推理和评估

`python/infer.py` 支持 ONNX 和 RKNN 两种模型输入。

### 1. ONNX 推理

```bash
cd examples/yolo26_segment/python
python3 infer.py \
  --model-path ../model/yolo26n_seg.onnx \
  --img-folder ../../../datasets/COCO/subset \
  --output-json results_yolo26_seg_onnx.json \
  --save-img-dir vis_onnx
```


### 2. RKNN 远程板端推理

```bash
cd examples/yolo26_segment/python
python3 infer.py \
  --model-path ../model/yolo26n_seg.rknn \
  --runtime board \
  --target rk1820 \
  --device-id ip:port \
  --core-mask 0x1 \
  --img-folder ../../../datasets/COCO/subset \
  --output-json results_yolo26_seg_rknn_board.json \
  --save-img-dir vis_rknn_board
```

主要参数说明：

- `--model-path`：ONNX 或 RKNN 模型路径
- `--runtime`：RKNN 运行方式，`sim` 为本地模拟器，`board` 为远程板端
- `--target`：目标平台，例如 `rk1820`、`rk3572`
- `--device-id`：远程板端设备 ID
- `--core-mask`：运行核心掩码
- `--img-folder`：图片目录
- `--dataset-list`：图片路径列表，设置后优先使用该列表
- `--output-json`：COCO 格式结果文件，包含 bbox 和 segmentation 字段
- `--save-img-dir`：可视化图片保存目录
- `--conf-thresh`：置信度阈值，默认 `0.001`
- `--nms-thresh`：NMS 阈值，默认 `0.70`
- `--coco-map-test`：推理结束后执行 COCO mAP 评估；结果包含 segmentation 时会同时评估 `bbox` 和 `segm`

## 数据集精度测试说明（可选）

测试数据集：COCO val2017

测试内容：

- bbox：目标框检测精度
- segm：实例分割精度

### 1. 准备 COCO 数据集

```bash
cd datasets/COCO
python3 download_eval_dataset.py
```

执行后会得到：

- `val2017/`：COCO val2017 原始图片
- `annotations/`：COCO 标注文件，其中包含 `instances_val2017.json`
- `coco_dataset_path.txt`：板端图片路径列表，默认指向 `/userdata/val2017`

### 2. Python 数据集评估

ONNX 评估：

```bash
cd examples/yolo26_segment/python
python3 infer.py \
  --model-path ../model/yolo26n_seg.onnx \
  --img-folder ../../../datasets/COCO/val2017 \
  --anno-json ../../../datasets/COCO/annotations/instances_val2017.json \
  --output-json results_yolo26_seg_onnx.json \
  --coco-map-test
```

RKNN 远程板端评估：

```bash
cd examples/yolo26_segment/python
python3 infer.py \
  --model-path ../model/yolo26n_seg.rknn \
  --runtime board \
  --target rk1820 \
  --device-id ip:port \
  --core-mask 0x1 \
  --img-folder ../../../datasets/COCO/val2017 \
  --anno-json ../../../datasets/COCO/annotations/instances_val2017.json \
  --output-json results_yolo26_seg_rknn_board.json \
  --coco-map-test
```

推理结束后会生成 `--output-json` 指定的 COCO 格式结果文件。`--coco-map-test` 会调用 `pycocotools` 输出 bbox 与 segm 的 mAP 指标。

### 3. C++ 板端数据集测试

`rknn_yolo26_segment_dataset_eval` 用于在板端批量运行 COCO val2017 图片，并输出 COCO detection 格式结果 JSON。


#### 准备测试数据

```bash
adb push datasets/COCO/val2017 /userdata/
adb push datasets/COCO/coco_dataset_path.txt /userdata/coco_dataset_test_path.txt
```

#### 编译和部署

```bash
./build-linux.sh -t rk3588 -a aarch64 -d yolo26_segment -b Release
adb push install/rk3588_linux_aarch64/rknn_yolo26_segment_demo /data/
```

#### 运行批量推理

```bash
adb shell
cd /data/rknn_yolo26_segment_demo
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH

./rknn_yolo26_segment_dataset_eval \
  model/yolo26n_seg.rknn \
  model/yolo26n_seg.weight \
  0x1 \
  --dataset-list /userdata/coco_dataset_test_path.txt \
  --output results_rknn_bbox.json
```

推理结束后会在当前目录生成 `results_rknn_bbox.json`，其中 `bbox` 坐标格式为 COCO 要求的 `[x, y, width, height]`。

可选参数：

- `--dataset-list path`：测试图片路径列表，默认 `/userdata/coco_dataset_test_path.txt`
- `--output json`：结果 JSON 文件，默认 `results_rknn_bbox.json`
- `--conf-thresh value`：置信度阈值，默认 `0.001`
- `--nms-thresh value`：NMS 阈值，默认 `0.70`

#### 获取结果

```bash
adb pull /data/rknn_yolo26_segment_demo/results_rknn_bbox.json ./
```


## 目录结构

```text
examples/yolo26_segment/
├── README.md
├── cpp/
│   ├── CMakeLists.txt
│   ├── main.cc
│   ├── dataset_eval.cc
│   ├── postprocess.cc
│   ├── postprocess.h
│   └── rknpu3/yolo26.cc
├── model/
│   ├── download_model.sh
│   ├── bus.jpg
│   ├── out.png
│   ├── coco_80_labels_list.txt
│   └── images.txt
└── python/
    ├── convert.py
    ├── export_onnx.py
    ├── infer.py
    └── requirements.txt
```

## 常见问题

### 1. 转换时找不到量化数据集

确认 `../model/images.txt` 中的图片路径可访问。当前列表指向 `../../../datasets/COCO/subset/` 下的校准图片。

### 2. 板端运行提示找不到动态库

进入安装目录后设置库路径：

```bash
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH
```

### 3. RK1820 运行核心与转换核心数不匹配

如果转换时使用 `--core-num 1`，板端运行使用 `0x1`。如果转换时使用 `--core-num 8`，板端运行使用 `0xff`。

### 4. C++ 推理结果较多或较少

单张图片推理默认阈值在 `cpp/postprocess.h` 中配置：

```c
#define NMS_THRESH 0.45
#define BOX_THRESH 0.25
#define SEG_MASK_THRESH 0.5f
```

Python 评估程序默认使用 `conf_thresh=0.001`、`nms_thresh=0.70`，可通过命令行参数调整。

### 5. Python 默认模型名不可用

`python/infer.py` 的默认模型路径为 `../model/yolo26s.rknn`，与当前示例的模型命名不一致。当前目录模型通常命名为 `yolo26n_seg.rknn`、`yolo26s_seg.rknn` 或 `yolo26m_seg.rknn`，建议运行时显式传入 `--model-path`。

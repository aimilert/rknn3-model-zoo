# ResNet 模型部署说明

模型地址：[resnet50-v2-7](https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/ResNet/resnet50-v2-7.onnx)

ResNet 为图像分类模型，本示例以 ResNet50-v2 为例。

## 1. 导出 RKNN 模型

### 1.1 下载 ONNX 模型

```bash
cd examples/resnet/model
bash download_model.sh
```

### 1.2 转换为 RKNN 模型

```bash
cd examples/resnet/python

# 参数：<onnx_model_path> <platform> [dtype] [output_rknn_path]
# platform 可选 rk1820, rk1828；dtype 可选 i8/u8/w8a8(量化) 或 fp(不量化)
python resnet.py ../model/resnet50-v2-7.onnx rk1820 i8
```

转换配置使用 W8A8 量化，输入为 224x224，按 ImageNet 均值方差归一化：

```python
rknn.config(mean_values=[[255*0.485, 255*0.456, 255*0.406]],
            std_values=[[255*0.229, 255*0.224, 255*0.225]],
            target_platform=platform,
            input_attrs={'input': {'dtype': 'uint8', 'layout': 'NHWC'}},
            quantized_dtype='w8a8', quantized_algorithm='normal', quantized_method='channel')
```

## 2. C++ 部署说明

### 2.1 编译

```bash
cd rknn3_model_zoo/

# 设置交叉编译工具链
export GCC_COMPILER=/path/to/aarch64-linux-gnu

# 编译
./build-linux.sh -t rk3588 -a aarch64 -d resnet
```

编译完成后，文件生成在 `install/rk3588_linux_aarch64/rknn_resnet_demo/` 目录。

### 2.2 运行示例

```bash
./rknn_resnet_demo <model_path> <weight_path> <image_path> <core_mask>
```

示例：

```bash
./rknn_resnet_demo model/resnet50-v2-7.rknn model/resnet50-v2-7.weight model/dog_224x224.jpg 0x1
```

输出 Top-5 分类结果，例如：

```
[0] score=0.95xxxx class=...
[1] score=0.02xxxx class=...
```

## 3. 数据集精度测试（可选）

### 3.1 数据集下载

https://image-net.org/download.php

### 3.2 rknn3 板端测试

将数据集解压并推到板子上，例如 `/data/imagenet` 目录，需包含：

```
ILSVRC2012_img_val_256   ILSVRC2012_img_val_256.txt
```

```bash
./dataset_eval model/resnet50-v2-7.rknn model/resnet50-v2-7.weight /data/imagenet/ 0x1
```

### 3.3 python 测试

参考 `python/dataset_eval.py`，根据实际环境调整数据集路径和模型路径即可。

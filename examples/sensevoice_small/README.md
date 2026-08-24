# Sensevoice-Small 模型部署说明


## onnx模型

### 导出onnx模型

将 SenseVoiceSmall模型转换为10秒固定长度音频输入的ONNX格式，同时导出嵌入矩阵和词表文件供推理使用

首先需安装requirements.txt
```
cd python
pip install -r requirements.txt
```
然后进行导出
```
cd python
python export_onnx.py 
```


### 测试onnx模型

测试 SenseVoice ONNX 语音识别模型的推理脚本，输入音频会自动重采样到16kHz，统一长度为10秒
```
cd python
python onnx_infer.py --audio_path ../model/zh.wav
```


## rknn3模型



### 导出rknn3模型

将 ONNX 模型转换为RKNN3，W4A16量化

```
cd python
python export_rknn.py ../model/sensevoice_fix_10s.onnx rk1820
```


## C++ 部署

编译可执行文件，将相关模型文件推入板端
```bash
cd ../..
./build-linux.sh -t rk3588 -a aarch64 -d sensevoice_small -b Release
adb push install/rk3588_linux_aarch64/rknn_sensevoice_small_demo /userdata/
```

进入板端推理
```bash
adb shell
cd /userdata/rknn_sensevoice_small_demo
export LD_LIBRARY_PATH=./lib
./rknn_sensevoice_demo ./model/sensevoice_fix_10s.rknn ./model/sensevoice_fix_10s.weight ./model/zh.wav 0x01
```

运行结果如下所示:
```
Sensevoice_small output: <|zh|><|NEUTRAL|><|Speech|><|withitn|>开放时间早上9点至下午5点。
```
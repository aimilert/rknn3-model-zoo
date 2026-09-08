# note: 模型文件名带"_rknn3"字段的模型为专门为rk1820进行优化，
#       不带"_rknn3"字段的是不带yolo后处理的模型，此时模型后处理需要在应用程序中完成，
#       将yolo的解码、候选框筛选排序以及nms放置到rk1820端进行计算，减少数据传输压力。
#
#       从以下链接下载完整的 PyTorch 示例工程：
#       https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolov5/yolov5-postprocess.tar.gz
#       下载完成后，参考其中的导出文档，按照步骤进行操作，即可生成适配 RK1820 的优化 ONNX 模型。

# 默认下载优化模型
wget -O ./yolov5n_rknn3.onnx https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolov5/yolov5n_rknn3.onnx
# wget -O ./yolov5n.onnx https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolov5/yolov5n.onnx

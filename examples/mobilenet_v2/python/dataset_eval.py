import onnxruntime as ort
import numpy as np
from PIL import Image
import os
from tqdm import tqdm


def preprocess_image(image_path, input_size=(224, 224)):
    """
    预处理输入图像以适应MobileNet模型的要求
    """
    # 打开图像并转换为RGB
    image = Image.open(image_path).convert('RGB')
    
    # 调整图像大小
    image = image.resize(input_size)
    
    # 转换为numpy数组并归一化
    image_array = np.array(image).astype(np.float32)
    
    # 归一化到[0, 1]范围
    image_array /= 255.0
    
    # 应用ImageNet的均值和标准差
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_array = (image_array - mean) / std
    
    # 调整维度为[batch, channels, height, width]
    image_array = np.transpose(image_array, (2, 0, 1))  # HWC -> CHW
    image_array = np.expand_dims(image_array, axis=0)   # 添加batch维度
    
    return image_array


if __name__ == "__main__":
    # 配置路径
    model_path = "onnx_models/mobilenetv2-12.onnx"
    # model_path = "onnx_models/resnet50-v2-7.onnx"
    data_root = "/data/imagenet"
    labels_path = os.path.join(data_root, "ILSVRC2012_img_val_256.txt") # ImageNet标签文件路径
    
    # 加载ONNX模型
    session = ort.InferenceSession(model_path)
    # 获取输入和输出节点名称
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    fp = open(labels_path, "r")
    lines = fp.readlines()
    fp.close()

    total = 0
    correct_1 = 0
    correct_5 = 0
    for line in tqdm(lines):
        path, label = line.strip().split()
        label = int(label)
        image_path = os.path.join(data_root, path[:-1])
        input_tensor = preprocess_image(image_path)
        outputs = session.run([output_name], {input_name: input_tensor})
        predictions = outputs[0][0]
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))  # Softmax
        top_indices = np.argsort(probabilities)[-5:][::-1]
        if top_indices[0] == label:
            correct_1 += 1
        if label in top_indices:
            correct_5 += 1
        total += 1
    
    print(f"top1: {correct_1/total:.4f}")
    print(f"top5: {correct_5/total:.4f}")



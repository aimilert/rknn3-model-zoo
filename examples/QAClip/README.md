# QA-CLIP ViT-L/14 导出 RKNN3 模型指南

模型地址：https://huggingface.co/TencentARC/QA-CLIP-ViT-L-14

## 模型导出命令

### LLM/Text 模型

```bash
cd examples/QAClip/python/llm
python export_llm.py --model_path=../../models/QA-CLIP-ViT-L-14 --export_llm_path=./chinese_clip_text.onnx
python export_rknn.py --onnx_path=chinese_clip_text.onnx --config=chinese_clip_text.config.pkl --rknn_path=chinese_clip_text.rknn --platform=rk1820
```

导出后会生成 Text RKNN 模型、权重文件、embedding 文件和 tokenizer 文件：

- `chinese_clip_text.rknn`
- `chinese_clip_text.weight`
- `chinese_clip_text.embed.bin`
- `chinese_clip_text.tokenizer.gguf`

### Vision 模型

```bash
cd examples/QAClip/python/vision
python export_vision.py --model_path=../../models/QA-CLIP-ViT-L-14 --export_vision_path=chinese_clip_vision.onnx --img_h=224 --img_w=224
python export_rknn.py --onnx_path=chinese_clip_vision.onnx --rknn_path=chinese_clip_vision.rknn --dataset_path=datasets.txt --platform=rk1820
```

导出后会生成 Vision RKNN 模型和权重文件：

- `chinese_clip_vision.rknn`
- `chinese_clip_vision.weight`

## C Demo

### 构建

在仓库根目录执行：

```bash
./build-linux.sh -t rk1820 -a aarch64 -d QAClip
```

如需在其他平台运行，请将 `-t` 参数和导出 RKNN 时的 `--platform` 参数改为对应平台。

### 运行

进入编译产物目录后执行：

```bash
./rknn_qaclipvit_demo chinese_clip_text.rknn chinese_clip_text.weight chinese_clip_text.embed.bin chinese_clip_text.tokenizer.gguf chinese_clip_vision.rknn chinese_clip_vision.weight demo.jpg 1
```

Demo 会分别计算图片与内置中文文本的相似度，并输出 softmax 后的概率。

## 所需模型文件

| 文件名 | 说明 |
|--------|------|
| chinese_clip_text.rknn | Text RKNN 模型 |
| chinese_clip_text.weight | Text 权重文件 |
| chinese_clip_text.embed.bin | Text embedding 文件 |
| chinese_clip_text.tokenizer.gguf | Text tokenizer 文件 |
| chinese_clip_vision.rknn | Vision RKNN 模型 |
| chinese_clip_vision.weight | Vision 权重文件 |
| demo.jpg | 测试图片 |

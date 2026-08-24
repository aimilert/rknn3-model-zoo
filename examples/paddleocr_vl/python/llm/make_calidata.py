import os
from PIL import Image
import json
import torch
import pickle
from tqdm import tqdm
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from py_utils.cali_data import capture_module_input
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modeling_paddleocr_vl import PaddleOCRVLForConditionalGeneration



parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='PaddlePaddle/PaddleOCR-VL', help='model path', required=False)
parser.add_argument('--datapath', type=str, default='../../../../datasets/OmniDocBench_ROI/llm/dataset.json', help='model path', required=False)
parser.add_argument('--export_datapath', type=str, default='../../../../datasets/OmniDocBench_ROI/llm/model_inputs.json', help='data path', required=False)
args = parser.parse_args()

## 加载模型
model = PaddleOCRVLForConditionalGeneration.from_pretrained(
    args.model_path, device_map="cpu",
    trust_remote_code=True).eval()

processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

## 生成量化校准数据
info = []
with open(args.datapath, "r") as f:
    datasets = json.load(f)


for idx, data in enumerate(tqdm(datasets)):
    image_name = data["image"].split(".")[0]
    path_ = "/".join(args.datapath.split("/")[:-1])
    imgp = os.path.join(path_, data["image_path"], data["image"])
    
    image = Image.open(imgp).convert("RGB")
    orig_w, orig_h = image.size
    spotting_upscale_threshold = 1500
    max_pixels = 2048 * 28 * 28 if data['input'] == "spotting" else 1280 * 28 * 28
    if data['input'] == "spotting" and orig_w < spotting_upscale_threshold and orig_h < spotting_upscale_threshold:
        process_w, process_h = orig_w * 2, orig_h * 2
        try:
            resample_filter = Image.Resampling.LANCZOS
        except AttributeError:
            resample_filter = Image.LANCZOS
        image = image.resize((process_w, process_h), resample_filter)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": data['input']},
            ]
        }
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        images_kwargs={"size": {"shortest_edge": processor.image_processor.min_pixels, "longest_edge": max_pixels}},
    ).to(model.device)
    
    with torch.inference_mode():
        temp = capture_module_input(
            model.model,
            lambda: model.generate(**inputs, max_new_tokens=512)
        )
    sample_name = "sample_{}".format(idx)
    path_ = "/".join(args.export_datapath.split("/")[:-1])
    path_dir = os.path.join(path_, "model_inputs")
    os.makedirs(path_dir, exist_ok=True)
    pickle_path = os.path.join(path_dir, sample_name)
    info.append({"sample":"model_inputs/{}".format(sample_name)})
    with open(pickle_path, 'wb') as f:
        pickle.dump(temp, f)
    
with open(args.export_datapath, 'w', encoding='utf-8') as f:
    json.dump(info, f, indent=2, ensure_ascii=False)

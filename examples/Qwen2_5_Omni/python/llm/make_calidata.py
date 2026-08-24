import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from py_utils.cali_data import capture_module_input
from PIL import Image
import json
import torch
import pickle
from tqdm import tqdm
from transformers import AutoProcessor
from qwen_omni_utils import process_mm_info
import argparse
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, Qwen2_5OmniForConditionalGeneration



parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-Omni-3B', help='model path', required=False)
parser.add_argument('--datapath', type=str, default='../../../../datasets/MMBench/llm/dataset.json', help='model path', required=False)
parser.add_argument('--export_datapath', type=str, default='./quant_data/model_inputs.json', help='model path', required=False)
args = parser.parse_args()

## 加载模型
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    args.model_path, torch_dtype=torch.bfloat16, device_map="cpu",
    trust_remote_code=True).eval() ## 注意校准数据生成时候模型类型应该与export_llm.py中模型类型一致，否则grq会报错

processor = AutoProcessor.from_pretrained(args.model_path)

## 生成量化校准数据
info = []
with open(args.datapath, "r") as f:
    datasets = json.load(f)
for idx, data in enumerate(tqdm(datasets)):
    image_name = data["image"].split(".")[0]
    path_ = "/".join(args.datapath.split("/")[:-1])
    imgp = os.path.join(path_, data["image_path"], data["image"])

    conversation = [
        {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": imgp
                },
                {"type": "text", "text": data["input"]},
            ],
        }
    ]
    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
    inputs = inputs.to(model.device).to(model.dtype)
    with torch.inference_mode():
        temp = capture_module_input(
            model.thinker.model,
            lambda: model.generate(**inputs, use_audio_in_video=False)
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

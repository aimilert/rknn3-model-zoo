import os
from PIL import Image
import json
import torch
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from py_utils.cali_data import capture_module_input
from llava.model import *  # 增加logits_to_keep输入

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='../../llava-fastvithd_1.5b_stage3', help='model path', required=False)
parser.add_argument('--datapath', type=str, default='../../../../datasets/MMBench/llm/dataset.json', help='model path', required=False)
parser.add_argument('--export_datapath', type=str, default='./quant_data/model_inputs.json', help='model path', required=False)
args = parser.parse_args()

## 加载模型
model = LlavaQwen2ForCausalLM.from_pretrained(
    args.model_path, device_map="cpu",
    trust_remote_code=True).eval()

tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
IMAGE_TOKEN_INDEX = 46465


## 生成量化校准数据
info = []
with open(args.datapath, "r") as f:
    datasets = json.load(f)
for idx, data in enumerate(tqdm(datasets)):
    image_name = data["image"].split(".")[0]
    path_ = "/".join(args.datapath.split("/")[:-1])
    imgp = os.path.join(path_, data["image_path"], data["image"])

    messages = [
        {"role": "user", "content": "<image>\n{}".format(data["input"])}
    ]
    rendered = tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    pre, post = rendered.split("<image", 1)
    # Tokenize the text *around* the image token (no extra specials!)
    pre_ids  = tok(pre,  return_tensors="pt", add_special_tokens=False).input_ids
    post_ids = tok(post, return_tensors="pt", add_special_tokens=False).input_ids
    # Splice in the IMAGE token id (-200) at the placeholder position
    img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
    input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)
    # Preprocess image via the model's own processor
    img = Image.open(imgp).convert("RGB")
    px = model.get_vision_tower().image_processor(images=img, return_tensors="pt")["pixel_values"]
    px = px.to(model.device, dtype=model.dtype)

    with torch.inference_mode():
        # 捕获进入 model.model 的输入并短路前向, 直接拿到 {"args","kwargs"}
        temp = capture_module_input(
            model.model,
            lambda: model.generate(inputs=input_ids, attention_mask=attention_mask, images=px, max_new_tokens=128)
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

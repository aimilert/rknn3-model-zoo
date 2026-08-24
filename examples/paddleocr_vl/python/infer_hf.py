import json
import math
import os
import time

import torch
from transformers import AutoProcessor, AutoTokenizer
from PIL import Image
from modeling_paddleocr_vl import PaddleOCRVLForConditionalGeneration

model_path = "PaddlePaddle/PaddleOCR-VL"
device = 'cuda:0'
VISION_CONFIG = "vision/vision_config.json"
PATCH_SIZE = 14
TEST_IMAGE = "../data/vision/test.png"
SAVE_IMAGE = None
prompt = "table"
DEFAULT_PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}
query = DEFAULT_PROMPTS.get(prompt, "OCR:")


def load_config(config_path: str):
    if not os.path.exists(config_path):
        return 504, 504
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    for key in ["img_h", "img_w"]:
        if key not in config:
            raise KeyError(f"Missing key '{key}' in config file")
    return config["img_h"], config["img_w"]


def find_best_size(original_width: int, original_height: int, patch_size: int = 28, target_size: int = 196):
    original_ratio = original_width / original_height
    best_pair = (1, target_size)
    best_score = float("inf")
    for a in range(1, int(math.sqrt(target_size)) + 1):
        if target_size % a != 0:
            continue
        for w_factor, h_factor in ((a, target_size // a), (target_size // a, a)):
            candidate_ratio = w_factor / h_factor
            score = abs(math.log(candidate_ratio / original_ratio))
            if score < best_score:
                best_score = score
                best_pair = (w_factor, h_factor)
    return best_pair[0] * patch_size, best_pair[1] * patch_size


def main():
    model = PaddleOCRVLForConditionalGeneration.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map=device,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print("model device:", model.device)
    print("model.model device:", model.model.device)

    image = Image.open(TEST_IMAGE).convert("RGB")
    old_w, old_h = image.size
    img_h, img_w = load_config(VISION_CONFIG)
    new_w, new_h = find_best_size(
        old_w,
        old_h,
        target_size=img_h // PATCH_SIZE // 2 * img_w // PATCH_SIZE // 2,
    )
    print(f"image size: src({old_w},{old_h}) -> new({new_w},{new_h})")
    image = image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    if SAVE_IMAGE is not None:
        image.save(SAVE_IMAGE)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": query}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        images_kwargs={
            "size": {
                "shortest_edge": processor.image_processor.min_pixels,
                "longest_edge": 1280 * 28 * 28,
            }
        },
    ).to(device)
    print("input_ids:", inputs["input_ids"])
    print("input_prompt", tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False))
    print("pixel_values:", inputs["pixel_values"].shape, inputs["pixel_values"].device)
    print("attention_mask:", inputs["attention_mask"].shape, inputs["attention_mask"].device)
    print("image_grid_thw:", inputs["image_grid_thw"], inputs["image_grid_thw"].device)
    start_time = time.time()
    with torch.inference_mode():
        generate_ids = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=1024)
    end_time = time.time()
    print("inference time:", end_time - start_time)
    print(generate_ids.shape)
    input_token_len = inputs["input_ids"].shape[-1]
    output_ids = generate_ids[0][input_token_len:]
    decoded_output = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(decoded_output)


if __name__ == "__main__":
    main()

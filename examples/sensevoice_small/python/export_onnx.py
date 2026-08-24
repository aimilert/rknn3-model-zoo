#!/usr/bin/env python3
"""
Modified from https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/sense-voice/export-onnx.py
"""

import os
from typing import Any, Dict, Tuple

import onnx
import torch
import numpy as np
from model import SenseVoiceSmall
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def add_meta_data(filename: str, meta_data: Dict[str, Any]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)
    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, filename)


def onnx_forward(
    self,
    x: torch.Tensor,
    x_masks: torch.Tensor
):

    encoder_out, encoder_out_lens = self.encoder.onnx_forward(x, x_masks)
    if isinstance(encoder_out, tuple):
        encoder_out = encoder_out[0]

    # return encoder_out
    ctc_logits = self.ctc.ctc_lo(encoder_out)

    return ctc_logits

def load_cmvn(filename) -> Tuple[str, str]:
    neg_mean = None
    inv_stddev = None

    with open(filename) as f:
        for line in f:
            if not line.startswith("<LearnRateCoef>"):
                continue
            t = line.split()[3:-1]

            if neg_mean is None:
                neg_mean = ",".join(t)
            else:
                inv_stddev = ",".join(t)

    return neg_mean, inv_stddev


def generate_tokens(params, path):
    sp = params["tokenizer"].sp
    with open(path, "w", encoding="utf-8") as f:
        for i in range(sp.vocab_size()):
            f.write(f"{sp.id_to_piece(i)} {i}\n")

    print(f"generate tokens to: {path}")
    # os.system("head tokens.txt; tail -n200 tokens.txt")


def display_params(params):
    print("----------params----------")
    print(params)

    print("----------frontend_conf----------")
    print(params["frontend_conf"])

    os.system(f"cat {params['frontend_conf']['cmvn_file']}")

    print("----------config----------")
    print(params["config"])

    os.system(f"cat {params['config']}")


def main():
    model, params = SenseVoiceSmall.from_pretrained(model="iic/SenseVoiceSmall")
    # display_params(params)

    os.makedirs("../model", exist_ok=True)
    # 生成token文件
    generate_tokens(params, "../model/tokens.txt")

    model.__class__.forward = onnx_forward

    # embed单独处理
    embedding_matrix = model.embed.weight.detach().cpu().numpy()
    np.save("../model/embedding_matrix.npy", embedding_matrix)
    np.savetxt("../model/embedding_matrix.txt", embedding_matrix.flatten(), fmt="%.9f")
    print("embedding saved to ../model/embedding_matrix.txt and ../model/embedding_matrix.npy")

    # 输入10s音频对应的shape
    x = torch.randn(1, 170, 560, dtype=torch.float32)
    x_masks = torch.ones((1, 170), dtype=torch.bool)

    opset_version = 13
    filename = "../model/sensevoice_fix_10s.onnx"

    torch.onnx.export(
        model,
        (x,x_masks),
        filename,
        opset_version=opset_version,
        input_names=["x", "x_masks"],
        output_names=["logits"],
    )

    lfr_window_size = params["frontend_conf"]["lfr_m"]
    lfr_window_shift = params["frontend_conf"]["lfr_n"]

    neg_mean, inv_stddev = load_cmvn(params["frontend_conf"]["cmvn_file"])
    vocab_size = params["tokenizer"].sp.vocab_size()

    meta_data = {
        "lfr_window_size": lfr_window_size,
        "lfr_window_shift": lfr_window_shift,
        "normalize_samples": 0,  # input should be in the range [-32768, 32767]
        "neg_mean": neg_mean,
        "inv_stddev": inv_stddev,
        "model_type": "sense_voice_ctc",
        # version 1: Use QInt8
        # version 2: Use QUInt8
        "version": "2",
        "model_author": "iic",
        "maintainer": "k2-fsa",
        "vocab_size": vocab_size,
        "comment": "iic/SenseVoiceSmall",
        "lang_auto": model.lid_dict["auto"],
        "lang_zh": model.lid_dict["zh"],
        "lang_en": model.lid_dict["en"],
        "lang_yue": model.lid_dict["yue"],  # cantonese
        "lang_ja": model.lid_dict["ja"],
        "lang_ko": model.lid_dict["ko"],
        "lang_nospeech": model.lid_dict["nospeech"],
        "with_itn": model.textnorm_dict["withitn"],
        "without_itn": model.textnorm_dict["woitn"],
        "url": "https://huggingface.co/FunAudioLLM/SenseVoiceSmall",
    }
    add_meta_data(filename=filename, meta_data=meta_data)
    print(f"export onnx to: {filename}")


if __name__ == "__main__":
    torch.manual_seed(20240717)
    main()

#!/usr/bin/env python3
"""
Modified from https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/sense-voice/test.py
"""

import argparse
from typing import Tuple

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime
import onnxruntime as ort
import soundfile as sf
import torch
import torch.nn.functional as F

class OnnxModel:
    def __init__(self, filename):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.session_opts = session_opts

        self.model = ort.InferenceSession(
            filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        meta = self.model.get_modelmeta().custom_metadata_map

        self.window_size = int(meta["lfr_window_size"])  # lfr_m
        self.window_shift = int(meta["lfr_window_shift"])  # lfr_n

        lang_zh = int(meta["lang_zh"])
        lang_en = int(meta["lang_en"])
        lang_ja = int(meta["lang_ja"])
        lang_ko = int(meta["lang_ko"])
        lang_auto = int(meta["lang_auto"])

        self.lang_id = {
            "zh": lang_zh,
            "en": lang_en,
            "ja": lang_ja,
            "ko": lang_ko,
            "auto": lang_auto,
        }
        self.with_itn = int(meta["with_itn"])
        self.without_itn = int(meta["without_itn"])

        neg_mean = meta["neg_mean"].split(",")
        neg_mean = list(map(lambda x: float(x), neg_mean))

        inv_stddev = meta["inv_stddev"].split(",")
        inv_stddev = list(map(lambda x: float(x), inv_stddev))

        self.neg_mean = np.array(neg_mean, dtype=np.float32)
        self.inv_stddev = np.array(inv_stddev, dtype=np.float32)

    def __call__(self, x, x_masks):
        logits = self.model.run(
            [
                self.model.get_outputs()[0].name,
            ],
            {
                self.model.get_inputs()[0].name: x.numpy(),
                self.model.get_inputs()[1].name: x_masks.numpy(),
            },
        )[0]

        return torch.from_numpy(logits)

# 读取音频文件
def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)
    return samples, sample_rate

# 加载token文件（词表）
def load_tokens(filename):
    ans = dict()
    i = 0
    with open(filename, encoding="utf-8") as f:
        for line in f:
            ans[i] = line.strip().split()[0]
            i += 1
    return ans


#音频特征提取
def compute_feat(
    samples,
    sample_rate,
    neg_mean: np.ndarray,
    inv_stddev: np.ndarray,
    window_size: int = 7,  # lfr_m
    window_shift: int = 6,  # lfr_n
):
    # 配置Kaldi风格fbank提取器
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 1.0
    opts.frame_opts.snip_edges = True
    opts.frame_opts.window_type = "hamming"
    opts.frame_opts.samp_freq = sample_rate
    opts.mel_opts.num_bins = 80

    online_fbank = knf.OnlineFbank(opts)
    online_fbank.accept_waveform(sample_rate, (samples * 32768).tolist())
    online_fbank.input_finished()

    features = np.stack(
        [online_fbank.get_frame(i) for i in range(online_fbank.num_frames_ready)]
    )
    assert features.data.contiguous is True
    assert features.dtype == np.float32, features.dtype

    T = (features.shape[0] - window_size) // window_shift + 1
    # lfr 下采样
    features = np.lib.stride_tricks.as_strided(
        features,
        shape=(T, features.shape[1] * window_size),
        strides=((window_shift * features.shape[1]) * 4, 4),
    )

    # 特征标准化
    features = (features + neg_mean) * inv_stddev

    return features


def main():

    parser = argparse.ArgumentParser(description='SenseVoice ONNX 模型推理测试')
    parser.add_argument('--audio_path', type=str, required=True, help='输入音频文件路径')
    parser.add_argument('--tokens_path', type=str, default='../model/tokens.txt', help='词表文件路径')
    parser.add_argument('--embedding_matrix_path', type=str, default='../model/embedding_matrix.npy', help='嵌入矩阵文件路径')
    parser.add_argument('--onnx_model_path', type=str, default='../model/sensevoice_fix_10s.onnx', help='ONNX模型文件路径')
    args = parser.parse_args()

    wave = args.audio_path
    tokens_path = args.tokens_path
    embedding_matrix_path = args.embedding_matrix_path
    onnx_model_path = args.onnx_model_path

    samples, sample_rate = load_audio(wave)
    if sample_rate != 16000:
        import librosa
        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    
    target_samples_num = 10 * sample_rate  

    # 如果音频比目标长度短，末尾补零；如果比目标长，截断
    if len(samples) < target_samples_num:
        pad_length = target_samples_num - len(samples)
        samples = np.pad(samples, (0, pad_length), mode='constant')
    elif len(samples) > target_samples_num:
        samples = samples[:target_samples_num] 

    model = OnnxModel(onnx_model_path)

    # 提取音频特征
    features = compute_feat(
        samples=samples,
        sample_rate=sample_rate,
        neg_mean=model.neg_mean,
        inv_stddev=model.inv_stddev,
        window_size=model.window_size,
        window_shift=model.window_shift,
    )

    features = torch.from_numpy(features).unsqueeze(0)
    language = model.lang_id["auto"]

    use_itn = True
    if use_itn:
        text_norm = model.with_itn
    else:
        text_norm = model.without_itn

    # 4个特殊token （language, event、emo, text_norm）对应的embedding
    language = torch.tensor([language], dtype=torch.int32)
    text_norm = torch.tensor([text_norm], dtype=torch.int32)

    embedding_matrix = np.load(embedding_matrix_path)
    embedding_matrix = torch.from_numpy(embedding_matrix)
    language_query = embedding_matrix[language].unsqueeze(1) 
    event_emo_query = torch.stack((embedding_matrix[1], embedding_matrix[2])).unsqueeze(0) 
    text_norm_query = embedding_matrix[text_norm].unsqueeze(1) 
    
    # 拼接完整输入：[LID, SER, AEC, ITN] + features
    x = torch.cat((language_query, event_emo_query, text_norm_query, features), dim=1)
    x_masks = torch.ones((1, x.shape[1]), dtype=torch.bool)

    logits = model(
        x=x,
        x_masks=x_masks,
    )

    # CTC 解码：取最大概率索引，去除重复和 blank
    idx = logits.squeeze(0).argmax(dim=-1)
    idx = torch.unique_consecutive(idx)
    blank_id = 0
    idx = idx[idx != blank_id].tolist()
    tokens = load_tokens(tokens_path)
    text = "".join([tokens[i] for i in idx])

    text = text.replace("▁", " ")
    print(text)


if __name__ == "__main__":
    main()

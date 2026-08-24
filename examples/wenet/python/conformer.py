"""Wenet Conformer: ONNX → RKNN conversion + streaming inference demo.

Workflow:
  1. Convert encoder.onnx / ctc.onnx / decoder.onnx → *.rknn
  2. Load RKNN models, init runtime on device
  3. Pre-process:  wav → 80-dim fbank (CMVN is built into encoder)
  4. Streaming inference: chunk-by-chunk encoder + CTC
  5. Post-process:  CTC prefix beam search + attention rescoring → text

Usage:
  # Full pipeline (convert + infer)
  python conformer.py \\
      --onnx_dir ori_onnx_models \\
      --wav test.wav \\
      --units 20210601_u2++_conformer_exp_aishell/units.txt \\
      --device_id xxxx

  # Convert only (skip inference)
  python conformer.py \\
      --onnx_dir ori_onnx_models \\
      --convert_only
"""

import argparse
import math
import os
import sys
import numpy as np

from rknn.api import RKNN

try:
    import torchaudio
    import torchaudio.compliance.kaldi as kaldi
except ImportError:
    sys.exit("Please install torchaudio")



MODEL_DEFAULTS = {
    "output_size": 256,
    "num_blocks": 12,
    "head": 4,
    "cnn_module_kernel": 8,
    "vocab_size": 4233,
    "subsampling_rate": 4,
    "right_context": 6,
}

ENCODER_OUT_LEN = 200  # fixed encoder output length for decoder input
LEFT_CHUNKS = 4
MAX_HYP_LEN = 13 

RKNN_CONFIG = {
    "platform": "rk1820",
    "core_num": 8,
    "quantization": False,
    "device_id": "172.16.10.74:35882",
    "core_mask" : 0xff
}


# ============================================================================
# RKNN conversion helpers
# ============================================================================

def convert_onnx_to_rknn(onnx_path, rknn_path):
    """Convert a single ONNX model to RKNN. Returns the RKNN object (not released)."""
    rknn = RKNN(verbose=True)
    if "encoder" in onnx_path:
        rknn.config(
            target_platform=RKNN_CONFIG["platform"],
            core_num=RKNN_CONFIG["core_num"],
            input_attrs={
                'att_cache': {'dtype': 'float16', 'layout': 'NCHW'},
                'cnn_cache': {'dtype': 'float16', 'layout': 'NCHW'}
                }
        )
    else:
        rknn.config(
            target_platform=RKNN_CONFIG["platform"],
            core_num=RKNN_CONFIG["core_num"],
        )

    ret = rknn.load_onnx(model=onnx_path) 


    if ret != 0:
        print(f"[ERROR] load_onnx {onnx_path} failed: {ret}")
        rknn.release()
        return None

    ret = rknn.build(do_quantization=RKNN_CONFIG["quantization"])
    if ret != 0:
        print(f"[ERROR] build {onnx_path} failed: {ret}")
        rknn.release()
        return None

    ret = rknn.export_rknn(rknn_path, save_ctx=True)
    if ret != 0:
        print(f"[ERROR] export {rknn_path} failed: {ret}")
        rknn.release()
        return None

    print(f"[OK] {onnx_path} -> {rknn_path}")
    return rknn


def convert_all(onnx_dir, rknn_dir):
    """Convert encoder / ctc / decoder ONNX models to RKNN.

    Returns dict {name: rknn_object} for successfully converted models.
    Caller is responsible for calling .release() on each object.
    """
    os.makedirs(rknn_dir, exist_ok=True)
    models = ["encoder", "ctc", "decoder"]
    rknn_objs = {}
    for name in models:
        onnx_path = os.path.join(onnx_dir, f"{name}.onnx")
        rknn_path = os.path.join(rknn_dir, f"{name}.rknn")
        if not os.path.isfile(onnx_path):
            print(f"[WARN] {onnx_path} not found, skip")
            continue
        if os.path.isfile(rknn_path):
            print(f"[SKIP] {rknn_path} already exists")
            continue
        rknn = convert_onnx_to_rknn(onnx_path, rknn_path)
        if rknn is not None:
            rknn_objs[name] = rknn
    return rknn_objs


# ============================================================================
# Feature extraction  
# ============================================================================

def compute_fbank(wav_path, num_mel_bins=80):
    """Compute 80-dim fbank. Returns (T, 80) float32."""
    wav, sr = torchaudio.load(wav_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav * (1 << 15)
    feat = kaldi.fbank(
        wav,
        num_mel_bins=num_mel_bins,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        window_type="povey",
    ) 
    return feat.numpy().astype(np.float32)


# ============================================================================
# Token table
# ============================================================================

def load_units(path):
    table = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                table[int(parts[1])] = parts[0]
    return table


# ============================================================================
# CTC decoding  
# ============================================================================

def _log_add(a, b):
    if a == -float("inf"):
        return b
    if b == -float("inf"):
        return a
    mx = max(a, b)
    return mx + math.log(math.exp(a - mx) + math.exp(b - mx))


def ctc_greedy_search(probs, blank=0):
    """Greedy CTC decode for partial display."""
    tokens, prev = [], blank
    for t in range(probs.shape[0]):
        idx = int(np.argmax(probs[t]))
        if idx != blank and idx != prev:
            tokens.append(idx)
        prev = idx
    return tokens


def ctc_prefix_beam_search(probs, beam_size, blank=0):
    """CTC prefix beam search.
    Returns list of (token_ids, log_score) sorted best-first.
    """
    T, V = probs.shape
    # cur: prefix_tuple -> (logP_blank, logP_nonblank)
    cur = {(): (0.0, -float("inf"))}

    for t in range(T):
        logp = probs[t]
        nxt = {}
        topk = np.argsort(logp)[-beam_size:]

        for u in topk:
            p = float(logp[u])
            for prefix, (s, ns) in cur.items():
                last = prefix[-1] if prefix else None

                if u == blank:
                    old_s, old_ns = nxt.get(
                        prefix, (-float("inf"), -float("inf")))
                    nxt[prefix] = (
                        _log_add(old_s, _log_add(s, ns) + p), old_ns)

                elif u == last:
                    # *uu -> *u  (merge repeat)
                    old_s, old_ns = nxt.get(
                        prefix, (-float("inf"), -float("inf")))
                    nxt[prefix] = (old_s, _log_add(old_ns, ns + p))
                    # *u-u -> *uu  (new repeat from blank path)
                    new_pf = prefix + (u,)
                    o2s, o2ns = nxt.get(
                        new_pf, (-float("inf"), -float("inf")))
                    nxt[new_pf] = (o2s, _log_add(o2ns, s + p))

                else:
                    new_pf = prefix + (u,)
                    o2s, o2ns = nxt.get(
                        new_pf, (-float("inf"), -float("inf")))
                    nxt[new_pf] = (
                        o2s, _log_add(o2ns, _log_add(s, ns) + p))

        cur = dict(
            sorted(nxt.items(),
                   key=lambda x: _log_add(x[1][0], x[1][1]),
                   reverse=True)[:beam_size])

    return [(list(pf), _log_add(s, ns)) for pf, (s, ns) in cur.items()]


# ============================================================================
# Attention rescoring 
# ============================================================================

def attention_rescoring(dec_rknn, nbest, all_enc, sos, eos,
                        reverse_weight, ctc_weight, enc_len=None):
    """Attention rescoring with fixed-shape encoder output.

    all_enc: (1, ENCODER_OUT_LEN, 256) zero-padded to fixed length.
    enc_len: actual (un-padded) encoder time steps; scoring only uses
             positions within this range.
    """
    N = len(nbest)
    if N == 0:
        return []

    L = MAX_HYP_LEN

    hyps = np.full((N, L), eos, dtype=np.int64)
    hyps[:, 0] = sos
    for i, (tok, _) in enumerate(nbest):
        tok_len = min(len(tok), L - 1)
        hyps[i, 1:tok_len + 1] = tok[:tok_len]

    inputs = [hyps, all_enc.copy()]
    outputs = dec_rknn.inference(inputs=inputs, data_format='nchw')
    scores = np.log(np.array(outputs[0], copy=True)).astype(np.float32)       # (N, L, V)
    r_scores = np.log(np.array(outputs[1], copy=True)).astype(np.float32) if len(outputs) > 1 else None

    best_sc, best_i = -float("inf"), 0
    for i, (tok, ctc_sc) in enumerate(nbest):
        tok_len = min(len(tok), L - 1)
        sc = 0.0
        for j in range(tok_len):
            sc += float(scores[i][j][tok[j]])
        sc += float(scores[i][tok_len][eos])
        if reverse_weight > 0 and r_scores is not None:
            rsc = 0.0
            for j in range(tok_len):
                rsc += float(r_scores[i][tok_len - j - 1][tok[j]])
            rsc += float(r_scores[i][tok_len][eos])
            sc = sc * (1.0 - reverse_weight) + rsc * reverse_weight
        sc += ctc_sc * ctc_weight
        if sc > best_sc:
            best_sc, best_i = sc, i
    return nbest[best_i][0]


# ============================================================================
# RKNN streaming ASR
# ============================================================================

class RknnAsr:
    """Streaming ASR using RKNN models on Rockchip NPU."""

    def __init__(self, rknn_dir, units_path, device_id,
                 chunk_size=16, left_chunks=-1,
                 beam_size=10, reverse_weight=0.3, ctc_weight=0.3,
                 rknn_objs=None, use_runtime=False, platform=None):
        self.chunk_size = chunk_size
        self.left_chunks = left_chunks
        self.beam_size = beam_size
        self.reverse_weight = reverse_weight
        self.ctc_weight = ctc_weight
        self.blank = 0
        self.sos = 2
        self.eos = 2

        # Model params (use defaults matching the exported ONNX)
        self.output_size = MODEL_DEFAULTS["output_size"]
        self.num_blocks = MODEL_DEFAULTS["num_blocks"]
        self.head = MODEL_DEFAULTS["head"]
        self.cnn_module_kernel = MODEL_DEFAULTS["cnn_module_kernel"]
        self.vocab_size = MODEL_DEFAULTS["vocab_size"]
        self.subsampling_rate = MODEL_DEFAULTS["subsampling_rate"]
        self.right_context = MODEL_DEFAULTS["right_context"]

        self.decoding_window = (
            (chunk_size - 1) * self.subsampling_rate + self.right_context + 1
        )
        self.stride = chunk_size * self.subsampling_rate

        self.id_to_token = load_units(units_path)

        self._own_rknn = rknn_objs is None

        # ---- Load or reuse RKNN models ----
        if platform is None:
            platform = RKNN_CONFIG["platform"]
        m_core_mask = RKNN_CONFIG["core_mask"]

        def _init_rt(rknn_obj, model):
            if use_runtime:
                ret = rknn_obj.init_runtime(target=platform, device_id=device_id, core_mask=m_core_mask)
            else:
                ret = rknn_obj.init_runtime()
            assert ret == 0, f"init {model} runtime failed: {ret}"

        if rknn_objs:
            # Reuse RKNN objects from convert_all (already in memory)
            self.enc_rknn = rknn_objs.get("encoder")
            self.ctc_rknn = rknn_objs.get("ctc")
            self.dec_rknn = rknn_objs.get("decoder")
            assert self.enc_rknn is not None, "encoder RKNN object not found"
            assert self.ctc_rknn is not None, "ctc RKNN object not found"
            assert self.dec_rknn is not None, "decoder RKNN object not found"

            _init_rt(self.enc_rknn, "encoder")
            _init_rt(self.ctc_rknn, "ctc")
            _init_rt(self.dec_rknn, "decoder")
        else:
            # Fallback: load from rknn
            self.enc_rknn = RKNN()
            ret = self.enc_rknn.load_rknn(model_path=os.path.join(rknn_dir, "encoder.rknn"),
                                          weight_path=os.path.join(rknn_dir, "encoder.weight"), load_ctx=True)
            assert ret == 0, f"load encoder.rknn failed: {ret}"
            _init_rt(self.enc_rknn, "encoder")

            self.ctc_rknn = RKNN()
            ret = self.ctc_rknn.load_rknn(model_path=os.path.join(rknn_dir, "ctc.rknn"),
                                          weight_path=os.path.join(rknn_dir, "ctc.weight"), load_ctx=True)
            assert ret == 0, f"load ctc.rknn failed: {ret}"
            _init_rt(self.ctc_rknn, "ctc")

            self.dec_rknn = RKNN()
            ret = self.dec_rknn.load_rknn(model_path=os.path.join(rknn_dir, "decoder.rknn"),
                                          weight_path=os.path.join(rknn_dir, "decoder.weight"), load_ctx=True)
            assert ret == 0, f"load decoder.rknn failed: {ret}"
            _init_rt(self.dec_rknn, "decoder")

        print("[INFO] RKNN models loaded. decoding_window={}, stride={}".format(
            self.decoding_window, self.stride))

    # ------------------------------------------------------------------
    # Encoder (single chunk)
    # ------------------------------------------------------------------

    def _run_encoder(self, chunk_feat, att_cache, cnn_cache):
        """Run encoder RKNN on one chunk.

        RKNN encoder inputs (static shapes, offset/required_cache_size/att_mask removed):
          [0] chunk:      (1, 67, 80)
          [1] att_cache:  (12, 4, 64, 128)
          [2] cnn_cache:  (12, 1, 256, 7)
        Outputs:
          [0] enc_out:     (1, 16, 256)
          [1] r_att_cache: (12, 4, 64, 128)
          [2] r_cnn_cache: (12, 1, 256, 7)
        """
        chunk = chunk_feat[np.newaxis].astype(np.float32)
        inputs = [chunk, att_cache.copy(), cnn_cache.copy()]
        outputs = self.enc_rknn.inference(inputs=inputs, data_format='nchw')
        enc_out = np.array(outputs[0], copy=True)
        r_att_cache = np.array(outputs[1], copy=True)
        r_cnn_cache = np.array(outputs[2], copy=True)
        return enc_out, r_att_cache, r_cnn_cache

    # ------------------------------------------------------------------
    # CTC
    # ------------------------------------------------------------------

    def _run_ctc(self, enc_out):
        """Run CTC RKNN. Returns (T', V) log probs in float32 to match C++."""
        outputs = self.ctc_rknn.inference(inputs=[enc_out.copy()], data_format='nchw')
        # Model outputs softmax; convert to log-space in float32 to match C++
        return np.log(np.array(outputs[0], copy=True, dtype=np.float32).squeeze(0)).astype(np.float32)

    # ------------------------------------------------------------------
    # Decode: CTC beam search + attention rescoring
    # ------------------------------------------------------------------

    def _decode(self, all_enc, all_ctc, enc_len=None):
        nbest = ctc_prefix_beam_search(all_ctc, self.beam_size, self.blank)
        if not nbest:
            return ""
        for rank, (tok, sc) in enumerate(nbest[:3]):
            txt = "".join(self.id_to_token.get(t, "?") for t in tok)
            print("  CTC nbest[{}]: score={:.2f}  {}".format(rank, sc, txt))
        token_ids = attention_rescoring(
            self.dec_rknn, nbest, all_enc,
            self.sos, self.eos, self.reverse_weight, self.ctc_weight,
            enc_len=enc_len)
        return "".join(self.id_to_token.get(t, "<unk>") for t in token_ids)

    # ------------------------------------------------------------------
    # Streaming recognize
    # ------------------------------------------------------------------

    def recognize(self, wav_path):
        """Chunk-by-chunk streaming ASR via RKNN on device."""
        # -- Init caches (static shapes, zeros) --
        att_cache = np.zeros(
            (self.num_blocks, self.head, LEFT_CHUNKS * self.chunk_size,
             self.output_size // self.head * 2), dtype=np.float32)
        cnn_cache = np.zeros(
            (self.num_blocks, 1, self.output_size,
             self.cnn_module_kernel - 1), dtype=np.float32)

        enc_outs = []
        ctc_outs = []

        # -- Pre-process: wav -> fbank --
        feat = compute_fbank(wav_path)
        T = feat.shape[0]
        print("fbank frames: {}".format(T))

        # Pad tail
        feat = np.concatenate([feat, np.zeros((self.decoding_window, 80), dtype=np.float32)])

        num_chunks = max(1, (T - self.decoding_window) // self.stride + 1)
        print("[STREAMING] {} frames => ~{} chunks".format(T, num_chunks))

        cid = 0
        while True:
            start = cid * self.stride
            if start >= T:
                break
            end = start + self.decoding_window

            enc_out, att_cache, cnn_cache = self._run_encoder(
                feat[start:end], att_cache, cnn_cache)
            T_enc = enc_out.shape[1]

            ctc_out = self._run_ctc(enc_out)
            enc_outs.append(enc_out)
            ctc_outs.append(ctc_out)

            partial_ids = ctc_greedy_search(ctc_out, self.blank)
            partial_txt = "".join(
                self.id_to_token.get(t, "<unk>") for t in partial_ids)
            print("  chunk {:>3d}: fbank [{},{}), enc {} frames, "
                  "partial: {}".format(
                      cid, start, end, T_enc,
                      partial_txt if partial_txt else "(silence)"))
            cid += 1

        # -- Post-process: CTC beam search + attention rescoring --
        all_enc = np.concatenate(enc_outs, axis=1)  # (1, T_actual, 256)
        all_ctc = np.concatenate(ctc_outs, axis=0)
        T_enc_actual = all_enc.shape[1]
        print("[INFO] encoder output total length: {}".format(T_enc_actual))

        if T_enc_actual < ENCODER_OUT_LEN:
            pad_len = ENCODER_OUT_LEN - T_enc_actual
            all_enc = np.concatenate(
                [all_enc,
                 np.zeros((1, pad_len, self.output_size), dtype=np.float32)],
                axis=1)
        elif T_enc_actual > ENCODER_OUT_LEN:
            print("[WARN] encoder output {} > fixed length {}, truncating".format(
                T_enc_actual, ENCODER_OUT_LEN))
            all_enc = all_enc[:, :ENCODER_OUT_LEN, :]
            T_enc_actual = ENCODER_OUT_LEN

        return self._decode(all_enc, all_ctc, enc_len=T_enc_actual)

    # ------------------------------------------------------------------
    def release(self):
        self.enc_rknn.release()
        self.ctc_rknn.release()
        self.dec_rknn.release()


# ============================================================================
# CLI
# ============================================================================

def get_args():
    p = argparse.ArgumentParser(
        description="Wenet Conformer ONNX->RKNN + streaming inference")
    p.add_argument("--onnx_dir", default="ori_onnx_models",
                   help="Directory with encoder/ctc/decoder.onnx")
    p.add_argument("--rknn_dir", default="rknn_models",
                   help="Output dir for *.rknn models")
    p.add_argument("--wav", default=None,
                   help="Input wav (16kHz mono) for inference")
    p.add_argument("--units", default=None,
                   help="Path to units.txt")
    p.add_argument("--convert_only", action="store_true",
                   help="Only convert, skip inference")
    p.add_argument("--use_runtime", action="store_true",
                   help="Run on device with target platform (default: simulator only)")
    p.add_argument("--device_id", default=RKNN_CONFIG["device_id"],
                   help="RKNN device id (e.g., 172.16.10.74:35882)")
    p.add_argument("--platform", default=RKNN_CONFIG["platform"],
                   help="RKNN target platform (e.g., rk1820, rk1828)")
    p.add_argument("--chunk_size", type=int, default=16)
    p.add_argument("--left_chunks", type=int, default=LEFT_CHUNKS)
    p.add_argument("--beam_size", type=int, default=10)
    p.add_argument("--reverse_weight", type=float, default=0.3)
    p.add_argument("--ctc_weight", type=float, default=0.3)
    return p.parse_args()


def main():
    args = get_args()

    # Update RKNN_CONFIG with provided platform
    RKNN_CONFIG["platform"] = args.platform
    print("[INFO] Using platform: {}".format(args.platform))

    # ---- Step 1: Convert ONNX -> RKNN ----
    print("=" * 60)
    print("[Step 1] Convert ONNX -> RKNN")
    print("=" * 60)
    rknn_objs = convert_all(args.onnx_dir, args.rknn_dir)

    if args.convert_only:
        # Release all RKNN objects after convert-only mode
        for _, rknn in rknn_objs.items():
            rknn.release()
        print("\n[convert_only] Done. RKNN models in {}".format(args.rknn_dir))
        return

    if not args.wav or not args.units:
        # Release before exit
        for _, rknn in rknn_objs.items():
            rknn.release()
        sys.exit("Need --wav and --units for inference (or --convert_only)")

    # ---- Step 2: RKNN streaming inference on device ----
    print("\n" + "=" * 60)
    print("[Step 2] RKNN streaming inference on device")
    print("=" * 60)

    asr = RknnAsr(
        rknn_dir=args.rknn_dir,
        units_path=args.units,
        device_id=args.device_id,
        chunk_size=args.chunk_size,
        left_chunks=args.left_chunks,
        beam_size=args.beam_size,
        reverse_weight=args.reverse_weight,
        ctc_weight=args.ctc_weight,
        rknn_objs=rknn_objs,
        use_runtime=args.use_runtime,
        platform=args.platform,
    )

    text = asr.recognize(args.wav)
    asr.release()

    print("\n" + "=" * 60)
    print("Result: {}".format(text))
    print("=" * 60)


if __name__ == "__main__":
    main()

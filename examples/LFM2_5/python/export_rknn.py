#!/usr/bin/env python3
"""Convert LFM2.5-1.2B-Instruct Conv Streaming ONNX to RKNN.

技术路线：
  - 标准 4 输入 convstream.onnx（Conv pads 已改为 [2,0]）
  - 自动收集 10 个 LFM2 ShortConv 输出
  - 传给 rknn.config(cvt_conv_streaming=...)
  - 不使用 input_initial_value / dynamic_input
  - W4A16 量化 (quantized_dtype="w4a16", quantized_algorithm="normal",
    quantized_method="group32")
  - Runtime 内部维护 Conv Cache
"""
import sys
import json
import copy
import argparse
import traceback
from pathlib import Path

from rknn.api import RKNN, DEFAULT_RKNN_LLM_CONFIG

_SCRIPT_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _SCRIPT_DIR.parent / "model"
OUTPUT_BASENAME = "LFM2.5-1.2B-Instruct"

DEFAULT_ONNX = _MODEL_DIR / "LFM2.5-1.2B-Instruct-convstream.onnx"
DEFAULT_CONFIG = _MODEL_DIR / f"{OUTPUT_BASENAME}.config.pkl"
DEFAULT_RKNN = _MODEL_DIR / "LFM2.5-1.2B-Instruct-convstream.rknn"

EXPECTED_CONV_COUNT = 10
EXPECTED_INPUT_NAMES = {"input_ids", "attention_mask", "position_ids", "num_logits_to_keep"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _onnx_io(model_proto):
    inputs = [inp.name for inp in model_proto.graph.input]
    outputs = [out.name for out in model_proto.graph.output]
    return inputs, outputs


def print_onnx_io(onnx_path: str):
    """打印 ONNX 输入输出（辅助信息）。"""
    try:
        import onnx
        model_proto = onnx.load(onnx_path, load_external_data=False)
        inputs, outputs = _onnx_io(model_proto)
        print(f"  ONNX inputs ({len(inputs)}): {inputs}")
        for inp in model_proto.graph.input:
            shape = [d.dim_param or d.dim_value for d in inp.type.tensor_type.shape.dim]
            print(f"    {inp.name}: shape={shape}")
        print(f"  ONNX outputs ({len(outputs)}): {outputs}")
        for out in model_proto.graph.output:
            shape = [d.dim_param or d.dim_value for d in out.type.tensor_type.shape.dim]
            print(f"    {out.name}: shape={shape}")
    except Exception as e:
        print(f"  Cannot load ONNX: {e}")


def collect_lfm2_streaming_conv_outputs(onnx_path: str):
    """从 convstream.onnx 中收集 10 个 LFM2 ShortConv 的输出 tensor 名。

    校验：
      - 恰好 10 个目标 Conv
      - kernel_shape=[3], dilations=[1], strides=[1], group=2048, pads=[2,0]
      - 输出名无重复

    任意校验失败 -> sys.exit(1)。
    """
    import onnx

    print(f"--> Collecting LFM2 streaming Conv outputs from: {onnx_path}")
    try:
        model_proto = onnx.load(onnx_path, load_external_data=False)
    except Exception as e:
        print(f"[ERROR] Cannot load ONNX for Conv collection: {e}")
        sys.exit(1)

    # 先校验输入输出协议
    input_names, output_names = _onnx_io(model_proto)
    if len(input_names) != 4:
        print(f"[ERROR] ONNX must have 4 inputs, got {len(input_names)}: {input_names}")
        sys.exit(1)
    if len(output_names) != 1:
        print(f"[ERROR] ONNX must have 1 output, got {len(output_names)}: {output_names}")
        sys.exit(1)
    missing = EXPECTED_INPUT_NAMES - set(input_names)
    if missing:
        print(f"[ERROR] Missing required inputs: {sorted(missing)}")
        sys.exit(1)
    for name in input_names + output_names:
        low = name.lower()
        if any(k in low for k in ("past_key", "past_value", "present_key", "present_value")):
            print(f"[ERROR] Explicit KV Cache tensor found: {name}")
            sys.exit(1)
        if "conv_state" in low:
            print(f"[ERROR] Explicit Conv State tensor found: {name}")
            sys.exit(1)

    conv_nodes = [n for n in model_proto.graph.node if n.op_type == "Conv"]
    print(f"  Total Conv nodes: {len(conv_nodes)}")

    target_outputs = []
    for node in conv_nodes:
        output_name = node.output[0] if node.output else ""
        attrs = {}
        for attr in node.attribute:
            if attr.name == "pads":
                attrs["pads"] = list(attr.ints)
            elif attr.name == "kernel_shape":
                attrs["kernel_shape"] = list(attr.ints)
            elif attr.name == "dilations":
                attrs["dilations"] = list(attr.ints)
            elif attr.name == "strides":
                attrs["strides"] = list(attr.ints)
            elif attr.name == "group":
                attrs["group"] = attr.i

        kernel = attrs.get("kernel_shape", [])
        pads = attrs.get("pads", [])
        dilations = attrs.get("dilations", [])
        strides = attrs.get("strides", [])
        group = attrs.get("group", 0)

        if (kernel == [3] and dilations == [1] and strides == [1]
                and group == 2048 and pads == [2, 0]):
            target_outputs.append(output_name)
            print(f"  MATCH [{len(target_outputs)-1}]: {output_name}")
            print(f"    kernel={kernel} pads={pads} dilation={dilations} "
                  f"strides={strides} group={group}")
        else:
            print(f"  SKIP: {output_name} (kernel={kernel} pads={pads} group={group})")

    if len(target_outputs) != EXPECTED_CONV_COUNT:
        print(f"\n[ERROR] Expected {EXPECTED_CONV_COUNT} ShortConv with pads=[2,0], "
              f"found {len(target_outputs)}")
        print("All Conv nodes:")
        for node in conv_nodes:
            attrs = {}
            for attr in node.attribute:
                if attr.name == "pads":
                    attrs["pads"] = list(attr.ints)
                elif attr.name == "kernel_shape":
                    attrs["kernel_shape"] = list(attr.ints)
                elif attr.name == "group":
                    attrs["group"] = attr.i
            print(f"  {node.output[0]}: {attrs}")
        sys.exit(1)

    # 检查重复
    if len(set(target_outputs)) != len(target_outputs):
        print(f"[ERROR] Duplicate Conv output names in cvt_conv_streaming list: {target_outputs}")
        sys.exit(1)

    print(f"\n  Collected {len(target_outputs)} Conv streaming outputs (no duplicates)")
    return target_outputs


def _fail(stage: str, ret, onnx_path, config_path, rknn_path, cvt_count):
    """统一失败打印。"""
    print(f"\n[FAILED] {stage}")
    print(f"  return value      : {ret}")
    print(f"  ONNX path         : {onnx_path}")
    print(f"  config path       : {config_path}")
    print(f"  RKNN output path  : {rknn_path}")
    print(f"  quantized_dtype   : w4a16")
    print(f"  quantized_algorithm: normal")
    print(f"  quantized_method  : group32")
    print(f"  cvt_conv_streaming: {cvt_count} nodes")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Convert LFM2.5-1.2B-Instruct Conv Streaming ONNX to RKNN."
    )
    parser.add_argument("--onnx_path", type=str, default=str(DEFAULT_ONNX),
                        help="Conv streaming ONNX path (default: ../model/LFM2.5-1.2B-Instruct-convstream.onnx)")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                        help="LLM config.pkl path (default: ../model/LFM2.5-1.2B-Instruct.config.pkl)")
    parser.add_argument("--rknn_path", type=str, default=str(DEFAULT_RKNN),
                        help="Output RKNN model path (default: ../model/LFM2.5-1.2B-Instruct-convstream.rknn)")
    parser.add_argument("--seq_len", type=int, default=128,
                        help="Prefill seq_len for load_llm (default: 128, must be > 1)")
    parser.add_argument("--target_platform", type=str, default="rk1820",
                        choices=["rk1820"],
                        help="Target platform (locked to rk1820)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 参数校验
    # ------------------------------------------------------------------
    if args.seq_len <= 1:
        print(f"[ERROR] --seq_len must be > 1, got {args.seq_len}")
        sys.exit(1)

    onnx_p = Path(args.onnx_path)
    config_p = Path(args.config)
    rknn_p = Path(args.rknn_path)
    weight_p = rknn_p.with_suffix(".weight")
    manifest_p = rknn_p.with_suffix(".export.json")

    # ------------------------------------------------------------------
    # 输入文件严格检查
    # ------------------------------------------------------------------
    if not onnx_p.exists():
        print(f"[ERROR] ONNX model not found: {onnx_p}")
        sys.exit(1)
    if onnx_p.stat().st_size == 0:
        print(f"[ERROR] ONNX model is empty (0 bytes): {onnx_p}")
        sys.exit(1)
    if not config_p.exists():
        print(f"[ERROR] Config file not found: {config_p}")
        sys.exit(1)
    if config_p.stat().st_size == 0:
        print(f"[ERROR] Config file is empty (0 bytes): {config_p}")
        sys.exit(1)

    # ONNX 可加载性检查
    try:
        import onnx
        onnx.load(str(onnx_p), load_external_data=False)
    except Exception as e:
        print(f"[ERROR] Cannot load ONNX [{onnx_p}]: {e}")
        sys.exit(1)

    print_onnx_io(args.onnx_path)

    # 收集 Conv streaming 输出（含 4in/1out/10conv/pads=[2,0]/无重复 校验）
    cvt_conv_streaming = collect_lfm2_streaming_conv_outputs(args.onnx_path)

    # 清理旧产物，避免失败后误用
    for p in (rknn_p, weight_p, manifest_p):
        if p.exists():
            p.unlink()

    print(f"\n--> Conversion parameters:")
    print(f"    Body quantization       : w4a16 / normal / group32")
    print(f"    KV cache buffer length  : 4096")
    print(f"    Max position embeddings : 8192")
    print(f"    KV cache dtype          : Float16")
    print(f"    Decode seq_len          : 1")
    print(f"    Prefill seq_len         : {args.seq_len}")
    print(f"    Conv streaming nodes    : {EXPECTED_CONV_COUNT}")

    # ------------------------------------------------------------------
    # RKNN 转换
    # ------------------------------------------------------------------
    rknn = RKNN(verbose=True)
    try:
        # Config
        print("--> Configuring model")
        llm_config = copy.deepcopy(DEFAULT_RKNN_LLM_CONFIG)
        attn_cfg = llm_config["attention_config"][0]
        attn_cfg["kvcache_buffer_len"] = 4096
        attn_cfg["max_position_embeddings"] = 8192
        attn_cfg["kvcache_dtype"] = "Float16"

        ret = rknn.config(
            target_platform=args.target_platform,
            quantized_dtype="w4a16",
            quantized_algorithm="normal",
            quantized_method="group32",
            core_num=8,
            llm_config=llm_config,
            cvt_conv_streaming=cvt_conv_streaming,
        )
        if ret != 0:
            _fail("rknn.config()", ret, args.onnx_path, args.config, args.rknn_path,
                  len(cvt_conv_streaming))
        print("done")

        # Load LLM
        print("--> Loading LLM model")
        ret = rknn.load_llm(
            model=args.onnx_path,
            config=args.config,
            seq_lens=[1, args.seq_len],
        )
        if ret != 0:
            _fail("rknn.load_llm()", ret, args.onnx_path, args.config, args.rknn_path,
                  len(cvt_conv_streaming))
        print("done")

        # Build
        print("--> Building model (w4a16 quantization)")
        ret = rknn.build(
            do_quantization=True,
        )
        if ret != 0:
            _fail("rknn.build()", ret, args.onnx_path, args.config, args.rknn_path,
                  len(cvt_conv_streaming))
        print("done")

        # Export
        print("--> Exporting RKNN model")
        rknn_out_dir = rknn_p.parent
        if rknn_out_dir and not rknn_out_dir.exists():
            rknn_out_dir.mkdir(parents=True, exist_ok=True)

        ret = rknn.export_rknn(args.rknn_path, save_ctx=True)
        if ret != 0:
            _fail("rknn.export_rknn()", ret, args.onnx_path, args.config, args.rknn_path,
                  len(cvt_conv_streaming))
        print("done")

    except SystemExit:
        raise
    except Exception as e:
        print(f"\n[ERROR] Unexpected exception during RKNN conversion: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        rknn.release()

    # ------------------------------------------------------------------
    # 保存 manifest (body quantization only, no LLM Head config)
    # ------------------------------------------------------------------
    # KV Cache 配置从 llm_config 实际值读取，不重复硬编码
    attn_cfg_final = llm_config["attention_config"][0]
    manifest = {
        "model": OUTPUT_BASENAME,
        "target_platform": args.target_platform,
        "core_num": 8,
        "body_quantization": {
            "dtype": "w4a16",
            "algorithm": "normal",
            "method": "group32",
        },
        "seq_lens": [1, args.seq_len],
        "kvcache_buffer_len": attn_cfg_final["kvcache_buffer_len"],
        "max_position_embeddings": attn_cfg_final["max_position_embeddings"],
        "kvcache_dtype": attn_cfg_final["kvcache_dtype"],
        "conv_streaming_node_count": len(cvt_conv_streaming),
        "onnx_path": str(onnx_p.resolve()),
        "rknn_path": str(rknn_p.resolve()),
        "weight_path": str(weight_p.resolve()),
    }
    try:
        with open(manifest_p, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"\n[FAILED] Cannot write manifest: {e}")
        print(f"  manifest path: {manifest_p}")
        sys.exit(1)

    # 校验 manifest 存在且非空
    if not manifest_p.exists() or manifest_p.stat().st_size == 0:
        print(f"\n[FAILED] Manifest file missing or empty: {manifest_p}")
        sys.exit(1)
    print(f"\n--> Manifest saved: {manifest_p}")

    # ------------------------------------------------------------------
    # 成功
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RKNN Conv Streaming export complete!")
    print("=" * 60)
    print(f"  cvt_conv_streaming: {len(cvt_conv_streaming)} conv nodes")
    print(f"  RKNN model : {rknn_p}")
    print(f"  Weight     : {weight_p}")
    print(f"  Manifest   : {manifest_p}")
    print(f"  RKNN size  : {rknn_p.stat().st_size:,} bytes "
          f"({rknn_p.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"  Weight size: {weight_p.stat().st_size:,} bytes "
          f"({weight_p.stat().st_size / 1024 / 1024:.2f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()

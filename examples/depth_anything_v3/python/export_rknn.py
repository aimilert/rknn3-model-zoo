#!/usr/bin/env python3
"""Export DA3-BASE ONNX partitions to RKNN3 FP16 models."""

import argparse
import shutil
from pathlib import Path

import onnx
from rknn.api import RKNN


MODEL_CONFIG = {
    "local": {
        "inputs": {"images": {"dtype": "uint8", "layout": "NHWC"}},
        "mean_values": [[0.485 * 255, 0.456 * 255, 0.406 * 255]],
        "std_values": [[0.229 * 255, 0.224 * 255, 0.225 * 255]],
    },
    "global": {
        "inputs": {
            "local_tokens": {"dtype": "float32", "layout": "NCHW"},
        },
    },
    "head": {
        "inputs": {
            "feat_5": {"dtype": "float16", "layout": "NCHW"},
            "feat_7": {"dtype": "float16", "layout": "NCHW"},
            "feat_9": {"dtype": "float16", "layout": "NCHW"},
            "feat_11": {"dtype": "float16", "layout": "NCHW"},
        },
    },
}
WARNING_COLOR = "\033[93m"
COLOR_RESET = "\033[0m"


def print_warning(message):
    print(f"{WARNING_COLOR}WARNING: {message}{COLOR_RESET}")


def find_stage_models(model_path):
    if model_path.is_dir():
        onnx_models = sorted(model_path.glob("*.onnx"))
    elif model_path.is_file() and model_path.suffix.lower() == ".onnx":
        onnx_models = [model_path]
    else:
        raise FileNotFoundError(f"model path is not an ONNX file or directory: {model_path}")

    stage_models = {}
    for stage in MODEL_CONFIG:
        matches = [path for path in onnx_models if stage in path.stem.lower()]
        if not matches:
            print_warning(f"no ONNX model containing keyword '{stage}' was found")
            continue
        if len(matches) > 1:
            names = ", ".join(path.name for path in matches)
            print_warning(
                f"multiple ONNX models contain keyword '{stage}': {names}; skipped"
            )
            continue
        stage_models[stage] = matches[0]
    return stage_models


def detect_stage(onnx_path):
    model = onnx.load(str(onnx_path), load_external_data=False)
    initializer_names = {value.name for value in model.graph.initializer}
    input_names = {
        value.name for value in model.graph.input if value.name not in initializer_names
    }
    expected_inputs = {
        stage: set(config["inputs"]) for stage, config in MODEL_CONFIG.items()
    }
    for stage, names in expected_inputs.items():
        if input_names == names:
            return stage
    raise RuntimeError(
        f"cannot identify DA3 stage from ONNX inputs: {sorted(input_names)}"
    )


def export_stage(stage, onnx_path, rknn_path, target_platform, core_num):
    spec = MODEL_CONFIG[stage]
    if not onnx_path.exists():
        raise FileNotFoundError(onnx_path)

    report_source = Path("tmp/model_report.html")
    report_source.unlink(missing_ok=True)
    rknn = RKNN(verbose=True)
    try:
        config_kwargs = dict(
            target_platform=target_platform,
            core_num=core_num,
            quantized_dtype="w16a16",
            distribute_strategy="best_perf",
            profile_mode=False,
            input_attrs=spec["inputs"],
        )
        if stage == "local":
            config_kwargs["mean_values"] = spec["mean_values"]
            config_kwargs["std_values"] = spec["std_values"]
        ret = rknn.config(**config_kwargs)
        if ret != 0:
            raise RuntimeError(f"config failed: {ret}")
        ret = rknn.load_onnx(model=str(onnx_path))
        if ret != 0:
            raise RuntimeError(f"load_onnx failed: {ret}")
        ret = rknn.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError(f"build failed: {ret}")
        rknn_path.parent.mkdir(parents=True, exist_ok=True)
        ret = rknn.export_rknn(str(rknn_path))
        if ret != 0:
            raise RuntimeError(f"export failed: {ret}")
    finally:
        rknn.release()

    weight_path = rknn_path.with_suffix(".weight")
    if not rknn_path.exists() or not weight_path.exists():
        raise RuntimeError(
            f"RKNN export did not create {rknn_path} and {weight_path}"
        )
    print(f"exported: {rknn_path}")
    print(f"weight: {weight_path}")

    if not report_source.exists():
        raise RuntimeError(f"RKNN model report was not generated: {report_source}")
    report_path = rknn_path.parent / f"{rknn_path.stem}_model_report.html"
    shutil.copy2(report_source, report_path)
    print(f"report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx_path",
        type=Path,
        required=True,
        help=(
            "ONNX model path or a directory containing DA3 local/global/head "
            "ONNX models"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="RKNN output directory; defaults to the ONNX model directory",
    )
    parser.add_argument(
        "--target_platform",
        default="rk1820",
        help="RKNN target platform (default: rk1820)",
    )
    parser.add_argument(
        "--core_num",
        type=int,
        default=8,
        help="NPU core number in the range 1-8 (default: 8)",
    )
    args = parser.parse_args()

    if args.core_num < 1 or args.core_num > 8:
        raise SystemExit("--core_num must be in the range [1, 8]")
    stage_models = find_stage_models(args.onnx_path)
    default_output_dir = (
        args.onnx_path if args.onnx_path.is_dir() else args.onnx_path.parent
    )
    output_dir = args.output_dir or default_output_dir

    for stage, onnx_path in stage_models.items():
        try:
            detected_stage = detect_stage(onnx_path)
        except RuntimeError as exc:
            print_warning(f"{onnx_path.name} has an unsupported interface: {exc}; skipped")
            continue
        if detected_stage != stage:
            print_warning(
                f"{onnx_path.name} contains keyword '{stage}', but its inputs "
                f"identify it as '{detected_stage}'; skipped"
            )
            continue
        print(f"exporting {stage}: {onnx_path}")
        export_stage(
            stage,
            onnx_path,
            output_dir / f"{onnx_path.stem}.rknn",
            args.target_platform,
            args.core_num,
        )


if __name__ == "__main__":
    main()

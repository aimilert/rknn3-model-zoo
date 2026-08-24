import os
import pickle

from rknn.api import DEFAULT_RKNN_LLM_CONFIG, RKNN


ONNX_MODEL = "../../model/llm/locateanything-3b-llm.onnx"
LLM_CONFIG = "../../model/llm/locateanything-3b-llm.config.pkl"
RKNN_MODEL = "../../model/llm/locateanything-3b-llm.rknn"
DATASET_PATH = "../../data/llm/dataset.txt"


def load_hidden_size(config_path):
    with open(config_path, "rb") as f:
        config = pickle.load(f)
    return int(config["hidden_size"])


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Convert LocateAnything-3B LLM ONNX to RKNN")
    parser.add_argument("--onnx_path", type=str, default=ONNX_MODEL)
    parser.add_argument("--config", type=str, default=LLM_CONFIG)
    parser.add_argument("--rknn_path", type=str, default=RKNN_MODEL)
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH)
    parser.add_argument("--platform", type=str, default="rk1828")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--kvcache_len", type=int, default=4096)
    parser.add_argument("--core_num", type=int, default=8)
    parser.add_argument("--no_quant", action="store_true", help="Build without quantization")
    args = parser.parse_args()

    hidden_size = load_hidden_size(args.config)
    dynamic_input = [
        [[1, 1, hidden_size], [1, 1], [1, 1]],
        [[1, args.seq_len, hidden_size], [1, args.seq_len], [1, args.seq_len]],
    ]

    llm_config = DEFAULT_RKNN_LLM_CONFIG.copy()
    llm_config["keep_one_logit"][0]["output_name"] = "output"
    llm_config["keep_one_logit"][0]["axis"] = 1
    llm_config["vocab"] = []
    llm_config["attention_config"][0]["kvcache_buffer_len"] = args.kvcache_len
    llm_config["attention_config"][0]["max_position_embeddings"] = args.kvcache_len
    llm_config["attention_config"][0]["kvcache_store_method"] = "GroupQuant"
    llm_config["attention_config"][0]["kvcache_dtype"] = "Int4_to_F16"
    llm_config["attention_config"][0]["kvcache_group_size"] = 16
    llm_config["attention_config"][0]["kvcache_residual_depth"] = 64

    rknn = RKNN(verbose=True)

    print("--> config model")
    rknn.config(
        target_platform=args.platform,
        dynamic_input=dynamic_input,
        quantized_dtype="w4a16",
        quantized_algorithm="normal",
        quantized_method="group32",
        core_num=args.core_num,
        llm_config=llm_config,
    )
    print("done")

    print("--> Loading model")
    ret = rknn.load_llm(model=args.onnx_path, config=args.config)
    if ret != 0:
        print("Load model failed!")
        exit(ret)
    print("done")

    print("--> Building model")
    ret = rknn.build(do_quantization=not args.no_quant, dataset=args.dataset_path)
    if ret != 0:
        print("Build model failed!")
        exit(ret)
    print("done")

    print("--> Export rknn model")
    rknn_dir = os.path.dirname(args.rknn_path)
    if rknn_dir:
        os.makedirs(rknn_dir, exist_ok=True)
    ret = rknn.export_rknn(args.rknn_path)
    if ret != 0:
        print("Export rknn model failed!")
        exit(ret)
    print("done")

    rknn.release()

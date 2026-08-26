import numpy as np
import os
from rknn.api import RKNN,DEFAULT_RKNN_LLM_CONFIG

ONNX_MODEL = './Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct-llm.onnx'
LORA_MODEL = None
LORA_CONFIG = None
LLM_CONFIG = './Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct-llm.config.pkl'
RKNN_MODEL = './Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct-llm.rknn'
DATASET_PATH = None

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Export Qwen/Qwen3-VL llm to RKNN model") 
    parser.add_argument("--onnx_path", type=str, help="onnx model path", required=False, default=ONNX_MODEL)
    parser.add_argument("--lora_path", type=str, help="lora model path", required=False, default=LORA_MODEL)
    parser.add_argument("--lora_config_path", type=str, help="lora config file path", required=False, default=LORA_CONFIG)
    parser.add_argument("--config", type=str, help="config file path", required=False, default=LLM_CONFIG)
    parser.add_argument("--rknn_path", type=str, help="output rknn model path", required=False, default=RKNN_MODEL)
    parser.add_argument("--dataset_path", type=str, help="model quantization dataset path", required=False, default=DATASET_PATH)
    parser.add_argument('--rebuild', action='store_true', required=False, default=False, help='Whether to rebuild the model')
    args = parser.parse_args()

    # Create RKNN object
    rknn = RKNN(verbose=True)
    if "2b" in args.onnx_path.lower():
        dynamic_input = [[[1, 1],   [1, 1],   [1, 1],   [1, 1, 2048],[1, 1, 2048],[1, 1, 2048], [1]], 
                            [[1, 128], [1, 128], [1, 128], [1, 128, 2048], [1, 128, 2048], [1, 128, 2048], [1]]]
    elif "4b" in args.onnx_path.lower():
        dynamic_input = [[[1, 1],   [1, 1],   [1, 1], [1, 1, 2560], [1, 1, 2560], [1, 1, 2560], [1]], 
                            [[1, 128], [1, 128], [1, 128], [1, 128, 2560], [1, 128, 2560], [1, 128, 2560], [1]]]

    llm_config = DEFAULT_RKNN_LLM_CONFIG.copy()
    llm_config['attention_config'][0]['mrope_type'] = 'Qwen3-VL'
    llm_config['attention_config'][0]['mrope_section'] = [24,20,20]
    llm_config['attention_config'][0]['mrope_new_id_name'] = 'mrope_id_input'
    # 设置最大的长上下文长度
    llm_config['attention_config'][0]['kvcache_buffer_len'] = int(3072)
    # 设置kvcache量化类型，默认是fp16，Int4_to_F16是int4量化
    llm_config['attention_config'][0]['kvcache_dtype'] = "Int4_to_F16"

    # pre-process config
    print('--> config model')
    rknn.config(
        target_platform = 'rk1820', dynamic_input = dynamic_input,
        quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32',llm_config=llm_config
    )
    print('done')

    if args.rebuild:
        print('--> Rebuilding model')
        ret = rknn.rebuild("./tmp")
        if ret != 0:
            print('Rebuild rknn model failed!')
            exit(ret)
        print('done')
    else:
        # Load model
        print('--> Loading model')
        ret = rknn.load_llm(model=args.onnx_path, config=args.config, seq=[1,128])

        if (args.lora_path is not None and args.lora_config_path is not None) and os.path.exists(args.lora_path):
            ret = rknn.load_lora(
                lora_paths=[[args.lora_path]],
                lora_config_paths=[[args.lora_config_path]],
                lora_patterns=[["lora0_pattern0"]],
                lora_quantized_dtype="float16",
                lora_quantized_method=None,
                lora_prefix="base_model.model.model.",
                lora_postfix_a=".lora_A.weight",
                lora_postfix_b=".lora_B.weight"
            )

        if ret != 0:
            print('Load model failed!')
            exit(ret)
        print('done')

        # Build model
        print('--> Building model')
        ret = rknn.build(do_quantization=True, dataset=args.dataset_path)
        if ret != 0:
            print('Build model failed!')
            exit(ret)
        print('done')

    # Export rknn model
    print('--> Export rknn model')
    export_rknn_dirname = os.path.dirname(args.rknn_path)
    if export_rknn_dirname and not os.path.exists(export_rknn_dirname):
        os.makedirs(export_rknn_dirname, exist_ok=True)
    ret = rknn.export_rknn(args.rknn_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    rknn.release()
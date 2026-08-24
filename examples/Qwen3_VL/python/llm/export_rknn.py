import numpy as np
import os
from rknn.api import RKNN,DEFAULT_RKNN_LLM_CONFIG

ONNX_MODEL = './Qwen3-VL-2B-llm.onnx'
LLM_CONFIG = './Qwen3-VL-2B-llm.config.pkl'
RKNN_MODEL = '../../model/llm_2B/Qwen3-VL-2B-llm.rknn'


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Export Qwen/Qwen3-VL llm to RKNN model") 
    parser.add_argument("--onnx_path", type=str, help="onnx model path", required=False, default=ONNX_MODEL)
    parser.add_argument("--config", type=str, help="config file path", required=False, default=LLM_CONFIG)
    parser.add_argument("--rknn_path", type=str, help="output rknn model path", required=False, default=RKNN_MODEL)
    parser.add_argument('--platform', type=str, required=False, help='Target platform (e.g. rk1820)')
    parser.add_argument('--rebuild', action='store_true', required=False, default=False, help='Whether to rebuild the model')
    parser.add_argument("--prune_mode", action="store_true", help="enable prune mode (two-stage: lm_head runs on host CPU to reduce coprocessor memory)")
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
    llm_config['attention_config'][0]['mrope_section'] = [24,20,20] # Please refer to config.json to configure this parameters. eg, https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/main/config.json
    llm_config['attention_config'][0]['mrope_new_id_name'] = 'mrope_id_input'
    llm_config['attention_config'][0]['kvcache_buffer_len'] = 1*1024
    llm_config['attention_config'][0]['max_position_embeddings'] = 1*1024
    # llm_config['attention_config'][0]['kvcache_dtype'] = 'Int4_to_F16' # 若有需要可开启kvcache量化

    if args.prune_mode:
        llm_config['llm_head'][0]['device'] = 'rk3588'

    # pre-process config
    print('--> config model')
    rknn.config(
        target_platform = args.platform, dynamic_input = dynamic_input,
        quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32',llm_config=llm_config
    )
    print('done')

    if args.rebuild:
        print('--> Rebuild rknn model')
        ret = rknn.rebuild("./tmp")
        if ret != 0:
            print('Rebuild rknn model failed!')
            exit(ret)
        print('done')
    else:
        # Load model
        print('--> Loading model')
        ret = rknn.load_llm(model=args.onnx_path, config=args.config, seq=[1,128])

        if ret != 0:
            print('Load model failed!')
            exit(ret)
        print('done')

        # Build model
        print('--> Building model')
        ret = rknn.build(do_quantization=True)
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
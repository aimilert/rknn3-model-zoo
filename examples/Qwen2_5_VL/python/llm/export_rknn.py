import numpy as np
from rknn.api import RKNN, DEFAULT_RKNN_LLM_CONFIG

ONNX_MODEL = '../../model/llm/Qwen2.5-VL-3B-llm.onnx'
LLM_CONFIG = '../../model/llm/Qwen2.5-VL-3B-llm.config.pkl'
RKNN_MODEL = '../../model/llm/Qwen2.5-VL-3B-llm.rknn'
DATASET_PATH = None

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Export Qwen/Qwen2.5-VL-3B-Instruct llm to RKNN model") 
    parser.add_argument("--onnx_path", type=str, help="onnx model path", required=False, default=ONNX_MODEL)
    parser.add_argument("--config", type=str, help="config file path", required=False, default=LLM_CONFIG)
    parser.add_argument("--rknn_path", type=str, help="output rknn model path", required=False, default=RKNN_MODEL)
    parser.add_argument('--platform', type=str, default= "rk1820", help='Target platform (e.g. rk1820)')
    parser.add_argument("--dataset_path", type=str, help="model quantization dataset path", required=False, default=DATASET_PATH)
    parser.add_argument("--no_prune_mode", dest="prune_mode", action="store_false", help="close prune mode")
    parser.set_defaults(prune_mode=True)
    args = parser.parse_args()

    # Create RKNN object
    rknn = RKNN(verbose=True)

    llm_config = DEFAULT_RKNN_LLM_CONFIG.copy()
    llm_config['attention_config'][0]['mrope_type'] = 'Qwen2.5-VL'
    llm_config['attention_config'][0]['mrope_section'] = [16,24,24] # Please refer to the config.json file in the downloaded HuggingFace file to configure this parameter. eg, https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/config.json
    llm_config['attention_config'][0]['mrope_new_id_name'] = 'mrope_id_input'


    # pre-process config
    print('--> config model')
    rknn.config(target_platform=args.platform, 
                quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32',llm_config=llm_config
                )
    print('done')

    # Load model
    print('--> Loading model')
    if args.prune_mode == True:
        ret = rknn.load_llm(model=args.onnx_path, config=args.config, llm_head_target = "rk3588") # 支持两段式
    else :
        ret = rknn.load_llm(model=args.onnx_path, config=args.config)

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
    ret = rknn.export_rknn(args.rknn_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    rknn.release()
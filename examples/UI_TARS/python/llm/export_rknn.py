import numpy as np
from rknn.api import RKNN, DEFAULT_RKNN_LLM_CONFIG

ONNX_MODEL = '../../model/llm/UI-TARS-2B-SFT-llm.onnx'
LLM_CONFIG = '../../model/llm/UI-TARS-2B-SFT-llm.config.pkl'
RKNN_MODEL = '../../model/llm/UI-TARS-2B-SFT-llm.rknn'
DATASET_PATH = None

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Export UI-TARS-2B-SFT llm to RKNN model") 
    parser.add_argument("--onnx_path", type=str, help="onnx model path", required=False, default=ONNX_MODEL)
    parser.add_argument("--config", type=str, help="config file path", required=False, default=LLM_CONFIG)
    parser.add_argument("--rknn_path", type=str, help="output rknn model path", required=False, default=RKNN_MODEL)
    parser.add_argument('--platform', type=str, default= "rk1820", help='Target platform (e.g. rk1820)')
    parser.add_argument("--dataset_path", type=str, help="model quantization dataset path", required=False, default=DATASET_PATH)
    args = parser.parse_args()

    # Create RKNN object
    rknn = RKNN(verbose=True)
    llm_config = DEFAULT_RKNN_LLM_CONFIG.copy()

    llm_config['attention_config'][0]['mrope_type'] = 'Qwen2.5-VL'
    llm_config['attention_config'][0]['mrope_section'] = [16,24,24] # Please refer to the config.json file in the downloaded HuggingFace file to configure this parameter.
    llm_config['attention_config'][0]['mrope_new_id_name'] = 'mrope_id_input'
    llm_config['attention_config'][0]['kvcache_buffer_len'] = 2048
    llm_config['attention_config'][0]['max_position_embeddings'] = 2048
    # pre-process config
    print('--> config model')
    rknn.config(target_platform=args.platform, 
                quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32',
                llm_config=llm_config
                )
    print('done')

    # Load model
    print('--> Loading model')
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
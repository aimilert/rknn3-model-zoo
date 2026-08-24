import numpy as np
from rknn.api import RKNN, DEFAULT_RKNN_LLM_CONFIG

KVCACHE_LEN = 4096

my_config = DEFAULT_RKNN_LLM_CONFIG.copy()
my_config['keep_one_logit'][0]['output_name'] = 'output'
my_config['keep_one_logit'][0]['axis'] = 1
my_config['attention_config'][0]['kvcache_buffer_len'] = KVCACHE_LEN
my_config['attention_config'][0]['max_position_embeddings'] = KVCACHE_LEN
my_config['attention_config'][0]['kvcache_store_method'] = 'GroupQuant'
my_config['attention_config'][0]['kvcache_dtype'] = 'Int4_to_F16'
my_config['attention_config'][0]['kvcache_group_size'] = 16
my_config['attention_config'][0]['kvcache_residual_depth'] = 64

ONNX_MODEL = '../model/llm/Nanbeige4.2-3B.onnx'
LLM_CONFIG = '../model/llm/Nanbeige4.2-3B.config.pkl'
RKNN_MODEL = '../model/llm/Nanbeige4.2-3B.rknn'
DATASET_PATH = '../../../datasets/CMMLU/dataset.txt'

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Export Nanbeige/Nanbeige4.2-3B llm to RKNN model") 
    parser.add_argument("--onnx_path", type=str, help="onnx model path", required=False, default=ONNX_MODEL)
    parser.add_argument("--config", type=str, help="config file path", required=False, default=LLM_CONFIG)
    parser.add_argument("--rknn_path", type=str, help="output rknn model path", required=False, default=RKNN_MODEL)
    parser.add_argument("--dataset_path", type=str, help="model quantization dataset path", required=False, default=DATASET_PATH)
    parser.add_argument('--platform', type=str, required=True, help='Target platform (e.g. rk1820)')
    args = parser.parse_args()

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> config model')
    rknn.config(target_platform=args.platform, 
                quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32',
                llm_config = my_config,
                dynamic_input=[[[1,128],[1,128], [1,128]],[[1,1],[1,1], [1,1]]]
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

    #Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(args.rknn_path)
    if ret != 0:
        print('Export rknn failed!')
        exit(ret)
    print('done')

    rknn.release()

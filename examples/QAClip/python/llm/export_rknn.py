import numpy as np
from rknn.api import RKNN
from rknn.api import DEFAULT_RKNN_LLM_CONFIG

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"

ONNX_MODEL = '../../models/chinese_clip_text.onnx'
LLM_CONFIG = '../../models/chinese_clip_text.config.pkl'
RKNN_MODEL = '../../rknn/chinese_clip_text.rknn'
DATASET_PATH = '../../../datasets/dataset.txt'

SEQ_LEN = 352
if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Export chinese_clip_text to RKNN model") 
    parser.add_argument("--onnx_path", type=str, help="onnx model path", required=False, default=ONNX_MODEL)
    parser.add_argument("--config", type=str, help="config file path", required=False, default=LLM_CONFIG)
    parser.add_argument("--rknn_path", type=str, help="output rknn model path", required=False, default=RKNN_MODEL)
    parser.add_argument("--dataset_path", type=str, help="model quantization dataset path", required=False, default=DATASET_PATH)
    parser.add_argument('--platform', type=str, default= "rk1820", help='Target platform (e.g. rk1820)')
    args = parser.parse_args()

    llm_config = DEFAULT_RKNN_LLM_CONFIG.copy()
    llm_config['attention_config'][0]['position_name'] = None
    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> config model')
    rknn.config(target_platform=args.platform, llm_config=llm_config,dynamic_input = [[[1,SEQ_LEN],[1,SEQ_LEN],[1,SEQ_LEN]]],
                quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32', core_num=1)
    print('done')

    # # # # Load model
    print('--> Loading model')
    ret = rknn.load_llm(model=args.onnx_path, config=args.config)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    rknn.build(do_quantization=False, dataset=args.dataset_path)
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


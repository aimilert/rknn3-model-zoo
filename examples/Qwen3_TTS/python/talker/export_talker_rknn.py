from rknn.api import RKNN, DEFAULT_RKNN_LLM_CONFIG
import pickle

ONNX_MODEL = '../../models/talker/talker_onnx/talker.onnx'
LLM_CONFIG = '../../models/talker/talker.config.pkl'
RKNN_MODEL = '../../models/talker/talker.rknn'
TARGET_PLATFORM = 'rk1820'
DO_QUANT = True
DATASET_PATH = None

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Export Qwen3-TTS talker to RKNN model") 
    parser.add_argument("--onnx_path", type=str, default=ONNX_MODEL, help="onnx model path")
    parser.add_argument("--config", type=str, default=LLM_CONFIG, help="config file path")
    parser.add_argument("--rknn_path", type=str, default=RKNN_MODEL, help="output rknn model path")
    parser.add_argument('--platform', type=str, default=TARGET_PLATFORM, help='Target platform (e.g. rk1820)')
    parser.add_argument('--do_quant', type=bool, default=DO_QUANT, help='whether to do quantization')
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH, help="model quantization dataset path")
    args = parser.parse_args()

    # Create RKNN object
    rknn = RKNN(verbose=False)

    llm_config = DEFAULT_RKNN_LLM_CONFIG.copy()
    llm_config['vocab'].pop()
    llm_config['attention_config'][0]['kvcache_buffer_len'] = 2048
    llm_config['attention_config'][0]['kvcache_dtype'] = 'Float16'

    with open(args.config, "rb") as f:
        loaded_config = pickle.load(f)
    hidden_size = loaded_config['hidden_size']
    dynamic_input = [
        [[1, 128, hidden_size], [1, 128], [1, 128], [1]],
        [[1, 1, hidden_size], [1, 1], [1, 1], [1]],
    ]

    # pre-process config
    print('--> config model')
    rknn.config(target_platform=args.platform, 
                llm_config=llm_config, 
                quantized_dtype='w4a16', 
                quantized_algorithm='normal', 
                quantized_method='group32',
                core_num=8,
                dynamic_input=dynamic_input,
                linear_attn_out_project_dtype=None, op_quantized_dtype='./layer_bit.json'
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
    ret = rknn.build(do_quantization=args.do_quant, dataset=args.dataset_path)
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

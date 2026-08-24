from rknn.api import RKNN, DEFAULT_RKNN_LLM_CONFIG
import pickle

ONNX_MODEL = 'llm.onnx'
LLM_CONFIG = 'llm.config.pkl'
RKNN_MODEL = 'llm.rknn'

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Export Qwen/Qwen3 llm to RKNN model") 
    parser.add_argument("--onnx_path", type=str, help="onnx model path", required=False, default=ONNX_MODEL)
    parser.add_argument("--config", type=str, help="config file path", required=False, default=LLM_CONFIG)
    parser.add_argument("--rknn_path", type=str, help="output rknn model path", required=False, default=RKNN_MODEL)
    parser.add_argument('--platform', type=str, default= "rk1820", help='Target platform (e.g. rk1820)')
    parser.add_argument("--dataset_path", type=str, help="model quantization dataset path", required=False)
    args = parser.parse_args()

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> config model')
    my_config = DEFAULT_RKNN_LLM_CONFIG.copy()
    my_config['keep_one_logit'][0]['output_name'] = "output"
    my_config['keep_one_logit'][0]['axis'] = 1
    my_config["vocab"] = []
    my_config['attention_config'][0]['kvcache_buffer_len'] = 1024*24
    my_config['attention_config'][0]['max_position_embeddings'] = 1024*24
    my_config['attention_config'][0]['kvcache_store_method'] = 'GroupQuant'
    my_config['attention_config'][0]['kvcache_dtype'] = 'Int4_to_F16'
    my_config['attention_config'][0]['kvcache_group_size'] = 16
    my_config['attention_config'][0]['kvcache_residual_depth'] = 64
    print(my_config)

    with open(args.config, "rb") as f:
        loaded_config = pickle.load(f)
    hidden_size = loaded_config['hidden_size']

    dynamic_input = [[[1, 1, hidden_size],   [1, 1],   [1, 1, 1]], [[1, 128, hidden_size], [1, 128], [1, 1, 128]]]
    rknn.config(target_platform=args.platform, dynamic_input = dynamic_input,
                quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32', 
                llm_config=my_config, 
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
    ret = rknn.build(do_quantization=True)
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


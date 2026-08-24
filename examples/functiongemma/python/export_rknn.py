import numpy as np
from rknn.api import RKNN

ONNX_MODEL = '../model/llm/functiongemma-270m-it.onnx'
LLM_CONFIG = '../model/llm/functiongemma-270m-it.config.pkl'
RKNN_MODEL = '../model/llm/functiongemma-270m-it.rknn'
DATASET_PATH = '../../../datasets/CMMLU/dataset.txt'

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Export gemma llm to RKNN model") 
    parser.add_argument("--onnx_path", type=str, help="onnx model path", required=False, default=ONNX_MODEL)
    parser.add_argument("--config", type=str, help="config file path", required=False, default=LLM_CONFIG)
    parser.add_argument("--rknn_path", type=str, help="output rknn model path", required=False, default=RKNN_MODEL)
    parser.add_argument("--dataset_path", type=str, help="model quantization dataset path", required=False, default=DATASET_PATH)
    parser.add_argument("--platform", type=str, help="target platform, e.g. rk1820/rk3572", required=False, default='rk1820')
    args = parser.parse_args()

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> config model')
    
    from rknn.api import DEFAULT_RKNN_LLM_CONFIG
    my_config = DEFAULT_RKNN_LLM_CONFIG.copy()
    MAX_CONTEXT_LEN = 1024
    kvcache_buffer_len = MAX_CONTEXT_LEN
    max_position_embeddings = MAX_CONTEXT_LEN
    sliding_window = 512

    attn_config = {                                   # accept multi internal kvcache as list(dict)
        'mask_name'              : 'attention_mask',         # required, this is used for recognize which attn belong to this config
        'position_name'          : 'position_ids',           # allow None if no position_ids input
        'kvcache_buffer_len'     : kvcache_buffer_len,                     # feed int or str "real_time"
        'max_position_embeddings': max_position_embeddings,
        'kvcache_dtype'          : 'Float16',                # Float16, Int8_to_F16, Int4_to_F16
        'kvcache_store_method'   : 'Normal',                 # GroupQuant, Normal
        'kvcache_group_size'     : 32,
        'kvcache_residual_depth' : 32,
        'sliding_window_size'    : -1,
        'attention_type'         : 'FullAttention',          # FullAttention, SlidingAttention
        'position_embeddings_host_storage'     : True,
    }

    attn_sld_config = {                                   # accept multi internal kvcache as list(dict)
        'mask_name'              : 'attention_mask_1',         # required, this is used for recognize which attn belong to this config
        'position_name'          : 'position_ids_1',           # allow None if no position_ids input
        'kvcache_buffer_len'     : sliding_window,                     # feed int or str "real_time"
        'max_position_embeddings': max_position_embeddings,
        'kvcache_dtype'          : 'Float16',                # Float16, Int8_to_F16, Int4_to_F16
        'kvcache_store_method'   : 'Normal',                 # GroupQuant, Normal
        'kvcache_group_size'     : 32,
        'kvcache_residual_depth' : 32,
        'sliding_window_size'    : sliding_window,
        'attention_type'         : 'SlidingAttention',          # FullAttention, SlidingAttention
        'position_embeddings_host_storage'     : True,
    }
    attention_config = []
    attention_config.append(attn_sld_config)
    attention_config.append(attn_config)

    my_config['attention_config'] = attention_config

    print('LLM config is:', my_config)
    
    # exit(0)
    dynamic_input = [[[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1]],
                        [[1, 128], [1, 128], [1, 128], [1, 128], [1, 128], [1]]]

    rknn.config(target_platform=args.platform, dynamic_input = dynamic_input,
                quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32',
                llm_config=my_config,
                # profile_mode=True,
                )
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_llm(model=args.onnx_path, config=args.config, seq=[1,128])
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
    ret = rknn.export_rknn(args.rknn_path, save_ctx=True)
    if ret != 0:
        print('Export rknn failed!')
        exit(ret)
    print('done')

    rknn.release()


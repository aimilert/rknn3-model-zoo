import numpy as np
from rknn.api import RKNN

ONNX_MODEL = '../../model/vision/SmolVLM-500M-vision.onnx'
RKNN_MODEL = '../../model/vision/SmolVLM-500M-vision.rknn'
DATASET_PATH = None


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="SmolVLM-500M vision convert rknn") 
    parser.add_argument("--onnx_path", type=str, help="onnx model path", required=False, default=ONNX_MODEL)
    parser.add_argument("--rknn_path", type=str, help="output rknn model path", required=False, default=RKNN_MODEL)
    parser.add_argument('--platform', type=str, default= "rk1820", help='Target platform (e.g. rk1820)')
    parser.add_argument("--dataset_path", type=str, help="model quantization dataset path", required=False, default=DATASET_PATH)
    parser.add_argument('--core_num', type=int, default=8, help='core_num (1-8)')
    parser.add_argument('--rebuild', action='store_true', required=False, default=False, help='Whether to rebuild the model')
    args = parser.parse_args()

    # Create RKNN object
    rknn = RKNN(verbose=True)

    print('--> config model')
    rknn.config(target_platform=args.platform, core_num=args.core_num,
                quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32',
                mean_values=[[127.5,127.5, 127.5]], #完整版
                std_values=[[127.5, 127.5, 127.5]],
                input_attrs={'pixel': {'dtype': 'uint8', 'layout': 'NHWC'}},
                disable_rules=['fuse_tp_rs_tp_rs_tp_to_conv']
                )

    if args.rebuild:
        print('--> Rebuilding model')
        ret = rknn.rebuild("./tmp")
        if ret != 0:
            print('Rebuild rknn model failed!')
            exit(ret)
        print('done')
    else:
        # Load model
        ret = rknn.load_onnx(model=args.onnx_path,
                            inputs=['pixel'], #完整版
                            input_size_list = [[1,3,512,512]])
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


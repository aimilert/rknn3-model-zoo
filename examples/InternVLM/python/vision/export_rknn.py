import numpy as np
from rknn.api import RKNN

ONNX_MODEL = './vision/InternVL3.5-2B-vision.onnx'
RKNN_MODEL = './vision/InternVL3.5-2B-vision.rknn'
DATASET_PATH = None

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Export InternVL3 vision to RKNN model")
    parser.add_argument("--onnx_path", type=str, help="onnx model path", required=False, default=ONNX_MODEL)
    parser.add_argument("--rknn_path", type=str, help="output rknn model path", required=False, default=RKNN_MODEL)
    parser.add_argument('--platform', type=str, default= "rk1820", help='Target platform (e.g. rk1820)')
    parser.add_argument("--dataset_path", type=str, help="model quantization dataset path", required=False, default=DATASET_PATH)
    parser.add_argument('--core_num', type=int, default=8, help='core_num (1-8)')
    args = parser.parse_args()

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> config model')
    rknn.config(target_platform=args.platform, 
                core_num=args.core_num,
                quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32',
                mean_values=[[123.675, 116.28, 103.53]],
                std_values=[[58.395, 57.12, 57.375]],
                input_attrs={'pixel_values': {'dtype': 'uint8', 'layout': 'NHWC'}},
                )
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=args.onnx_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False, dataset=args.dataset_path)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(args.rknn_path)
    if ret != 0:
        print('Export rknn failed!')
        exit(ret)
    print('done')

    rknn.release()


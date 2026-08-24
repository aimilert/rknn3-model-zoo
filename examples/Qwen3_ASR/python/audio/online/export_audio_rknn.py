from argparse import ArgumentParser

from rknn.api import RKNN


DEFAULT_ONNX_PATH = '../../../models/encoder_online.onnx'
DEFAULT_RKNN_PATH = '../../../models/encoder_online.rknn'
DEFAULT_QUANT = False

def parse_arg():
    parser = ArgumentParser(description='Export Qwen3-ASR audio encoder (online) to RKNN model')
    parser.add_argument('--onnx_path', type=str, default=DEFAULT_ONNX_PATH, help='onnx model path')
    parser.add_argument('--rknn_path', type=str, default=DEFAULT_RKNN_PATH, help='output rknn model path')
    parser.add_argument('--platform', type=str, default="rk1820", help='target platform (e.g. rk1820)')
    parser.add_argument('--do_quant', action='store_true', default=DEFAULT_QUANT, help='enable quantization when building')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arg()

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0]], std_values=[[1]], target_platform=args.platform,
                input_attrs={'x': {'dtype': 'float32', 'layout': 'NCHW'}},
                quantized_dtype='w4a16',
                quantized_algorithm='normal',
                quantized_method='group32',
                core_num=8,
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
    ret = rknn.build(do_quantization=args.do_quant)
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

    # Release
    rknn.release()

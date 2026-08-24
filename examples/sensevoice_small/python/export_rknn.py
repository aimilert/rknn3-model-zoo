import sys
import os
from rknn.api import RKNN

def parse_arg():
    if len(sys.argv) < 3:
        print("Usage: python3 {} onnx_model_path [platform] [output_rknn_path(optional)]".format(sys.argv[0]))
        print("       platform choose from [rk1820]")
        exit(1)

    model_path = sys.argv[1]
    platform = sys.argv[2]

    if len(sys.argv) > 3:
        output_path = sys.argv[3]
    else:
        output_path = os.path.splitext(model_path)[0] + '.rknn'

    return model_path, platform, output_path

if __name__ == '__main__':
    model_path, platform, output_path = parse_arg()

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print('--> Config model')

    rknn.config(target_platform=platform, quantized_dtype="w4a16", quantized_algorithm="normal", quantized_method="group32", core_num=1)
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True)
    if (ret != 0):
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path, save_ctx=True)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Release
    rknn.release()

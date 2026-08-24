import json
import os
import sys

from rknn.api import RKNN

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

ONNX_MODEL = '../../model/vision/locateanything-3b-vision_pixel.onnx'
RKNN_MODEL = '../../model/vision/locateanything-3b-vision_pixel.rknn'
DATASET_PATH = '../../../../datasets/MMBench/vision/datasets.txt'
DEFAULT_IMG_H = 448
DEFAULT_IMG_W = 448
DEFAULT_PATCH_SIZE = 14
DEFAULT_MERGE_KERNEL_SIZE = [2, 2]
MEAN_VALUES = [[0.5 * 255, 0.5 * 255, 0.5 * 255]]
STD_VALUES = [[0.5 * 255, 0.5 * 255, 0.5 * 255]]


def load_config(config_path: str):
    if not os.path.exists(config_path):
        config = {
            "img_h": DEFAULT_IMG_H,
            "img_w": DEFAULT_IMG_W,
            "patch_size": DEFAULT_PATCH_SIZE,
            "merge_kernel_size": DEFAULT_MERGE_KERNEL_SIZE,
        }
    else:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

    grid_h = config["img_h"] // config.get("patch_size", DEFAULT_PATCH_SIZE)
    grid_w = config["img_w"] // config.get("patch_size", DEFAULT_PATCH_SIZE)
    merge = config.get("merge_kernel_size", DEFAULT_MERGE_KERNEL_SIZE)
    num_patches = grid_h * grid_w
    num_tokens = (grid_h // merge[0]) * (grid_w // merge[1])
    return config, num_patches, num_tokens


vision_config, num_patches, num_tokens = load_config('vision_config.json')
input_shape = vision_config.get('input_shape', [1, 3, vision_config['img_h'], vision_config['img_w']])
print(
    f"Using img_h={vision_config['img_h']}, img_w={vision_config['img_w']}, "
    f"input_shape={input_shape}, num_patches={num_patches}, num_tokens={num_tokens}"
)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="LocateAnything-3B vision convert rknn")
    parser.add_argument("--onnx_path", type=str, help="onnx model path", required=False, default=ONNX_MODEL)
    parser.add_argument("--rknn_path", type=str, help="output rknn model path", required=False, default=RKNN_MODEL)
    parser.add_argument("--dataset_path", type=str, help="model quantization dataset path", required=False, default=DATASET_PATH)
    parser.add_argument('--platform', type=str, default="rk1828", help='Target platform (e.g. rk1820)')
    parser.add_argument('--core_num', type=int, default=8, help='core_num (1-8)')
    args = parser.parse_args()

    rknn = RKNN(verbose=True)

    print('--> config model')
    rknn.config(
        target_platform=args.platform,
        core_num=args.core_num,
        quantized_dtype='w8a16',
        quantized_algorithm='normal',
        quantized_method='group32',
        mean_values=MEAN_VALUES,
        std_values=STD_VALUES,
        # disable_rules=['merge_parallel_op_after_split', 'merge_parallel_op_before_concat', "reduce_reshape_op_around_split"],
        input_attrs={'image': {'dtype': 'uint8', 'layout': 'NHWC'}},
        output_attrs={"vision_output": {'dtype': 'float16', 'layout': 'UNDEFINED'}}
    )
    print('done')

    print('--> Loading model')
    ret = rknn.load_onnx(
        model=args.onnx_path,
        inputs=['image'],
        input_size_list=[input_shape],
    )
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset=args.dataset_path)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    print('--> Export rknn model')
    ret = rknn.export_rknn(args.rknn_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    rknn.release()

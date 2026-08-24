import numpy as np
import os,sys
from rknn.api import RKNN
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from py_utils.tools import gen_qwen_vl_vision_prune_quantize_dataset

ONNX_MODEL = 'Qwen3-VL-2B-vision.onnx'
RKNN_MODEL = 'Qwen3-VL-2B-vision.rknn'
DATASET_PATH = None

def load_config(config_path: str):
    import json
    import os
    if not os.path.exists(config_path):
        return 384, 384, np.array([[1,24,24]], dtype=np.int64)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    required_keys = ['img_h', 'img_w', 'grid_thw']
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing key '{key}' in config file")
    return config["img_h"], config["img_w"], np.array([config["grid_thw"]], dtype=np.int64)

img_h, img_w, grid_thw = load_config('vision_config.json')
print(f"Using img_h={img_h}, img_w={img_w}, grid_thw={grid_thw}")

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Qwen/Qwen3-VL vision convert rknn") 
    parser.add_argument("--onnx_path", type=str, help="onnx model path", required=False, default=ONNX_MODEL)
    parser.add_argument("--rknn_path", type=str, help="output rknn model path", required=False, default=RKNN_MODEL)
    parser.add_argument('--platform', type=str, default= "rk1820", help='Target platform (e.g. rk1820)')
    parser.add_argument("--dataset_path", type=str, help="model quantization dataset path", required=False, default=DATASET_PATH)
    parser.add_argument("--no_prune_mode", dest="prune_mode", action="store_false", help="close prune mode")
    parser.set_defaults(prune_mode=True)
    parser.add_argument('--rebuild', action='store_true', required=False, default=False, help='Whether to rebuild the model')
    args = parser.parse_args()

    # Create RKNN object
    rknn = RKNN(verbose=True)

    print('--> config model')
    if args.prune_mode == True:
        rknn.config(target_platform=args.platform, core_num=8,
                quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32')
    else:
        rknn.config(target_platform=args.platform, core_num=8,
                    quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32',
                    mean_values=[[0.5 * 255, 0.5 * 255, 0.5 * 255]], #完整版
                    std_values=[[0.5 * 255, 0.5 * 255, 0.5 * 255]],
                    input_attrs={'pixel': {'dtype': 'uint8', 'layout': 'NHWC'}}
                    )
    print('done')

    if args.rebuild:
        print('--> Rebuilding model')
        ret = rknn.rebuild("./tmp")
        if ret != 0:
            print('Rebuild rknn model failed!')
            exit(ret)
        print('done')
    else:
        # Load model
        print('--> Loading model')
        if args.prune_mode == True:
            ret = rknn.load_onnx(model=args.onnx_path,
                             inputs=['/Reshape_1_output_0', 'grid_thw'], #裁剪版
                             input_size_list = [[int(grid_thw[0][1] * grid_thw[0][2]),1536],[1,3]],
                             input_initial_val=[None, grid_thw])
        else:
            ret = rknn.load_onnx(model=args.onnx_path,
                             inputs=['pixel', 'grid_thw'], #完整版
                             input_size_list = [[1,3,img_h,img_w],[1,3]],
                             input_initial_val=[None, grid_thw])
        if ret != 0:
            print('Load model failed!')
            exit(ret)
        print('done')

        if args.prune_mode == True:
            if args.dataset_path is not None:
                ret = gen_qwen_vl_vision_prune_quantize_dataset(args.dataset_path, "../../data/vision/datasets_npy.txt", img_h, img_w, [0.5 * 255, 0.5 * 255, 0.5 * 255], [0.5 * 255, 0.5 * 255, 0.5 * 255], 16)
                if ret == 0:
                    args.dataset_path = "../../data/vision/datasets_npy.txt"
            else:
                args.dataset_path = "../../data/vision/datasets_npy.txt"

        # Build model
        print('--> Building model')
        ret = rknn.build(do_quantization=True, dataset=args.dataset_path)
        if ret != 0:
            print('Build model failed!')
            exit(ret)
        print('done')

    # Export rknn model
    print('--> Export rknn model')
    export_rknn_dirname = os.path.dirname(args.rknn_path)
    if export_rknn_dirname and not os.path.exists(export_rknn_dirname):
        print(f"create export_rknn_dirname: {export_rknn_dirname}")
        os.makedirs(export_rknn_dirname, exist_ok=True)

    ret = rknn.export_rknn(args.rknn_path)
    if ret != 0:
        print(f"Export rknn model failed! {args.rknn_path}")
        exit(ret)
    print(f"done {args.rknn_path}")
    
    rknn.release()


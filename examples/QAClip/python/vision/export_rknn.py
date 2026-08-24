from rknn.api import RKNN

ONNX_MODEL = 'chinese_clip_vision.onnx'
RKNN_MODEL = 'chinese_clip_vision.rknn'
DATASET_PATH = '../../datasets.txt'

img_h, img_w = 224, 224

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="chinese clip vision convert rknn") 
    parser.add_argument("--onnx_path", type=str, help="onnx model path", required=False, default=ONNX_MODEL)
    parser.add_argument("--rknn_path", type=str, help="output rknn model path", required=False, default=RKNN_MODEL)
    parser.add_argument("--dataset_path", type=str, help="model quantization dataset path", required=False, default=DATASET_PATH)
    parser.add_argument('--platform', type=str, default= "rk1820", help='Target platform (e.g. rk1820)')
    args = parser.parse_args()

    # Create RKNN object
    rknn = RKNN(verbose=True)

    print('--> config model')
    rknn.config(target_platform=args.platform, core_num=1,
                quantized_dtype='w6a16', quantized_algorithm='normal', quantized_method='group32',
                mean_values=[[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]], #完整版
                std_values=[[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255]],
                input_attrs={'pixel': {'dtype': 'uint8', 'layout': 'NHWC'}},
                output_attrs={"image_features": {'dtype': 'float16', 'layout': 'UNDEFINED'}}, profile_mode = True
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


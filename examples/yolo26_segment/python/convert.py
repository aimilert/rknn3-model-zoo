import sys
import argparse
import os
import time
import numpy as np
from rknn.api import RKNN
from pathlib import Path
import onnx
from onnx import helper, shape_inference, TensorProto

DEFAULT_DATASET = '../model/images.txt'
DEFAULT_OUTPUT = '../model/yolo26n.rknn'

YOLO26_SEG_RAW_OUTPUTS = [
    '/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0',
    '/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0',
    '/model.23/one2one_cv4.0/one2one_cv4.0.2/Conv_output_0',
    '/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0',
    '/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0',
    '/model.23/one2one_cv4.1/one2one_cv4.1.2/Conv_output_0',
    '/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0',
    '/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0',
    '/model.23/one2one_cv4.2/one2one_cv4.2.2/Conv_output_0',
    'output1',
]

YOLO26_SEG_OUTPUT_SPECS = [
    ('box_1', YOLO26_SEG_RAW_OUTPUTS[0], False),
    ('score_1_sigmoid', YOLO26_SEG_RAW_OUTPUTS[1], True),
    ('seg_1', YOLO26_SEG_RAW_OUTPUTS[2], False),
    ('box_2', YOLO26_SEG_RAW_OUTPUTS[3], False),
    ('score_2_sigmoid', YOLO26_SEG_RAW_OUTPUTS[4], True),
    ('seg_2', YOLO26_SEG_RAW_OUTPUTS[5], False),
    ('box_3', YOLO26_SEG_RAW_OUTPUTS[6], False),
    ('score_3_sigmoid', YOLO26_SEG_RAW_OUTPUTS[7], True),
    ('seg_3', YOLO26_SEG_RAW_OUTPUTS[8], False),
    ('proto', YOLO26_SEG_RAW_OUTPUTS[9], False),
]

def parse_arg():
    parser = argparse.ArgumentParser(
        description='Convert ONNX model to RKNN format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic conversion (no inference test)
  python3 convert.py --onnx model.onnx --platform rk1820 --dtype w8a8

  # With custom output path and dataset
  python3 convert.py --onnx model.onnx --platform rk1820 --dtype w8a8 --output yolo26n.rknn --dataset dataset.txt

  # Enable inference test on remote device (platform is also the target)
  python3 convert.py --onnx model.onnx --platform rk3572 --dtype w8a8 --device-id 172.16.15.199:5555

  # Enable accuracy analysis on RKNN3 platform (requires --device-id)
  python3 convert.py --onnx model.onnx --platform rk3572 --dtype w8a8 --device-id 172.16.15.199:5555 --accuracy-analysis

  # Enable inference test with local simulator
  python3 convert.py --onnx model.onnx --platform rk1820 --dtype w8a8 --inference
        '''
    )

    parser.add_argument('--onnx', required=True, help='Path to ONNX model file')
    parser.add_argument('--platform', required=True,
                        choices=['rk1820', 'rk1828', 'rk3572'],
                        help='Target platform (also used as inference target if --inference or --device-id is set)')
    parser.add_argument('--dtype', choices=['w8a8', 'fp', 'w8a16', 'w4a16', 'w6a16'], default='w8a8',
                        help='Data type: w8a8, w8a16, w6a16, w4a16, or fp (float). Default: w8a8')
    parser.add_argument('--output', default=DEFAULT_OUTPUT,
                        help=f'Output RKNN path. Default: {DEFAULT_OUTPUT}')
    parser.add_argument('--dataset', default=DEFAULT_DATASET,
                        help=f'Quantization dataset file path. Default: {DEFAULT_DATASET}')
    parser.add_argument('--core-num', type=int, default=1,
                        help='Number of NPU cores to compile for RK1820 (1 or 8, default: 1)')
    parser.add_argument('--inference', action='store_true',
                        help='Enable inference test with local simulator (uses --platform as target)')
    parser.add_argument('--device-id',
                        help='Device ID for remote connection (e.g., 172.16.15.190:5555). Enables inference test')
    parser.add_argument('--accuracy-analysis', action='store_true',
                        help='Enable accuracy analysis (requires --device-id). Only for RKNN3 platforms (rk1820, rk3572)')

    args = parser.parse_args()

    do_quant = args.dtype != 'fp'
    quantized_dtype = args.dtype
    if args.accuracy_analysis and not args.device_id:
        parser.error('--accuracy-analysis requires --device-id')

    do_inference = args.inference or args.device_id is not None

    # Validate dataset file exists if quantization is enabled
    if do_quant and not Path(args.dataset).exists():
        print(f"ERROR: Dataset file not found: {args.dataset}")
        sys.exit(1)

    return args.onnx, args.platform, do_quant, args.output, args.dataset, args.core_num, do_inference, args.device_id, args.accuracy_analysis, quantized_dtype



def add_sigmoid_to_score_heads(onnx_path):
    model = onnx.load(onnx_path)
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f'WARNING: ONNX shape inference failed: {e}')

    graph = model.graph

    new_output_infos = []
    new_output_names = []

    value_infos = {}
    for info in list(graph.input) + list(graph.value_info) + list(graph.output):
        value_infos[info.name] = info

    node_outputs = {name for node in graph.node for name in node.output}
    available_values = set(value_infos) | node_outputs | {init.name for init in graph.initializer}

    missing_outputs = [raw_name for _, raw_name, _ in YOLO26_SEG_OUTPUT_SPECS if raw_name not in available_values]
    if missing_outputs:
        print('ERROR: ONNX model does not contain required YOLO26 segment outputs:')
        for name in missing_outputs:
            print(f'  - {name}')
        sys.exit(1)

    def make_output_info(output_name, raw_name):
        output_info = onnx.ValueInfoProto()
        output_info.name = output_name
        if raw_name in value_infos:
            output_info.type.CopyFrom(value_infos[raw_name].type)
        else:
            output_info.type.tensor_type.elem_type = TensorProto.FLOAT
        return output_info

    for output_name, raw_name, needs_sigmoid in YOLO26_SEG_OUTPUT_SPECS:
        if needs_sigmoid:
            node = helper.make_node(
                'Sigmoid',
                inputs=[raw_name],
                outputs=[output_name],
                name='Sigmoid_' + output_name,
            )
        else:
            node = helper.make_node(
                'Identity',
                inputs=[raw_name],
                outputs=[output_name],
                name='Identity_' + output_name,
            )
        graph.node.append(node)
        new_output_infos.append(make_output_info(output_name, raw_name))
        new_output_names.append(output_name)

    # 替换 graph.output
    while len(graph.output) > 0:
        graph.output.pop()
    graph.output.extend(new_output_infos)

    src = Path(onnx_path)
    sigmoid_onnx_path = src.with_name(src.stem + "_sigmoid.onnx")

    extractor = onnx.utils.Extractor(model)
    model = extractor.extract_model([i.name for i in graph.input], new_output_names)
    onnx.checker.check_model(model)
    onnx.save(model, sigmoid_onnx_path)
    print(f"Modified ONNX saved to {sigmoid_onnx_path}")
    print(f"New ONNX outputs: {new_output_names}")

    return str(sigmoid_onnx_path), new_output_names


if __name__ == '__main__':
    model_path, platform, do_quant, output_path, dataset_path, core_num, do_inference, device_id, accuracy_analysis, quantized_dtype = parse_arg()

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # Pre-process config
    # mean_values=[0,0,0], std_values=[255,255,255]: Normalize input to [0,1] range
    # This matches PyTorch preprocessing: img / 255.0
    print('--> Config model')

    # Build config parameters based on toolkit version
    # RKNN3 (rk1820, rk3572): supports profile_mode for accuracy analysis
    # RKNN2 (others): does not support profile_mode
    config_params = {
        'mean_values': [[0, 0, 0]],
        'std_values': [[255, 255, 255]],
        'quantized_dtype': quantized_dtype,
        'quantized_algorithm': 'normal',
        'quantized_method': 'channel',
        'optimization_level' : 3,
        'target_platform': platform
    }

    # Only add profile_mode for RKNN3 platforms when doing accuracy analysis
    if accuracy_analysis:
        config_params['profile_mode'] = True

    # RK1820 supports one or eight NPU cores; the default is one core.
    if platform == 'rk1820':
        config_params['core_num'] = core_num

        config_params['distribute_strategy'] = 'best_perf'
        print(f'  RK1820: Compiling for {core_num} NPU cores (distribute_strategy=best_perf)')
        
    subgraph = None
    if do_quant and platform == 'rk1820':
        

        subgraph_inputs1 = [
        '/model.10/m/m.0/attn/qkv/conv/Conv_output_0',
        ]

        subgraph_outputs1 = [
            '/model.10/m/m.0/attn/proj/conv/Conv_output_0'
        ]

        subgraph_inputs2 = [
        '/model.22/m.0/m.0.1/attn/qkv/conv/Conv_output_0',
        ]

        subgraph_outputs2 = [
            '/model.22/m.0/m.0.1/attn/proj/conv/Conv_output_0'
        ]

        subgraph_inputs3 = [
        '/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.0/conv/Conv_output_0',
        ]

        subgraph_outputs3 = [
            '/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/conv/Conv_output_0'
        ]

        subgraph_inputs4 = [
        '/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.0/conv/Conv_output_0',
        ]

        subgraph_outputs4 = [
            '/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.1/conv/Conv_output_0'
        ]

        subgraph_inputs5 = [
        '/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.0/conv/Conv_output_0',
        ]

        subgraph_outputs5 = [
        '/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.1/conv/Conv_output_0'
        ]



        subgraph = [[subgraph_inputs1, subgraph_outputs1, 'w16a16'],
                    [subgraph_inputs2, subgraph_outputs2, 'w16a16'],
                    [subgraph_inputs3, subgraph_outputs3, 'w16a16'],
                    [subgraph_inputs4, subgraph_outputs4, 'w16a16'],
                    [subgraph_inputs5, subgraph_outputs5, 'w16a16']]
                    
        print('text fp16 subgraph:', subgraph)

    if subgraph is not None:
        config_params['subgraph_dtype_target'] = subgraph

    ret = rknn.config(**config_params)
    if ret != 0:
        print('Config failed!')
        sys.exit(ret)
    print('done')

    # Load model
    # Outputs are 10 segment heads: 3 box heads + 3 score heads + 3 mask coeff heads + proto.
    # DFL decode, NMS, and mask decode are implemented in post-processing.
    print('--> Loading model')

    load_model_path, outputs = add_sigmoid_to_score_heads(model_path)
    ret = rknn.load_onnx(model=load_model_path, outputs=outputs)

    if ret != 0:
        print('Load model failed!')
        sys.exit(ret)


    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=do_quant, dataset=dataset_path)
    if ret != 0:
        print('Build model failed!')
        sys.exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        sys.exit(ret)
    print('done')

    if load_model_path is not None and os.path.exists(load_model_path):
        os.remove(load_model_path)
        print(f'Removed temporary ONNX: {load_model_path}')

    # Inference test (optional)
    if do_inference:
        print('--> Init runtime for inference test')

        if device_id:
            ret = rknn.init_runtime(target=platform, device_id=device_id, core_mask=0x1)
        else:
            ret = rknn.init_runtime(target=platform)

        if ret != 0:
            print('Init runtime failed!')
            sys.exit(ret)
        print('done')

        print('--> Running inference test')
        # Generate random input data for testing
        data = np.random.randint(0, 256, (1, 640, 640, 3), dtype=np.uint8)

        if accuracy_analysis :
            # Accuracy analysis only supported on RKNN3 platforms
            output = rknn.inference(inputs=[data], accuracy_analysis=True)
            print(f'Accuracy analysis done, output shapes: {[o.shape for o in output]}')
        else:
            output = rknn.inference(inputs=[data])
            print(f'Inference test done, output shapes: {[o.shape for o in output]}')

    # Release
    rknn.release()

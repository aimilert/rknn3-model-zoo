import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from py_utils.export_vision_helper import export_qwen2_vl_vision
from py_utils.tools import clear_llm_external_weight_in_dir
from transformers import AutoConfig, Qwen3_5ForConditionalGeneration
import torch
import torch.nn as nn



class Qwen3_5VisionExportWrapper(nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor):
        outputs = self.visual(
            pixel_values,
            grid_thw=grid_thw,
            return_dict=True,
        )
        return outputs.pooler_output

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export Qwen/Qwen3.5-0.8B vision configuration and onnx model for RKNN")
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--export_vision_path", type=str, help="export vision onnx model path", required=False, default="../../model/vision/Qwen3.5-0.8B-vision.onnx")
    parser.add_argument("--quant", action='store_true', help="Whether use GRQ quantization")
    parser.add_argument("--cali_dataset", default='./quant_data/model_inputs.json', help="some samples for grq quantized_algorithm")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    parser.add_argument("--img_h", type=int, help="Input image size (e.g., 384, 416, 448). Must be a multiple of 32.", required=False, default=384)
    parser.add_argument("--img_w", type=int, help="Input image size (e.g., 384, 416, 448). Must be a multiple of 32.", required=False, default=384)
    
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    kwargs = {
        'trust_remote_code': True,
    }
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    kwargs['config'] = config
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.float32,
        # 注意此处的数据类型，由于 rknn 目前仅支持 float32 ，因此需要指定；若是在加载权重时限制了数据类型，需要自行修改config.json中的 "use_flash_attn" 参数为 false
        low_cpu_mem_usage=True, _attn_implementation="eager",
        trust_remote_code=True)
    
    if args.quant and torch.cuda.is_available():
        from rknn.quantization.api import RKQuantizer

        ## 初始化量化工具
        QuantTool = RKQuantizer(verbose=True)
        
        ## 量化工具加载模型
        ret = QuantTool.load_model(model=model.model.visual, tokenizer=None, device='cuda')
        if ret != 0:
            print('Load model failed!')
            exit(ret)
        
        ## 执行量化算法
        dataset = args.cali_dataset
        model.model.visual = QuantTool.quantize(quantized_dtype="w4a16", quantized_method="group32", quantized_algorithm="grq", dataset=dataset)
        

        model = model.cpu()

    export_vision_dirname = os.path.dirname(args.export_vision_path)
    if export_vision_dirname and not os.path.exists(export_vision_dirname):
        print(f"create export_vision_dirname: {export_vision_dirname}")
        os.makedirs(export_vision_dirname)

    # patch_size 为 16；若 config 里有值则以 config 为准
    patch_size = 16
    if getattr(config, "vision_config", None) is not None and config.vision_config.patch_size is not None:
        patch_size = config.vision_config.patch_size
    else:
        print(f"vision_config.patch_size is None, use default value {patch_size}")

    vision_wrapper = Qwen3_5VisionExportWrapper(model.model.visual).eval()

    # export vision model
    export_qwen2_vl_vision(vision_wrapper, args, patch_size) # 添加grid_thw对输入图片的patch

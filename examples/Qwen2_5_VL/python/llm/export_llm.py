import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from py_utils.export_llm_helper import causal_llm_to_onnx, update_config, export_tokenizer, export_llm_config, export_embed_weight
from py_utils.tools import clear_llm_external_weight_in_dir, gen_qwen2_5_vl_quantize_dataset
from transformers import AutoConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration # 增加 num_logits_to_keep 输入

prompt = "RKLLM"
chat_context = {
    "messages":[
        {
            "role": "user",
            "content": [
                {"type": "image",},
                {"type": "text", "text": prompt},
            ],
        }
    ],
    "add_generation_prompt": True,
}

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export Qwen/Qwen2.5-VL llm configuration and onnx model for RKNN")
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--export_llm_path", type=str, help="export llm onnx model path", required=False, default="../../model/llm/Qwen2.5-VL-3B-llm.onnx")
    parser.add_argument("--quant", action='store_true', help="Whether GRQ quantization")
    parser.add_argument("--cali_dataset", default='./quant_data/model_inputs.json', help="some samples for grq quantized_algorithm")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    ATTN_IMPL = None
    try:
        ## 使用flashattention可以极大减少grq运行显存占用，请自行安装，注意flashattention要求模型加载类型为bf16/fp16，建议使用bf16。
        import flash_attn
        ATTN_IMPL = "flash_attention_2"
        print("Use FlashAttention2")
    except Exception as e:
        print(
            f"FlashAttention2 is unavailable ({e}), "
            "falling back to standard attention."
            "It is recommended to install FlashAttention2, as standard attention requires a large amount of video memory."
    )
    kwargs = {
        'trust_remote_code': True,
        'torch_dtype': torch.bfloat16 
    }
    if ATTN_IMPL is not None:
        kwargs["attn_implementation"] = ATTN_IMPL
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    update_config(config, ['use_cache'], False)
    update_config(config, ['_attn_implementation_autoset'], False)
    kwargs['config'] = config
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, **kwargs)
    
    if args.quant and torch.cuda.is_available():
        from rknn.quantization.api import RKQuantizer


        ## 初始化量化工具
        QuantTool = RKQuantizer(verbose=True)
        
        ## 量化工具加载模型
        ret = QuantTool.load_model(model=model.model, tokenizer=None, device='cuda')
        if ret != 0:
            print('Load model failed!')
            exit(ret)
        
        ## 执行量化算法
        dataset = args.cali_dataset
        model.model = QuantTool.quantize(quantized_dtype="w4a16", quantized_method="group32", quantized_algorithm="grq", dataset=dataset)

        model = model.cpu()
        
        if ATTN_IMPL is not None:
            ## 后续导出onnx模型不能使用flashattention
            fakeq_state = model.state_dict()
            del model
            torch.cuda.empty_cache()
            kwargs = {
                'trust_remote_code': True,
                "attn_implementation": 'eager'
            }
            config = AutoConfig.from_pretrained(args.model_path, **kwargs)
            update_config(config, ['use_cache'], False)
            update_config(config, ['_attn_implementation_autoset'], False)
            kwargs['config'] = config
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, **kwargs)
            model.load_state_dict(
                fakeq_state,
                strict=False
            )

    export_llm_dirname = os.path.dirname(args.export_llm_path)
    if not os.path.exists(export_llm_dirname):
        os.makedirs(export_llm_dirname)

    # Export llm to onnx
    causal_llm_to_onnx(model, args)

    # Export LLM configuration 
    export_llm_config(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.config.pkl', chat_context, prompt)

    # Export tokenizer
    export_tokenizer(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.tokenizer.gguf')

    # Export embedding weight
    export_embed_weight(model.model.embed_tokens.weight, os.path.splitext(args.export_llm_path)[0] + '.embed.bin')
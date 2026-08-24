import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
import torch
import importlib.util
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from py_utils.export_llm_helper import update_config, export_tokenizer, export_llm_config, export_embed_weight
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

from export_onnx import causal_gemma3_to_onnx

# Get the custom module path
custom_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'gemma3'))
print(f"Custom gemma3 module path: {custom_path}")

# Load our custom modeling_gemma3 module using importlib
spec = importlib.util.spec_from_file_location(
    'transformers.models.gemma3.modeling_gemma3',
    os.path.join(custom_path, 'modeling_gemma3.py')
)
modeling_module = importlib.util.module_from_spec(spec)
sys.modules['transformers.models.gemma3.modeling_gemma3'] = modeling_module
spec.loader.exec_module(modeling_module)
print(f"Loaded custom modeling_gemma3 from: {modeling_module.__file__}")

def fuse_gemma3_embed_scale(model):
    """FunctionGemma/Gemma3 ties input embeddings and lm_head by default, and
    embed_tokens carries an `embed_scale` that must be folded into the weight
    before export. lm_head is detached first so the scale only affects the
    input embedding stream. This mirrors the conversion_scripts pipeline."""
    embed_tokens = model.model.embed_tokens
    embed_scale = getattr(embed_tokens, "embed_scale", None)
    if embed_scale is None:
        return
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    with torch.no_grad():
        scale = embed_scale.to(embed_tokens.weight.dtype).to(embed_tokens.weight.device)
        embed_tokens.weight.mul_(scale)
        embed_scale.fill_(1.0)
    print("Fused Gemma3 embedding scale into embed_tokens.weight")


prompt = "RKLLM"
chat_context = {
    "messages":[
        {"role": "user", "content": prompt}
    ],
    "add_generation_prompt": True,
}

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export google/functiongemma-270m-it llm configuration and onnx model for RKNN")
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="google/functiongemma-270m-it")
    parser.add_argument("--export_llm_path", type=str, help="export llm onnx model path", required=False, default="../model/llm/functiongemma-270m-it.onnx")
    parser.add_argument("--quant", action='store_true', help="Whether use GRQ quantization")
    parser.add_argument("--cali_dataset", default='../../../datasets/llm_quant.json', help="some samples for grq quantized_algorithm")
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
    kwargs['config'] = config
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
    fuse_gemma3_embed_scale(model)

    if args.quant and torch.cuda.is_available():
        from rknn.quantization.api import RKQuantizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

        ## 初始化量化工具
        QuantTool = RKQuantizer(verbose=True)

        ## 量化工具加载模型
        ret = QuantTool.load_model(model=model, tokenizer=tokenizer, device='cuda',
                                   system_prompt=None, tools=None, system_role='system', user_role='user')
        if ret != 0:
            print('Load model failed!')
            exit(ret)

        ## 执行量化算法
        dataset = args.cali_dataset
        model = QuantTool.quantize(quantized_dtype="w4a16", quantized_method="group32", quantized_algorithm='grq', dataset=dataset)

        model = model.cpu()

    export_dirname = os.path.dirname(args.export_llm_path)
    if not os.path.exists(export_dirname):
        os.makedirs(export_dirname)

    # Export llm to onnx
    ## 导出onnx时，要求attention实现方式必须为eager
    if ATTN_IMPL is not None:
        model.config._attn_implementation = "eager"
        for layer in model.model.layers:
            layer.self_attn.config._attn_implementation = "eager"
    causal_gemma3_to_onnx(model, args)

    # Export LLM configuration
    export_llm_config(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.config.pkl', chat_context, prompt)

    # Export tokenizer
    export_tokenizer(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.tokenizer.gguf')

    # Export embedding weight
    export_embed_weight(model.model.embed_tokens.weight, os.path.splitext(args.export_llm_path)[0] + '.embed.bin')

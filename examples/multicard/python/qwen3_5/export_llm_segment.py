import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
import torch
import importlib.util

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from py_utils.export_llm_helper import causal_qwen_3_5_llm_to_onnx, causal_qwen_3_5_llm_to_onnx_multi_segment, update_config, export_tokenizer, export_llm_config, export_embed_weight
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

try:
    import transformers.modeling_flash_attention_utils as _fa_utils
    _orig_is_packed_sequence = _fa_utils._is_packed_sequence
    def _is_packed_sequence(position_ids, batch_size):
        return False
    _fa_utils._is_packed_sequence = _is_packed_sequence
except:
    pass

prompt = "RKLLM"
chat_context = {"role": "user", "content": prompt}

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export Qwen3_5 multi-segment ONNX model for multicard deployment")
    parser.add_argument("--model_path", type=str, required=False, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--quant", action='store_true')
    parser.add_argument("--cali_dataset", default='../../../../datasets/llm_quant.json')
    parser.add_argument("--export_llm_path", type=str, required=False, default="../../model/llm_qwen/Qwen3.5-9B-llm.onnx")
    parser.add_argument("--multi_segment", action='store_true')
    parser.add_argument("--num_segments", type=int, default=0)
    parser.add_argument("--modelscope", action='store_true')
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    fakeq_state = None
    if args.quant and torch.cuda.is_available():
        ATTN_IMPL = None
        try:
            import flash_attn
            ATTN_IMPL = "flash_attention_2"
            print("Use FlashAttention2")
        except Exception as e:
            print(f"FlashAttention2 is unavailable ({e}), falling back to standard attention.")
        from rknn.quantization.api import RKQuantizer
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            _attn_implementation=ATTN_IMPL if ATTN_IMPL is not None else "eager",
            trust_remote_code=True)
        QuantTool = RKQuantizer(verbose=True)
        ret = QuantTool.load_model(model=model, tokenizer=tokenizer, device='cuda',
                                   system_prompt=None, tools=None, system_role='system', user_role='user')
        if ret != 0:
            print('Load model failed!')
            sys.exit(ret)
        dataset = args.cali_dataset
        quant_model = QuantTool.quantize(quantized_dtype="w4a16", quantized_method="group32",
                                         quantized_algorithm='grq', dataset=dataset,
                                         auto_hybrid_rate=0.)
        quant_model = quant_model.cpu()
        fakeq_state = quant_model.state_dict()
        del quant_model

    custom_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    modeling_spec = importlib.util.spec_from_file_location(
        'transformers.models.qwen3_5.modeling_qwen3_5',
        os.path.join(custom_path, 'modeling_qwen3_5_export.py'))
    modeling_module = importlib.util.module_from_spec(modeling_spec)
    sys.modules['transformers.models.qwen3_5.modeling_qwen3_5'] = modeling_module
    modeling_spec.loader.exec_module(modeling_module)

    def pack_dynamic_cache(conv_states, recurrent_states, key_cache, value_cache, model_config):
        past_key_values = modeling_module.Qwen3_5DynamicCache(model_config)
        assert conv_states.shape[0] == past_key_values.layer_types.count('linear_attention'), "conv_states mismatch"
        conv_states = list(torch.split(conv_states, dim=0, split_size_or_sections=1))
        for i, layer_types in enumerate(past_key_values.layer_types):
            if layer_types == 'linear_attention':
                past_key_values.conv_states[i] = conv_states.pop(0)
        return past_key_values

    class Qwen3_5_export_model_cst(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.config = model.config

        def forward(self, input_ids, position_ids, attention_mask, Tc, logits_to_keep, conv_status,
                    recurrent_status=None, key_cache=None, value_cache=None):
            past_key_values = pack_dynamic_cache(conv_status, recurrent_status, key_cache, value_cache, self.model.config)
            model_out = self.model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask,
                                   past_key_values=past_key_values, Tc=Tc, logits_to_keep=logits_to_keep)
            logits = model_out.logits
            conv_status = model_out.past_key_values.conv_states
            conv_status = [v for v in conv_status if v is not None]
            conv_status = torch.concat(conv_status, dim=0)
            return logits, conv_status, None, None, None

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model = modeling_module.Qwen3_5ForCausalLM.from_pretrained(args.model_path, torch_dtype="float32", device_map="cpu")

    if fakeq_state is not None:
        model.load_state_dict(fakeq_state, strict=False)

    export_dirname = os.path.dirname(args.export_llm_path)
    if not os.path.exists(export_dirname):
        os.makedirs(export_dirname)

    exported_model = Qwen3_5_export_model_cst(model)
    args.tokenizer = tokenizer
    args.chat_context = [chat_context]

    if args.multi_segment:
        causal_qwen_3_5_llm_to_onnx_multi_segment(exported_model, args)
    else:
        causal_qwen_3_5_llm_to_onnx(exported_model, args)

    if args.quant and torch.cuda.is_available():
        QuantTool.export_op_quantized_dtype(args.export_llm_path, op_dtype_path='layer_bit.json')

    export_llm_config(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.config.pkl', chat_context, prompt)
    export_tokenizer(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.tokenizer.gguf')
    export_embed_weight(model.get_input_embeddings().weight, os.path.splitext(args.export_llm_path)[0] + '.embed.bin')
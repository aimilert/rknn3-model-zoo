import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from py_utils.export_llm_helper import causal_llm_to_onnx, export_llm_config, export_embed_weight, export_tokenizer
from py_utils.tools import clear_llm_external_weight_in_dir
from transformers import AutoConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import ChineseCLIPModel


def _get_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    square_tensor = torch.pow(tensor, 2)
    sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
    normed_tensor = torch.pow(sum_tensor, 0.5)
    return normed_tensor


class ChineseCLIPTextModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config.text_config

    def forward(self, input_ids, attention_mask=None, position_ids=None, num_logits_to_keep=None):
        if attention_mask is not None and attention_mask.dim() == 2:
            seq_len = attention_mask.shape[-1]
            attention_mask = attention_mask[:, None, :].expand(-1, seq_len, -1)

        text_outputs = self.model.text_model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, return_dict=True
        )

        # get sequence output robustly (ModelOutput or tuple)
        if isinstance(text_outputs, tuple):
            sequence_output = text_outputs[0]
        else:
            sequence_output = getattr(text_outputs, 'last_hidden_state', text_outputs[0])

        # CLIP-style pooling normally takes the first token. During RKNN export,
        # keep the requested token index in the graph so num_logits_to_keep is an ONNX input.
        # if num_logits_to_keep is not None:
        #     pooled = sequence_output[:, num_logits_to_keep:, :]
        # else:
        #     pooled = sequence_output[:, -1, :]
        pooled = sequence_output[:, 0, :]

        text_embeds = self.model.text_projection(pooled)
        text_embeds = text_embeds / _get_vector_norm(text_embeds)
        return text_embeds


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export Chinese-CLIP text model to ONNX for RKNN")
    parser.add_argument("--load_weight", type=int, help="Whether load model weight", required=False, default=False)
    parser.add_argument("--model_path", type=str, help="model path or name", required=False,
                       default="../../models/QA-CLIP-ViT-L-14")
    parser.add_argument("--export_llm_path", type=str, help="export llm onnx model path", required=False,
                       default="./chinese_clip_text.onnx")
    args = parser.parse_args()

    kwargs = {
        'trust_remote_code': True,
    }

    config = AutoConfig.from_pretrained(args.model_path, **kwargs)

    model = ChineseCLIPModel.from_pretrained(args.model_path, torch_dtype=torch.float16, trust_remote_code=True)

    export_llm_dirname = os.path.dirname(args.export_llm_path)
    if export_llm_dirname and not os.path.exists(export_llm_dirname):
        os.makedirs(export_llm_dirname)

    wrapped_model = ChineseCLIPTextModelWrapper(model)
    wrapped_model.eval()

    # set some expected fields for exporter
    args.hidden_size = config.text_config.hidden_size
    args.vocab_size = config.text_config.vocab_size

    print("Exporting Chinese-CLIP text model to ONNX...")
    causal_llm_to_onnx(wrapped_model, args)
    # causal_llm_to_qaclipvitl14_text_onnx(wrapped_model, args)

    # export config info
    embedding_config = {
        "model_type": "chinese_clip_text",
        "hidden_size": config.text_config.hidden_size,
        "projection_dim": config.projection_dim if hasattr(config, 'projection_dim') else config.text_config.hidden_size,
        "max_position_embeddings": config.text_config.max_position_embeddings,
        "vocab_size": config.text_config.vocab_size,
    }

    user_config = {"task_type": 1}
    export_llm_config(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.config.pkl',
                     embedding_config, None, user_config)

    # Export tokenizer
    export_tokenizer(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.tokenizer.gguf')

    # export embedding weight if available
    try:
        embed_weight = wrapped_model.model.text_model.embeddings.word_embeddings.weight
        export_embed_weight(embed_weight, os.path.splitext(args.export_llm_path)[0] + '.embed.bin')
    except Exception:
        print('No embedding weight exported (attribute not found)')

    if not args.load_weight:
        clear_llm_external_weight_in_dir(export_llm_dirname)

    print(f"Chinese-CLIP text model exported successfully to {args.export_llm_path}")

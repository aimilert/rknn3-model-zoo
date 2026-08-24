"""
使用torch.jit.trace导出静态shape的Encoder模型
"""

import os
import sys
import torch
import argparse
import yaml


HYPS_LEN=13
ENCODER_TIME=200
VOCAB_SIZE=4233
WENET_PATH='<path2wenet>/Wenet/wenet'


sys.path.insert(0, WENET_PATH)
from wenet.utils.init_model import init_model

def fix_decoder_for_rknn(input_onnx, output_onnx):
    """将模型中所有LogSoftmax替换为Softmax，保留原始axis"""
    import onnx
    from onnx import helper

    model = onnx.load(input_onnx)

    to_replace = []
    for node in model.graph.node:
        if node.op_type == 'LogSoftmax':
            to_replace.append(node)

    for old in to_replace:
        axis = -1
        for attr in old.attribute:
            if attr.name == 'axis':
                axis = attr.i
                break
        new_node = helper.make_node(
            'Softmax',
            inputs=list(old.input),
            outputs=list(old.output),
            name=old.name.replace('LogSoftmax', 'Softmax'),
            axis=axis,
        )
        model.graph.node.remove(old)
        model.graph.node.append(new_node)
        print(f"  LogSoftmax -> Softmax  axis={axis}  name={old.name}")

    onnx.save(model, output_onnx)

def export_ctc(model, output_dir):
    ctc = model.ctc
    #todo
    ctc.forward = ctc.log_softmax

    hidden = torch.randn((1, 16, 256))  # batch, time, hidden
    out_onnx_path = os.path.join(output_dir, 'ctc.onnx') 

    torch.onnx.export(
        ctc,
        hidden,
        out_onnx_path,
        opset_version=14,
        input_names=['hidden'],
        output_names=['probs'],
        dynamic_axes=None,
    )
    fix_decoder_for_rknn(out_onnx_path, out_onnx_path)

class DecoderWrapper(torch.nn.Module):
    def __init__(self, model, batch_size):
        super().__init__()
        self.model = model
        self.batch_size = batch_size

    def forward(self, hyps, encoder_out, reverse_weight):
        # 固定hyps_lens为batch_size
        hyps_len = hyps.shape[1]
        hyps_lens = torch.full((self.batch_size,), hyps_len,
                               dtype=torch.long, device=hyps.device)
        score, r_score = self.model.forward_attention_decoder(
            hyps, hyps_lens, encoder_out, reverse_weight
        )
        return score, r_score


def export_decoder(model, output_dir, batch_size=10, dict_size=5538, hyps_len=20, encoder_time=200):
    output_size = model.encoder.output_size()

    decoder = DecoderWrapper(model, batch_size)
    decoder.eval()

    hyps = torch.randint(0, dict_size, (batch_size, hyps_len))
    encoder_out = torch.randn((1, encoder_time, output_size))
    reverse_weight = torch.tensor([0.3]) # for open-source model

    with torch.no_grad():
        traced = torch.jit.trace(decoder, (hyps, encoder_out, reverse_weight))
    
    out_onnx_path = os.path.join(output_dir, 'decoder.onnx')

    torch.onnx.export(
        traced,
        (hyps, encoder_out, reverse_weight),
        out_onnx_path, # os.path.join(output_dir, 'decoder.onnx'),
        opset_version=14,
        input_names=['hyps', 'encoder_out', 'encoder_out'],
        output_names=['score', 'r_score'],
        dynamic_axes=None,
    )
    fix_decoder_for_rknn(out_onnx_path, out_onnx_path)



class EncoderWrapper(torch.nn.Module):
    """
    Encoder包装器，将动态参数固化为常量
    """
    def __init__(self, encoder, required_cache_size):
        super().__init__()
        self.encoder = encoder
        self.required_cache_size = required_cache_size

    def forward(self, chunk, att_cache, cnn_cache):
        # 固定offset和required_cache_size
        offset = self.required_cache_size
        required_cache_size = self.required_cache_size

        # 在forward内部创建att_mask，避免作为输入
        if required_cache_size > 0:
            total_len = required_cache_size + chunk.size(1)
            positions = torch.arange(total_len, device=chunk.device)
            att_mask = (positions >= required_cache_size).float().unsqueeze(0).unsqueeze(0)
        else:
            att_mask = torch.ones(0, 0, 0, dtype=torch.float32, device=chunk.device)

        output, r_att_cache, r_cnn_cache = self.encoder.forward_chunk(
            chunk, offset, required_cache_size, att_cache, cnn_cache, att_mask
        )
        return output, r_att_cache, r_cnn_cache


def export_encoder(config_path, checkpoint_path, output_dir,
                   chunk_size=16, num_decoding_left_chunks=4):
    """导出静态shape的Encoder"""

    # 加载配置
    with open(config_path, 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    # 获取模型参数
    output_size = configs['encoder_conf']['output_size']
    num_blocks = configs['encoder_conf']['num_blocks']
    head = configs['encoder_conf']['attention_heads']
    cnn_module_kernel = configs['encoder_conf'].get('cnn_module_kernel', 1)
    feature_size = configs['input_dim']

    # 计算shape
    decoding_window = (chunk_size - 1) * 4 + 6 + 1  # subsampling_rate=4, right_context=6
    required_cache_size = chunk_size * num_decoding_left_chunks
    att_cache_shape = (num_blocks, head, required_cache_size, output_size // head * 2)
    # 12, 1, 512, 14
    cnn_cache_shape = (num_blocks, 1, output_size, cnn_module_kernel - 1)

    print(f"decoding_window: {decoding_window}")
    print(f"att_cache_shape: {att_cache_shape}")
    print(f"cnn_cache_shape: {cnn_cache_shape}")

    # 加载模型
    class Args:
        pass
    args = Args()
    args.config = config_path
    args.checkpoint = checkpoint_path
    args.model = configs.get('model', 'conformer')
    args.cmvn = None
    args.idx = 0

    model, _ = init_model(args, configs)
    model.eval()

    # 创建wrapper
    wrapper = EncoderWrapper(model.encoder, required_cache_size)
    wrapper.eval()

    # 创建静态输入
    chunk = torch.randn((1, decoding_window, feature_size))
    att_cache = torch.zeros(att_cache_shape)
    cnn_cache = torch.zeros(cnn_cache_shape)

    # 使用torch.jit.trace
    print("Tracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (chunk, att_cache, cnn_cache))

    # 导出ONNX
    output_path = os.path.join(output_dir, 'encoder.onnx')
    torch.onnx.export(
        traced,
        (chunk, att_cache, cnn_cache),
        output_path,
        # opset_version=14,
        export_params=True,
        do_constant_folding=True,
        input_names=['chunk', 'att_cache', 'cnn_cache'],
        output_names=['output', 'r_att_cache', 'r_cnn_cache'],
        dynamic_axes=None,  # 无动态轴
    )

    # export decoder
    _dict_size = VOCAB_SIZE 
    # note: encoder time is not fix , should be based on fbank frame
    export_decoder(model, output_dir, batch_size=10, dict_size=_dict_size, hyps_len=HYPS_LEN, encoder_time=ENCODER_TIME)

    #export ctc model
    export_ctc(model, output_dir)

    print(f"Exported to {output_path}")
    return output_path




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='ONNX Export for Conformer')
    parser.add_argument('--config_path', required=True, help='config path')
    parser.add_argument('--checkpoint_path', required=True, help='check point path')
    parser.add_argument('--output_dir', required=True, help='onnx output path')

    args = parser.parse_args()

    conf_path = args.config_path 
    check_points_path = args.checkpoint_path
    out_path = args.output_dir

    args = parser.parse_args()
    export_encoder(
        config_path=conf_path,
        checkpoint_path=check_points_path,
        output_dir=out_path,
    )
   
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from py_utils.export_llm_helper import check_gptq, register_bitwise_right_shift


def causal_gemma4_to_onnx(model, args):
    import torch

    # Create a wrapper class to avoid DynamicCache issues
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            # del self.model.vision_tower
            # del self.model.audio_tower
            
        def forward(self, input_ids, per_layer_inputs, attention_mask_global, position_ids_global,
                    attention_mask_local, position_ids_local, num_logits_to_keep):
            # 参数名称必须与 input_names 中的名称一致
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                cache_position=None,
                attention_mask_global=attention_mask_global,
                attention_mask_local=attention_mask_local,
                position_ids_global=position_ids_global,
                position_ids_local=position_ids_local,
                per_layer_inputs=per_layer_inputs,
                logits_to_keep = num_logits_to_keep,
            )
            return outputs.logits
    
    # Wrap the model
    model.eval().float()
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    model.model.language_model.embed_tokens.fuse_scale()
    wrapped_model = ModelWrapper(model)

    # Debug Parameter
    args.prompt_size = 64
    args.dynamic_shape = True

    model.eval()
    wrapped_model.eval()
    in_len = args.prompt_size

    dummy_input = torch.zeros((1, in_len), dtype=torch.long)
    per_layer_inputs = torch.randn(1, in_len, model.config.text_config.num_hidden_layers, model.config.text_config.hidden_size_per_layer_input, dtype=torch.float32)  # 假设有30层，每层256维的输入
    
    attention_mask_global = torch.ones((1, in_len), dtype=torch.float)
    position_ids_global = torch.arange(0, in_len, dtype=torch.long).unsqueeze(0)

    attention_mask_local = torch.ones((1, in_len), dtype=torch.float)
    position_ids_local = torch.arange(0, in_len, dtype=torch.long).unsqueeze(0)

    inputs = (dummy_input, 
              per_layer_inputs,
              attention_mask_global, 
              position_ids_global, 
              attention_mask_local, 
              position_ids_local
              )
    #打印inputs的成员的shape
    for i, input_tensor in enumerate(inputs):
        print(f"Input {i} shape: {input_tensor.shape}")
    
    input_names = ["input_ids", "per_layer_inputs","attention_mask", "position_ids", "attention_mask_1", "position_ids_1"]
    
    dynamic_axes = {}
    if args.dynamic_shape:
        dynamic_axes.update({
            'input_ids': {1: 'sequence'},
            'per_layer_inputs': {1: 'sequence'},
            'attention_mask': {1: 'sequence'},
            'position_ids': {1: 'sequence'},
            'attention_mask_1': {1: 'sequence'},
            'position_ids_1': {1: 'sequence'},
        })

    # 获取 forward 参数
    forward_args = model.forward.__code__.co_varnames
    for i in range(3):
        name = f"deepstack_embeds{i}"
        if name in forward_args:
            embed = torch.randn(1, in_len, args.hidden_size, dtype=torch.float32)
            inputs = (*inputs, embed)
            input_names.append(name)
            dynamic_axes[name] = {1: "sequence"}
            print(f"Added optional input: {name}")

    output_names = ["output"]

    logit_keep_keys = ['logits_to_keep', 'num_logits_to_keep']
    logit_keep_key  = None
    _forward_func = model.forward
    while hasattr(_forward_func, '__wrapped__'):
        _forward_func = _forward_func.__wrapped__

    for key in logit_keep_keys:
        if key in _forward_func.__code__.co_varnames:
            logit_keep_key = key
            break
    if logit_keep_key:
        # 只留最后一个 token 的 logits，减少计算量
        num_logits_to_keep = torch.tensor(-1, dtype=torch.int32).reshape(1)
        insert_nones = [None]* (_forward_func.__code__.co_varnames.index(logit_keep_key) - len(inputs) -1)
        # inputs = (*inputs, *insert_nones, num_logits_to_keep)
        inputs = (*inputs, num_logits_to_keep)
        input_names.append('num_logits_to_keep')


    if getattr(args, 'output_hidden_states', False) and 'output_hidden_states' in _forward_func.__code__.co_varnames:
        idx = _forward_func.__code__.co_varnames.index('output_hidden_states') - 1
        if idx < len(inputs):
            # 如果有 output_hidden_states 参数，则需要在输入中添加一个 None
            inputs = (*inputs[:idx], True, *inputs[idx+1:])
        else:
            inputs = (*inputs, *((None,)*(_forward_func.__code__.co_varnames.index('output_hidden_states') - len(inputs) - 1)), True)

    if hasattr(model.config, 'quantization_config'):
        q_config = model.config.quantization_config
        if check_gptq(q_config.bits, q_config.group_size) == False:
            print("GRQ model quantization not supported. Only W4A16 quantization grouped or channel asymmetric/symmetric with group_size in {32, 64, 128} or -1 (per-channel) is supported.")
            exit(1)
        register_bitwise_right_shift()
    else:
        model.float()

    # out = model(*inputs)
    # if len(out) != len(output_names):
    #     print(f"WARNING: output number not match, expect {len(output_names)}, got {len(out)}")
    #     print(f"WARNING: try only keep one output")
    #     output_names = output_names[:1]

    with torch.no_grad():
        torch.onnx.export(
            wrapped_model,
            inputs,
            args.export_llm_path,
            export_params=True,
            opset_version=19,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamo=True
        )

    if False:
        ort.set_default_logger_severity(3)  # 设置 ONNX Runtime 日志级别为 WARNING 及以上
        # class MyLoggingHandler(ort.LoggingHandler):
        #     def __init__(self):
        #         super().__init__()
            
        #     def log(self, severity, category, logid, code_location, message):
        #         # 在这里自定义处理日志消息
        #         if severity < 3:  # 只记录 WARNING 及以上级别的日志
        #             print(f"[ONNX Runtime] {message}")

        # # 注册自定义日志处理器
        # ort.set_default_logger(MyLoggingHandler())

        # check onnx model result
        print("Checking ONNX model output...")
        sess_onnx = ort.InferenceSession(args.export_llm_path, providers=['CPUExecutionProvider'])
        valid_inputs = [v.numpy() if isinstance(v, torch.Tensor) else v for v in inputs if v is not None]
        input_feed   = {sess_onnx.get_inputs()[i].name: valid_inputs[i] for i in range(len(valid_inputs))}
        for k, v in input_feed.items():
            if isinstance(v, np.ndarray):
                print(f"Input {k}: {v.shape} {v.dtype} {v}")
            else:
                print(f"Input {k}: {type(v)} {v}")

        output_onnx  = sess_onnx.run(output_names, input_feed)
        output_name  = sess_onnx.get_outputs()[0].name
        cos_sim      = torch.cosine_similarity(torch.tensor(output_onnx[0]).reshape(1,-1), out[0].detach().cpu().reshape(1,-1), dim=-1)
        euler_dist   = torch.dist(torch.tensor(output_onnx[0]).reshape(1,-1), out[0].detach().cpu().reshape(1,-1), p=2)
        abs_diff     = torch.abs(torch.tensor(output_onnx[0]) - out[0].detach().cpu())
        print(f"Cosine Similarity       : {cos_sim.item()}")
        print(f"Euclidean Distance      : {euler_dist.item()}")
        print(f"Max Absolute Difference : {abs_diff.max().item()}")
        print("first 10 elements of output:")
        print(f"ONNX output: {output_onnx[0].reshape(-1)[:10]}")
        print(f"PyTorch output: {out[0].detach().cpu().numpy().reshape(-1)[:10]}")

        if True:
            save_dir = os.path.join(os.path.dirname(args.export_llm_path), "src_io")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for k, v in input_feed.items():
                if isinstance(v, np.ndarray):
                    np.save(os.path.join(save_dir, f"{k}.npy"), v)
                else:
                    with open(os.path.join(save_dir, f"{k}.txt"), 'w') as f:
                        f.write(str(v))
            np.save(os.path.join(save_dir, f"{output_name}_onnx.npy"), output_onnx[0])
            np.save(os.path.join(save_dir, f"{output_name}_torch.npy"), out[0].detach().cpu().numpy())
            print(f"ONNX model input/output saved to {save_dir}")

    print(f"Exported to {os.path.abspath(args.export_llm_path)}")

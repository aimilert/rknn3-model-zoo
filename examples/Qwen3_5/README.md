# Qwen3.5 模型部署说明

## 1. 环境要求

Qwen3.5 模型导出依赖固定版本环境。请优先使用如下版本，避免因 transformers、ONNX 或
ONNXRuntime 版本差异导致模型结构、动态维度或算子导出不一致。

```bash
torch                             2.7.0
transformers                      5.3.0
onnx                              1.18.0
onnxruntime                       1.22.1
```

> ⚠️ **注意**：
> 
> - 若需要从 ModelScope 下载模型，可在导出 ONNX 时增加 `--modelscope` 参数。

## 2. 支持的模型

目前支持 Qwen3.5 系列 LLM 模型。导出时请指定对应的模型路径。

以 **Qwen3.5-2B** 为例：

```bash
# 进入脚本目录
cd examples/Qwen3_5/python

# 导出 LLM ONNX 模型，--quant表示使用grq量化，取消则导出float模型
python export_llm.py \
    --model_path Qwen/Qwen3.5-2B \
    --export_llm_path ../model/Qwen3.5-2B.onnx --quant

# 导出 LLM RKNN 模型
python export_rknn.py \
    --onnx_path ../model/Qwen3.5-2B.onnx \
    --config ../model/Qwen3.5-2B.config.pkl \
    --rknn_path ../model/Qwen3.5-2B.rknn \
    --platform rk1820

# 如需重新导出（仅修改 profile_mode 或 kvcache 相关参数），可使用 --rebuild 跳过 ONNX 加载和图优化等步骤，加速模型导出：
python export_rknn.py \
    --onnx_path ../model/Qwen3.5-2B.onnx \
    --config ../model/Qwen3.5-2B.config.pkl \
    --rknn_path ../model/Qwen3.5-2B.rknn \
    --platform rk1820 \
    --rebuild
```

> ⚠️ **`--rebuild` 说明**：当前仅支持重置 `profile_mode` 和 `llm_config` 中 kvcache 相关参数（如 `kvcache_buffer_len`或者`max_position_embeddings` 等）。其他参数变更需走完整导出流程。

执行 `export_llm.py` 后，会在 ONNX 同目录下同步生成以下文件：

```bash
Qwen3.5-2B.onnx
Qwen3.5-2B.config.pkl
Qwen3.5-2B.tokenizer.gguf
Qwen3.5-2B.embed.bin
```

## 3. RKNN LLM 量化配置

Qwen3.5 的 RKNN 转换默认使用 W4A16 量化，并采用 group32 量化方式。
转换脚本中默认目标平台为 `rk1820`，同时会根据 ONNX 输入信息自动生成动态输入配置。

示例配置如下：

```python
llm_config = DEFAULT_RKNN_LLM_CONFIG.copy()
llm_config['attention_config'][0]['kvcache_buffer_len'] = 3 * 1024
llm_config['attention_config'][0]['max_position_embeddings'] = 3 * 1024

rknn.config(
    target_platform='rk1820',
    dynamic_input=dynamic_shapes,
    quantized_dtype='w4a16',
    quantized_algorithm='normal',
    quantized_method='group32',
    llm_config=llm_config,
)
```

> ⚠️ **注意**：
> - 上述配置位于 `examples/Qwen3_5/python/export_rknn.py` 文件中，请根据实际需求调整
  `kvcache_buffer_len`、`max_position_embeddings` 等参数。
> - 目前模型不支持外部和内部grq

## 4. KV Cache INT4 量化

在大规模语言模型推理过程中，KV Cache（Key/Value Cache）用于存储历史的注意力键值，以避免重
复计算，从而提高推理速度。随着序列长度增长，KV Cache 的内存占用会快速增加。为了减少 KV
Cache 的存储带宽与内存访问开销，可以采用量化方式将其从 FP16/FP32 转换为 INT8 或更低位宽表
示。

若需支持更长的上下文长度并进一步压缩 KV Cache 内存，可启用 `Int4_to_F16` 模式。
启用方式如下：

```python
llm_config = DEFAULT_RKNN_LLM_CONFIG.copy()
llm_config['attention_config'][0]['kvcache_buffer_len'] = 3 * 1024
llm_config['attention_config'][0]['max_position_embeddings'] = 3 * 1024
llm_config['attention_config'][0]['kvcache_dtype'] = 'Int4_to_F16'
llm_config['attention_config'][0]['kvcache_group_size'] = 16
llm_config['attention_config'][0]['kvcache_residual_depth'] = 64

rknn.config(
    target_platform='rk1820',
    dynamic_input=dynamic_shapes,
    quantized_dtype='w4a16',
    quantized_algorithm='normal',
    quantized_method='group32',
    llm_config=llm_config,
)
```

> ⚠️ **注意**：
> - `Int4_to_F16` 适用于更长上下文场景，但可能带来一定精度损失。
> - 修改 KV Cache 配置后，需要重新执行 RKNN 转换。

## 5. 量化数据集说明

`export_llm.py` 在启用grq量化(--quant)时必须要填入量化校准数据路径(--cali_dataset)，对于纯文本模型，可以参考datasets/llm_quant.json文件中的数据格式构建。需要注意地，量化校准数据对于grq量化精度十分重要（尤其是在垂类任务上微调过的模型），建议用户将实际应用场景中模型的输入（包含system prompt，function call等信息）作为样本，数量在20~128即可。更多关于量化数据集的说明可以参考RKNN3-Toolkit使用文档。

## 6. 混合量化说明

在默认的 W4A16 全量量化基础上，`export_llm.py` 还支持**混合量化**（mixed-precision），
即对不同层分配不同的量化位宽，从而在精度与压缩率之间取得平衡。混合量化通过
`RKQuantizer.quantize()` 的以下两个参数控制（见 `export_llm.py`）：

```python
quant_model = QuantTool.quantize(
    quantized_dtype="w4a16",
    quantized_method="group32",
    quantized_algorithm='grq',
    dataset=dataset,
    # layer_quant_config="layer_quant_config.json",  ## 手动指定每层量化位宽
    auto_hybrid_rate=0.,                             ## >0 时开启自动量化位宽分配
)
```

- **`auto_hybrid_rate`**：自动混合量化比例。设为 `0.`（默认）表示关闭；设为大于 0 的值
  （例如 `0.2`）时，量化工具会根据各层对量化误差的敏感度，自动将部分敏感层提升到更高
  位宽（如 W8A16），其余层保持 W4A16。值越大，被分配高精度的层越多，整体精度越高但模型
  体积也越大。
- **`layer_quant_config`**：手动指定每层量化位宽的 JSON 配置文件路径。适用于已经明确知道
  哪些层需要特殊位宽的场景，可精确控制每个算子的量化策略，优先级高于 `auto_hybrid_rate`
  的自动分配结果。

开启混合量化后，ONNX 导出完成后还必须调用以下接口，将每层的实际量化位宽导出为配置文件：

```python
if args.quant and torch.cuda.is_available():
    QuantTool.export_op_quantized_dtype(args.export_llm_path, op_dtype_path='layer_bit.json')
```

该调用会在导出目录下生成 `layer_bit.json`，记录每个算子最终使用的量化位宽。

随后在 `export_rknn.py` 转换 RKNN 时，需要通过 `rknn.config()` 的 `op_quantized_dtype`
参数指定该配置文件，否则 RKNN 转换不会使用混合位宽：

```python
rknn.config(
    target_platform='rk1820',
    dynamic_input=dynamic_shapes,
    quantized_dtype='w4a16',
    quantized_algorithm='normal',
    quantized_method='group32',
    llm_config=llm_config,
    op_quantized_dtype='./layer_bit.json',  ## 开启混合量化时必须指定
)
```

> ⚠️ **注意**：
> - 混合量化依赖 grq 量化流程，因此必须同时启用 `--quant` 且运行环境具备 CUDA（GPU）。
> - `export_op_quantized_dtype` 必须在 ONNX 导出之后执行，生成的 `layer_bit.json` 需与
>   RKNN 模型配套使用，更换模型或重新量化后需重新生成。
> - `auto_hybrid_rate` 与 `layer_quant_config` 可二选一或配合使用；手动配置优先级更高。

## 7. C++ 部署说明

### 7.1 编译

```bash
cd rknn3_model_zoo/

# 设置交叉编译工具链
export GCC_COMPILER=/path/to/aarch64-linux-gnu

# 编译
./build-linux.sh -t rk3588 -a aarch64 -d Qwen3_5
```

编译完成后，文件生成在 `install/rk3588_linux_aarch64/rknn_Qwen3_5_demo/` 目录。

### 7.2 所需文件

C++ 推理侧需使用 RKNN 模型、RKNN 权重文件、tokenizer 文件和 embedding 权重文件进行部署。

```bash
<llm_model_path>      ../model/Qwen3.5-2B.rknn
<llm_weight_path>     ../model/Qwen3.5-2B.weight
<tokenizer_path>      ../model/Qwen3.5-2B.tokenizer.gguf
<embedding_path>      ../model/Qwen3.5-2B.embed.bin
```

其中 `tokenizer.gguf` 和 `embed.bin` 由 `export_llm.py` 自动导出，部署时需与 RKNN 模型保持同一
模型版本。`weight_path` 为 LLM RKNN 运行时使用的权重文件，请填写实际导出的权重文件路径。

C++ demo 命令行参数如下：

```bash
./rknn_qwen3_5_demo \
    <model_path> <weight_path> \
    <tokenizer_path> <embedding_path> \
    <core_mask> <prompt>
```

参数说明：

```bash
model_path       LLM RKNN 模型路径，例如 ../model/Qwen3.5-2B.rknn
weight_path      LLM RKNN 权重文件路径，例如 ../model/Qwen3.5-2B.weight
tokenizer_path   tokenizer 文件路径，例如 ../model/Qwen3.5-2B.tokenizer.gguf
embedding_path   embedding 权重文件路径，例如 ../model/Qwen3.5-2B.embed.bin
core_mask        NPU 核心掩码，按 16 进制填写，例如 0x1、0x2、0xff
prompt           用户输入文本
```

以 **Qwen3.5-2B** 为例：

```bash
./rknn_qwen3_5_demo \
    ../model/Qwen3.5-2B.rknn \
    ../model/Qwen3.5-2B.weight \
    ../model/Qwen3.5-2B.tokenizer.gguf \
    ../model/Qwen3.5-2B.embed.bin \
    0xff \
    "解释相对论"
```

> ⚠️ **注意**：
>
> - `prompt` 中如果包含空格，需要使用英文双引号包起来。
> - `core_mask` 在 demo 中通过 `strtoul(argv[5], nullptr, 16)` 解析，因此建议按 16 进制格式填写。
> - demo 默认使用 `top_k=1`、`top_p=0.9`、`temperature=1.0`、`repeat_penalty=1.2`。
> - demo 中 `enable_thinking=false`，如需开启 thinking 模式，请在 C++ 代码中修改对应字段。


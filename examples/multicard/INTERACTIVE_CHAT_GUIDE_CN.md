# 多卡多轮对话操作手册（Interactive Chat）

> 适用版本：RKNN3 SDK V1.1.0（RK1820 / RK1828 / RK3572 Model Zoo V1.1.0）
> 示例模型：`Qwen/Qwen3.5-27B`（4 张 RK182X 加速卡，`--stage-count 4`）
> 代码入口：`examples/multicard/cpp/main.cc`

本文介绍 `rknn_multicard_demo` 的**多轮对话模式**：通过命令行启动后，在终端内持续与模型对话，历史上下文自动累积，直到你主动结束。

---

## 1. 功能简介

在原有「单次 prefill + decode」之外，demo 新增了 `--interactive`（简写 `-i`）多轮对话模式：

- 程序启动并完成模型/设备初始化后，进入 `while` 循环。
- 每一轮：从标准输入（stdin）读取一行用户输入 → 拼成 Qwen chat 格式 → 以 prefill 方式送入模型 → decode 生成回复并打印。
- 历史通过推理参数 `keep_history=1` 由 runtime **自动累积 KV cache**，因此后续轮次无需重复送入历史文本，只需送入**本轮新增的用户输入**。
- 资源（session、KV cache、embedding mmap、tokenizer 等）在多轮循环期间**不会**被释放；只有退出程序时才统一清理。

> 非交互模式（`--prompt` / `--perf`）的原有行为完全保留，不受影响。

---

## 2. 编译

在服务器端（x86_64 交叉编译到 aarch64）执行：

```bash
cd rknn3-model-zoo

# 设置交叉编译工具链
export GCC_COMPILER=/opt/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu

# 编译 multicard demo
./build-linux.sh -t rk3588 -a aarch64 -d multicard
```

编译产物位于：

```text
install/rk3588_linux_aarch64/rknn_multicard_demo/rknn_multicard_demo
install/rk3588_linux_aarch64/rknn_multicard_demo/lib/librknn3_api.so
install/rk3588_linux_aarch64/rknn_multicard_demo/lib/librknn3_api_rkcp.so
install/rk3588_linux_aarch64/rknn_multicard_demo/lib/librga.so
```

> 若构建目录残留旧路径（例如项目搬移后 CMakeCache 仍指向 `/data1/...`），请先删除
> `build/build_rknn_multicard_demo_*` 与 `install/rk3588_linux_aarch64/rknn_multicard_demo` 再重新构建。

---

## 3. 部署

将整个 demo 目录推送到开发板（`scp` / `rsync` / `adb push` 均可），并与模型文件放在同一工作区。以下为示例目录结构（以 Qwen3.5-27B 为例）：

```text
~/xyf_workspace/4rk1828_Qwen3.5-27B_test/
├── install/rk3588_linux_aarch64/rknn_multicard_demo/
│   ├── rknn_multicard_demo
│   └── lib/...
└── Qwen3.5-27B-RKNN3/
    ├── Qwen3.5-27B-llm.tokenizer.gguf
    ├── Qwen3.5-27B-llm.embed.bin
    ├── Qwen3.5-27B-llm_seg0.rknn / .weight / .safetensors
    ├── Qwen3.5-27B-llm_seg1.rknn / .weight / .safetensors
    ├── Qwen3.5-27B-llm_seg2.rknn / .weight / .safetensors
    └── Qwen3.5-27B-llm_seg3.rknn / .weight / .safetensors
```

---

## 4. 启动多轮对话

在开发板上执行：

```bash
cd ~/xyf_workspace/4rk1828_Qwen3.5-27B_test/install/rk3588_linux_aarch64/rknn_multicard_demo
export LD_LIBRARY_PATH=./lib
export MODEL_DIR=~/xyf_workspace/4rk1828_Qwen3.5-27B_test/Qwen3.5-27B-RKNN3

taskset f0 ./rknn_multicard_demo \
    --model ${MODEL_DIR}/Qwen3.5-27B-llm_seg0.rknn \
    --weight ${MODEL_DIR}/Qwen3.5-27B-llm_seg0.weight \
    --vocab ${MODEL_DIR}/Qwen3.5-27B-llm.tokenizer.gguf \
    --embed ${MODEL_DIR}/Qwen3.5-27B-llm.embed.bin \
    --ctx-size 4096 \
    --core-mask 0xff \
    --stage-count 4 \
    --bucket-size 128 \
    --rope-tensor ${MODEL_DIR}/Qwen3.5-27B-llm_seg0.safetensors \
    --interactive \
    --predict 128
```

启动后终端会先打印模型/设备初始化日志，然后出现交互提示：

```text
=== Interactive Chat Mode ===
Type your message and press Enter (Ctrl-D to exit).

User:
```

此时即可开始对话。

---

## 5. 如何持续对话

1. 在 `User:` 提示符后**输入一句话**，按回车。
2. 程序把这句话拼成 Qwen chat 格式作为 prefill 送入，随后逐 token 打印 `Assistant:` 的回复。
3. 回复结束后程序回到 `User:`，等待下一轮输入。
4. 重复上述过程即可**持续多轮对话**，模型会记住之前的上下文（KV cache 自动累积）。

示例：

```text
User: 你好，介绍一下你自己
Assistant: <think>...</think>你好！我是 Qwen3.5，由阿里巴巴集团旗下的通义实验室...

User: 好的，谢谢
Assistant: <think>...</think>不客气！如果你有任何问题或需要帮助，随时告诉我...
```

> 输入**空行**会被直接跳过（不会触发推理），继续等待有效输入。

---

## 6. 如何结束对话

在 `User:` 提示符下按 **`Ctrl-D`**（发送 EOF），程序会：

1. 退出多轮循环；
2. 打印分 stage 的性能统计；
3. 统一清理 KV cache 与全部资源；
4. 正常返回退出。

```text
User: (按下 Ctrl-D)

Per-Stage Performance Statistics:
...
```

> 说明：多轮循环期间 KV cache 与 session 资源都被保留，**只有退出时才统一 `rknn3_session_clear_kvcache` 与 `release_resources`**，这与单次模式不同。

---

## 7. 命令行参数说明

| 参数 | 含义 | 多轮模式取值示例 |
|------|------|------------------|
| `-m, --model <path>` | 第 0 段（seg0）RKNN 模型路径，自动推导 seg1..segN | `${MODEL_DIR}/Qwen3.5-27B-llm_seg0.rknn` |
| `--weight <path>` | 第 0 段（seg0）权重路径 | `${MODEL_DIR}/Qwen3.5-27B-llm_seg0.weight` |
| `--vocab <path>` | Tokenizer 文件 | `${MODEL_DIR}/Qwen3.5-27B-llm.tokenizer.gguf` |
| `--embed <path>` | Token Embedding 权重 | `${MODEL_DIR}/Qwen3.5-27B-llm.embed.bin` |
| `-c, --ctx-size <tokens>` | 最大上下文长度（token 数） | `4096` |
| `--core-mask <mask>` | NPU 核心掩码 | `0xff` |
| `--stage-count <count>` | 段数（= 使用加速卡数） | `4` |
| `--bucket-size <tokens>` | 每次推理 token 桶大小 | `128` |
| `--rope-tensor <safetensors>` | 外置 Rope Cache（Qwen3.5/Gemma-4 必填） | `${MODEL_DIR}/..._seg0.safetensors` |
| `-i, --interactive` | **开启多轮对话模式** | （开关） |
| `-n, --predict <count>` | 每轮最多生成的 token 数 | `128`（按需调整） |
| `--device-id <id0#id1#...>` | 手动指定设备 ID（可选，默认自动分配） | 通常省略 |

> 若不传 `--interactive`，则保持原有单次推理行为（`--prompt` 或 `--perf`）。

---

## 8. 工作原理（代码层面）

多轮对话在 `main.cc` 中的关键点：

1. **`run_chat_turn()`**：封装「prefill + decode」一轮完整对话，内部不清理 KV cache。
2. **外层 `while (true)`**：读取 stdin → 拼 Qwen chat 格式 → 调用 `run_chat_turn()` → 回到循环。
3. **chat 格式拼接**：
   - 首轮：`<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n{输入}<|im_end|>\n<|im_start|>assistant\n`
   - 后续轮：`<|im_start|>user\n{输入}<|im_end|>\n<|im_start|>assistant\n`
4. **`keep_history=1`**（在 `run_pipeline_once` 内设置）：让 runtime 在每次 `rknn3_session_run` 之间保留历史 KV cache，因此每轮只传新增用户输入即可。
5. **性能统计**：多轮结束后会打印两部分统计——`Performance Statistics`（整段对话累计的 prefill/decode 总耗时、总 token、tokens/s）与 `Per-Stage Performance Statistics`（各 stage 分阶段明细）。
6. **上下文溢出保护**：循环内累计当前上下文 token 数（prefill + decode），当剩余上下文不足以再容纳一轮时，会打印 `[warning] ... clearing KV cache to start a fresh conversation` 并自动清空 KV cache 开启新对话，避免因上下文写满导致 runtime 静默失败（表现为 `prefill/decode did not return token`）。
7. **资源生命周期**：循环内不调用 `release_resources`；仅在「主动清空上下文」时调用 `rknn3_session_clear_kvcache`，其余资源统一在循环退出后释放。

---

## 9. 常见问题

**Q1：如何在一行里输入中文/空格？**
直接输入即可，程序按整行读取（`std::getline`），内部使用 UTF-8 编码的 tokenizer。

**Q2：能设置每轮回复长度吗？**
可以，用 `--predict <count>` 设置每轮最大生成 token 数；命中 EOS 会提前结束。

**Q3：历史能累积多少轮？**
受 `--ctx-size`（`max_context_len`）限制。累计 token 接近上限后，程序会**自动清空 KV cache 并开启一段新对话**（会打印 warning），而不是让 runtime 静默失败。若需保留更长的上下文，请增大 `--ctx-size`（受模型导出时的 `kvcache_buffer_len` 约束）。

**Q4：多轮模式会不会比单次慢？**
首轮 prefill 与单次一致；后续轮次因历史 KV cache 已存在，prefill 只需处理新增输入，但总上下文越长，decode 阶段注意力计算开销越大。

**Q5：如何非交互地批量测试多轮？**
可通过管道把多行输入喂给程序，例如：

```bash
printf '第一句话\n第二句话\n' | taskset f0 ./rknn_multicard_demo ... --interactive
```

程序读到 EOF 后自动结束并打印性能统计。

**Q6：为什么多轮后报 `prefill/decode did not return token`？**
这是上下文写满的典型症状：累计 token（prefill + decode）达到 `--ctx-size` 上限后，runtime 无法继续写入 KV cache，于是不再产出 token。新版 demo 会在接近上限时主动清空 KV cache（见第 8 节第 6 点），避免该错误；若仍出现，请确认 `--ctx-size` 与模型 `kvcache_buffer_len` 是否匹配。

# 多 Session 并发推理方案（Qwen3.5-27B / 4×RK1828）

> 项目：`rknn3-model-zoo/examples/multicard`
> 核心代码：`cpp/main.cc`（4823 行）、`python/qwen3_5/`（模型转换）
> 目标模型：**Qwen3.5-27B**，4 段流水线 / 4 张 RK1828 卡
> 部署现场：`CM3588-Plus:<板卡上模型目录>`（板卡在独立内网）
> 安装目录：`../install-Qwen/rk3588_linux_aarch64`
> 前置文档：本目录 `多卡推理方案解析.md`、`多卡推理优化方案.md`（实测基线数据来源）

---

## 0. 结论摘要（TL;DR）

1. **可行性：中等偏高，但不是"改几行就能跑"。** 卡在 decode 的算力/带宽利用率只有 **25.5%**（每卡 20.6ms 忙 / 80.75ms 周期），多会话交错正是把这 74.5% 的空闲填满的手段，机制上成立。
2. **收益上限被"卡数"锁死：N 个会话最多约 N 倍（N ≤ 4）。** decode 12.38 tok/s → N=2 约 24、N=3 约 36、N=4 约 48 tok/s。**不存在超过 4× 的空间**，因为 4 张卡在 4 路并发时已被占满。
   > ✅ **实测（2026-09-14，P2 完成）**：N=1 10.73 → N=2 **19.78** → N=3 **28.63** → N=4 **35.74** tok/s（同一 workload 口径，相对 N=1 为 **3.33×**）。N=5 反而降到 **34.94**，证实 N=4 就是拐点。绝对数字低于上面的理论值，缺口来自每轮 prefill 气泡与卡级串行的排队（见 §5 的 P2 完成记录）。
3. **一票否决的前置条件已实测通过（见 §4）：权重共享、KV 隔离、4 路满上下文并发全部成立。** 结论：**每卡硬上限 5 个 session，方案的 4 路目标已验证可行**（4 个会话各填 3894 token 上下文，prefill + decode 全部成功）。`rknn3_session_init` **不复制权重**，每会话只增加 KV cache。
4. **代码层面的主要障碍是 4 处全局状态**：`g_last_stage_result`、单个 `PipelineState`、`EmbedCallbackContext`/`StageCallbackContext` 里的 `pipeline` 指针、以及 `init_output_tensors` 分配的**每 context 共享**输出 tensor 内存。前三处是"状态串话"，第四处是**真数据竞争**（见 §3.2）。
   > ✅ **进度（2026-09-14）**：前三处已在 **P1** 里随 `Conversation` 一起下沉（§5 的 P1 完成记录），回归全部通过；第四处（`output_tensors` 共享）已在 **P2** 里用卡级锁解决（§5 的 P2 完成记录）。
5. **SDK 明确要求调用方自己做线程安全**：文档 3.5 节「同一个 rknn3_context 或 rknn3_session 的并发使用需调用方自行保证线程安全」。所以**并发粒度必须锁在"卡"上**——同一张卡同一时刻只允许一个 session 在跑，靠"卡间不同会话并行 + 卡内串行"来拿收益。
   > ✅ **已落地**：`StageContext::run_mutex`，持锁范围覆盖整个 `rknn3_session_run`（回调里读 `output_tensors` 也在锁内）。实测这套"卡内串行"拿到了 3.33×，说明 SDK 这条限制并不吃掉收益。
6. **模型文件不需要重新转换。** 多会话是纯运行时改造，`.rknn`/`.weight`/`.safetensors`/`embed.bin`/`.tokenizer.gguf` 全部不动。这让版本管理简单很多（见 §8）。
7. **内存不是瓶颈，不需要重新导出模型。** 实测每会话 KV 仅 **64.4–70.7 MB/卡**，且**在 `rknn3_session_init` 时按满上下文一次性预分配**——填入 3894 token 上下文后新增分配为 **0.0 MB**。所以容量只与「会话数」有关，与实际用掉多少上下文**完全解耦**；4 会话总开销固定在 ~283 MB/卡（4×70.7），与上下文填满与否无关。
   > ⚠️ **第 8 点的退路已作废**：卡内上限 5 个 session **不由内存决定**（失败时最紧 node 仍剩 93.7 MB，占基线 74%，够再放约 11 个会话的 KV）。因此**调小 `kvcache_len` 提高并发数这条路是无效的**——它降低的是每会话内存，而限制并发的不是内存。详见 §4.4。
8. **P3（交互式多会话）已实现并验收**（见 §5 的 P3 完成记录）。`--sessions N --interactive` 解开了 P2 时期写死的互斥：一条输入线程 + 每会话一个 worker 抢同一个待办队列（**"谁空闲给谁"**），输出按**整段成块 + `[s<i>]` 前缀**打印，N 路不互相穿插。
   > ✅ **实测（2026-09-14）**：N=1 时与 P1 交互黄金**逐 token 等价**（1539/1539、跨清 KV 3078/3078，清 KV 仍在 3826/4096）；N=4 + 12 行输入 → **恰好 12 个块、无穿插、4 个会话各分到 3 轮**（"谁空闲给谁"的正面证据），聚合 **37.70 tok/s**。
   > 同批补齐了 §9.3 的 **30 分钟稳定性**（通过）、§9.5 的 **TSan 两轮**（补轮把 R11 新加的锁那段代码真正压进去了；但首轮"我方零竞争"的说法已按 4 条"写侧在我方"的报告收窄，边界见 §9.5）与 §9.3 的 **ASan 四阶段 + 阳性对照**（零报告）。**`--perf` 仍与 `--sessions` 互斥。**
9. **M5（服务化）已实现并实测**：`main.cc --serve` 提供带 `rid` 的**帧协议**（走独立 fd，避开 stdout 上的日志污染），板端 `python3` 网关（`examples/multicard/serve/`）对外提供 **OpenAI 兼容** `/v1/chat/completions`（SSE + 非流式）、`/v1/models`、`/health`，并做会话粘性 + 跨轮 KV 复用。同一份脚本量 HTTP 层并发伸缩：**10.96 / 20.22 / 36.44 tok/s = 3.33×**（连量 4 轮，伸缩比 3.32~3.38 不再漂），对照官方 `rkllm3-server` 同板同条件 **10.63 / 10.98 / 10.98（1.03×）**——**收益来自我们自己的并发执行器，不是 HTTP 层**；HTTP 层要证明的只是"没把它吃掉"。踩到的四个坑**全部只影响速度/并发度、答案完全正确**，第 4 个（会话租约算错 → 并发悄悄串行化，**偶发**）曾让伸缩量到 `1.00×`（详见 §9.7）。
10. **多用户接入（把板卡当边缘服务器，§9.8）**：多人各带身份（`X-Conversation-Id` 等，**不要再靠对话内容认身份**）从终端接 API，各占一个会话；`NSESSION` 以内谁问谁拿，**超过只排队、不抢占**，等超过 `QUEUE_TIMEOUT` 返 503；一段对话静默超过 `IDLE_TTL` 就把它占的会话交还给排队者，或者客户端**问完主动调 `POST /v1/conversations/close`** 立刻交还（网关不知道对话何时结束，所以"主动说一声"这条路必须有——没有它，一个连问 4 个问题的脚本就把 4 个会话占满 5 分钟）。`GET /v1/pool` 是"谁在占用、谁在排队"的唯一观测点（排队在吞吐上完全看不出来），槽位带原始 `key` 给匿名对话用。**⚠️ 本轮验证全在本地桩后端完成，板上真后端还没跑过多用户演示；且没有鉴权 ⇒ 一个客户端拿 5 个 id 就能占满全部会话、还不交还，把真人挤进队列（R12）。**
11. **Phase 1（工具调用）已实现并真机验收（2026-09-17，§9.9）**：Agent 化的第一步。**格式不查文档、取自模型自己的 `tokenizer.chat_template`**（GGUF 元数据键）——这一代是 **XML**
    （`<tool_call><function=…><parameter=…>…</parameter></function></tool_call>`），**不是** Qwen3 早期的 Hermes JSON；写错一个字节的表现不是"调得别扭"，而是**根本不调、且不报错**。工具目录在**网关侧**拼（`main.cc` 那份 `QWEN35_CHAT_TEMPLATE` 对工具零支持），所以**后端二进制一行没动**。板上三个探针 3/3 解出可解析调用、参数名 ⊆ 声明的 properties、`required` 一个不缺；用**模型自己那次调用**的历史回显，命中 KV 复用 **热=32 vs 冷=630**。
    > **值得记的是过程**：桩上全绿、板上**第一遍就红**——`serve_http_test.py` 回显助手轮时丢了 `tool_calls`，而网关的前缀判据是**全有或全无**（没有"最长公共前缀"的部分复用），于是**静默整段重算**（469/469，无报错）。修正后经 A/B/C 变异测试证明这条检查不是空转。同一天模型换成 **8192 导出**（判据是 `rope_cos_cache` 形状 `F16 [1,4,1,8192,16]`），5013 / 6853 token 的 prompt 都能从**中段**取回暗号。
    > **没验的**：流式路径下的工具调用、`tool_choice`（未实现）、并行多工具调用、以及**参数非规范形式**（`"北京"` 带引号、`1.50`）时的回显——最后一条是**换模型/换导出后要重验的第一件事**。

---

## 1. 收益模型：为什么多会话能提速

### 1.1 实测基线（来自 `多卡推理方案解析.md`）

| 指标 | 数值 |
|---|---|
| Prefill | 1512.24 ms / 512 tok = 2.95 ms/tok = **338.57 tok/s** |
| Decode | 10255.51 ms / 127 tok = 80.75 ms/tok = **12.38 tok/s** |
| stage0 decode | 20.60 ms/tok |
| stage1 decode | 20.68 ms/tok |
| stage2 decode | 20.62 ms/tok |
| stage3 decode | 18.71 ms/tok |
| 四段之和 | **80.61 ms/tok** |
| host 开销 | 80.75 − 80.61 = **0.14 ms**（0.17%） |

**内存基线（`--probe-sessions` 实测，见 §4）**：

| 指标 | 数值 |
|---|---|
| 单卡 NPU node 数 | **8**，每 node 总量 614.5–639.5 MB |
| 单卡可用内存 | **~4.93 GB**（8 node 合计） |
| 每卡权重（seg0–2） | 3611.6 MB（451.4–451.8 MB/core） |
| 每卡权重（seg3） | 3783.4 MB（含 norm + lm_head，比 seg0–2 多 171.8 MB） |
| 每卡 internal | 35.9–42.0 MB |
| 每卡 KV cache（上报值） | 70.7 MB（seg0–2，8.8 MB/core）／64.4 MB（seg3，8.0 MB/core） |
| **建 session 前最紧 node 空闲** | **125.9 MB**（seg3 的 node[5]，全段最紧） |
| **每 session KV 成本（实测）** | **8.05 MB/node = 64.4 MB/卡** |
| **每卡 session 硬上限** | **5 个**（第 6 个 `rknn3_session_init` 失败） |

> ⚠️ 注意 seg3 的权重比 seg0–2 大 171.8 MB，所以**最紧的 node 在 seg3 而不是 seg0**（125.9 vs 146.8 MB）。做容量规划必须取全段最小值。

### 1.2 关键观察

单序列自回归下，token N+1 依赖 token N，四段必须严格串行——**流水线并行在 decode 上拿不到任何跨 token 重叠**。于是：

```
单会话 decode 时间轴（80.75ms 一个周期）

卡0  ████░░░░░░░░░░░░░░░░  20.6ms / 80.75ms = 25.5%
卡1  ░░░░████░░░░░░░░░░░░  20.7ms / 80.75ms = 25.6%
卡2  ░░░░░░░░████░░░░░░░░  20.6ms / 80.75ms = 25.5%
卡3  ░░░░░░░░░░░░████░░░░  18.7ms / 80.75ms = 23.2%
                │
          同时只有 1 张卡在干活
```

**每张卡有约 74.5% 的时间在等下游/上游，这就是可以卖给第二个会话的空闲。**

### 1.3 多会话交错后的时间轴（N=2）

```
        周期 0        周期 1        周期 2
卡0  █A███B█░░░░  █A███B█░░░░  █A███B█░░░░   ← A、B 各占 20.6ms，卡0 满
卡1  █A███B█░░░░  █A███B█░░░░  ...
卡2  █A███B█░░░░  ...
卡3  █A███B█░░░░  ...
```

两个会话相位错开 20.6ms，每张卡在一个 40.3ms 的周期里干 2×20.6=41.2ms —— **卡被填满，吞吐翻倍到 ~24 tok/s**。

推到 N 路：

| 并发会话数 N | 每卡占用率 | Decode 吞吐（理论） | Decode 吞吐（留 15% 余量） |
|---|---|---|---|
| 1 | 25.5% | 12.38 tok/s | 12.38 |
| 2 | 51% | 24.8 tok/s | ~21 |
| 3 | 76% | 37.1 tok/s | ~32 |
| 4 | **102%（饱和）** | 49.5 tok/s | ~42 |

> ⚠️ N=4 时理论值已 >100%，说明**没有任何调度余量**，任何一点抖动都会变成排队。工程上建议**目标 N=2，验证 N=3**，N=4 当作压力测试而非产品配置。

### 1.4 顺带修好的东西：prefill 气泡被摊薄

prefill 每个用户轮的固定气泡约 **650ms**（最后一桶排空 stage1→2→3，`多卡推理方案解析.md` §4）。单会话交互时这 650ms 是纯等待；多会话下它被其他会话的 decode 填掉，**交互首 token 延迟（TTFT）在多路负载下几乎不变，而总吞吐线性上涨**。这是多会话方案除吞吐外的第二个收益。

---

## 2. 现状代码分析：哪些状态是"全局"的

`cpp/main.cc` 目前是**单会话假设**写死的。要支持 N 会话，必须先把下列状态"下沉"到会话对象里。

### 2.1 全局状态清单

| 符号 | 位置 | 作用 | 多会话下的问题 |
|---|---|---|---|
| `static LastStageResultState g_last_stage_result` | `main.cc:262` | 末段采样出的 next_token 暂存 | **所有会话共用一个 token 槽**，A 的 token 会被 B 覆盖 |
| `PipelineState pipeline(g_stage_count)` | `main.cc:2221` | 4 个 `StageSlot` 队列 | 队列里的 `expected_tokens`/`producer_done` 是**会话级**语义，共用会互相串桶 |
| `StageRuntime::callback_ctx` | `main.cc:154`，绑定于 `1646` | `PipelineState*` + `stage_index` | 一个 stage 只有一个回调上下文 → 只能服务一个会话 |
| `EmbedCallbackContext embed_ctx` | `main.cc:2259`，绑定于 `2261` | 持 `pipeline` 指针 | 同上 |
| `StageRuntime::output_tensors` | `init_output_tensors`，`main.cc:1497-1578` | 每 context 一份输出 tensor 内存 | **两个会话同时跑会写同一块内存 → 真实数据竞争**（见 §3.2） |
| `stage.performance` | `main.cc:155` | 性能统计 | 混在一起，统计口径失效 |
| `g_*` 配置项（`g_stage_count` 等） | `main.cc:49-59` | 启动后只读 | ✅ 无问题，保持全局 |
| `input_cb_data`（rope cache mmap） | `main.cc:2263` | 只读 mmap | ✅ 只读可共享，但**写入 input tensor 的动作**受 §3.2 约束 |
| `embed_info`（embedding.bin mmap） | `main.cc:2230` | 只读 mmap | ✅ 可共享 |
| `Tokenizer` / `VocabInfo` | `main.cc:2227-2235` | 只读 | ✅ 可共享（需确认 `libtokenizer.a` 编码线程安全，见 §7） |

### 2.2 关键调用链（改造要碰的地方）

```
main()                                            main.cc:2221 pipeline 构造
 ├─ init_stage(stages[i], pipeline, ...)          main.cc:1637   绑定 callback_ctx.pipeline
 │   ├─ rknn3_init / load_model_from_path / model_init      ← 每卡一次，多会话应复用
 │   └─ rknn3_session_init(ctx, param, 1)         main.cc:1701   ← 多会话在这里循环 N 次
 │       └─ rknn3_session_set_callback(sess, cb)  main.cc:1741   ← per-session，天然的挂载点
 ├─ g_last_stage_result.tokenizer = tokenizer     main.cc:2238
 └─ g_interactive 循环                             main.cc:2385
     └─ run_chat_turn(stages, pipeline, ...)      main.cc:1940
         └─ run_pipeline_once(stages, pipeline, …) main.cc:1855
             ├─ reset_pipeline(pipeline)                    ← 会话级
             ├─ reset_last_stage_result()                   ← 会话级
             ├─ spawn workers: run_stage_worker(i, …)       main.cc:1893  ← 每 token 新建 3 线程
             ├─ rknn3_session_run(stages[0].session, …)     main.cc:1903
             └─ join + 读 slots[0]->expected_tokens         main.cc:1911-1921
```

### 2.3 一个容易被忽略的并发敏感点

`run_stage_worker` 里末段判断"是否最后一桶"的逻辑：

```cpp
// main.cc:1818-1826
if (is_last_stage) {
  uint64_t total_tokens;
  { std::lock_guard<std::mutex> lock(pipeline.slots[0]->mutex);
    total_tokens = pipeline.slots[0]->expected_tokens; }
  local_param.disable_sampling = (consumed_tokens + batch.n_tokens < total_tokens);
}
```

`pipeline.slots[0]->expected_tokens` 在 `embed_callback` 里按桶累加（`main.cc:1766-1771` 区间）。**这是"本会话这一轮 prefill 一共几个 token"的会话级量**。只要 `pipeline` 下沉到会话对象，这段逻辑天然正确；如果偷懒保留全局 `pipeline`，会出现"A 会话的采样被 B 会话的桶数把关"这种极难排查的错误。

---

## 3. 目标架构

### 3.1 分层：Context（卡）↔ Session（会话）

RKNN3 的 API 分层给了天然支持：

```
┌──────────────────── 卡 i（rknn3_context） ────────────────────┐
│  rknn3_init / load_model_from_path / model_init   ← 权重，1 份   │
│  output_tensors（可选：改为 per-session）                        │
│                                                                │
│  ┌──── Session A（rknn3_session*）────┐  ┌── Session B ──┐     │
│  │ KV cache A                         │  │ KV cache B    │     │
│  │ chat template A                    │  │ chat template B│    │
│  │ RKLLMCallback + userdata A         │  │ callback B     │    │
│  └────────────────────────────────────┘  └───────────────┘     │
└────────────────────────────────────────────────────────────────┘
```

- **权重、模型结构**：每卡一份，N 会话共享（**前提是 P0 验证通过**）。
- **KV cache、prompt 历史、采样状态、回调**：每会话一份。`rknn3_session_set_callback` 是 per-session 的，把 `PipelineState*` 塞进 `callback.output_userdata` 即可实现"回调按会话路由"，这是改造的最关键抓手。

### 3.2 数据竞争：`output_tensors` 必须处理

`init_output_tensors`（`main.cc:1497`）用 `rknn3_create_mem(stage.ctx, ...)` 为每卡的每个输出 tensor 分配**一块**内存，并挂到 `callback.output_tensors`（`main.cc:1714`）。`stage_output_callback` 从这块内存里把 hidden states 拷进 `StageBatch` 入队。

**如果 A、B 两个 session 同时在同一 context 上跑，它们会写同一块 `output_tensors[i].mem` —— 后完成的那个会看到对方的 hidden states。**

两种解法，二选一：

| 方案 | 做法 | 代价 |
|---|---|---|
| **A. 卡级互斥锁（推荐起步）** | 保证同一 `rknn3_context` 同一时刻只有一个 session 在 `rknn3_session_run`。共享 output mem 天然安全。 | 需严格锁序（§3.4）；SDK 本来就要求调用方做这件事 |
| **B. per-session output mem** | 每个 session 各自 `rknn3_create_mem` 一套输出 tensor，`callback.output_tensors` 指向自己那套 | 多占 `N × n_output × bucket × hidden` 内存（128 tok × 5120 × 2B × 输出数 ≈ 1.3MB/份，可忽略）；但**仍需卡级锁**，因为 SDK 不保证同 context 并发安全 |

**结论：卡级互斥锁是必须的，方案 B 只是把偶发的读脏风险也消掉。建议 P1 用 A，P2 视稳定性决定是否升 B。**

### 3.3 Threading 模型（推荐方案：每会话一条驱动线程）

**方案 A —— 每会话一组线程 + 每卡一把锁**

```
会话 A 线程  ─┐
会话 B 线程  ─┤  各自跑完整的 run_pipeline_once
会话 C 线程  ─┘  只在 session_run 前后加卡锁

每个会话内部：主驱动线程(stage0) + 3 个 worker 线程
卡锁：std::mutex g_card_lock[4]
```

- 改动最小：`run_pipeline_once` 的骨架不变，只是参数从全局 `pipeline` 换成会话自己的。
- 死锁分析：单个会话的 stage 推进**严格升序**（0→1→2→3），锁申请严格升序。环形等待需要"持有高序号卡、等低序号卡"，在本模型里不存在，因此**无死锁**。
- 缺点：N=4 时 16 个线程，线程调度开销上升。但实测 host 开销仅 0.14ms/tok（0.17%），**不是瓶颈**。

**方案 B —— 全局调度器 + 常驻 worker（想做长远就上）**

```
调度线程：round-robin 从就绪会话队列取一个"step 任务"
每卡一个常驻 worker 线程 + 任务队列，任务结构体携带 {conversation_id, session*, pipeline*, batch}
```

- 删除 `main.cc:1893-1898` 每 token 重建 3 线程的做法，`run_stage_worker` 变成常驻循环。
- 好处：线程数固定、调度可控、方便做优先级/公平性。
- 代价：`StageSlot`、`close_stage_slot`、`fail_pipeline`、`reset_pipeline` 全要改成带 conversation_id 的形式，工作量约为方案 A 的 2–3 倍。

**建议：P1 用方案 A 打通，P3 再评估是否换 B。**

### 3.4 锁序约定（务必写进代码注释）

> 所有会话必须**按 stage 索引升序**获取卡锁；禁止跨 stage 反序加锁；`rknn3_session_run` 全程持锁，不得在持锁期间等待非本会话的资源。

---

## 4. P0 前置验证：**已完成，全部通过** ✅

在真实板卡（CM3588-Plus + 4×RK1828，Qwen3.5-27B 4 段流水线）上实测完成。**结论：方案可行，4 路并发目标已验证。** 每卡硬上限 5 个 session，方案的 4 路目标有 1 个 session 的余量。

### 4.0 复现方式

探针已内置在 `main.cc` 里，通过 `--probe-sessions N` 触发（N ∈ [2,16]），跑完即退出，不进入对话/压测模式：

```bash
--probe-sessions 6      # 每卡创建 5 个额外 session，第 6 个会失败 → 定出天花板
```

探针依次回答 5 个问题：
1. 每卡能容纳几个 session（抬高 N 到某个 `rknn3_session_init` 失败）
2. 权重是否 per-session 复制
3. 每 session 的真实内存成本
4. 两块 KV cache 是否真正隔离
5. **N 路满上下文并发是否成立**（方案的真正闸门）

### 4.1 实测结论

| # | 问题 | 结论 | 证据 |
|---|---|---|---|
| 1 | 每卡 session 上限 | **5 个**（第 6 个失败） | 5 次独立运行一致（1×`N=8` + 4×`N=6`），每次都在建第 6 个时于 stage0 失败 |
| 2 | 权重是否 per-session 复制 | **共享** ✅ | 建 4 个额外 session 只消耗 32.2 MB/node = 4×8.05 MB，**恰好等于上报的 per-core KV**。若复制权重，每个需 451 MB/core，而最紧 node 只剩 125.9 MB → 第 2 个就会 OOM |
| 3 | 每 session KV 成本 | **8.05 MB/node = 64.4 MB/卡** | 建 session 前后 node 空闲差值实测；与上报值 8.8 MB/core 同量级 ✅ |
| 4 | KV cache 隔离性 | **隔离** ✅ | 两个不同 prompt 各 12 步 decode，逐步交错（A→B→A→B…），交错序列与各自单独跑的序列**逐 token 一致**；两个对照各重跑一遍也一致（排除采样非确定性）。A/B 从第 5 个 token 起分叉，说明两会话确实持有各自不同的上下文 |
| 5 | **4 路满上下文并发** | **成立** ✅ | 4 个会话各填 **3894 token**（模型上限 4096 的 95.1%），prefill + decode **全部成功** |
| 6 | 销毁 session 后内存归还 | **完整归还** | 与基线残留差 +0.0 MB |

**关键发现：KV cache 在 `rknn3_session_init` 时按满上下文一次性预分配，与实际用掉多少上下文无关。**

填满 3894 token 上下文后，node 空闲内存**变化为 0.0 MB**。这与上报值在量级上也自洽（**推算**，非直接测得）：64 层 / 4 段 / 8 core = 每 core 2 层；每 token 每层 K+V = 2 × 4 KV-head × 256 ≈ 2048 元素，INT4 下约 1 KB/token/层 → 2 层 × 4096 token × 1 KB ≈ 8.4 MB ≈ 上报的 8.8 MB/core。即上报值本身就对应**满上下文**的 KV，不是当前用量的 KV。

> **这条结论对容量规划非常重要**：容量只与「会话数」有关，与上下文长度**完全解耦**。4 会话的总开销固定在 ~283 MB/卡（4×70.7），无论 4 个上下文是空的还是都填满。

### 4.2 查询接口可用性（踩过的坑）

设计探针时假设的「建 session 前后对比设备内存」这条路，**在本平台上一半是死路**：

| 接口 | 是否反映 session 级分配 | 说明 |
|---|---|---|
| `rknn3_find_devices().devices[i].mem_info.sys_free` | ❌ **恒为 0** | 完全不可用 |
| `RKNN3_QUERY_ALLOCATION_INFO`（weight/internal/kvcache 明细） | ❌ **恒 +0.0** | 是模型加载时的静态快照，建 session 后不更新 |
| `RKNN3_QUERY_DEVICE_MEM_INFO.sys_total/sys_free` | ❌ 无意义 | 返回 19.0 MB（本机侧小块内存，不是设备内存） |
| **`RKNN3_QUERY_DEVICE_MEM_INFO.node_mem_info[n].free`** | ✅ **准确** | 建 session 前后差值精确等于上报的 per-core KV（误差 <0.05 MB） |

**所以判定只能靠 `node_mem_info`，且必须注意两点：**

1. **取全段最小值，不能只看 stage0。** seg3 的权重比 seg0 大 171.8 MB，所以最紧的 node 在 seg3（125.9 MB）而非 seg0（146.8 MB）。实测中这个差异会直接影响结论。
2. **`node_mem_info` 不反映上下文增长**——但这不是接口缺陷，而是因为 KV 已预分配（见 §4.1）。

### 4.3 `--ctx-size` 的实际行为（重要）

**传 `--ctx-size` 大于模型固化的 `kvcache_buffer_lens` 是无效的，会被静默降级。**

本模型的元数据实测：

```
max_ctx_len = 4096, max_position_embeddings = 4096
attention_kvcache_lens[0]: n_lens = 1, lens = [4096]     ← 单 KV group，无可选
kvcache_dtype = 1 (INT4_TO_F16), store_method = 2 (GroupQuant)
kvcache_group_size = 1, kvcache_residual_depth = 64
model_type = qwen3_5, vocab = 248320, emb_dim = 5120, cores/card = 8
```

传 `--ctx-size 8192` 时 runtime 只打一条 warning，然后**使用 4096**：

```
No exact kvcache group id found for max_context_len=8192 ...
Using closest attention kvcache group_id = 0 ... (chosen kvcache_buffer_lens: 4096)
```

> ⚠️ **因此本模型是 4096 上下文，不是 8192。** 若现场需要 8192，必须**重新导出模型**（导出时指定 `kvcache_len=8192`），改运行时参数无效。

### 4.4 未解释的问题：5 这个上限**不由内存决定** ⚠️

这是本次验证留下的**唯一未闭环项**，也是后续最需要注意的一点：

- 5 个 session 只吃掉 5×8.05 = 40.3 MB/node，而基线预算是 125.9 MB/node
- 第 6 个 session 失败时，最紧 node **仍剩 93.7 MB（占基线 74%）**，按实测成本够再放约 11 个会话的 KV

**结论：卡内上限 5 不由 node 空闲量解释**，它来自别的池子（未被 `node_mem_info` 统计）或一个硬上限。

**由此得出两条必须写进规划的推论：**

1. **不要用「剩余内存 ÷ 单会话成本」估算可容纳的会话数**——在这个平台上会算出 ~11，而实际是 5。容量必须按「**实测会话数上限**」来规划，即 **4 路目标 + 1 个余量**。
2. **调小 `kvcache_len` / `--ctx-size` 重新导出，无法提高并发数。** 本文档早期版本把「降 ctx 重导」当作容量不足时的退路——**该退路已作废**：它降低的是每会话内存占用，而限制并发的不是内存。这个方向的唯一价值是省内存（若将来需要给别的功能腾空间），不是提高并发。

**后续若要闭环**，可在 P1 阶段顺带做一次定位：逐段单独建 session 看是哪一段先失败；或用 `rknn3_session_query_state`（`rknn3_api.h:1644`）查失败时的 session 状态；或向 RK 确认是否存在 per-context 的 session 数硬上限。**但这不是阻塞项**——4 路目标已验证可行。

### 4.5 原「多进程备选路线」已不需要

早期版本为「权重被 per-session 复制」这种情况准备的备选路线（减小分段数 / 等 SDK / 退回串行）**全部不需要**——权重确认共享。保留在此仅作记录：`rknn3_context` 与进程绑定，一张卡同一时刻只能被一个进程打开，所以「每卡一个进程」会把 4 卡流水线拆成 4 个进程，反而破坏流水线，从来就不是好路线。

---

## 5. 改造清单（按阶段）

### P1：状态下沉（不改行为，可完整回归测试）

**目标：把 `main.cc` 从"单会话"重构为"会话对象"，但**仍然只跑一个会话**，输出必须与基线逐 token 一致。**

| # | 文件/符号 | 位置 | 动作 |
|---|---|---|---|
| 1.1 | 新增 `struct Conversation` | 新代码 | 聚合 `session`、`PipelineState pipeline`、`LastStageResultState result`、`StageCallbackContext cb`、`EmbedCallbackContext embed`、`StagePerformanceStatistics perf`、`context_tokens`、`first_turn` |
| 1.2 | `StageRuntime` | `main.cc:140` | 拆成 `StageContext`（`ctx`/`model_path`/`output_tensors`/`embedding_dim`/`max_ctx_len`，**每卡一份**）+ `Conversation`（per-session） |
| 1.3 | `g_last_stage_result` | `main.cc:262` | 删除全局；`Conversation::result` 取代，`callback.result_userdata` 指向它（`main.cc:1723`） |
| 1.4 | `reset_last_stage_result` / `get_last_stage_token` | `main.cc:491-502` | 加 `Conversation&` 参数 |
| 1.5 | `run_pipeline_once` | `main.cc:1855` | 签名改为 `(StageContexts&, Conversation&, ...)`；`reset_pipeline`/`reset_last_stage_result` 作用于 `conv.pipeline` |
| 1.6 | `run_stage_worker` | `main.cc:1768` | 参数改为 `(stage_idx, contexts, Conversation&, ...)`；末段 `disable_sampling` 读 `conv.pipeline.slots[0]`（`main.cc:1822`） |
| 1.7 | `init_stage` | `main.cc:1637` | 拆成 `init_stage_context()`（`rknn3_init`→`load`→`model_init`→`init_output_tensors`）+ `init_conversation()`（`rknn3_session_init`→`set_chat_template`→`set_callback`） |
| 1.8 | `run_chat_turn` | `main.cc:1940` | 加 `Conversation&` 参数 |
| 1.9 | 交互循环 | `main.cc:2385-2462` | 用 `conv.context_tokens` 等替换裸局部变量，行为不变 |

**验收标准（硬性）：**
- `--perf 512 128` 的 decode 必须仍是 **80.75 ± 1 ms/tok**、prefill **338 ± 10 tok/s**（回归，不能变慢）。
- 与基线二进制跑同一批 prompt，**逐 token ID 完全一致**（写个 `--dump-tokens` 对比脚本）。
- 多轮对话清 KV cache 的分支（`main.cc:2423-2431`）仍正确触发。

#### P1 完成记录：**已完成，验收全部通过** ✅（2026-09-14）

代码：commit `6347f57`，分支 `feature/multisession-concurrency`，tag `p1-session-split`。
源码 md5 `dfad38d5bbe49dae699bc2da52708e01`；二进制 md5 `6f5aa9d73dee48ebcafd6a03a2a9d71d`
（BuildID `6cc50c5c34890f1a9ca1ed12c05c65c2cbaf660d`；同一源码连编两次 md5 一致，构建可复现）。

**落地形态**：`StageRuntime` → `StageContext`（每卡：ctx / 输出张量 / 形状 / 卡级统计）
+ `StageSession`（每会话每卡：session 句柄 + **指向本会话**的 `RKLLMCallback` 与
`StageCallbackContext` / `EmbedCallbackContext`）。新增 `Conversation`，持有
`std::vector<StageSession> stages`、`PipelineState pipeline`、`LastStageResultState result`、
`context_tokens`、`first_turn` 与四个累计统计量；`g_last_stage_result` 全局已删除。
`init_stage` 拆为 `init_stage_context()`（`rknn3_init`→`load`→`model_init`→`init_output_tensors`
→`query_ext_input_indices`，每卡一次）+ `init_conversation()`（`rknn3_session_init`→
`set_chat_template`→`set_callback`，每会话每卡一次）。`main()` 里仍是**一个** `Conversation`。

> 注意 `RKLLMCallback` 必须随会话走而不是随卡走：它同时携带 per-card 的
> `output_tensors` / `ext_input_indices` / `tokenizer` 与 per-conversation 的
> `callback_ctx.pipeline` / `embed_userdata` / `result_userdata`。挂在 `StageContext`
> 上会让 N 个会话的回调互相指错上下文——这是 P2 最容易踩的坑。

**回归证据**（本地任务目录 `rt_work/reg1/`，未入库；测试在板卡现场执行）：

| 判据 | 方法 | 结果 |
|---|---|---|
| 逐 token ID 一致 | 两个 prompt 各 48 token | 49/49 ID 完全相同 ✅ |
| 逐 token ID 一致（多轮） | 3 轮交互 | 1539/1539 ID 完全相同 ✅ |
| 逐 token ID 一致（跨清 KV） | 6 轮交互 | 3078/3078 ID 完全相同 ✅ |
| 生成文本一致 | 同上三组 stdout 逐字节 diff | 完全相同 ✅ |
| decode 无回归 | `--perf 512 128` | **80.82** ms/tok（P0 同机 80.78）✅ |
| prefill 无回归 | `--perf 512 128` | **341.06** tok/s（P0 同机 341.10）✅ |
| 清 KV cache 分支 | 6 轮交互逼到门限 | 两侧都在 `3826/4096` **恰好触发 1 次** ✅ |

**逐 token ID 对比是怎么做到可信的**：板卡上原有的 P0 二进制没有导 ID 的能力，直接比 ID 无从谈起。
所以额外用 **P0 源码 + 只加 `--dump-tokens` 插桩**编了一个 `rknn_multicard_demo.p0dump`
（对 `main.cc.p0_baseline` 净增 36 行，插桩代码与 P1 的**逐字符相同**），先证明插桩本身行为中性
——未插桩的 P0 与插桩版 P0 的生成文本逐字节相同——再用它的 ID 序列去比 P1 的 ID 序列。
链路是「P0 ≡ P0+插桩（文本）」+「P0+插桩 ≡ P1（ID）」，结论才落到「P1 ≡ P0（ID）」。

**板卡上的三个二进制（都留着，便于回滚与复测）**：

| 文件 | md5 | 说明 |
|---|---|---|
| `rknn_multicard_demo.p0_a6739a6e` | `a6739a6e01092d15ca75fb6683be0275` | 改造前的 P0 产物（无 `--dump-tokens`） |
| `rknn_multicard_demo.p0dump` | `205232b8e9836e5919b091f9f85d23c5` | P0 + 插桩，回归参照 |
| `rknn_multicard_demo` | `6f5aa9d73dee48ebcafd6a03a2a9d71d` | **当前 P1** |

**顺带新增的工具**：`--dump-tokens <path>`，把每步采样到的 token id 逐行写文件。
P2 必须靠它——N 路会话并发时 stdout 会交错，文本对比不再可行，只能按会话比 ID。

**一个操作坑（浪费了一轮跑测）**：`pscp` 不复刻权限位，新上传的文件默认 644，
执行时报 `rc=126`。上传后要 `chmod +x`（覆盖已有文件时旧权限会保留，所以只有新文件名才中招）。



### P2：并发执行

| # | 动作 | 状态 | 说明 |
|---|---|---|---|
| 2.1 | 新增 `--sessions N`（范围 1..16） | ✅ | 未指定时走原单会话路径（零回归）；显式指定（含 N=1）才进并发执行器 |
| 2.2 | `std::vector<Conversation>` + 每会话一个驱动线程 | ✅ | 复用 P1 的 `run_pipeline_once`，外层并发；加了 `StartGate` 让 N 条线程同时起跑 |
| 2.3 | 每卡一把 `run_mutex` | ✅ | 放在 `StageContext` 上（不是全局数组，语义更清楚）；`run_stage_worker` 与 stage0 的 `session_run` 前后 `lock_guard`。**没有**出现锁序问题：任何线程同时只持有一把卡锁，且从不在持锁时去取 slot 锁 |
| 2.4 | 首轮相位错开 | ❌ **不做** | 实测不需要。`StartGate` 反而让 N 个会话**同时**起跑（最坏情况），N=4 仍有 83% 效率，"N 个会话同时挤 stage0"的代价已被卡级串行吸收 |
| 2.5 | 输入源改造（stdin 分发到 N 个会话） | ❌ **未做** | `--sessions` 与 `--interactive` 互斥，参数校验里直接拒绝，避免"参数接受了但行为没测过" |
| 2.6 | 性能统计按会话分列 | ⚠️ **部分** | 已有每会话 prefill/decode token 数、纯 decode tok/s、墙钟；**聚合**吞吐按"所有会话从最早起跑到最晚结束"的墙钟算。`print_stage_performance_statistics` 仍是卡级聚合（同一张卡上多个会话的耗时会累加在一起） |
| 2.7 | 自动化压测开关 | ✅ | 合并成 `--sessions N --rounds M`（不另设 `--benchmark-sessions`）：每个会话跑 M 轮固定 prompt，输出每会话 + 聚合吞吐。`--rounds` 首轮用 `--prompt` 原文（与单会话路径同源，token 才可比对），后续轮用 chat 模板续写 |

**验收标准：**
- N=2 聚合 decode 吞吐 ≥ **20 tok/s**（理论 24.8，留 20% 余量）；N=3 ≥ **30 tok/s**。
- 每个会话的输出与"该会话单独运行"时逐 token 一致（**正确性优先级高于吞吐**）。
- 连续跑 30 分钟无 `malloc(): unaligned tcache chunk`、无段错误、无内存增长。

#### P2 完成记录：**已实现，正确性全部通过，吞吐达标（2026-09-14）** ✅

**版本指纹**

| 项 | 值 |
|---|---|
| commit | `446cbdf` multicard: drive N conversations concurrently (P2) |
| 分支 / tag | `feature/multisession-concurrency` / `p2-concurrent` |
| 源码 md5 | `ef943467048fd5f985a89e1ec725f248` |
| 二进制 md5 | `8f6e900dcb30b55edd607507f09f5394` |
| Build ID | `dc876cf266a3a99a720075fbc65160086edcc078` |

**落地形态**

- `StageContext::run_mutex`（**卡级串行锁**，方案 §3.2 的方案 A）：持锁范围覆盖整个
  `rknn3_session_run`，所以回调里对共享 `output_tensors` 的读取也在锁内。
  这是 P2 唯一的真数据竞争，到此关闭。
- `std::vector<std::unique_ptr<Conversation>>`，每会话一条驱动线程 + `StartGate`
  同时起跑。**每会话的 4 段仍各占一张卡并行跑**，被串行化的只是"同一张卡上的不同会话"。
- `g_tokenizer_mutex`：`Tokenizer` 是 3rdparty 预编译库、没有线程安全承诺，
  decode 路径的 `TokenToPiece`/`Decode` 由 N 个线程同时调用，串行化掉
  （微秒级，相对 80ms/token 可忽略）。
- 每会话自己的 token dump 文件（N>1 时为 `<path>.s<i>`）；并发模式下关掉逐 token
  的 stdout 打印——N 路必然交错，且 `printf` 在热路径上会污染吞吐测量。

**验收证据**

*（1）跨会话无串扰——这是最关键的一条*

同一 prompt、同 `-n 48 --ignore-eos`，每路 dump 出的 token id 与**单会话路径**逐字节比对：

| 配置 | 输出文件 | 结果 |
|---|---|---|
| legacy（不指定 `--sessions`） | `p2_single.tok` | 49 token（基准） |
| `--sessions 1` | `p2_s1.tok` | **IDENTICAL** 49/49 |
| `--sessions 2` | `p2_s2.tok.s0` / `.s1` | **IDENTICAL** 49/49、49/49 |
| `--sessions 4` | `p2_s4.tok.s0..s3` | **IDENTICAL** 49/49 ×4 |

7 路输出全部一致。任何跨会话串扰（`output_tensors` 被覆盖、KV 混用、流水线队列串台）
都会让 token 变掉——这条过不了，后面的吞吐数字没有意义。

> 注意 `--sessions 1` 单凭自身不算证据（R1 已经证明 P2 二进制在 legacy 路径下与 P1
> 黄金文件一致），它的作用是**隔离变量**：把"并发执行器本身"的引入与"多会话"分开，
> 这样 N=2/4 的差异只能归因于会话数。

*（2）单会话路径零回归——卡级锁是加在共享路径上的，必须证明它无害*

用 P2 二进制跑**不指定 `--sessions`** 的原路径，与 P1 时期的黄金产物比对：

| 项 | P1 黄金 | P2 本次 | 结论 |
|---|---|---|---|
| prompt A token ids | `a_p1.tok`（49） | `p2r_a.tok`（49） | **IDENTICAL** |
| 交互 3 轮 token ids | `it_p1.tok`（1539） | `p2r_it.tok`（1539） | **IDENTICAL** |
| 交互 6 轮（跨清 KV）token ids | `kv_p1.tok`（3078） | `p2r_kv.tok`（3078） | **IDENTICAL** |
| `--perf 512 128` prefill | 341.06 tok/s | 340.58 tok/s | 噪声内 |
| `--perf 512 128` decode | 80.82 ms/tok | 80.77 ms/tok（12.38 tok/s） | 噪声内 |
| 清 KV 分支触发 | 恰好 1 次 @ 3826/4096 | 恰好 1 次 @ 3826/4096 | 一致 |

*（3）吞吐标定*（`--sessions N --rounds 4 -n 64 --ignore-eos`，同 workload）

| N | 聚合 decode 吞吐 | 相对 N=1 | 每会话纯 decode（实测） | 扩展效率 |
|---|---|---|---|---|
| 1 | 10.73 tok/s | 1.00× | **12.38 tok/s** | — |
| 2 | **19.78** | 1.84× | 11.40 / 11.32 | 92% |
| 3 | **28.63** | 2.67× | 11.06 / 11.28 / 10.86 | 89% |
| 4 | **35.74** | **3.33×** | 10.55 / 10.34 / 10.16 / 10.53 | 83% |
| 5 | 34.94 | 3.26× | — | **反而下降（65%）** |

三点值得记下来：

1. **N=1 的每会话纯 decode = 12.38 tok/s，与 P0 实测基线 12.38 完全吻合。**
   这是对并发执行器的独立交叉验证——换了一套线程模型，单会话性能没有变。
2. **N=4 时每会话只从 12.38 掉到 10.16~10.55 tok/s（单会话延迟劣化约 17~22%），
   而总吞吐 3.33×**。这正是"卡内串行 + 卡间并行"该有的样子：吞吐换来的代价很小。
3. **N=5 比 N=4 低**（34.94 < 35.74）——4 张卡在 N=4 已经喂满，第 5 路只增加排队。
   这条实测确认了 §0 第 2 条的"上限被卡数锁死"。
4. 上面表里的"同 workload 口径"= 256 token/会话（4 轮 × 64）**含 4 次 prefill 气泡**，
   所以聚合绝对值（N=1 只有 10.73）低于"纯 decode × N"的理论值；
   **相对 N=1 的 3.33× 才是并发增益的正确读数**（纯 decode 口径见每会话那一列）。
   "扩展效率"= (聚合吞吐 / N=1 聚合吞吐) / N。

*（4）边界与失败路径*

- `--sessions 5`：rc=0，5 个会话全部建成功（与 §4.1 的"每卡上限 5"一致）。
- `--sessions 6`：第 6 个会话在 stage0 `rknn3_session_init` 失败，进程 **rc=255 干净退出**，
  日志里没有 segmentation fault / abort / double free。已建的 5 个会话按
  "先销毁会话、再销毁 context"的顺序释放。

**P2 没做的事情（明确记账，别当成已完成）**

> 括号内为 2026-09-14 的更新状态；未标注的仍是欠账。

1. ~~**`--sessions` 与 `--interactive` 互斥。** 单线程 `read_line_utf8` 没法喂 N 路会话，
   需要一输入线程 + 按会话分发 + N 路输出各自成行。参数校验里直接拒绝了这个组合。~~
   → ✅ **已在 P3 解开**（§5 的 P3 完成记录）。
2. **`--sessions` 与 `--perf` 互斥**，同理。→ **仍然保留**（perf 模式是单会话标定用途）。
3. ~~**没有测 30 分钟稳定性**（验收标准的第 3 条）。目前最长的单次运行是 6 轮交互
   （3078 token，约 4 分钟）和 N=4 的 1024 token 压测（约 29 秒）。~~
   → ✅ **已测，通过**（§9.3）。
4. **没有验证方案 B**（每会话一份 `output_tensors`，每份约 1.3 MB，可去掉卡锁）。
   卡级锁已经拿到 83% 效率，方案 B 的潜在收益只有那剩下的 17%，而它要赌
   "runtime 支持同 context 真并发"——不值得先做。→ **仍然没做**。
5. 卡级统计仍是多会话混在一起（见表中 2.6 的 ⚠️）。→ **仍然没做**。
6. **没有跑 ASan/TSan** → ✅ **两者都已跑**（TSan 两轮见 §9.5；ASan 四阶段 + 阳性对照见 §9.3）。

### P3：可选增强

| # | 动作 | 依赖 | 状态 |
|---|---|---|---|
| 3.0 | **交互式多会话**：`--sessions N --interactive`，stdin 分发到 N 路 | 无（P2 已完成调度） | ✅ **已完成（2026-09-14）**，见下 |
| 3.1 | `rknn3_session_run_async` + `rknn3_session_pause/resume` 替换同步 `session_run` | 需先确认 async 语义与 callback 时序 | 未做 |
| 3.2 | 会话持久化：`rknn3_session_save_kvcache` / `load_kvcache_from_path`（`rknn3_api.h:1610-1633`） | 让会话可挂起/恢复/迁移 | 未做 |
| 3.3 | 前端服务化（HTTP/WebSocket），每连接一个 `Conversation` | ~~需先有 P2 的调度~~ ✅ 已具备 | 未做（3.0 是它的前置） |
| 3.4 | 常驻 worker 线程池（§3.3 方案 B） | 若线程抖动成为瓶颈 | 未做（实测线程开销 0.17%，暂无必要） |
| 3.5 | 动态会话数（按负载创建/回收） | 受 KV cache 总容量硬约束 | 未做 |

#### P3 完成记录：**交互式多会话已实现，验收全部通过（2026-09-14）** ✅

**版本指纹**

| 项 | 值 |
|---|---|
| commit | `075a7a4` multicard: interactive multi-session over --sessions (P3) |
| 分支 / tag | `feature/multisession-concurrency` / `p3-interactive` |
| 源码 md5 | `ff85aa377f91eb3b5e543dfa8754e4c6` |
| 二进制 md5 | `d30ce8af8b508dd783cfa934100b462f`（板上文件名 `rknn_multicard_demo.p3`） |

**落地形态**

- **互斥解开**：`validate_command_line_options` 里去掉了 `--interactive` + `--sessions` 的拒绝；
  `--perf` + `--sessions` 的拒绝**保留**。新增 `--rounds` + `--interactive` 的拒绝
  （两种驱动方式不该同时生效，rc=255）。
- **分派：谁空闲给谁。** 一条输入线程读 stdin，把整行推进 `InteractiveDispatcher`
  的 `pending` 队列；每条会话一个 worker 线程按"抢到就干"消费。**不预先指派**
  哪一行归哪个会话——这是与"轮询均摊"不同的语义，也是用户选定的方案。
- **输出：整段成块 + `[s<i>]` 前缀。** 生成过程全程不打印，一轮生成完整回复后
  一次性 `printf("[s%d] %s\n")` + `fflush`。
  实现上复用了 `LastStageResultState::block_out`：只有最后一阶段会采样
  （`disable_sampling` 只在非末阶段为 true），所以**只有一个线程会往 `block_out` 追加**。
- **背压**：`pending` 上限 `kInteractivePendingCap = 32`，输入线程在队列满时等 `cv_space`。
- **防挂死的两个细节**（都是先审出来、后写进去的）：
  ① 若所有会话都提前失败，输入线程可能正卡在 `cv_space` 上——靠 `active_workers`
  计数进谓词并在归零时唤醒，保证一定退得出；
  ② 若 stdin 是 TTY 且会话全挂了，`getline` 会一直等用户输入——检测到这种情况时
  打印一行提示让用户按回车/Ctrl-D。
- **统计口径**：交互模式的耗时窗口是**"首个真实轮次开始 → 最后一个轮次结束"**，
  从线程启动开始计时会把人的思考时间算进去。

**验收证据**（脚本 `run_p3_interactive.sh`，板卡 `<板卡临时目录>`）

| 测试 | 方法 | 通过标准 | 实测 |
|---|---|---|---|
| IT-A 等价性 | `--sessions 1 --interactive -n 512 --ignore-eos < 3 行` | 与 P1 交互黄金逐 token 一致 | ✅ **IDENTICAL** `it_a.tok == it_p1.tok`（1539 token） |
| IT-B 等价性（跨清 KV） | 同上，6 行输入 | 同上，且清 KV 行为一致 | ✅ **IDENTICAL** `it_b.tok == kv_p1.tok`（3078 token）；清 KV 恰好 1 次 @ 3826/4096 |
| IT-C 结构性正确 | N=4 + 12 行输入，`-n 96 --ignore-eos` | 12 个块、无穿插、四路都分到活 | ✅ **12 个 `[s<i>]` 块**；含前缀行数 12 = 行首块数 12（**无穿插**）；**s0/s1/s2/s3 各 3 轮** |
| IT-C token 总量 | 四路 dump 求和 | 12 × (1 prefill + 96 decode) = 1164 | ✅ **恰好 1164**（每会话 291） |
| IT-D CLI 门 | `--sessions 4 --perf 512 128`；`--sessions 4 --interactive --rounds 2` | 两个组合都拒绝 | ✅ 均 rc=255 |
| IT-E 崩溃检查 | 日志 grep | 无 segfault/abort/堆损坏 | ✅ 全部 0 命中 |

> **IT-C 能验什么、不能验什么**：**哪些行落到哪个会话是不可预期的**（取决于当时谁空闲），
> 所以它只验"结构与总量"，不能拿 token 内容跟任何黄金比。等价性必须靠 IT-A/IT-B 的 N=1
> 构造来验——N=1 时所有输入必然按顺序落到 s0，与单会话路径一一对应。

> ⚠️ **IT-C 里 `--ignore-eos` 的副作用**：这个 flag 让模型越过轮次边界继续生成，
> 于是回复里会出现下一轮的模板标记。这是**测试 flag 的后果，不是缺陷**；
> 真实验收用的是正常停止条件。

> **原待确认项已关闭（R11，2026-09-14）**：worker 里的 `rknn3_session_clear_kvcache`
> 当时**没有持有卡级锁**——从已验收的 `--rounds` 路径原样继承，未独立验证过。
> 后来专门做了验证，**结论是需要锁**，并且**两个逃在锁外的站点都已补上**（见 §9.6）。
> 补齐时同步给 `--rounds` 的清 KV 分支加了一行 `VLOG`：那条分支原先**没有任何日志**，
> 而默认输出必须逐字节不变，所以只能走 `--verbose`。**这正是先前 30 分钟稳定性测试的
> 盲区**——那轮 workload 每会话只有 2048 token，根本没够到清 KV 阈值 3584，
> 测了 30 分钟也没进过这条分支。

**同批完成的两项验证**（P2 的欠账）

- ☑ **30 分钟稳定性**：见 §9.3。6 轮 N=4 全部 rc=0，吞吐无衰减，**首末轮 token 逐字节一致**。
- ☑ **TSan 跑两轮**：见 §9.5。**R11 新加锁的那段代码已真正跑过 TSan**（首轮"从未执行到
  `clear_kvcache`"的盲区已关闭）；多会话路径的报告**全部落在 SDK 的 16 MB
  `FreeListAllocator` arena 内**，控制组（`ignore_noninstrumented_modules=1`）把并发盘的
  报告**整体压到 0**，即没有一条是"两侧都在我方插桩代码内"的竞争。**但有 4 条报告的
  写侧是我方 `input_callback`（写的是 SDK 给的缓冲）——首轮"我方零竞争"的说法已按此收窄。**
  边界必须连同结论一起引用。
- ☑ **ASan 跑一轮**：见 §9.3。四个阶段零报告，**且阳性对照证明这套 ASan 在这块板上真会报**
  （堆越界 / use-after-free / LSan 三项都报出来了）；AS-2 的四路 dump 与 Release 版
  R11 T2 dump **逐字节一致** → 排除"构建不同、跑了另一条路"这个解释。

### M5：服务化（已实现，设计与实测见 §9.7）

本节的改造清单到 M4 为止。**M5（板端 OpenAI 兼容服务）已实现并实测**，不在这里重复展开：

- C++ 侧 `--serve` 提供**帧协议**（独立 fd，`REQ`/`DELTA`/`DONE`/`CLEAR`/`ERR`/`QUIT`），
  **复用的是本节 P3 那套 dispatcher**，唯一语义差别是 `PendingInput::target_session`；
- 门面是板端 `python3` 网关（标准库，无第三方依赖，**已入库 `examples/multicard/serve/`**），
  对外 `/v1/chat/completions`（SSE + 非流式）、
  `/v1/models`、`/health`；
- **HTTP 层 N=1/2/4 = 10.96 / 20.22 / 36.44 tok/s（3.33×）**（连量 4 轮，伸缩比 3.32~3.38），
  对照官方 `rkllm3-server` 同板同条件 **10.63 / 10.98 / 10.98（1.03×）**；
- 四个"只影响速度/并发度、答案完全正确"的坑（第 4 个是**偶发**把并发串行化，必须先修掉才能报出上面这个稳定值）、
  不可逐字节复现这条验收前提、未验证项（`tool` 角色等）与**网关位置与 §8.1 红线的冲突**，都记在 §9.7。

### 工作量估算

| 阶段 | 人日 | 风险 |
|---|---|---|
| P0 探针 | 0.5 | 结果可能否决整个方案 |
| P1 状态下沉 | 2–3 | 低（可逐步回归） |
| P2 并发执行 | 3–5 | 中（锁、时序、SDK 未文档化行为） |
| P3 增强 | 4–6 | 中高 |
| **合计（P0–P2）** | **6–9 人日** | |

---

## 6. 分阶段收益与里程碑

| 里程碑 | 交付物 | 可观测指标 |
|---|---|---|
| M0 | P0 探针报告 | 权重共享 ✅/❌、每会话 KV 实测大小 —— **✅ 已完成** |
| M1 | P1 重构（单会话） | decode 80.75ms/tok、逐 token 一致 —— **✅ 已完成（80.82 ms/tok，token 全中）** |
| M2 | P2 并发（N=2） | 聚合 decode ≥ 20 tok/s，2 会话输出正确 —— **✅ 已完成（19.78 tok/s 同口径；纯 decode 每会话 11.4，token 全中）** |
| M3 | P2 完成（N=3~4） | 聚合 decode ≥ 30 tok/s，30 分钟稳定性 —— **✅ 全部完成（N=3 28.63、N=4 35.74；30 分钟稳定性已测，§9.3）** |
| M4 | P3 交互式多会话 | `--sessions N --interactive`，N 路各自成块输出 —— **✅ 已完成（§5 的 P3 完成记录）** |
| M5 | P3 服务化（OpenAI 兼容 HTTP） | `--serve` 帧协议 + 板端网关，多连接并发 —— **✅ 已完成（§9.7）：`/v1/chat/completions`（SSE + 非流式）/ `/v1/models` / `/health`，N=4 聚合 36.44 tok/s = 3.33×（4 轮 3.32~3.38；官方 server 同条件 1.03×）** |

---

## 7. 风险清单

| # | 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|---|
| R1 | ~~**权重被 per-session 复制 → OOM**~~ ✅ **已排除** | — | — | P0 探针实测：权重共享，建 4 个额外 session 只消耗 4×8.05 MB/node（§4.1） |
| R2 | ~~**KV cache 总量超卡内存**~~ ✅ **已排除** | — | — | 实测每会话仅 64.4–70.7 MB/卡，4 会话共 ~283 MB，余量充足；且 KV 预分配、与上下文长度解耦（§4.1） |
| R2b | **卡内 session 上限 5，且不由内存决定** | 已确知（非概率） | 中 | 目标 4 路有 1 个 session 余量；**不要超配到 5**。成因未定位（§4.4），调 ctx 无效 |
| R3 | 同 context 并发导致 hidden states 串话 | 高（不处理必现） | 高 | §3.2 卡级锁 + 可选 per-session output mem |
| R4 | **`malloc(): unaligned tcache chunk`** —— 此前已在本 demo 中出现过的堆损坏，根因未定位 | 中 → **低** | 高 | 已开 `-DENABLE_ASAN`（`cpp/CMakeLists.txt:10`）跑完四个阶段（含最重的 4 会话 × 4800 decode + 4 次清 KV）：**0 报告**，且**阳性对照证明这套 ASan 在这块板上会报**（§9.3）；30 分钟稳定性 0 命中（§9.3）。**概率下调，但不关闭**：① ASan 会改变堆布局与时序，"竞争诱发型"损坏可能因此不出现；② ASan 下累计只跑了约 10 分钟，**没做满 30 分钟 soak**；③ SDK 自己缓冲内部的越界看不见（同 §9.5 的边界）。 |
| R5 | 死锁（锁序错误） | 中 | 高 | 严格 stage 升序锁；加 `--sessions 1` 回退开关；持锁超时告警 |
| R6 | SDK 未文档化的同 context 并发限制 | 中 | 中 | 卡级锁已把并发"降级"为卡内串行，风险大幅降低。**但 TSan 实测发现：卡级锁只能串行化调用方，管不到 SDK 自己那条回调/传输线程**——竞争报告**全部落在 `librknn3_api_rkcp.so` 的 16 MB `FreeListAllocator` arena 内**，且两侧持的常是**两把不同的 SDK 锁**（说明那两把锁没有互斥这段内存，§9.5）。**其中 4 条的写侧是我方的 `input_callback`（写 SDK 给的缓冲）**——静态看指向 SDK 内部的缓冲管理，但**未能定性真伪（库未插桩），不视为已关闭** |
| R7 | `libtokenizer.a` 是否线程安全未知 | 低 | 中 | 每会话各自持一个 `Tokenizer` 实例（而非共享），或给解码加锁；`init_tokenizer_and_embedding`（`main.cc:1459`）要改成可重入 |
| R8 | 每 token 新建线程（`main.cc:1893`）× N 会话 → 线程风暴 | 低 | 低 | 实测 host 开销仅 0.17%，暂不处理；P3 换常驻池 |
| R9 | N=4 时利用率 >100%，排队抖动放大 | 高 | 中 | 产品配置定为 N=2~3，N=4 仅压测 |
| R10 | ~~交互输入单线程阻塞（`read_line_utf8`）~~ ✅ **已解决** | — | — | P3 拆出独立输入线程 + 待办队列 + 背压（§5 的 P3 完成记录）。**注意交互输入用的是普通 `std::getline`，不是 `read_line_utf8`**——后者逐字符回显，会与 N 路输出块互相穿插 |
| R11 | ~~**交互 worker 里的 `clear_kvcache` 未持卡锁**~~ ✅ **已解决（09-14 补锁 / 09-15 验证完成）** | — | — | **验证结论：需要锁。** 依据是 SDK 自己的契约——N 个会话的 session 共用一张卡的**同一个 `rknn3_context`**，A 会话清 KV 而 B 会话在同一张卡上跑 `session_run`，就是对该 context 的并发使用（§9.6）。全仓 6 处 `clear_kvcache` 里只有 2 处逃在锁外，均已补上；其余 4 处要么单线程、要么在所有 worker `join` 之后 |
| R12 | **网关无鉴权**（M5 新增面；**多人接入后升级**） | 已确知（非概率） | **中 → 高**（若真的对多人开放） | `/v1/chat/completions` 一旦 `HOST=0.0.0.0` 暴露，局域网内任何人可占满 4 张卡。**§9.8 之后多了一层后果**：网关只能按请求上带的身份分会话、无法核实身份，所以一个客户端用 5 个不同 id 各发一次就能占满全部会话、把所有真人挤进队列等 `QUEUE_TIMEOUT`——**"超过 5 个就排队"这条策略只在参与者都守规矩时成立**。缓解：Agent/演示同机时用 `HOST=127.0.0.1`；要对外提供服务**必须先**加反向代理/鉴权，**本次未做**（§9.7 / §9.8） |
| R13 | **网关侧会话账本与真实状态不一致 → 静默退化（全量 prefill / 并发变串行）** | 已发生 4 次（均已修） | 低（只慢不错） | 前 3 次的症状是"答案对、慢 3~4 倍"；第 4 次的症状是"答案对、单会话速度正常、**聚合吞吐不随 N 涨**"（会话租约算错 → 两段对话钉在同一会话上被串行执行），且**取决于到达顺序**（同二进制量到过 3.40× 与 1.00×）。缓解：响应头 `X-KV-Reuse` 暴露 `reset/sent/base/full`、网关日志暴露**租约落点**、`selftest()` 里 4 条回归用例（均在修复前代码上验证过失败），并用 `usage` 里的两个数（整段 `prompt_tokens` 与 `prompt_tokens_details.cached_tokens`）在 HTTP 层做端到端核对（§9.7） |

| R14 | **prefill 分块大小取自 `--bucket-size`，与运行时的固定分块脱节 → `embed buffer too small`** | 已发生（修复处注释记录的现场症状） | 中（`--bucket-size` 非 128 时必现；**默认值下不触发**） | `stage_output_callback` 用 `min(remaining, g_bucket_size)` 记"这次回调吐了几个 token"，但**运行时是按模型的 max dynamic seq len（128）切的，与 `--bucket-size` 无关**。两者不一致时记账偏小，多出来的 token 被算进下一批，累积错位后报 `embed buffer too small`。改为从 output tensor 的元素数（÷ `embedding_dim`）**反推真实分块**，`g_bucket_size` 只作兜底。**默认 128 下两个口径恰好相等，所以这个 bug 一直没被默认配置撞到**——它的触发条件不是"代码坏了"，而是"有人改了 `--bucket-size`"。修法在 `main.cc` 的 `stage_output_callback`。**另注（2026-09-16）**：这个修复原先**只存在于构建服务器那份检出上**（未提交），本轮才移植进仓库——一个只活在某一台机器磁盘上的修复，等同于不存在。**同日已在板上复现并验证修好**：`--bucket-size 64` + 一段跨 128 token 分块的 prompt，改动前 `e223ea8f` 回 503 且后端日志 `[stage1] embed buffer too small: need=1361920, got=1310720` + `prefill failed`，改动后 `3355dc4f` 正常作答（详见 §9.7 末）。 |

---

## 8. 版本管理

### 8.1 代码分支策略（实际执行情况）

**原计划**是「先固基线、再在改造分支上把 P1 拆成 p1-1..p1-5 五个小 commit，每个 commit 都过一遍回归」。
**实际做法与之不同，如实记录**：

```bash
# 实际：基于 main(b6a47da) 开分支，一个 commit 完成一个阶段，跑完回归才推进
git checkout -b feature/multisession-concurrency
#   d59a239  multicard: generalize chat template, add multi-session feasibility probe   (P0)
#   6347f57  multicard: sink session state into a Conversation object (P1)
#   446cbdf  multicard: drive N conversations concurrently (P2)
#   fa8aa97  docs: 多 Session 并发推理方案（脱敏版）v1.1
#   8f1cdbb  docs: P2 并发完成记录 + 修正 v1.1 里被实测推翻的预期
#   075a7a4  multicard: interactive multi-session over --sessions (P3)
#   4690d3a  docs: P3 交互式多会话完成记录 + 30 分钟稳定性 / TSan 实测（v1.3）
#   9d2d167  multicard: take the card lock around clear_kvcache in both workers (R11)
#   15f160e  docs: R11 清 KV 卡锁验证与补锁记录（v1.4）
#   6d0666a  multicard: framed --serve protocol for the board-side OpenAI gateway (M5)
#   db3fe3d  multicard: move the OpenAI gateway into the repo (examples/multicard/serve/)
#   213a131  multicard/serve: stop the session pool from handing a bound session to a new conversation
#   462cec8  multicard/serve: add a demo runbook and the concurrent-streaming demo it drives
#   570c406  multicard/serve: serve the 4-panel chat demo from the gateway itself
#   bc7f013  multicard/serve: multiuser edge-server mode (identity, queue, release)
#   1dde346  multicard: reject an over-long turn instead of silently clearing KV  (服务侧收尾加固 + 一次由执行发现的竞态)
#   b13bbdb  multicard: derive the prefill chunk from the output tensor, not --bucket-size  (R14；原先只在构建服务器磁盘上)
git tag p0-baseline     # 指向 d59a239，回归对照点
git tag p1-session-split
git tag p2-concurrent
git tag p3-interactive  # 指向 P3 的代码 commit
git tag r11-kvlock      # 指向 R11 的代码 commit
git tag m5-serve        # 指向 M5 的代码 commit
git tag m5-multiuser    # 指向多用户接入（身份/排队/交还）的代码 commit
```

> **文档记录为什么总在下一个 commit**：完成记录里要写**本阶段 commit 的哈希**，
> 而文档自己就在那个 commit 里——哈希写不进去（改一个字哈希就变）。
> 所以沿用 P2 的做法：**代码一个 commit、文档记录一个 commit**，两者都推。
> 顺序上文档在**工作区先改好**，哈希落地后才随第二个 commit 提交。

**P1 为什么没有拆成 5 个 commit**：P1 的五步互相咬合——`Conversation` 一旦引入，
`PipelineState`/`result`/`callback` 就必须同时改到 `Conversation` 上，否则编译不过；
中间态（保留 `g_last_stage_result` 作 alias）要额外写一版临时双写逻辑，属于**为分子而分子**。
更关键的是：**板卡上一轮完整回归要 40 分钟以上**（每次都要过 PCIe 同步 3.6GB 权重），
每个小 commit 都跑一遍不现实。折中办法是**用「逐 token ID 一致」代替 commit 粒度做 bisect**：
真出问题时，bisect 的对象是「二进制 × prompt 的 token 序列」，而不是 commit。

**P2 同样是一个 commit**，理由同上（卡级锁、并发执行器、每会话 dump、统计若要拆开，
每一步单独都无法在板卡上验证"并发不串扰"——只有全部到位才能跑出有意义的对照）。
**并发的不确定性这次没有成为问题**：N=2/4 的 7 路 token 与单会话路径逐字节一致，
说明卡级锁把不确定性窗口关干净了；如果以后测到偶发漂移，才需要回到 commit 级 bisect。

**回归对照物是独立保存的，不依赖 commit 粒度**（这是代替小 commit 的关键）：

| 对照物 | 位置 | md5 |
|---|---|---|
| P0 源码快照 | `rt_work/main.cc.p0_baseline` | `ec1402fc7f9e7134a5f23483516812e8` |
| P1 源码快照 | `rt_work/main.cc.p1_session_split` | `dfad38d5bbe49dae699bc2da52708e01` |
| P0+插桩源码 | `rt_work/main.cc.p0_dump` | `81857bf2fa962a85d54a5a89cb6929c2` |
| P2 源码快照 | `rt_work/main.cc.p2_concurrent` | `ef943467048fd5f985a89e1ec725f248` |
| P3 源码快照 | `rt_work/main.cc.p3_interactive` | `ff85aa377f91eb3b5e543dfa8754e4c6` |
| R11 源码快照 | `rt_work/main.cc.r11_kvlock` | `c1fcc366958c34941b99c46c8f22cd62` |
| M5 源码快照（`--serve` 帧协议 / `TokenSink`） | `rt_work/main.cc.m5_serve` | `108706c0c96c231f1632c489fc5963b3` |
| M5 板端网关（**已入库**）演进 | `examples/multicard/serve/rkllm_gateway.py` | M5 入库 `db3fe3d` = `e09ac2b941de767443c0d1d635bd305b` → 修坑 4 `213a131` = `1eb4347a3aae6ec9e69e0f624bdaf62e` → 加网页演示 `570c406` = `a7dde57a99cdb89d343e3f984000cec6` → 多用户版 `bc7f013` = `f35fd2b84ec9968d83795c70cc3d6e0d` → 修 ERR 帧解析 `5c167f5` = `fcb9dcd3b15bf26190d142296055593c` → 加 REJECT 帧解析 `1dde346` = `7a06e5ca2b9efb230d119582bbc56961` → **加工具调用/工具目录/助手轮回显 `c1fd763` = `166d1543000f764b505d04eb40b72fba`（2026-09-17 上板验收所用的版本）**（本行 md5 **2026-09-16 逐版本重算过，都对**；另注：`5c167f5` 那格上的"当前 HEAD"**写的时候是对的，`1dde346` 之后就过期了**——一本流水账里最容易悄悄变错的就是"当前"这个词，2026-09-16 上板时才发现） |
| └ **板卡上的部署副本（会漂移，以这一行为准）** | 板卡上的部署副本 | **2026-09-16 上板实测**：原副本 md5 `009d0e265c2530c4a44573f42611b810`——早前那句"`009d0e26` 一说"是对的，`e09ac2b9` 不是。它与 `570c406` **只差 `_static_demo` 的一处 docstring**（功能无差别，不是功能漂移；**何时分叉的没查**），所以 §9.7 里"入库后两份副本 md5 相同"那句**在 2026-09-16 实测时不成立**。它**没有** §9.8 的身份/排队/`close`/`/v1/pool` 的 `key`。**同日已就地打上 ERR 解析补丁**（见 §9.7 末），原件留 `rkllm_gateway.py.bak_err_parse`，补丁后 = `8fc1b7076c99c2f226a251a554b90137`——**没有拿 HEAD 整体覆盖**，因为那会顺带推上只在桩后端验过的排队/`close`/`pool`。要用 §9.8 那套仍需重新下发 + 重启网关（重载模型约 240s）。**同日稍后（全套件验收时）已整体换新**：后端 `3355dc4f` + 网关 `7a06e5ca` + 演示页 `1db0882d` + 测试脚本，旧件全存 `bak_20260916/`（回滚 = 换回 `rknn_multicard_demo.bak_e223ea8f` 与 `bak_20260916/` 里的文件，再重启）。也就是说"**没有拿 HEAD 整体覆盖**"只在**当天前半段**成立，别按这行去推断板卡此刻的状态。**2026-09-17 整体重换一次**（工具调用版网关 + 8192 上下文模型）：`serve/` 下 10 个文件与工作区**逐字节相同**（md5 逐一核对，表见 §9.9），旧件在 `serve/bak_20260917_tools/`——**所以"板卡上不是仓库 HEAD 的副本"这句话，从 2026-09-17 起不成立**；它是 09-16 之前的历史，别再当现状引用 |
| 网页演示页（**已入库**） | `examples/multicard/serve/demo_4chat.html` | `570c406` = `88a53e8b1935c7ef27bb04959ec7c211`（旧版，四个匿名对话）→ **`bc7f013` = `1db0882d98850b20232d0f96e325c9a1`**。**板卡上的是旧版** `af7240627777f576a24d5f95bd522b38`（2026-09-16 核实，与早前记的 `af72406…` 吻合）——它会以四个匿名对话占住全部 4 个会话（§9.8 咬到 5）；**2026-09-16 已换成 `1db0882d…`**（不换的话，新网关的排队会让旧页面的四个对话互相堵住，演示当场卡死） |
| M5 启动/构建/验收脚本（**已入库**） | `examples/multicard/serve/*.sh`、`*.py`、`README.md` | 见 `git ls-tree`；板卡路径全部走环境变量 |
| M5 板端二进制（Release） | 板卡 `<板卡安装目录>/rknn_multicard_demo.serve` | `e223ea8f3f5d38e82a37cd560c29fb96`（1091328 B，M5）→ **`3355dc4f44a5f47332c8e0d7528be84d`（1076496 B，`b13bbdb`，2026-09-16 起在板上运行）** |
| 板卡二进制 | 见 §5 各阶段完成记录的表 | P0 `a6739a6e…` / P0+插桩 `205232b8…` / P1 `6f5aa9d7…` / P2 `8f6e900d…` / P3 `d30ce8af…` / **R11 补锁后 `1b01b960904c7412f64a173b222240a1`** |
| TSan 二进制（首轮，**不含 R11 锁**） | 板卡 `<板卡临时目录>/rknn_multicard_demo.tsan` | `acb78c120353c73f74725ed1c7a30ed2` |
| TSan 二进制（补轮，**含 R11 锁**） | 板卡 `<板卡临时目录>/rknn_multicard_demo.tsanr11` | **`7f19e6d505ac443829604765bcb2b142`** |
| ASan 二进制 | 板卡 `<板卡临时目录>/rknn_multicard_demo.asan` | `80344d4bdc81d8b7a6aa94a78f62eeaa` |
| ASan 运行时（GCC 11 的 `libasan.so.6`） | 板卡 `<板卡临时目录>/libasan.so.6` | `1dc1577ea7882fbd5239039432b34c1e` |
| ASan 阳性对照源码 | `rt_work/asan_probe.cc` | 编译产物 `asan_probe2` @ `-O0` |

**红线（2026-09-14 更新）：**

- **`rt_work/` 一律不进 git、不推 GitHub、不上云、不上板卡。** 它是任务过程记录：
  部署日志、探针输出、实测原始数据、源码快照。
  > ⚠️ **M5 曾破这条线，已纠正**：网关一开始写在 `rt_work/serve/`，而它是**可部署件**——
  > 为了让它在板上跑，那份副本被下发到了板卡（违反了"不上板卡"）。**处置：可部署件移进仓库**
  > `examples/multicard/serve/`（脚本里板卡/云路径全部参数化），`rt_work/` 只留过程记录。
  > 判据：**要落到板卡上的东西属于仓库，不属于过程记录目录。**
- **本方案文档（`多Session并发推理方案.md`）已脱敏后入库**，并经作者确认后推送——
  内网地址、板卡用户名/路径、云服务商目录、SSH 端口全部换成占位符。
- **同目录另外三篇**（`多卡推理方案解析.md`、`多卡推理优化方案.md`、
  `重导出调小分块-压prefill气泡.md`）**仍含内网地址、板卡用户名与绝对路径，未脱敏、未入库**。
  它们被本文按文件名引用，所以公开仓库里那两处引用是**断链**。
  要一起推的话，必须先做同样的脱敏处理。
- **不向上游 Rockchip 仓库发 PR。** 这是从 Rockchip fork 出来的个人分支，两个 remote
  （`origin` / `origin-ssh`）都指向自己的 fork。

### 8.2 模型产物的版本管理（**本次改造不涉及**）

多会话是纯运行时改造，模型文件不动。但为了防止"改崩了不知道是哪一版"，建议固化清单：

```bash
# 在板卡上生成模型指纹（cd 到板卡上的模型目录）
cd <板卡模型目录>
md5sum Qwen3.5-27B-llm_seg*.rknn Qwen3.5-27B-llm_seg*.weight \
       Qwen3.5-27B-llm_seg*.safetensors Qwen3.5-27B-llm.embed.bin \
       Qwen3.5-27B-llm.tokenizer.gguf > MANIFEST.md5
```

现场文件清单（已确认齐备）：

```
Qwen3.5-27B-llm_seg0.rknn / .weight / .safetensors   ← 4 段，seg0..seg3 各 3 个文件
Qwen3.5-27B-llm.embed.bin                             ← 词嵌入
Qwen3.5-27B-llm.tokenizer.gguf                        ← 分词器
```

**模型内固化、必须记录在案的元数据**（多会话容量规划依赖它，实测值如下）：

```
max_ctx_len = 4096              ← 实际可用上下文，--ctx-size 传更大无效（§4.3）
kvcache_buffer_lens = [4096]    ← 单 KV group，运行时可选的只有这一个
kvcache_dtype = 1 (INT4_TO_F16)
kvcache_store_method = 2 (GroupQuant)
kvcache_group_size = 1, kvcache_residual_depth = 64
vocab = 248320, emb_dim = 5120, cores/card = 8
```

> ⚠️ 注意 `max_ctx_len` 与 `--ctx-size` 的**从属关系**：`--ctx-size` 只在 ≤ `max_ctx_len` 时生效，超过会被**静默降级**到 `kvcache_buffer_lens` 里最接近的档位并只打一条 warning。多会话下每会话的可用上下文就是这个 `max_ctx_len`，**不是** `--ctx-size`。建议启动时打印这两个值，避免现场误判。

建议把导出时用的参数写进模型目录的 `README`（模型文件本身不改，但**这个数值决定多会话规划**）。

> ⚠️ **2026-09-17 补充：多会话改造本身确实不动模型，但"换一份导出"是会发生的。**
> 当天板上换成了 **8192 上下文的 Qwen3.5-27B** 导出——上面那张表里的
> `max_ctx_len = 4096` 是**旧导出**的值，换模型时必须重新确认，别沿用手抄的数字。
> **确认的判据不是配置文件里的数字，而是模型文件里的物理证据**：
> `-llm_segN.safetensors` 的 `rope_cos_cache` 形状（本次为 `F16 [1, 4, 1, 8192, 16]`）
> 就是"这份导出编码了多少个位置"的答案。`--ctx-size` 必须与它一致——
> **调大了后端静默降级（白给），调小了越界读位置编码（收下了但答非所问，而 HTTP 层一路 200）**。
> 启动脚本用 `CTX` 环境变量传，验收实测见 §9.9.4。

### 8.3 转换环境版本矩阵（**重要：三套环境互斥**）

三个模型族的依赖是**冲突的**，必须分 conda 环境：

| 模型 | 环境名（建议） | torch | transformers | 备注 |
|---|---|---|---|---|
| **qwen3.5-27B** | `rknn3`（现用） | 2.7.0 | **5.3.0** | 需 `torch.distributed.checkpoint.hf_storage` 存在 |
| **qwen3.8-27B** | `rknn3`（现用） | 2.7.0 | **5.3.0** | 同上；另需 `_patch_torch_hf_storage()` shim |
| **gemma4-31B** | `rknn3_gemma` | 2.8.0 | 5.10.0 | 无 shim 需求 |

**踩过的坑（务必写进 `requirements.txt` 注释，已落实）：**

1. **`torch.distributed.checkpoint.hf_storage` 公开名是 torch 2.8 才有的。** transformers 5.10.0 的守卫写的是 `is_torch_greater_or_equal("2.7")`，**差了一个版本号**，导致 torch 2.7 下 import 失败。绕法：`examples/multicard/python/qwen3_8/export_llm_segment.py` 顶部的 `_patch_torch_hf_storage()`，用 `importlib.import_module` 把私有 `_hf_storage` 别名到公开名。
   > ⚠️ **注意**：函数内**不能**写 `import torch.distributed...`，会触发 `UnboundLocalError`（该名字在本函数作用域内被绑定成局部变量）。这是实际踩过的坑。
2. **transformers ≥ 5.8 在 ONNX tracing 下必炸 `IndexError: tuple index out of range`。** 根因：tracing 时 `tensor.shape` 返回 0 维 tensor，于是 `q_length = inputs_embeds.shape[1]` 是 `tensor(64)`；5.8+ 的 BC 分支 `isinstance(q_length, torch.Tensor)` 把它误判成已废弃的 `cache_position`，去索引 `torch.Size([])[0]`。**5.3.0 没有这个分支，所以只有 5.3.0 能过。** 5.8.1 / 5.10.0 均失败。
3. `py_utils/export_llm_helper.py` 里的 `create_causal_mask` 调用已改为**按签名自适应**（用 `inspect.signature` 判断 `cache_position` 是否为必填），以兼容不同 transformers 版本。

### 8.4 编译与产物版本命名

```bash
# 交叉编译（服务器上）。build-linux.sh 在仓库根目录，不是 examples/multicard
cd <云服务器上的仓库目录>
export GCC_COMPILER=aarch64-linux-gnu
./build-linux.sh -t rk3588 -a aarch64 -d multicard
```

- 产物：`install/rk3588_linux_aarch64/rknn_multicard_demo/`，板卡侧对应 `../install-Qwen/rk3588_linux_aarch64`。
- 安装目录末尾的 `W RKNN model can not be found in .../model` 是**无害**警告（demo 不用该内置模型目录），构建仍成功。

#### 部署链路（三跳，实测可用）

板卡在独立内网，**云服务器无法直连板卡**，所以是 本地 → 云服务器 → 本地 → 板卡：

```bash
# 1) 本地 → 云服务器（改代码）
pscp -P <ssh端口> main.cc <云服务器>:.../examples/multicard/cpp/main.cc
plink -ssh -P <ssh端口> <云服务器> "md5sum .../main.cc"        # 校验，确认传的是新文件

# 2) 云服务器交叉编译 + 打包
plink ... "./build-linux.sh -t rk3588 -a aarch64 -d multicard"
plink ... "cd install/rk3588_linux_aarch64 && tar czf /root/pkg.tar.gz rknn_multicard_demo"

# 3) 云服务器 → 本地 → 板卡
pscp ... root@<云服务器>:/root/pkg.tar.gz ./
pscp ... ./pkg.tar.gz <板卡用户>@<板卡IP>:<板卡临时目录>
```

#### 每一步都必须校验 md5 —— 这不是形式主义

实测中这条链路**没有一次**是「传了就等于到位」的。踩过的坑：

1. **`pscp` 静默截断**：出现过传输进度显示 100% 但文件不完整的情况，md5 对比才发现。
2. **MSYS 路径转换**：Windows 的 Git Bash 会把 `/root/...` 改写成 `C:/Program Files/...`，必须 `export MSYS_NO_PATHCONV=1`。
3. **中文经命令行传递乱码**：用 plink 直接跑含中文的 `grep` 模式会匹配不到（本地 shell 编码污染了 pattern，不是二进制里没有）。**验证含中文的字符串必须写成脚本文件再上传执行**，不能走命令行。
4. **CRLF**：pscp 传上去的 `.sh` 脚本带 `\r`，板卡上 `bash` 会报 `$'\r': command not found`。上传后必须 `sed -i 's/\r$//' script.sh`。

#### 版本戳：用 md5 + BuildID，不要只靠文件名

二进制加了 git sha 也好，但**现场判定的第一依据应该是内容哈希**——文件名会骗人（同一路径反复覆盖），哈希不会：

```bash
md5sum   rknn_multicard_demo/rknn_multicard_demo
readelf -n rknn_multicard_demo/rknn_multicard_demo | grep -i 'build id'
```

本次 P0 验证共下发 6 个版本，板卡侧每次覆盖前都留了备份，回滚只需 `cp` 回来。**板卡实测清单（`md5sum` 为准）**：

| 文件 | md5 | 说明 |
|---|---|---|
| `rknn_multicard_demo` | `a6739a6e01092d15ca75fb6683be0275` | **当前已安装 = 探针终版**（BuildID `83d4cca876b615c71f815a8445835c0d2c71e0e3`） |
| `.baseline_e73d3ac6` | `f37d39ddc13e4f414bf8a5d0ed1d52ee` | 原有的干净基线（本次改造之前） |
| `.bak` | `19d148ff89f6106009ded6b05446907d` | 原有的旧备份 |
| `.bak_bef4edfa` | `70be79f862891cdea06821899e4d6bd8` | 原有备份（按 BuildID `bef4edfa…` 命名） |
| `.bak_11418d04` | `11418d043fc15016e30c7228bf2c3c18` | 探针 v1（首个探针版本，BuildID `32fdabd5…`） |
| `.bak_76853bc6` | `76853bc67447434ee9114487e96afda0` | 探针 v2 之前一版 |
| `.bak_9bce613b` | `9bce613b184682ed288e22f3b9747b09` | 探针 v2（修隔离测试判据，BuildID `ae358511…`） |
| `.bak_4c2f9f5f` | `4c2f9f5fa7a8369ec964057ba6c76cdf` | 探针 v3（加满上下文测试） |
| `.bak_e9b638c0` | `e9b638c0ade20ac2442571fbd5b7be96` | 探针 v4（tokenizer 自校验） |

> ⚠️ **注意 `bak_bef4edfa` 是按 BuildID 前 8 位命名的**（不是 md5），所以文件名和 md5 对不上。要统一命名规范，否则下次一定搞混。
>
> ⚠️ **板卡 `rknn_multicard_demo` 现为 P1 版**（md5 `6f5aa9d73dee48ebcafd6a03a2a9d71d`，见 §5 的 P1 完成记录；
> `a6739a6e…` 是更早的探针版）。**P2 版是并列存在的新文件 `rknn_multicard_demo.p2`**
> （md5 `8f6e900dcb30b55edd607507f09f5394`），没有覆盖 P1，方便随时回到零回归对照。
> 回归对照物命名要显式（`<名字>.p0_a6739a6e` / `.p0dump` / `.p2`），不要混进 `.bak_` 那一堆，否则会被轮换掉。

- **发布包命名**：`rknn_multicard_demo_<YYYYMMDD>_<git-sha7>_sess<N>.tar.gz`。
- **探针是常驻工具，建议保留**：`--probe-sessions` 是排查“现场为什么建不出会话”的第一手段，不要在多会话功能完成后删掉它。

### 8.5 配置版本管理

多会话引入的新参数必须有明确的默认值，且**默认值必须保持单会话行为完全不变**（向后兼容）：

**实际实现：**

| 参数 | 默认 | 范围 | 说明 |
|---|---|---|---|
| `--sessions N` | **不指定**（走原单会话路径） | 1..16 | 注意实现与计划不同：**不指定 ≠ 指定 1**。不指定时完全不进并发执行器，保证与改造前逐字节一致；显式指定（含 N=1）才并发化 |
| `--rounds M` | `1` | 1..1000 | 每会话轮数；M>1 时必须同时给 `--prompt`（首轮输入） |
| `--benchmark-sessions N` | — | — | **没有这个参数**，合并进 `--sessions N --rounds M` |
| `--card-lock-timeout-ms` | — | — | **没有实现**。死锁排查靠"任何线程同时只持一把卡锁、且不在持锁时取 slot 锁"这个不变量，实测未出现死锁 |

校验里显式拒绝的组合（避免出现"参数被接受但行为没测过"）：
`--sessions` × `--interactive`、`--sessions` × `--perf`、`--sessions` × `--probe-sessions`、
`--rounds>1` 不带 `--sessions`、`--rounds>1` 不带 `--prompt`。

---

## 9. 验证方案

### 9.1 正确性（最高优先级）

| 测试 | 方法 | 通过标准 | 实测 |
|---|---|---|---|
| 单会话回归 | 与 P1 二进制/黄金 token 文件对比 | 逐 token ID 完全一致 | ✅ 49/49、1539/1539、3078/3078 全中；perf 340.58 vs 341.06 tok/s |
| 并发隔离（N=1） | `--sessions 1` vs 单会话路径 | 逐 token 一致 | ✅ 49/49 |
| 并发隔离（N=2） | 两路**同 prompt 同时**起跑，各自 dump | 两路都与单会话一致 | ✅ 49/49 ×2 |
| 并发隔离（N=4） | 四路同上 | 四路都与单会话一致 | ✅ 49/49 ×4 |
| **交互等价（N=1）** | `--sessions 1 --interactive` vs P1 交互黄金 | 逐 token 一致 | ✅ **IDENTICAL** 1539/1539 |
| **交互等价（跨清 KV）** | 同上，6 轮输入 | 逐 token 一致 + 清 KV 行为一致 | ✅ **IDENTICAL** 3078/3078；清 KV 1 次 @3826/4096 |
| **交互分派结构（N=4）** | 12 行输入，逼着排队 | 12 个块、无穿插、四路都分到活 | ✅ **12 块**；含前缀行数=行首块数（**无穿插**）；s0~s3 **各 3 轮**；token 总量 **1164** |
| 上下文边界 | 会话接近 `max_ctx_len` | KV cache 清理正确触发 | ✅ 交互 6 轮在 3826/4096 触发，恰好 1 次，与 P1 一致 |
| 会话数超限 | `--sessions 6` | 优雅失败、不崩溃 | ✅ rc=255，`rknn3_session_init failed`，无 segfault/abort |
| 长对话 × N | 每会话 20 轮 | 无 KV 溢出、无串话、无崩溃 | ❌ **未测**（最长只到 6 轮 × 单会话） |
| 多轮 × 并发 | N 路各跑多轮 | 同上 | ⚠️ 只做了 N=4 × 4 轮（1024 token），token 未逐轮核对（该轮次没开 dump） |

> **测试设计的要点**：N 路用**同一个 prompt**、**同时**起跑。这是对 `output_tensors`
> 竞争最狠的构造——如果卡级锁漏了或者 `output_tensors` 被两个会话共用而没串行化，
> 两路会读到对方同一位置的隐状态，token 必然分叉。用不同 prompt 反而更容易蒙混过关。

### 9.2 性能

| 场景 | 指标 | 目标 | 实测 |
|---|---|---|---|
| N=1 baseline | decode | 80.75 ms/tok（不能退化） | ✅ 80.82→80.77 ms/tok；`--sessions 1` 纯 decode 12.38 tok/s |
| N=2 | 聚合 decode | ≥ 20 tok/s（理论 24.8） | ⚠️ 同 workload 口径 **19.78**（含 4 次 prefill 气泡）；纯 decode 每会话 11.4 tok/s |
| N=3 | 聚合 decode | ≥ 30 tok/s（理论 37.1） | ⚠️ **28.63**（同口径）；相对 N=1 为 2.67× |
| N=4 | 聚合 decode | ≥ 40 tok/s（压测，允许抖动） | ⚠️ **35.74**（同口径）；相对 N=1 为 **3.33×** |
| N=5 | 聚合 decode | — | ✅ 34.94，**低于 N=4**，确认 N=4 是拐点 |
| N=2 | 单会话 TTFT | 与 N=1 相比劣化 ≤ 30% | ❌ **未测**（没单独量 TTFT；只有整轮墙钟） |
| N=4 | 单会话延迟劣化 | — | ✅ 每会话 12.38 → 10.16~10.55 tok/s（劣化约 17~22%） |
| **N=4 交互式** | 聚合 decode（12 轮 × 96 token） | 无目标（首次标定） | **37.70 tok/s**（口径：首轮开始 → 末轮结束，已扣掉人的思考时间，但**含 12 次 prefill 气泡**，故不与上表直接可比） |
| **N=4 长稳定** | 6 轮 × 30 分钟聚合 | 30 分钟不退化 | ✅ 41.26 → 41.62 tok/s（§9.3） |

> **口径提醒**：上表"同 workload 口径"= 每会话 4 轮 × 64 token，**墙钟里含 4 次 prefill 气泡**，
> 所以 N=1 只有 10.73 而不是 12.38。绝对数字与"纯 decode × N"不可直接比；**跨 N 的比值
> （3.33×）才是并发增益的读数**。目标值 20/30/40 是按"纯 decode 理论值 ×0.8"定的，
> 与实测口径不同——**按同口径，N=2/3/4 分别差 1%、5%、11%**，处于测量口径差异范围内。

### 9.3 稳定性

#### 30 分钟压测：**已做，通过** ✅（2026-09-14）

方法：`--sessions 4 --rounds 8 -n 256 --ignore-eos` 连跑 6 轮，覆盖 30 分钟；
每轮 dump token 并采样 RSS/温度。**全部 rc=0。**

| 轮次 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| 聚合吞吐 tok/s | 41.26 | 41.41 | 41.39 | 41.42 | 41.50 | 41.62 |
| 峰值 RSS（MB） | 465 | 662 | 469 | **1181** | 462 | 463 |

- **无衰减趋势**（吞吐反而微升 41.26 → 41.62），SoC 温度 44.4 → 49.9 °C，**未降频**。
- **最强的一条证据是首末轮逐字节一致**：第 1 轮与第 6 轮的 4 路 token dump 完全相同
  （每路各 2056 token），6 轮之间 18 对跨会话组合也全部一致。跑 30 分钟之后数值行为没有漂移。
- 崩溃/堆破坏模式（`Segmentation`/`Aborted`/`double free`/`corrupted`/`unaligned tcache`）**0 命中**；
  最差会话的纯 decode 10.69 → 10.78 tok/s，未劣化。

> ⚠️ **RSS 那一列必须诚实地看**：1181 MB 是一次**孤立尖峰**，且整列**非单调**、
> 末轮回落到 463 MB——**与泄漏的形状不符**。但采样是**每 5 秒一个单点**，
> 无法排除这是模型加载/teardown 的一次性峰值。所以结论只能说到
> **"没有与泄漏相符的证据"**，不能写成"无泄漏"。要定性需要更密的采样或分阶段（加载/推理/销毁）拆开量。

#### ASan：**已跑，四个阶段零报告；且阳性对照已证明这套 ASan 在这块板上真会报** ✅（2026-09-15）

`cmake -DENABLE_ASAN=ON`（`cpp/CMakeLists.txt:10` 已支持；注意它只追加到 `CMAKE_CXX_FLAGS_DEBUG`，
必须配 `-d Debug` 才生效）。ASan 能查的是 `unaligned tcache chunk` 这类堆损坏（R4），
**和 TSan 查的数据竞争是两件事**——TSan 跑过不等于 ASan 的需求消失。

**先说这个工具在这里的能见度**（与 TSan 那条边界同理，不重复论证）：
ASan 用**进程级替换 malloc/free**（interposition）+ 编译期红区/影子内存检查。
**前者对所有代码生效**——包括闭源的 `librknn3_api_rkcp.so`：它越界写到红区、或 free 掉已释放的指针，
ASan 在分配器这一层拦得住；**后者只插桩我们自己的代码**——SDK 自己缓冲**内部**的一次普通越界
（没碰到红区）不会在访问那一刻被报。所以结论只能写成「ASan 没报」而不是「一定没有堆损坏」。

**运行时依赖的坑与 TSan 完全同形**：板卡自带 `libasan.so.8`（GCC 13），而交叉工具链是 GCC 11.4
→ 需要 `libasan.so.6`，SONAME 对不上。自带一份并让 `LD_LIBRARY_PATH` 把它排在 `./lib` **前面**。

> ⚠️ **构建系统的一个陷阱（踩到了）**：`build-linux.sh` 的 `INSTALL_DIR` 是
> `install/${TARGET_PLATFORM}/${TARGET_SDK}`，**不含 `BUILD_TYPE`**。所以一次 `-d Debug` 构建
> 会 `rm -rf` 掉同一路径下的 Release 安装目录——**ASan/TSan 构建会把已验收的 Release 二进制删掉**。
> 必须在构建前把 Release 的安装目录整体另存一份，跑完再拷回来。

| 阶段 | 形态 | 结果 |
|---|---|---|
| AS-0 | 指纹与运行时确认 | 二进制 `80344d4bdc81d8b7a6aa94a78f62eeaa`；`libasan.so.6` `1dc1577ea7882fbd5239039432b34c1e` |
| AS-1 | 单会话黄金等价（`--interactive`，与 P1 黄金同路径） | ✅ **逐字节 IDENTICAL，1539 token** |
| AS-2 | 并发 + 强制清 KV（**与 R11 的 T2 完全同 workload**） | ✅ rc=0；**恰好 4 次清 KV @ `context 3987/4096`**；四会话 prefill 504 / decode 4800，全部 `ok`，纯 decode 9.60~9.66 tok/s |
| AS-3 | 交互式多会话 soak（`--interactive`，48 行输入） | ✅ 52 块 = 48 回合 + 4 次清 KV 提示；**无穿插**（52 = 52）；四路各 4812 token dump |
| AS-4 | 泄漏检查（单独 `detect_leaks=1`，`--sessions 2 --rounds 2 -n 32`） | ✅ LSan **一行未打印**（无泄漏） |
| AS-5 | 汇总 | **`asan_report.*` 文件数 = 0** |

**AS-1 顺带是个 UB 探针**：ASan/Debug 改了优化级别与内存布局，token 仍与黄金逐字节一致，
说明这条路径里没有依赖未定义行为的成分。

**交叉验证（本次最硬的一条）**：AS-2 的四路 dump（各 4812 token）与 **R11 的 T2 post dump
（由 Release 二进制产出）逐字节一致**（4/4）。即 **ASan/Debug 构建没有改变任何数值行为**——
所以 ASan 的"零报告"不能被"构建不同跑了另一条路"解释掉。

##### 阳性对照：证明"零报告"是有意义的零，而不是工具没生效

一整轮 ASan 下来一个报告都没出，这个事实有两种解释：① 真的没有堆错误；② **工具压根没生效**
（插桩被优化掉、SONAME 没接上、影子内存没映射……）。只跑被测程序**区分不了这两种**。
这与 TSan 那轮是同一条纪律（先证明"故意制造的竞争能报出来"，才轮得到"我们代码干净"）。

`rt_work/asan_probe.cc` 在**这块板卡上、用随二进制下发的那份 `libasan.so.6`** 实测：

| 对照 | 机制 | 结果 |
|---|---|---|
| 堆越界写（`malloc(16)` 写偏移 24） | 编译期红区/影子内存检查 | ✅ 报出 heap-buffer-overflow |
| use-after-free | 分配器层拦截 | ✅ 报出 heap-use-after-free |
| 故意泄漏 4096 字节 | LSan | ✅ 报出（证 AS-4 的 `detect_leaks=1` 在工作） |
| 栈越界写 | 栈红区 | ❌ 未报 —— **我的构造有问题**（数组是死代码，GCC 把 store 删了），**不作为证据** |

> **第一版对照失败留档**：第一版用 `p[16] = 'X'` 这种**常量下标**，`-O1` 下 GCC 把访存整个优化掉了
> （`nm` 里只有 1 个 `__asan_report_*` 引用，而 demo 二进制有 10 个）→ 前两个对照一个都没报。
> **这是对照写坏了，不是 ASan 坏了**，不能据此说"工具的地址检查无效"。改法：下标走 `volatile`，
> 编译器无法折叠成常量；并在 `-O0` 下编译（与 demo 的 Debug 构建对齐）。
> 这条同时也说明：**`-O1` + ASan 下，编译期能算出来的越界会被优化掉**，探针必须防这个。

**R4 的判定**：ASan 已跑、阳性对照成立、最重的那条 workload（4 会话 × 4800 decode + 4 次清 KV）
零报告，因此 **R4 的概率从"中"下调到"低"**（影响等级不变）。但**不能就此关闭**：
① ASan 会改变堆布局与时序，这类"竞争诱发型"堆损坏可能因此**不出现**；
② ASan 下累计只跑了约 10 分钟（AS-2 约 8.5 分钟 + AS-3），**没有**做满 30 分钟 soak；
③ 边界同 §9.5——SDK 自己缓冲内部的越界看不见。要关闭需要在 ASan 下跑满长时 soak。

### 9.4 调试辅助

| 开关 | 状态 | 作用 |
|---|---|---|
| `--dump-tokens <file>` | ✅ 已有 | 每步输出 token ID；并发模式下自动变成 `<file>.s<i>`（每会话一个） |
| 每会话 perf 表 | ✅ 已有 | prefill/decode token 数、纯 decode tok/s、墙钟，区分"哪个会话慢" |
| `--conv-trace` | ❌ 未加 | 打印 `[conv A][stage2] enter/exit card_lock`，排查死锁与相位 |
| TSan 构建 | ✅ 已加 | `-fsanitize=thread` 独立构建（板上 `rknn_multicard_demo.tsanr11`，**含 R11 锁**），见 §9.5 |
| ASan 构建 | ✅ 已加 | `cmake -DENABLE_ASAN=ON` + `-d Debug`；注意 `INSTALL_DIR` 不含 `BUILD_TYPE`，**Debug 构建会删掉 Release 安装目录，先另存**，见 §9.3 |

### 9.5 TSan 两轮：**多会话路径（含 R11 清 KV）没有"两侧都在我方代码内"的竞争；SDK 内部报告未定性**（2026-09-14 首轮 / 2026-09-15 补轮）

**先说这个工具能看到什么**：TSan 靠**编译器插桩** + 运行时影子内存算 happens-before。
它**只能插桩我们自己编的代码**——`librknn3_api_rkcp.so` 是闭源预编译库、**没有被插桩**，
它内部的访存 TSan 既看不见、也不会成为别人 happens-before 链上的边。
**所以"TSan 干净"只证明我们的代码干净，不证明 SDK 干净**（反之，它报的 SDK 内部"竞争"
也可能是"看不见原子操作"导致的假阳性）。

> ⚠️ **首轮对速度的估计过于乐观**：首轮记的是"慢约 2.5×（90s → 233s）"，那是**短会话**的数字。
> 补轮实测：长上下文下每会话纯 decode 只有 **0.10 tok/s**（3754 prefill + 96 decode 用掉
> 单会话 1252 s 墙钟），比 Release 的 12.38 tok/s 慢**两个数量级**；另有**每进程约 230~240 s
> 的固定成本**（模型加载）。所以 TSan 下的 tok/s 无参考价值，本文不引用；跑时长必须按
> "decode 产出"倒推，不能照 Release 估。

**为什么要补第二轮**：首轮（2026-09-14）的三个阶段**一次都没走到 `clear_kvcache`**——
负载太短够不到阈值。而 **R11 恰恰是在 `clear_kvcache` 上新加了两把卡锁**
（`run_conversation_worker` / `run_interactive_session_worker`），板上那份首轮二进制
（`acb78c12…`）里**根本没有那把锁**。也就是说：**R11 那段新代码从未在 TSan 下执行过。**
补轮换用含锁的重编版本（`7f19e6d5…`），把清 KV 路径真正压进去，并补一个控制组。

| 轮次 | 阶段 | 形态 | 报告数 | 竞争访存在**我们代码**里 | 清 KV |
|---|---|---|---|---|---|
| 首轮 | TS-A | `--sessions 2 --rounds 1`（并发路径） | 9 | 0 | ✗ 未到达 |
| 首轮 | TS-B | `--sessions 2 --interactive`（**P3 新代码**） | 11 | 0 | ✗ 未到达 |
| 首轮 | TS-C | **不指定 `--sessions`**（单会话老路径，对照） | 4 | 0 | ✗ 未到达 |
| 补轮 | TS-A | `--sessions 2 --rounds 1 -n 8`（复刻首轮 TS-A） | 8 | 0 | ✗ |
| 补轮 | **TS-B** | **同 TS-A + `ignore_noninstrumented_modules=1`（控制组）** | **0**（rc=0） | — | ✗ |
| 补轮 | TS-C（**中止**） | `--sessions 4 --rounds 12 -n 400`（靠 decode 累积上下文） | 11 | 1 | ✗ 未到达（跑 36 min 仅约 1/5，已中止，报告单独留档） |
| 补轮 | TS-C2 | `--sessions 4 --rounds 10 -n 8`（**轮数不够，没撞到阈值**） | 18 | 1 | ✗ **0 次**（见下） |
| 补轮 | TS-C3 | `--sessions 4 --rounds 12 -n 8`，中等长度 prompt | 14 | 1 | ✅ **恰好 4 次 @ `context 3765/4096`** |
| 补轮 | TS-D3 | `--interactive` 48 行长输入 / 4 会话 | 16 | 1 | ✅ **每会话 13 块 = 12 回合 + 1 次清 KV，各清 1 次、52 块无穿插** |
| 补轮 | TS-E | `--sessions 1 -n 8`（**单会话对照，复刻首轮 TS-C**） | 4 | 0 | ✗ |

（首轮条数以 TSan 页脚 `reported N warnings` 为准；补轮条数为分类脚本逐条解析调用栈所得，
两者口径一致——首轮 TS-A/TS-B/TS-C 的 9/11/4 与脚本解析的 5+4 / 7+4 / 0+4 完全吻合。）

**为什么要用中等长度 prompt 去顶清 KV 阈值**（这是补轮的关键设计）：清 KV 的守卫是
`context_tokens + prefill_reserve(512) >= 4096`，即 `context_tokens >= 3584`。
补轮第一版 TS-C 那种"靠 decode 累积"的路子，在 TSan 的 0.10 tok/s 下要跑 **100 分钟以上**（实测
36 分钟才到约 1/5，已主动中止）。但 **prefill 每个 token 比 decode 快一个数量级**，而
`--rounds` 每轮都会把 `--prompt` **重新追加一遍**（不是只发一次）——所以用一个 368 token
的 prompt，每轮上下文增 376，第 11 轮前撞到 3584 清一次，decode 只需 96 token/会话。
全程最高上下文 `10 × 376 = 3760 ≤ 4096`，**不会顶穿**（prompt 一旦超过 512 就可能顶穿，
落在守卫包线之外，所以不能更长）。

> 补轮第一次（TS-C2）**清了 0 次**：我按 4 字符/token 估 prompt 是 443 token，实测是 **368**
> （1776 字符 ≈ 368 token，约 4.8 字符/token），10 轮只到 3384 < 3584——**差一轮**。
> 改成 `--rounds 12` 后正好清 4 次。这条数字是硬的（`prefill 3754 / decode 96` 反推出来的），
> 不依赖对 tokenizer 的估计精度。

**结论一：控制组把"我们代码干净"从一个说法变成了一个有边界的结论。**
补轮 TS-B 是**同负载 + `ignore_noninstrumented_modules=1`**：该选项把"两侧访存都在未插桩模块内"
的报告**整体抑制**。结果是 **rc=0、0 条**——TS-A 那 8 条**全被压掉**。这精确说明：
**那些报告没有一条是"两侧都在我方插桩代码内"的竞争**。卡级锁、tokenizer 互斥、
每会话队列与 `block_out` 都干净。

**结论二（**对首轮结论的收窄，务必看这条**）：首轮"我方代码零竞争"的说法在补轮里被推翻了一部分。**
首轮报告里确实没有任何一条竞争的访存第一处本地帧落在我方代码；但补轮在**4 会话并发**盘里
出现了 **4 条**这样的竞争（3 条在完成轮 + 1 条在中止轮），形态完全一致：

```
Read of size 8  by T1 (mutexes: write M301):           ← 读侧
  #2 rknn3_session_output_callback(Context*, RK::MsgHeader*)   [SDK+0xd9be4]
Previous write of size 1 by T2 (mutexes: write M305):  ← 写侧
  #2 memcpy (string_fortified.h:29)
  #3 input_callback  main.cc:1545            ← **我方代码在写**
  #4 rknn3_session_input_callback(Context*, RK::MsgHeader*)    [SDK+0xdd428]
Location is heap block of size 16777216 ... allocated by FreeListAllocator::Init()
```

`main.cc:1545` 就是 `input_callback` 里把 rope cache 拷进 SDK 给的输入 tensor 的那句
`memcpy(dst, src, copy_stride)`，`dst = input_tensors[i].mem->virt_addr`
（**地址是 SDK 给的、缓冲是 SDK 的**）。所以准确的写法是：

- **我方代码确实执行了这 4 次竞争写**——但写的是 **SDK 交给我们的那块的缓冲**，
  是在 SDK 自己调进来的回调里、持着 SDK 自己的锁做的，**正是回调契约要求我们做的事**；
- **两侧持的是两把不同的 SDK `recursive_mutex`**（实测组合：M301/M305、M301/M307、
  M301/M304、M307/M311，创建点都是同一个 SDK 位置）→ **那两把锁没有在互斥这段内存**；
- 读写地址相隔 **7 字节**（`0x…4310` 读 8 字节 / `0x…4317` 写 1 字节），落在
  同一个 16 MB arena 里。**这到底是"同一对象"还是"相邻对象的假共享"，只有 SDK 源码能定。**

**结论三：聚合统计（去重后）。** 首轮 3 份 + 补轮 7 份（剔除中止轮的重复副本），
**完成轮共 52 条数据竞争 + 40 条 `unlock of an unlocked mutex`**；中止轮另留档 11 条
（其中 1 条为我方访存侧）→ 合计 63 + 40。地址形态高度一致：

| 统计项 | 结果 |
|---|---|
| 两处访存区间**确实重叠** | **52/52**（中止轮 11/11） |
| 落在 SDK 的 **16 MB `FreeListAllocator` arena** | **52/52**（中止轮 11/11） |
| 访存宽度 | 只有 size=1 与 size=8，没有其他宽度 |
| 一侧是 `rknn3_session_output_callback` | **52/52**（每条都有一侧是它） |
| **一侧是我方 `input_callback`（main.cc:1545）** | **3/52**（中止轮 1/11） |
| 另一侧的其他本地帧 | `rknn3_session_input_callback`、`rknn3_mem_sync_range`、`utils_read`、`rknn3_session_get_next_input_embed_callback`，均在 SDK 内 |

> **重叠率 100% 不是发现，是预期**：TSan 只在两处访存落到同一地址时才报竞争，
> 所以"区间重叠"必然 100%。**真正有信息量的是"全部落在那一个 16 MB arena 里"
> 和"一侧永远是 output_callback"这两件。** 这条写在这里是为了防止后人把 100% 当成异常。

**结论四：条数不是一个稳定指标，别拿它做判据。** 同一形态的负载，实测条数为
**首轮 TS-A 5 / 补轮 TS-A 4**、4 会话并发 **14 / 10 / 12**（TS-C2/C3/D3，且 TS-C2 与 TS-C3
是同一 workload 只差轮数）。**稳定的是"位置与形态"，不是条数**——所以本文的结论一律
按"落在哪里/两侧是谁"来写，不写"共 N 条"作为论据。

**结论五：有一组报告与本次改造无关。** 单会话对照在**两轮里都稳定复现**：
**4 条 `rknn3_destroy+0xfb068` 的 "unlock of an unlocked mutex"**，每卡 1 条，
全部在**主线程单线程收尾**时触发（栈：`rknn3_destroy` ← `destroy_contexts` main.cc:528 ←
`main`），**数据竞争 0 条**。首轮 TS-C、补轮 TS-E 形状完全一致 → **是 SDK 收尾路径的固有
现象，不是 P2/P3/R11 引入的**。（`pthread_mutex_unlock` 是被 TSan 拦截的，"unlock of an
unlocked mutex"不能简单当作未插桩导致的假阳性；但它**在零并发的对照里也稳定复现**，
所以"与改造无关"这一点是确定的。）

**结论六：数据竞争只在 ≥2 会话并发时出现。** 单会话（首轮 TS-C、补轮 TS-E）**0 条数据竞争**，
4 条 unlock 是收尾路径的固有现象；两轮 TS-A（2 会话）都是 4~5 条；4 会话并发 10~14 条。

> ⚠️ **为什么这些不能简单判定为"SDK 的 bug"**：我们的卡级锁只能串行化**调用方**，
> 而报告里至少有一侧是 SDK **自己创建的**回调/传输线程——**它不在我们的锁内，
> 任何应用层的锁都管不到它**。这些位置要么是 SDK 内部有我们看不见的同步协议
> （lock-free 环 + 原子操作，TSan 盲区），要么是真竞争。**TSan 到此为止，
> 定性需要 SDK 源码或原厂答复。** 因此 R6 **不视为已关闭**。

**这一轮真正拿到的正面结论是**：**R11 新加锁的那段代码（两处 `clear_kvcache`）在 TSan 下
真的执行到了**（TS-C3 清 4 次、TS-D3 四路各清 1 次），**而它没有引入任何新的竞争形态**——
清 KV 路径的出现没有让报告的形状发生变化（仍全在 SDK arena 内、一侧仍是 `output_callback`）。
首轮那个"R11 代码从未在 TSan 下跑过"的盲区，**已关闭**。

**复现**（红线内的中间产物，**不进 git**）：脚本 `rt_work/run_tsan.sh`（首轮）、
`rt_work/run_tsan2.sh`/`run_tsan3.sh`/`run_tsan4.sh`（补轮，板上 `<板卡临时目录>`），
分类脚本 `rt_work/tsan/classify.py` + 原始报告 `rt_work/tsan/report_ts*.txt`、
`rt_work/tsan_all/tsan{2,3,4}/report.*`。
板上另需一份 GCC 11 的 `libtsan.so.0`（板卡自带的是 `libtsan.so.2`，SONAME 对不上，
必须随二进制一起下发并让 `LD_LIBRARY_PATH` 把它排在 `./lib` **前面**）。
**补轮二进制 `7f19e6d505ac443829604765bcb2b142`（含 R11 锁）**，首轮旧件
`acb78c120353c73f74725ed1c7a30ed2`（不含 R11，仅留档）。

### 9.6 R11：回合之间清 KV 要不要卡锁 —— 验证与补锁（2026-09-14）

**问题**：两个 worker 里的 `rknn3_session_clear_kvcache` 没有持卡级锁——从已验收的
`--rounds` 路径**原样继承**、此前没有独立验证过。P3 当时刻意没改，怕**悄悄变更已验收路径的行为**，
所以把它列为待确认项（R11）留到后面单独验。

**结论：需要锁。**

依据**不是**"测出了竞争"（下面会说明为什么测不出来），而是 **SDK 自己的契约**：

1. N 个会话的 session 共用一张卡的**同一个 `rknn3_context`**（P1 的分层就是这么落的）。
2. SDK 外部文档 §3.5：「同一个 `rknn3_context` 或 `rknn3_session` 的并发使用需调用方自行保证线程安全」。
   注意**头文件里没有这句话**，只在文档里，所以这条依据的来源要写清楚。
3. 于是「A 会话清 KV」与「B 会话在同一张卡上跑 `rknn3_session_run`」就是对该 context 的并发使用。
4. **关键点**：清 KV 只动本会话自己的 KV 缓冲（P0 已证 KV 隔离）——但**隔离的是数据，不是调用**。
   它是否并发安全，由 SDK 的契约决定，**不能由"两边数据不重叠"反推**。

**全部 6 处 `clear_kvcache` 的处置**（逐一排查，不是抽样）：

| # | 位置 | 是否并发场景 | 处置 |
|---|---|---|---|
| 1 | `main.cc` `run_conversation_worker`（`--rounds`，`main.cc:2347`） | **是**——N 个 worker 同时跑，别的会话可能正持着同一张卡 | **✅ 补锁**（本次） |
| 2 | `main.cc` `run_interactive_session_worker`（`main.cc:2489`） | **是**——同上，R11 的现场 | **✅ 补锁**（本次） |
| 3 | `main.cc` `probe_clear_group`（P0 探针，`main.cc:2931`） | 否——单线程顺序执行 | 不动 |
| 4 | `main.cc:4090` 并发模式收尾 | 否——在所有 `driver.join()` **之后** | 不动 |
| 5 | `main.cc:4159` 单会话交互路径 | 否——主线程一对一 | 不动 |
| 6 | `main.cc:4276` 进程收尾 | 否——主线程单线程 | 不动 |

处置就是给 #1/#2 各加一圈**逐卡** `std::lock_guard`（取一张放一张），
粒度仍是「同时只持一张卡的锁」——§3.4 的锁序不变量**没有被破坏**。

**另外补了一行 `VLOG`**（在 `--rounds` 的清 KV 分支上）：那条分支**原先没有任何日志**，
而 `--rounds` 是已验收路径、默认输出必须逐字节不变，所以只能用 `VLOG`（`--verbose` 才打）。
补它的直接动因见下面那条"测试盲区"。

**验收（板卡实测）**

| 测试 | 构造 | 通过标准 | 实测 |
|---|---|---|---|
| T1-A 单会话黄金 | `--sessions 1 --interactive -n 512 < 3 行` | 与 P1 交互黄金逐 token 一致 | ✅ **IDENTICAL** `t1a.tok == it_p1.tok`（1539 token） |
| T1-B 单会话黄金（**跨清 KV**） | 同上，6 行输入 | 同上，且清 KV 行为不变 | ✅ **IDENTICAL** `t1b.tok == kv_p1.tok`（3078 token）；清 KV **恰好 1 次** |
| T2 补锁前后等价性 | 同一 workload（`--sessions 4 --rounds 12 -n 400 --ignore-eos`，每会话 4800 decode，**跨第 10 轮必清一次 KV**）分别喂修复前 `p3` 与修复后 `p3lock` | 每会话 token dump 逐字节一致，**且清 KV 确实发生** | ✅ **四路全部 IDENTICAL（各 4812 行）**；`pre` rc=0/582s、`post` rc=0/578s；**post 侧清 KV 恰好 4 次、全部 `context 3987/4096`**（每会话 1 次）。每会话纯 decode：pre 10.79~10.86 → post 10.73~10.78 tok/s（差 ~0.5%，在噪声内） |
| T3 交互式多会话 soak（**R11 现场**） | `--sessions 4 --interactive -n 400 --ignore-eos`，96 行输入 | 无穿插、无崩溃、**清 KV 次数 > 0** | ✅ 见下 |

**T2 怎么读（几点必须一起说，否则会高估这条证据）**

- **`pre` 那一份是复用的，不是 2026-09-15 重跑的**：它在 2026-09-14 就已经 rc=0 跑完、
  四路各 4812 行完整。本次只重跑了 `post`。**可以复用**的理由是 workload 确定
  （同 prompt、同 `-n`、同 `--ignore-eos`），token 输出不依赖板卡温度或"第几次跑"。
  想推翻这个前提就删掉 `eq_pre.tok.*` 重跑一次 pre，代价约 10 分钟。
- **`pre` 的清 KV 无法直接计数**（那个分支当时没有日志），它的清 KV 是**由三条旁证合起来的**：
  ① guard 条件两个二进制完全相同；② `post` 在同一 workload 上确实报了 4 次、且位置算出得
  `3987/4096`；③ 四路 token 逐字节一致。**不要写成"pre 也实测清过 4 次"**。
- **T2 证明的是"补锁是行为中性的"，不是"不加锁就会坏"。** 后一句这类窗口靠几次压测
  本来撞不到，本文不声称。
- 顺带对了一下**清 KV 的阈值推算**：按 perf 表反推的每轮平均 `42(prefill) + 400(decode) = 442`，
  9 轮累计 3978 ≥ 3584 → 第 10 轮之前清一次。日志实测是 `3987/4096`，**比推算多 9 个 token**。
  量与位置都对得上（确实是在第 10 轮前、且只清一次），**这 9 个的来路没有细究**——
  各轮 prefill 并不等长（首轮直接用 `--prompt` 原文，后续轮才套 chat 模板），
  用平均值反推本来就会有零头。**所以这里只声称"位置一致"，不声称"逐 token 精确复现"。**

**T3 结果**（这一轮才是真正压在 R11 上的证据）：

- `rc=0`，96 行输入全部消化，**每会话 24 轮**、dump 各 9624 token。
- **104 个 `[s<i>]` 块 = 96 个真实回合 + 8 条清 KV 提示**；
  行首块数 104 = 含前缀行数 104 → **没有任何一次输出穿插**。
  （注：清 KV 提示本身是经 `print_session_block` 打印的，带 `[s<i>] ` 前缀，
  所以对块数时要把这 8 条算进去，否则会误以为多了 8 个回合。）
- **清 KV 共 8 次（每会话 2 次）**——**>0 是这条测试成立的前提**：如果一次都没清，
  这轮就根本没测到 R11 这条路径。次数与阈值算得也对得上：每轮 `上下文 += 42(prefill) + 400(decode)`，
  `3584 ÷ 442 ≈ 8.1` → 第 10 轮与第 19 轮之前各清一次，24 轮共 2 次。
- 峰值 RSS 1476 MB、峰值温度 38.8 °C；采样从首 13 MB（还没加载完）到中段 1466 MB、
  末段 1476 MB——**中段到末段只涨 0.7%，与"泄漏"不符**，但采样间隔 30 s，
  仍不足以排除一次性峰值，**结论只写到"没有与泄漏相符的证据"**。
- 崩溃检查全 0：Segmentation / Aborted / double free / corrupted / unaligned tcache / failed
  六项均 0 命中，`session_init` 失败 0 次。

> ⚠️ **测试盲区（必须记下来，否则下次还会踩）**：**上一轮 30 分钟稳定性测试根本没走到清 KV**——
> 那轮 workload 是 `--rounds 8 -n 256`，每会话 2048 token + 每轮 42 prefill ≈ 2384，
> **够不到阈值 3584**。跑了 30 分钟、看起来"通过"，但清 KV 这条分支一次都没被执行。
> 而且当时那条分支**连日志都没有**，所以"没报错"完全不能作为"测过"的证据。
> 本次补 `VLOG` 就是为了让这种"没测到"以后能被一眼看出来。

> ⚠️ **另一条测试陷阱**：修复前的二进制在那个分支上**没有日志**，
> 所以拿它跑"清 KV 次数"**永远是 0**，**不能理解成"没清"**。
> T2 里 `pre` 的计数必须靠修复后二进制在同一确定 workload 上的日志来反推。

> **这条结论的边界（别过度引用）**：锁是**依据 SDK 契约**加的，**不是**因为观测到了竞争。
> 也**无法**用实验反证"不加锁就会出错"——这类窗口靠跑几次压测本来就撞不到。
> 反过来说，T1-B 与 T2 的价值在于证明**补锁是行为中性的**（逐字节一致），
> 而不是证明"不加锁会坏"。

### 9.7 M5：板端 OpenAI 兼容服务化 —— 自研门面 + 帧协议（2026-09-15）

**为什么不用官方 `rkllm3-server`。** 它的 OpenAI 接口是现成的，少写一堆代码；但文档 §4.5.2 写明
「推理执行排队串行」，实测 `--n-session 1/2/4` 聚合都是 **10.63 / 10.98 / 10.98 tok/s（1.03×）**——
**slot 提供的是隔离，不是并行**。同一台板、同一个执行器（P2 起）是 3.33×，所以收益只能在
"容器"这一层自己写：**HTTP 门面自研，推理仍走已验证的多会话执行器。**

**分层**

| 层 | 产物 | 职责 |
|---|---|---|
| 执行器 | `cpp/main.cc --serve`（C++，入库） | 帧协议收发、会话钉选、KV 复用、token 增量上报 |
| 门面 | `rkllm_gateway.py`（python3 标准库，**无第三方依赖**，板上有 python3 即可） | OpenAI 兼容 HTTP/SSE、**chat 模板渲染**、会话粘性账本、think 摘除、请求排队 |

网关是**父进程**，自己拉起 `rknn_multicard_demo --serve` 子进程，所以只有一个进程要管。

**帧协议为什么走独立 fd（`--serve-fd`，默认 3）**

`main.cc` 与 SDK 会往 **stdout** 上打日志（`print`/`printf` 散落各处），把协议放 stdout
**一定会被污染**。所以帧走单独一条 fd，网关用 `subprocess` 的 `pass_fds` 把它交给子进程。
`--serve-fd`（默认 3，取值 1~1023）**做成参数而不是写死**：`pass_fds` 给子进程的是**父进程里的
fd 号**，写死 3 就得在多线程进程里做 `dup2` + `preexec_fn`，那是不安全的做法。
该 fd 打不开时后端**退回 stdout 并告警**——能跑，但不再保证帧流不被日志插花。
> 这不是假想：实测网关这条管道**落在 fd 5**（`--serve-fd 5`），不是 3——写死 3 当场就废。

| 方向 | 帧 | 语义 |
|---|---|---|
| 网关→后端 | `REQ <rid> <session\|-1> <max_new_tokens> <reset> <prompt_len>` + `prompt_len` 字节原始 payload | 一次请求（`session=-1` = 由后端挑空闲会话） |
| | `QUIT` | 收工（不必等 stdin EOF，便于优雅关闭） |
| 后端→网关 | `READY <nsessions> <default_max_new_tokens>` | 后端就绪。**必须先于任何 `DELTA` 发出**，否则网关启动期就会把帧读串 |
| | `DELTA <rid> <n>` + `n` 字节 | 增量文本（逐段，长度前缀、不转义） |
| | `DONE <rid> <session> <finish_reason> <prefill_tok> <decode_tok> <prefill_ms> <decode_ms> <ctx_tok>` | 回合结束 |
| | `CLEAR <rid> <session> <ctx_tok> <ctx_limit>` | **本回合开头**清了该会话的 KV |
| | `ERR <rid> <session> <n>` + `n` 字节 | 错误（含本回合已产出的部分文本；`rid=0` 表示"头行本身都没解析出来"）。**语义 = 这个会话死了** |
| | `REJECT <rid> <session> <n>` + `n` 字节 | 本回合**在 prefill 之前被拒**（上下文装不下）。**语义 = 会话还活着**，网关只失败这一条请求 |

协议上的几个刻意取舍：

- **载荷不定长、不做转义**：prompt 里可以有换行、制表符、任意字节，所以不能用 `getline` 读完整条
  请求——头行定长可解析，载荷按头行声明的长度**原样**读。网关能用中文/多行 prompt 就是靠这个。
- **头行解析失败必须回 `ERR`，不能静默丢弃**：丢一条会让网关永远等那个 `rid`，现场表现为
  "服务卡住"，实际是网关自己发错了。同理，`prompt_len` 超过 1 MiB 上限、或载荷短读，
  都回 `ERR` 后**终止输入线程**——流已经错位，再读只会把后面的头行当载荷。
- **长度上限 `kServeMaxPromptBytes = 1 MiB`**：没有上限的话，头行里一个手滑的数字（比如 2⁶⁰）
  就能让本进程按那个数字去 reserve，直接把服务打死。
- **`ERR` 与 `REJECT` 不能混用，区别只有一条：会话死没死。** 网关收到 `ERR` 会把该会话
  `mark_dead`（它确实不该再用）；收到 `REJECT` 只把这一条请求判失败（HTTP 400 + `pool.discard`），
  会话原地不动。用错方向的代价是不对称的：该 `REJECT` 而发了 `ERR`，**每撞一次就永久少一个会话**；
  该 `ERR` 而发了 `REJECT`，网关会把请求转给一个已经坏掉的会话。越界校验用 `ERR`（防线被踩 =
  请求方的问题），上下文装不下用 `REJECT`（不是会话的错）。
- **`finish_reason` 是反推的，不是执行器给的**：`run_chat_turn` 在"步数用尽"和"采样到结束符"
  两种情况下是同一个 `break`、没把原因带出来，改签名又会动到被逐字节验收过的那条路径。
  所以按 `decode_tok >= max_new_tokens` 判 `length`、否则 `stop`。**对 OpenAI 客户端够用，
  但它确实是推断值**（§9.1 的黄金 token 对照保护的是执行器，不是这个字段）。

C++ 侧把原先直接往 `block_out` 字符串里拼的输出抽成 `TokenSink`：交互路径用 `StringTokenSink`
（**行为逐字节不变**，P1/P3 的黄金 token 对照因此仍然有效），serve 路径用 `ServeTokenSink` 发 `DELTA`。
`CLEAR` 的时序在 `main.cc` 里是固定先清后 prefill（清 KV 分支在 prefill 之前），所以网关收到
`CLEAR` 时，该回合的 prefill 尚未发生——这正是下面那条记账规则的依据。

**`--serve` 复用的是 P3 那套驱动，不是另写一套。** 一条输入线程 + N 个会话线程抢同一条待办队列，
与 `--interactive` 是**同一个** `InteractiveDispatcher`、同一个 `run_interactive_session_worker`；
唯一的语义差别是多了一个 `PendingInput::target_session`（交互模式一律 `-1` = 谁空闲给谁，
服务模式钉住指定会话，因为**只有钉住会话才能在同一段对话上复用 KV**）。这个设计带来两个好处：
M4 已在板上验收过的调度/成块/清 KV 逻辑被完全继承（不是"重写一遍再验一遍"），
而且那个改动很小的差异点让回归范围缩小到"钉住 + 帧编解码"两件事。
一个必须配套的细节：因为待办可能钉给某个特定会话，**通知要 `notify_all` 而不是 `notify_one`**，
否则被唤醒的会话若接不住队首，这条活就没人取了（会一直躺在队列里）。

**服务模式下 prompt 由网关渲染、后端逐字节透传。** 原因：`main.cc` 自己的 chat 模板只有
"首轮 / 后续轮"两态，表达不了 OpenAI `messages` 里的任意角色历史（多轮 assistant、以及将来的
tool）。所以网关按模板把 `messages` 拼成一段 prompt 交给后端。**代价是多了一条必须守住的不变量**：
网关渲染的字节流必须与 `main.cc` 模板**逐字节一致**，否则同一段对话在交互前端和服务前端的
行为会悄悄分叉——`check_template.py` 就是守这条线的（首轮 / 多轮两组逐字节比对）。
另注意 `--serve` 与 `--interactive` **互斥**（同一套驱动的两个前端）、`--serve` 必须配 `--sessions N`、
`--rounds` 对 `--serve` 不适用；服务模式**保持静音**（`g_suppress_generation_output` 不受影响），
因为 `result_callback` 里的 sink 分支排在静音判断**之前**，帧通道与这个开关无关。

**会话粘性与 KV 复用**

网关为每个 session 维护三态账本 `known[s]`：**字符串** = 已知的 KV 尾部文本、**`""`** = KV 为空、
**`None`** = 刚被清过、内容未知。只有 `prompt.startswith(base)` 才复用（发差异后缀），
否则带 `RESET` 重发全文。三态里 `None` 与 `""` **必须分开**：`reset` 会触发后端清 KV，
但**清 KV 只是这个回合历史的起点**，不是"历史为空"——若把 `None` 当 `""` 处理，
下一轮就会把整段 prompt 追加到一段脏 KV 后面（见下表坑 1）。

**实测（板卡，2026-09-15；修掉坑 4 后**重下发、重启、重跑全量套件**，`FAIL 行数：0`，5 组测试 rc 全 0）**

| 测试 | 通过标准 | 实测 |
|---|---|---|
| `check_template.py` 模板逐字节 | 与 `main.cc` 的 chat 模板一致 | ✅ 首轮（system+user）/ 多轮（user/assistant/user）全中；`/no_think` 只贴 user |
| `serve_http_test.py` 接口面 | health / models / 非流式 / SSE 全通 | ✅ `finish=stop`；SSE 13 个 delta、首块带 role、有 `finish_reason` |
| KV 复用（HTTP 级证据） | 续聊 prefill **远小于**全量 | ✅ 开思考 **12 vs 154 tok（0.08）**；关思考 **16 vs 158 tok（0.10）** |
| `sticky_check.py` 语义正确性 | 暗号在"只走 KV"的路径下存活 | ✅ 粘性轮 prefill **23 vs 75 tok（0.31）**、`sent=118 < full=366`、`base=248`；两条路径都答出暗号 |
| `nothink_check.py` 关思考粘性 | `reset=0` 且 `base>0` | ✅ `sent=113 / full=352`、`base=239`；暗号还在 |
| `http_scaling.py` HTTP 层伸缩 | 聚合吞吐随 N 增长 | ✅ 见下表（另连量 3 轮确认不漂） |
| 网关 `--selftest` 会话池回归 | 租约/复用/清除的语义 | ✅ 22 PASS / 0 FAIL；其中坑 4 的 2 条用例在**修复前代码上复现失败** |

> **口径备注（2026-09-16）**：上表里的"续聊 prefill 12 vs 全量 154"这个 **12** = 本轮**真正
> 重算**的 token 数，不是 `usage.prompt_tokens`。同一天起 `usage.prompt_tokens` 改成了
> **OpenAI 口径 = 整段 prompt 的长度**（复用部分另记在 `prompt_tokens_details.cached_tokens`），
> 所以现在要把"重算量"算出来得自己减：`prompt_tokens - prompt_tokens_details.cached_tokens`。
> 在此之前网关报的是增量，脚本里直接拿 `prompt_tokens` 当重算量用——那些判据已同步改口径
> （`sticky_check.py` / `serve_http_test.py` 里都留了注释）。**结论不变，读数方式变了。**

**HTTP 层伸缩（同一份脚本、同一台板、同样的 N；这是自研门面存在的唯一理由）**

下面取 4 轮里的代表轮，括号内是 4 轮的实测范围（含套件自带的那次）：

| N | 墙钟(s) | 合计 tok | 聚合 tok/s | 伸缩比 | 单请求延迟(s) |
|---|---|---|---|---|---|
| 1 | 8.8 | 96 | **10.96**（10.96~10.97） | 1.00× | 8.8 |
| 2 | 9.5 | 192 | **20.21**（20.20~20.26） | 1.85× | 9.5 |
| 4 | 10.5 | 384 | **36.44**（36.44~37.05） | **3.33×**（3.32~3.38） | 10.5 |

**对照（官方 `rkllm3-server`，同板同条件）：10.63 / 10.98 / 10.98 tok/s，伸缩 1.03×。**

- 为什么必须从 HTTP 这一层量：**网关/SSE/排队都有可能把并发吃掉**（比如把请求排成一队、
  或持着全局锁发帧）。后端日志只证明"执行器能重叠"，不证明"用户真的拿到并发"。
- **为什么要连量 4 轮而不是量一轮**：这个数曾经是**偶发**的——同一构建、同一脚本量到过
  `3.40×`（运气好）也量到过 `1.90×` / `1.00×`（运气不好，见坑 4）。**修掉坑 4 之后 4 轮落在
  3.32~3.38 且每轮 N 值相差 <0.6 tok/s**，取整口径与 P2 的 3.33× 一致——所以这条"稳定"是被
  反复量出来的，不是挑出来的。**在修掉坑 4 之前，任何单轮数字（包括那个更漂亮的 3.40×）都不该作为结论。**
- 代价与 P2 相同：N=4 时单请求延迟 8.8 → 10.5s（+19%）。

**四个坑（全部只影响速度、答案完全正确）**

这四个都**不会被正确性测试抓到**——输出是对的。坑 1~3 的表现是每轮退化成全量 prefill（慢 3~4 倍），
坑 4 的表现是**并发被悄悄串行化**（吞吐掉回 1×）。能抓到它们的是响应头
`X-KV-Reuse`（`session/reset/sent/base/full`）、`usage` 里的两个数、以及**网关日志里的租约落点**：

| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| 1 | 首轮落在**脏**会话上时，第二轮把整段 prompt 又拼到脏 KV 后面 | `RESET` 的 `CLEAR` 把 `known` 抹成"空"而非"未知" | 三态 `known` + `release()` 记账（见上） |
| 2 | `enable_thinking=false` 时粘性完全失效（`reset=1; base=0; sent=361=full`） | `/no_think` 只贴在**最后一条** user 上 → 同一段历史跨轮渲染**不一致**（第一轮的历史里没有、第二轮的历史里有）→ 前缀直接断 | 每条 user 都贴 `/no_think` |
| 3 | 同上（坑 2 修好了仍然 `base=0`） | 关思考时网关把 `<think>…</think>` 从正文里摘掉，但账本记的是**模型吐出的原始文本** → 客户端下一轮回显的是摘标签后的文本，前缀对不上 | `release()` 前先 `strip_think(...)` |
| 4 | **并发伸缩偶发塌陷**：同一构建、同一脚本量到 `3.40×` / `1.90×` / `1.00×`；日志里租约落点是 `0,0`（N=2）与 `0,1,2,0`（N=4） | `acquire()` 从空闲会话里按 `known` 挑，**完全不看 `bound`** → 一个"空闲但已有主人"的会话被派给了另一段对话，两段对话钉在同一会话上，后端只能**串行**跑它们，于是互相等 | 选会话时排除已被占用的；~~真的全占满要窃取时**解绑旧主人**~~（**v1.9 起这一支改为排队，见本节末尾的补注**）；给 `bound` 设上限（见下） |

每一条都在网关 `selftest()` 里留了回归用例，并**在修复前的代码上验证过它会失败**（不是"补了测试恰好是绿的"）。
坑 2、3 是**顺序耦合**的：不先修坑 2，坑 3 的现象被坑 2 掩盖着，看不出来。

坑 4 的性质和 1~3 不同，值得单独说：它**不制造错误、也不制造慢的答案**，它是**把并发悄悄退化成串行**。
N=1 时它完全无害（只有一个对话，`bound` 里就一条），所以只看 N=1 的量测永远发现不了；
而它又是**取决于到达顺序**的偶发：同一份二进制，先量到 3.40×（运气好），重跑同一套测试就塌到 1.00×/1.90×
（运气不好）。**这就是"每会话 decode 都是 11 tok/s、伸缩却只有 1×"这个矛盾现象的来源**——
看到"答案全对 + 单会话速度正常 + 聚合不涨"这个组合，先去看租约落点，别怀疑硬件。

坑 4 的修法（`SessionPool.acquire()`）：从空闲会话里挑时**先排掉 `bound` 里已有的主人**
（`candidates = [s for s in free if s not in owned] or free`），挑不出干净的才允许窃取；
窃取时**把旧主人从 `bound` 里删掉**——否则一个会话会留下两条绑定，下一轮又派重。
另给 `bound` 加了上限（4096，超了丢最旧的），避免长时间跑下来这张表无限长。
回归用例直接**摆好池状态再断言**（`known=[None]*n` + 指定 `bound`），不依赖前面几节残留的状态——
这一点是被坑出来的：最初写的用例"恰好是绿的"，因为前一个用例留下的池状态让断言失去了意义。

> **这句"允许窃取"后来被删掉了（v1.9，见 §9.8）。** 多人接入的需求进来之后，
> 「挑不出干净的才允许窃取」这个兜底变成了排队场景里的**唯一**行为，而它有两个问题：
> 受害者下一轮要付完整重新 prefill，且**抢谁取决于到达顺序**（同一份代码量到过 3.40×/1.90×/1.00×）。
> 现在空闲会话全都有主时**只排队、不窃取**。坑 4 的判定（选会话要排除已有主人的）没变，
> 变的是"全占满"这一支。

`ThinkStripper` 的实现要点：标签可能**跨 `DELTA` 帧**（甚至逐字符）到达，所以必须是**位置状态机**
（lead/pre/body/post/done），不能靠一次性正则；若生成在 `<think>` 内部就被 `max_tokens` 截断，
**原样吐出**（宁可露出标签，也不静默吞内容）。另有一处刻意的**有界偏差**要记着：
关思考时，客户端收到的正文（已摘 think）与 KV 里实际存的文本（含 think）**不同**；
复用判定用的是客户端回显的文本，所以自洽，但客户端若自己改写历史就会退化成全量——
**只变慢，不会算错**。

**不具可复现性（重要：验收标准必须据此设定）**

`top_k=1` 贪婪解码在本栈上**不是逐字节可复现**的：同一 prompt、同一会话、同一参数，
实测出现过两个不同答案（`determinism_probe.py` 的 A/B/C 三组）。因此：

- 验收标准只能是**语义级**的（暗号存活、`finish_reason`、prefill 计数、伸缩比），
  **不能**是"token 逐字节一致"；
- §9.1 那套黄金 token 对照**不能**直接搬到 HTTP 层；它仍然有效，是因为它比的是**同一进程内的
  两条路径**（单会话 vs 并发），而不是两次独立运行。

**已知限制与未验证项（如实列出）**

- **`tool` 角色的渲染未测**：`render_messages` 只处理 system/user/assistant，带 tool 调用的历史
  没有跑过一遍。工作流 Agent 若要接 function calling，**这一项要先补测**。
- **网关没有鉴权**，`--host 0.0.0.0` 时局域网内谁能连上谁就能用满 4 张卡。Agent 跑在板卡本机时
  设 `HOST=127.0.0.1` 更稳妥（`start_gateway.sh` 已支持）。
- **CORS 是 `*`**（为演示页开箱可用）。它不改变"谁能访问"——没有鉴权时本来就谁都能访问——
  但它意味着**任意网页**都能在访客浏览器里驱动这块板卡。演示只在受控内网里跑，别暴露到不可信网络。
- **没有 `--no-sticky` 回退开关**：粘性复用关不掉（要关只能换对话 id）。出问题时只能靠重启网关。
- 开思考（默认）时是**流式原样透传**，推理过程不过滤、也不单独成字段（协议里没有 `reasoning_content`）。
- **`/no_think` 是软开关**：实测关思考下模型仍可能吐出一段 thinking（`nothink_check` 第二轮就出现了），
  若这段没闭合就被截断，`ThinkStripper` 会原样吐出带 `<think>` 的文本——这是上面那条有界偏差的另一面。
- `pass_fds` 在 Windows 上不可用（本地只能用桩后端 `fake_backend.py` + `--frames-stdout` 做回归）。

**⚠️ 帧协议的一个硬缺陷：`ERR` 被按 `DELTA` 解析（2026-09-16 板上撞到，已修 `5c167f5`）**

写侧两种帧的字段数不同：`DELTA <rid> <n>` 对 `ERR <rid> <session> <n>`（`serve_err` 多发一个
`session`）。读侧原先共用一套两字段解析（`rest.partition(b" ")`），于是 `ERR` 的长度字段拿到的
是 `"<session> <n>"`，`int()` 抛 `ValueError`。后果不是"这一帧丢了"：异常终止读线程，`finally`
把**所有在途请求**标成 `backend exited` 并置 `_dead`，**此后每个请求都 503，直到重启网关**。
板上现场的错值 `b'3 11'` 就是钥匙——`3` 是会话号、`11` 是 `"turn failed"`（`main.cc:2754`
在某轮失败时发的消息）的字节数。**一次"某轮失败"能打掉四会话的服务**：伤的是可用性，
不是正确性（已在流的答案是好的）。修法：两种帧的 `rid` 都是第一个字段、`n` 都是最后一个，
按位置取即可兼容。回归 `rt_work/repro_gateway_err_frame.py`（真 `_read_loop` 跑在真管道上：
修前 4 FAIL / 修后 0；含"ERR 后面还排着另一个请求"那条——它会一直等到 600s 超时；网关自检
33 PASS / 0 FAIL）。**触发那轮失败的原因仍未定位**：它紧跟一条 `sent=0/208 (base=208)`
（重复提示词、KV 已完整 → 后端收到 `prompt_len=0` 的空 user 轮）；后端日志在重启时被覆盖，
下次要先拷走 `gateway_backend.log` 再重启。**板卡副本是就地打补丁的**（原件与补丁后 md5
见 §8.1 表），补丁后连着两轮 4 路并发没再出现 ERR——但**那两轮里没有 ERR 帧到达**，
所以板卡侧是"按构造验过"，不是"现场复现过"。

**产物与位置（一处与 §8.1 红线的冲突，已按"移进仓库"处理）**

- C++ 侧（`--serve` / `--serve-fd` / `TokenSink`）随 `cpp/main.cc` 入库，**帧协议就是网关的契约**。
- **网关与板端脚本已入库 `examples/multicard/serve/`**：`rkllm_gateway.py`、`start_gateway.sh`、
  `restart_gateway.sh`、`build_serve.sh`、`run_all_board_tests.sh`、5 个测试脚本、
  `fake_backend.py`（本地桩后端）与 `README.md`（使用说明；**md 不下发到板卡**）。
- **演示层（2026-09-15 补）**：`demo_4session.py`（命令行 4 行进度条）、**`demo_4chat.html`
  （网页 4 个对话框，四路同时流式输出）**、`DEMO.md`（演示手册，**md 不下发到板卡**）、
  `page_check.js`（用桩 DOM 跑页面里的真实 JS，需 node，不进 `run_all_board_tests.sh`）。
  网页版由**网关自己发**（`GET /demo`，读同目录的 `demo_4chat.html`）——另起静态服务器会引入
  跨域，同源则天然没有这个问题；`OPTIONS` 预检回 204、响应带 `Access-Control-Allow-Origin: *`
  与 **`Access-Control-Expose-Headers: X-KV-Reuse`**（少了最后这个头，前端读不到复用台账）。
- 起因是原先它们只在 `rt_work/serve/`，而 §8.1 的红线是「`rt_work/` 不上板卡」；为了让网关在板上跑，
  那份副本已经被下发过（**已发生**）。处置：把**可部署件**移进仓库，`rt_work/` 只留过程记录
  （部署日志、探针输出、实测原始数据、源码快照、早期探索脚本）。
- 脚本里的板卡/云路径**全部参数化**（`INSTALL_DIR` / `MODEL_DIR` / `GATEWAY_DIR` / `LOG` …，
  见 `start_gateway.sh` 头部），仓库里不出现任何内网地址与个人目录名。**入库后重新下发到板卡、
  用环境变量重启并重跑了一遍全量套件**（不是"搬完就算"）——板卡上的 `rkllm_gateway.py` 与
  仓库副本 **md5 相同**，两份副本不再有漂移（**2026-09-16 更正**：实测两者差一处 docstring、
  md5 并不相同，见 §8.1 表末行——**何时分叉的没查**）。加网页演示那一层之后**又重跑了一遍全量套件**
  （FAIL 0，同轮伸缩 3.37×），确认这一层没有吃掉并发。

> ⚠️ **不要把"3.33×"当成 HTTP 层的功劳**：收益全部来自 P2 的卡级并发执行器，
> HTTP 层要做的是**别把它吃掉**（排队、全局锁、串行发帧都会吃）。§9.7 的伸缩表是
> "没吃掉"的证据，不是"新增了"的证据。

**服务侧收尾加固（2026-09-16）**

上面那批是网关侧的；这次补的是 **C++ 服务侧**的四件事，全部由"防线被踩时会发生什么"来定义：

- **`REJECT` 帧**：上下文将满时不再"自动清 KV 再把这一轮跑完"。那样做的后果是模型拿到一段
  **没有开头**的对话，然后以 `finish_reason=stop` 返回一个自信的错答案——`CLEAR` 帧救的是
  **下一轮**（网关据此作废粘性记录、下轮全量重发），**本轮已经错了，而且不报错、不重试**。
  改成先 `CLEAR` 再 `REJECT`，把策略交回网关。同时**顺手清掉 KV**：这段对话本身就装不下，
  留着满 KV 只会让下一轮同样顶穿。后端线程走 `continue` 而不是 `break`——`break` 等于把一个
  健康的会话摘掉（服务容量永久少一份），钉给它的活从此没人接。
- **会话线程退出清扫**：线程退出的两条路径（正常收工 / 推理失败）都要把队列里**钉给本会话**的活
  取出来回 `ERR` 并丢掉。留着不动的后果是它们占住 `kInteractivePendingCap`（32）的额度，攒满后
  输入线程卡在 `cv_space` 上——而它等的谓词是 `pending.size() < 32 || active_workers == 0`，
  **只有部分会话死掉时两条都不成立**，两侧互等，连 `input_thread.join()` 都出不去。
- **越界校验**：`max_new > kServeMaxNewTokens`（2³¹−1）与 `session` 越界都必须**先丢掉载荷**
  再回 `ERR`——不读掉的话，下一次循环会把载荷开头的字节当成头行，整条帧流**从此永久错位**，
  表现为"之后每个请求都立刻失败"。`prompt_len` 上限（1 MiB）必须**最先**判：反过来的话，
  「会话号越界 + `prompt_len` 声明成 2⁶⁰」会先去跳一个天文数字。
- **清 KV 收敛成一个实现**：原先交互模式与服务模式各有一份，`clear_conversation_kv` 现在是唯一
  入口。清法不统一时，"到底清了几次"这种事在测试里没法数——而它正是好几条分支的**行为判据**。

**一次「以为修好了」的竞态（同一次改动）**

上面那条清扫我先前只做了一半：`dispatcher_push` 只在**等队列空间之前**判过目标会话死没死。

```
目标会话还活着 → 队列满 → 输入线程等在 cv_space 上
     ↓ 这段等待里目标会话死了（它退出时把钉给它的活扫掉、腾出位置）
醒来 → 这条活被推进队列，而**能消费它的线程刚刚退出了**
```

要点是：**恰好只有目标会话死掉才能解开这个等待**（它的清扫腾出额度），所以"醒来"这一刻正是
"它已经死了"最可能成立的一刻；而对方的清扫在此之前已经跑完，**不会回来捞这条活**。后果是那条
请求**永远没有回帧**——网关侧白等 `REQUEST_TIMEOUT`（默认 1800s），期间还白占一个租约。
修法是等到位置后**再判一次**。为什么这样就闭环：置 `worker_alive=false` 与清扫是**同一把锁**里的
两件事，`dispatcher_push` 全程持这把锁，两者只能一前一后——我方在前则推进去的那条随后被对方捞走，
对方在前则这次判断看到 `false`。两个方向都有回帧。

**这条不是读代码读出来的，是执行出来的。**

**这次的验证：把服务侧路径搬到主机上执行（2026-09-16）**

这批 C++ 改动此前只过了 `-fsyntax-only`，**一次都没被执行过**。板卡要 240s 加载模型、且当时正被
拿去演示，不能为了验证动它；所以把 **SDK 那一层换成桩**，让**工作区那份 `main.cc` 原文**在主机上
跑起来（`#define main` 改名后 `#include` 进测试 TU，`--gc-sections` 之后需要补的外部符号正好 21 个；
入口 `rt_work/hosttest/run_host_tests.sh`，`rt_work/` 不进仓库）。三组一起跑，缺一不可：

| 组 | 期望 | 实测 |
|---|---|---|
| 工作区版（被测的那份源码） | 全通过 | **64 PASS / 0 FAIL**，构建 0 warning |
| `git show HEAD:` 导出的改动前版本 | 在覆盖本次修复的六条上失败 | t1/t2/t5/t6/t7/t8 **如预期失败** |
| └ 同上，四条对照测试 | 通过 | t3/t4/t9/t10 通过——排除"HEAD 根本跑不动"这个解释 |
| 四个变异体（各自只去掉一处修复） | 对应测试变红 | 全部变红（去掉载荷丢弃 / `REJECT` 写成 `ERR` / 不做清扫 / 入队只判一次） |

**只跑第一组什么也说明不了**——把某个修复删掉、测试照样绿，那这套断言就是白写。变异体那一步才是
"断言抓得住缺陷"的判据；上面那条竞态就是它先红出来的（去掉二次判断 → 40 条活里 39 条入队、
1 条永远没人取）。

**夹具自己也出过一次错，值得记一笔**：收尾时"测试卡住"的救援手段原本是
`dispatcher.active_workers = 0`。而 `active_workers` 在 `main.cc` 里是**不变量驱动**的计数、只由
会话线程自己 `--`——手工拧成 0 之后，还没退出的会话线程接着把它减成 **-1**，于是 `dispatcher_push`
里 `active_workers == 0` 这条退出条件**再也不成立**：救援反而把要救的那个等待锁得更死。同一个变异体
一次能放行、一次卡到看门狗（退出码 99）超时，看着像"偶发"，其实是这条救援在**自己制造死锁**。有效的
放行是**清空待办队列**——两侧谓词各有一半立刻为真，一个不变量都不用伪造。**测试装置里的"帮忙"和产品
代码里的"帮忙"一样危险，它也会静默地把前提条件改掉。**

**边界（说清楚比说满重要）**：桩只保证**调用次数与成败**，不保证张量语义——所以这套证明的是
「服务侧状态机 + 帧协议」，**证明不了推理结果对不对**。`main.cc` 的**启动**路径（`init_stage`、
张量/metadata 查询、embedding）也不在覆盖内：夹具是手工构造 stage 的，覆盖的是"会话已经开始
服务之后"。**本轮未做 aarch64 交叉编译，也未上板**；板上的 `run_all_board_tests.sh` 仍是最终判据。

#### 交叉编译与上板验收：把上面那段边界补掉（2026-09-16）

**证据链**（每一环都可复核）：

| 环节 | 值 |
|---|---|
| 源码 | `examples/multicard/cpp/main.cc` md5 `e0c49989c1fc5d1123d6f8201e2ee514` |
| 交叉编译 | 构建服务器 `GCC_COMPILER=aarch64-linux-gnu`（linaro 6.3.1），走仓库自带的 `build_serve.sh`（Release，不带 sanitizer） |
| 产物 | `rknn_multicard_demo.serve` md5 `3355dc4f44a5f47332c8e0d7528be84d`，1076496 B，aarch64 ELF |
| 产物自检 | 比本次构建开始时刻新、`NEEDED` 里没有 libasan/libtsan、`--serve-fd` 与 `REJECT` 各出现 3 次（三条都是脚本里的断言，不满足就退出） |
| 网关 | `rkllm_gateway.py` md5 `7a06e5ca2b9efb230d119582bbc56961`（= `1dde346`，与后端同一次改动） |

**为什么网关必须和后端一起换，而不是"只换二进制更保守"**：这次后端会发 `REJECT` 帧，而板卡上那份
网关是旧版、**不认识这个 tag**——不匹配任何分支时它既不处理、**也不读掉载荷**，payload 会被当成
下一行读走，帧流自此错位（最坏情况读线程抛异常退出 → 之后每个请求 503）。所以混搭不是保守，是
制造一个**已知会坏**的组合。板上据此做的是全栈替换：后端、网关、演示页、测试脚本一起换成仓库版本，
旧文件全部留了备份（`bak_20260916/` 与 `rknn_multicard_demo.bak_e223ea8f`），回滚就是换回来重启。

**`run_all_board_tests.sh` 全套（连跑两轮，状态即上面那份产物 + 网关）**：

| 段 | 结果 |
|---|---|
| `check_template.py` | rc=0 |
| `serve_http_test.py`（非流式 / 流式 / KV 复用 / 并发 / 关思考 / 多用户身份 / 池与 `close`） | rc=0，41s |
| `serve_http_test.py … bigmax` | rc=0 |
| `sticky_check.py` | rc=0 |
| `nothink_check.py` | rc=0 |
| `http_scaling.py … 96 1,2,4` | rc=0，30s |
| 汇总 | **FAIL 0 / 非零 rc 0 / traceback 0——两轮都全绿** |

两轮的伸缩：N=1 `10.97` / N=2 `20.26`、`20.22`（1.85×、1.84×）/ N=4 **`37.40`、`37.13`（3.41×、3.38×）**，
对照官方 rkllm3-server 同板同条件 `1.03×`。两轮差 1% 以内，不是单次侥幸。

**这一轮头一次在真后端上跑到的**（此前只在桩后端验过）：§9.8 的身份显式化 / 排队不抢占 /
`POST /v1/conversations/close` / `GET /v1/pool`。另外 `usage` 的新口径在板上拿到了非零的
`prompt_tokens_details.cached_tokens`（开思考 180 tok 里复用 168、关思考 188 里复用 172），
这是"新网关 + 新测试脚本"配对成功的直接证据——旧判据（`prompt_tokens` 比全量小）在旧网关上**必然**
失败，因为它把 `prompt_tokens` 当成了"本轮增量"。

**R14 在板上复现、也修好了——这是本批唯一一条"改动前 vs 改动后"的板上对照**

`--bucket-size` 默认 128 时，两种记账口径**恰好相等**，所以上面那份全绿**完全说明不了 R14 有没有修**。
要让它现形必须把 `--bucket-size` 改成别的值，再喂一段跨过 128 token 分块的 prompt：

| 后端 | `--bucket-size` | 同一段 prompt（366 字，`prompt_tokens`=261） |
|---|---|---|
| 改动前 `e223ea8f` | 64 | HTTP 503；后端日志 `[stage1] embed buffer too small: need=1361920, got=1310720` + `prefill failed` |
| 改动后 `3355dc4f` | 64 | HTTP 200，2.1s，答出暗号，后端无失败记录 |

同一份改动前的二进制在这段 prompt 不长时（15 / 42 token）**照常答对**——触发条件是**跨过一个分块**，
不是"代码肉眼可见地坏了"。这与 R14 那条"默认值下不触发"是同一件事的两面。

**边界（说清楚比说满重要）**：覆盖的是服务侧状态机、帧协议、HTTP 语义与分块记账；**张量语义仍不在
覆盖内**——推理结果对不对还是靠答案本身目视。`main.cc` 的**启动**路径在板上是真跑的（240s 加载），
但主机夹具不覆盖它。另外这一轮把 §9.8 那套多用户行为**第一次放到了真后端上**，而它在 4 路演示场景下
是**新的默认行为**（旧演示页那种四个匿名对话会占满会话、被排队咬到），所以演示页也必须一起换新。

---

### 9.8 多用户接入：把板卡当边缘服务器（2026-09-15）

**需求（这是这套东西真正要解决的问题）。** 不是"演示 4 路并发"，而是**把开发板做成一台边缘
服务器**：多个用户各自从终端接进来提问、各自拥有一段对话；**不要求同时**，随时来随时问；
人数在会话数以内时谁提问谁就立刻拿到一个会话；**超过就排队等**。
"同时 4 路"是**并发能力**的展示（第 3/5 步），不是日常形态。
客户端**只要终端接 API**，不做网页前后端——所以这一节没有任何前端产物。

**诊断：原实现有三个缺口，而且都是"答案全对、现象静默"的那一类。**

| # | 缺口 | 现象（不报错） |
|---|---|---|
| 1 | **身份只能靠内容认** | 两个人问同一句话 → `sha1(system+首问)` 相同 → 钉在同一会话上串行。答案全对，聚合吞吐掉到接近单路 |
| 2 | **会话全占满时是"窃取"** | 新对话把别人的会话抢走，被抢的人下一轮付完整 prefill；**抢谁取决于到达顺序** |
| 3 | **排队这件事看不见** | 排队的人不是变慢了，是还没拿到——他没输出，吞吐数字上完全看不出来 |

**修法**

- **身份显式化**：`conversation_key()` 早就支持 `explicit`，`do_POST` 也早就在读
  `conversation_id` / `X-Conversation-Id`——**但 README 从未提过**，等于没有。
  现在补文档，并多接一个 OpenAI 既有字段 `user` 作为兜底（优先级 `conversation_id` >
  `X-Conversation-Id` > `user` > 内容摘要）。**这一条是文档修复，不是代码修复**——
  值得单独记一笔：能力存在但没写进文档，在多人场景里和不存在没有区别。
- **去掉窃取，改成排队**（见 §9.7 坑 4 下面那条补注）。空闲会话全都有主 →
  `cv.wait()` 等别人释放，最长等 `QUEUE_TIMEOUT`（默认 600s），超了返回 **503** 并明确
  告诉客户端"复用已有 `conversation_id` 或稍后重试"，不让终端用户无限期干等。
- **空闲回收（`IDLE_TTL`，默认 300s）**：**这是"排队"能成立的前提**，不是可选项。
  网关**没有"对话结束"这个信号**（客户端不发送关闭消息），所以如果只是"占着不放"，
  5 段对话会把 5 个会话永久占死，第 6 个人等到 1800 秒超时也拿不到。
  一段对话静默超过 TTL → 收回它占的会话，`known[s]` 置 `None`（内容未知 ⇒ 下一轮强制 RESET）。
  被回收的人下次提问会多付一次完整 prefill（约 2–3s），**正确性不受影响**——上下文一直在客户端手上。
- **`GET /v1/pool`**：回答"现在是谁在占着、谁在排队"。这是缺口 3 的唯一解法。
  `/health` 也补了 `sessions_busy` / `sessions_bound` / `waiting`。
- **CORS 预检放行 `X-Conversation-Id`**（原先只有 `Content-Type`）。浏览器里发这个头会被预检
  拦掉，而 curl/python **不发预检**——所以这个问题只会在网页端现形，方向很容易查错。
- **`POST /v1/conversations/close` —— 这一条是"去掉抢占"之后**才**暴露出来的缺口。**
  抢占时代，一段不再提问的对话占着会话不影响别人：下一个人直接把它抢走。改成排队之后，
  它就**永久占住**了——只能等 `IDLE_TTL`。对"来问一句就走"的客户端（脚本、Agent、测试套件）
  这是致命的：一个脚本连问 4 个不相干的问题，就把 4 个会话全占住、把后面所有人挡在队列里
  整整一个 `IDLE_TTL`（默认 5 分钟）。**所以"排队"要有意义，必须同时给客户端一条说"我走了"的路。**
  `close` 幂等（没占会话时返回 `closed: false`），正在生成时返回 409（不能抽走跑着的会话
  脚下的 KV）。匿名对话（没有显式身份）用 `id` 关不掉——它的身份是网关按内容算的摘要，
  客户端算不出来；所以 `/v1/pool` 的每个槽位同时给出给人看的 `conversation` 标签和机器用的
  原始 `key`，匿名对话按 `key` 关。**"看得见却踢不掉"是不行的**：没有这条，
  排障的人只能看着一段不知道是谁的对话占着会话干等到超时。

**排队 vs 抢占：为什么选排队**（`--stagger` 错开提问就是为了把这个演出来）

- 抢占要付的代价是把受害者的 KV 丢掉，它下一轮要**重新 prefill**——对"随时来随时问"的
  使用形态来说，这等于把成本转嫁给一个**当时什么都没做**的人。
- 更糟的是**抢谁取决于到达顺序**：这正好是坑 4 的那种不确定性（同一份代码 3.40×/1.90×/1.00×）。
  演示场合最不该有的就是"重跑一次结果不一样"。
- 排队的代价是后来的人等 —— 但这是**可见的、有限的**（`/v1/pool` 能看、`QUEUE_TIMEOUT` 封顶），
  而抢占的代价是**不可见的、转嫁给他人的**。

**验证（本地桩后端，`fake_backend.py`，非板卡）**

| 断言 | 结果 |
|---|---|
| 5 用户 2 会话：前 2 个立刻拿到，后 3 个排队 | ✅ 7.1 / 14.8 / 6.5s 后依次补上；落点 `0,1,0,0,1` |
| **不抢占**：全占满时 `bound` 与排队前逐项相同 | ✅ 回归用例断言 `dict(pool.bound) == before` |
| 静默超时回收后排队者立刻拿到 | ✅ `idle_ttl=0.2`、回拨 `last_used` 10s |
| 回收后该会话内容未知 ⇒ 强制 RESET | ✅ `known[s] is None` |
| 正在跑的会话不被回收 | ✅ |
| 无身份的请求**不记绑定** | ✅ 回归用例（漏了这条，所有匿名请求会共用一个会话） |
| 排队超时返回 503、不留残留 | ✅ |
| `serve_http_test.py` 全量（桩 + `--think-prefix`） | ✅ 全绿，含 HTTP 级 KV 复用证据（14 vs 131 tok） |
| **套件在"关掉回收"（`IDLE_TTL=0`）+ 4 会话下连跑两遍全绿** | ✅ 22 PASS ×2，两遍都全绿——证明它靠的是自己 `close`，不是碰巧被 TTL 救 |
| 同一句话、3 个不同身份 => 3 个不同会话 | ✅ 落到 `['0','1','2']` |
| 同一句话、**不给身份** => 全落到 1 个会话 | ✅ 落到 `['0','0','0']`——退化的实证 |
| `close` 语义（幂等 / 400 无身份 / 409 正在跑 / 交还后可被新对话立刻拿到） | ✅ 6 条回归用例 + HTTP 层各一条 |
| `demo_multiuser.py` 端到端（5 用户 2 会话、`--no-id` 对照、`--rounds 2`） | ✅ 4 人 `--no-id` 时**全落到 1 个会话**（`落到 1 个不同会话 ['0']`）——正是要演示的退化 |
| `demo_multiuser.py` 的 `--close` 对照（本地桩，5 用户 2 会话） | ✅ 不交还：墙钟 26.6s、第 5 人 **503**；交还：墙钟 **2.0s**、第 5 人立刻拿到会话 |
| TTL 回收路径（`--idle-ttl 3`，5 用户 2 会话） | ✅ u3/u4/u5 各排队 2.5/2.8/5.2s 后被补上，日志有"会话回收→排队结束"成对记录 |
| 网页演示页四路带出去的身份 | ✅ `["web-1","web-2","web-3","web-4"]`；`/v1/pool` 四个槽位分别记着 `id:web-1..4` |
| 关页面（`sendBeacon`）/ 点「清空」（`fetch`）都能把会话还回去 | ✅ 两条路各自验：交还后池子里再无 `id:web-*` |
| 上述三条断言**能报警**（负向对照） | ✅ 把页面里的身份头那一行删掉重跑 → 身份 + 归属共 **5 项 FAIL**；不是摆设 |
| "占满 → 排队 → 交还 → 立刻补上"端到端 | ✅ 4 个 `id:web-*` 占满，第 5 个进 `waiting`（未被抢占），`close` 一个后立刻拿到会话 |

**过程中被自己写的东西咬到的五处**（都留在代码注释里）

1. 重写 `acquire()` 时**丢掉了 `key is not None` 保护**，于是所有匿名请求共享 `bound[None]`、
   被钉到一个会话上串行——**和坑 4 同一类故障**，且同样静默。补回保护 + 回归用例，
   并让 `/v1/pool` 的 `conversation` 对匿名请求显示 `null` 而不是 `(anon)`
   （后者看起来像一个真实用户，会误导诊断）。
2. **删掉抢占之后，`serve_http_test.py` 自己把自己堵死了。** 它原先每问一个新问题就开一段
   新对话（一共 9 段），抢占时代无所谓——反正能抢。改成排队的当天，这些请求就会一路排到
   `IDLE_TTL`。**修法是让套件像真实客户端那样工作**：每段对话显式命名、用完 `close`，
   全程占用有界（≤ 会话数）。顺带把 N 路并发那一段也改成"N 段各带身份的对话"——那本来
   就是它要测的东西。
   这件事的意义超出测试：**它说明"只排队不抢占"是一个会改变所有现有调用方行为的改动**，
   而不只是一个新策略。任何"每问一次就换一段对话"的既有调用方都会撞上它。
3. `--stagger 0.6` 的**值被当成位置参数**吃掉（`ARGS` 只过滤 `--` 开头的 token）→ `int("0.6")` 崩。
   带值开关必须连值一起跳过。
4. Windows GBK 控制台上 `⚠️` 让 `print` **抛 `UnicodeEncodeError` 直接崩**（不是显示乱码）。
   演示脚本常从 Windows 终端连板卡，所以 `sys.stdout.reconfigure(errors="replace")`。
5. **网页演示页把整套多用户演示饿死了。** 改完之后 `demo_multiuser.py 6 40 --close` 六个用户
   **全部 503、合计 0 token**；`/v1/pool` 一看，四个槽位全是**匿名**对话
   （`h:5155b7…` 之类）——`demo_4chat.html` 的 `streamChat()` 只发 `Content-Type`，四个面板
   于是成了四段匿名对话，把 4 个会话全扣在手里。
   **这条是缺口 3（不可见）的一个变种**：抢占时代看不出来（下一个人直接抢走），改成排队后
   它就成了"一个人把服务器占满"。修法两件：给每个面板**稳定的身份**（`web-1..web-4`，
   基线单独一个 `web-baseline`）、**不用的会话立刻交还**（基线量完就还；点「清空」还；
   关页面用 `sendBeacon` 还——`beforeunload` 里 `fetch` 会被浏览器取消，这段代码等于没写）。
   面板跑完**故意不还**：同一页里再点一次「②」就是这四段的第二轮，KV 复用的演示靠它们活着。
   顺带把 `page_check.js` 从"4 个问题互不相同"（**现在已经不是并发的前提**）改成验
   "身份发出去没有 / 池子里记的是谁 / 会话还没还"，并**做了负向对照**证明它们会 FAIL。

**未验证 / 已知限制**

- **板卡上尚未跑过端到端的多人演示**：上面这一整节都是**桩后端**上验的（本地网关 + `fake_backend.py`）。
  逻辑与板卡共用同一份 `rkllm_gateway.py`，但"板上真后端 + 多人"这一组合**还没跑过**。
  网页页的身份/交还改动（咬到 5）同样只在桩上验过；板卡上跑的 `demo_4chat.html` 还是旧版。
- **排队策略架不住没有鉴权**：网关只能按请求上带的身份分会话，**无法核实身份是真的**。
  一个客户端用 5 个不同的 id 各发一次就能占满全部会话，把真人挤到队列里。
  **"超过 5 个就排队"只在参与者都守规矩时成立**——要对外服务必须先加鉴权（反向代理）。
- **排队不保证严格 FIFO**：醒来后谁先抢到锁谁先拿（实测 5 人 2 会话的完成顺序是 u3/u5/u4）。
- `IDLE_TTL` / `QUEUE_TIMEOUT` 的默认值（300 / 600）是**拍出来的**，没有实测依据；
  它们必须满足 `QUEUE_TIMEOUT > IDLE_TTL`，否则会出现"马上轮到你了却先超时"。
- **`IDLE_TTL=0` 是"永不回收"**（本轮测试就用的它，为了证明套件不依赖 TTL）。这个档位下
  唯一的释放手段是 `close`；**用匿名对话的客户端没有这个手段**，会话会被它占到网关重启。
  默认 300，不要为了"省事"设成 0 跑多人场景。
- **`close` 是"客户端自愿"的**：网关没有任何办法强制一个不守规矩的客户端交还会话，
  这和下面那条"没有鉴权"是同一个根因。

---

### 9.9 Phase 1：工具调用（function calling）+ 8192 上下文 —— 2026-09-17 真机验收

"边缘服务器、多人终端接 API"（§9.8）之后，Agent 化的第一步是让模型能**调工具**。
这一步最反直觉的地方是：**工具调用的格式不是我们定的，也不是任何一篇文档定的，
而是这份权重训练时见过的那个格式**。写错一个字节，模型的反应不是"调得别扭"，
而是**根本不调**——它会把格式示例当普通文本吐出来，或者干脆不用工具，而且**不报错**。

#### 9.9.1 格式的权威来源：模型自己的 `tokenizer.chat_template`

**没去翻 Qwen 的文档，也没抄别的项目**，而是从板卡上那个 GGUF 的
`tokenizer.chat_template` 元数据键里把 Jinja 模板原文抠出来（提取脚本 `rt_work/gguf_tmpl.py`）。
理由很简单：文档描述的是"某一代 Qwen 长什么样"，而**权重里那份模板描述的是这一份权重**。

这一代是 **XML 形式**，不是 Qwen3 早期那种 Hermes JSON：

```
<tool_call>
<function=get_weather>
<parameter=city>
北京
</parameter>
</function>
</tool_call>
```

> ⚠️ **`main.cc` 里那份 `QWEN35_CHAT_TEMPLATE` 对工具没有任何支持**——它是为单轮对话提取的。
> 所以工具目录（system 里的 `<tools>` 段）只能在**网关侧**拼。好消息是：**工具这条路一行都
> 不用碰后端二进制**，板卡上正在跑的 `3355dc4f` 直接可用，不需要重编重传。

三个容易写错的点（都照抄模板，一处不改）：

| 点 | 模板怎么做 | 直觉会怎么写 |
|---|---|---|
| 工具定义进 prompt | `tojson`：**键排序** + `ensure_ascii=True`（中文变 `\uXXXX`）+ HTML 转义（`<` `>` `&` `'` → `<` 等） | 直接 `json.dumps(tools)` |
| 助手轮里第一条 tool_call 前的分隔 | `\n\n`（**仅当正文非空**），后续每条是 `\n` | 统一 `\n` |
| `tool` 角色怎么渲 | **连续的 tool 消息合成一个 user 轮**，每条各自包一层 `<tool_response>` | 渲成 `<|im_start|>tool`——**模板里根本没有 tool 轮** |

`_TOOLS_PROLOGUE` 的正确性**不是靠人眼看**：`check_template.py` 拿真 Jinja2 渲模型自己的
模板，与网关拼出来的串**逐字节对拍**（49 条 PASS，板卡 Python 3.12.3 无 jinja2 也能跑，
因为夹具是预渲染好的 golden `tool_golden.json`）。

#### 9.9.2 与模板的两处**刻意**偏差，都是为了 KV 复用

工具调用把 §9.6 那条"KV 复用不变量"推到了一个**新的、更脆**的地方。

**偏差一：助手轮在历史里原样保留模型当时生成的那段文本**，不做模板那套
"把 `<think>` 段拆出来重新包成 `<think>\n{推理}\n</think>\n\n"的改写。

理由：**改写 = 下一轮 prompt 的前缀对不上 = KV 复用全废**，而这条路径**只慢不错**——
没有错误、没有日志、只有一个悄悄变长的耗时（板上表现见 §9.9.6）。
实测里粘性比这点格式保真度值钱（长上下文下是"全量重算"和"只发差异"的区别）。
代价写清楚：**多步工具循环里模型看到的助手轮少了 `<think>` 包装**，属于已知偏差。

**偏差二：`add_generation_prompt` 不带模板里的 `<think>\n`**（本项目用 `/no_think` 软开关 +
回复侧摘除，已在板上验收）。改它会让**之前所有吞吐数字失去可比性**，所以不动。

#### 9.9.3 新的裂缝：客户端的回显是**不对称**的

网关的判据是 `prompt.startswith(known[session])`（`rkllm_gateway.py` 的 `_make_lease`）——
**全有或全无，没有"最长公共前缀"的部分复用**。于是工具调用引入了一个单轮对话里不存在的问题：

> **模型自己那一轮**，参数文本是从它生成的 token 里**逐字抠出来的**（`parse_call_body` 保留
> `raw`，复现时按原文渲染）。**客户端回显这一轮**时走的是 `normalize_tool_calls`——它把参数
> **解析成有类型的 dict**，`raw` 就丢了，只能按规范形式重新渲染（`_param_text`）。
> **两者逐字节相等，当且仅当模型当时的原始文本本来就是规范形式。**

`北京` 是；`"北京"`（带引号）、`1.50`（数字）、`true` 不是。
不等的话，助手轮中间断一个字节 → 整段重算 → `cached_tokens` 归零。

**板卡实测（2026-09-17）：Qwen3.5-27B 在板上是规范形式**，所以这个静默降级没有显现。
但这**不是保证**，是这一份权重的采样习惯，换个导出、加个约束解码就可能变。

**判据必须用冷会话对照**（热会话重算 `<` 冷会话重算 **且** 热会话 `cached > 0`），
**不能只看 `cached_tokens > 0`**——早前桩上量到 44/443 那次的教训就是：部分命中
看起来也叫"有复用"，而这次失败是**完全归零**，两者用同一个阈值判不出来。

#### 9.9.4 8192 上下文导出上板

"能不能跑 8192"**不能看配置文件里写的数字**，要看模型文件里的物理证据：

```bash
# 读 safetensors 头（不用加载权重）
# → rope_cos_cache  F16  [1, 4, 1, 8192, 16]
```

**这个形状就是"这份导出编码了多少个位置"的答案**，`--ctx-size` 必须与它一致：
调大了后端静默降级（白给），调小了就是越界读位置编码——典型表现是**收下了但答非所问**，
而 HTTP 层一路 200。为此给 `start_gateway.sh` 加了 `CTX`（默认 4096），板上按
`CTX=8192` 启动。栈就绪实测约 **90 秒**（脚本注释里写的 240s 是保守值），无 OOM。

**光起得来还不够，要验"用得到"**：跑批里每一件的 prompt 都远小于 4096，
`--ctx-size 8192` 只是被**接受**了，没被**用到**。所以单独写了一次性探针
`rt_work/longctx_probe.py`——暗号放在**中段**，首尾都是填充：

| 填充段数 | prompt_tokens | 结果 |
|---|---|---|
| 220 | 5013 | 取回 `7391` ✅ |
| 300 | 6853 | 取回 `7391` ✅ |

> 探针第一版是**不确定的**：给 64 token，模型把预算全花在 `<think>` 里（`finish='length'`，
> 暗号一个字节没吐）。那看起来像"上下文没用"，其实只是**没轮到回答**。
> 加 `chat_template_kw.enable_thinking=false` + `max_tokens=256` 之后才是有意义的测量。
> **这种"看起来是功能问题、其实是测量问题"的失败，在 LLM 场景里是常态。**

#### 9.9.5 板卡跑批：全绿

模型换成 8192 导出、网关换成工具调用版之后，`run_all_board_tests.sh` **整套重跑**：

| 环节 | 耗时 |
|---|---|
| `check_template.py`（模板对拍） | 0s |
| `serve_http_test.py full` | 41s |
| `serve_http_test.py bigmax` | 10s |
| `serve_http_test.py tools` | 30s |
| `toolcall_check.py`（**新**，只有真模型答得了） | 24s |
| `serve_http_test.py sticky` | 5s |
| `serve_http_test.py nothink` | 10s |
| `serve_http_test.py http_scaling` | 30s |

**结论：0 条 FAIL、0 个非零 rc、0 个 traceback。** 伸缩性没被工具调用拖累：
N=1 10.95 / N=2 20.34（1.86×）/ N=4 37.59（3.43×）tok/s。

`toolcall_check.py` 是**新增的板卡侧检查**，三个探针（单参数 / 双参数 / 另一个工具），
问的是桩后端永远答不了的问题：**真模型会不会按这个格式回话**。

| 判据 | 结果 |
|---|---|
| 三个探针都解得出可解析的调用 | 3/3 ✅ |
| 参数名 ⊆ 声明的 properties | 全部 ✅ |
| `required` 参数一个不缺 | 全部 ✅ |
| 用模型自己那次调用的历史做回显，命中 KV 复用 | 热=**32** vs 冷=**630** ✅ |

> **"选中该用的工具"（3/3）记成 NOTE，不记成判据。** 那是模型质量，不是我们的契约：
> 把它写成 FAIL 会造出两类假红——桩后端上（固定回一个 `get_weather`）**永远红**，
> 板上则是**采样走偏就红一次**。一个会随机红的检查，最后一定会被人忽略。

#### 9.9.6 真机抓到的那条 FAIL（本节最值钱的一段）

桩上全绿、板上第一遍就红：

```
[FAIL] 带工具的多轮历史命中 KV 复用（本轮重算部分远小于冷会话）
       热会话重算=469 tok（复用 0）, 冷会话重算=469 tok
```

**根因**：`serve_http_test.py tools` 构造多轮历史时写的是

```python
follow = hist + [{"role": "assistant", "content": c2}, ...]   # ← 丢了 tool_calls
```

在**真模型**上，模型对"工具结果回灌"那一轮的回复**是又一个工具调用**
（`content` 只有 `<think>  </think>  `）。于是回显的助手轮**少了一个字段**，
和记账串在这一轮分叉 → 网关的全匹配判据判定"前缀不成立" → **整段重算，`cached=0`**。

而**桩上从来没复现过**——因为桩看到 `<tool_response>` 就收手，不会"再调一次"。
原注释那句"前缀断在助手轮时，它前面的部分照样命中缓存"**是错的**（正是 §9.9.3 那条：
没有部分复用）。两件事都修了：`follow` 从完整消息构建 `tail`、保留 `tool_calls` 并追加工具结果；
`fake_backend.py` 加 `--tool-call-repeats` 让桩也能"每一轮都回工具调用"。

> 计数锚点也有坑：`<tool_call>` **也出现在工具目录前言里**（格式示例），
> 拿它计数不安全；`<tool_response>` 只出现在工具结果里，是安全锚点。

**变异测试闭环（A/B/C）**——证明这条检查不是空转：

| 状态 | 结果 |
|---|---|
| 修复 + `--tool-call-repeats` | 绿（35 vs 439）✅ |
| 旧写法 + `--tool-call-repeats` | **红（395 vs 395），复现板上那次** |
| 还原 | 绿，且文件**逐字节等于变异前** |

同一天对 `toolcall_check.py` 的回显往返也做了同样的变异：非规范形式的桩上**红**（453 vs 453），
还原后**绿**（33 vs 453）。**两条检查都被证明是活的。**

#### 9.9.7 顺带修掉的 `restart_gateway.sh` 自杀

`restart_gateway.sh:16` 原来是 `pkill -f 'rknn_multicard_demo'`。板上 `INSTALL_DIR`
必须显式传，而它里面就有 `.../aarch64/rknn_multicard_demo` 这一段字——**pkill 会命中
调用者自己的 cmdline，于是把执行它的那个 shell 一起杀了**：`rm` 和重启全没执行，
现场表现是"进程没了、日志还是上一次的、看起来像网关自己崩了"。

修法：模式加括号 `[.]`（命令行里的字面文本不匹配该正则），并**锚到带 `.serve` 的完整二进制名**
（后端 cmdline 是 `./rknn_multicard_demo.serve ...`，目录名不含 `.serve`）。
证据是 `pgrep -af` 对照：**旧模式匹配到了跑脚本的那个 shell 自己（PID 61872），新模式只匹配
网关与后端**。修完端到端跑了一遍——包括"在命令行上带 `INSTALL_DIR`"这个**修复前必然自杀**的场景
——rc=0，网关起来，`--ctx-size 8192` 确认。

顺带实测到一条**有用的**事实：**网关退出时会带走后端子进程**（kill 掉网关，两个都没了）。
所以这行只是"网关非正常死亡（SIGKILL / 崩溃）留下孤儿后端"的保险——那种情况下
四张卡还占着，不杀干净起不来新的。**别因为它"通常没用"就删掉。**

#### 本节的版本对照（工作区 == 板卡，逐字节）

`serve/` 下 10 个文件在这一轮**逐一核对 md5，板卡副本与工作区完全相同**（不再是 §8.1 那种
"板上跑的其实是另一份"）。回滚件留 `serve/bak_20260917_tools/`。

| 件 | md5 |
|---|---|
| `rkllm_gateway.py`（`c1fd763`，含工具调用） | `166d1543000f764b505d04eb40b72fba` |
| `toolcalls.py` | `b1ab7c207e2b104a355ed542da31ec14` |
| `toolcall_check.py`（新） | `739d05a35f22c603452f53956d1647b9` |
| `serve_http_test.py`（回显往返修复） | `9efad17bd74efeb29aabd1d02779475f` |
| `fake_backend.py`（`--tool-call-repeats`） | `52c235afa0268f06e997c7ad33bd09c3` |
| `start_gateway.sh`（`CTX`） | `6842e7deb1dac2aaea646922b0ca480e` |
| `restart_gateway.sh`（pkill 自匹配修复） | `be2eb23c0124a2d1a98544fa781bb4a5` |
| 后端二进制（**本轮未变**，仍是 `b13bbdb`） | `3355dc4f44a5f47332c8e0d7528be84d` |

#### 未验证 / 已知限制

- **只验了非流式路径**（`stream: false`）。流式与非流式共用同一份解析，但**流式下的工具调用
  没单独跑过**。
- **工具集在对话中途变化会让前缀失效**：工具目录在 system 消息里，位置最靠前，
  一变等于整段前缀作废。多轮对话里**不要中途换工具集**（换 = 下一轮全量重算，静默变慢）。
- **`tool_choice` 没实现**：不能强制调用、也不能禁止调用。唯一约束是模板前言里那句
  "如果没有可用函数就正常回答"。
- **并行多工具调用没构造过**：解析侧按 `</tool_call>` 逐个切，理论上支持一段里有多个
  `function` 块，但**没有真的逼模型一次调两个**。
- **参数类型的规范化是有损的**（§9.9.3）：数字 `1.50` → `1.5`、`true` → 布尔，
  回显时渲染不回原样。当前板上没触发，但这是**换模型/换导出要重验的第一件事**。
- `--tool-call-repeats` **只是桩后端的开关**，板上用不到；它是为"在本地复现板上那条红"而加的。
- 桩后端固定只回 `get_weather`，所以**"选对工具"这件事桩上永远验不了**——那是
  `toolcall_check.py` 存在的唯一理由。

---

## 10. 一页纸总结

| 项 | 结论 |
|---|---|
| **能不能做** | ✅ **已实现并实测通过**（P0 可行性 + P1 重构 + P2 并发，见 §4/§5） |
| **收益** | 相对单会话同 workload：N=2 **1.84×**、N=3 **2.67×**、N=4 **3.33×**；N=5 反而下降（34.94 < 35.74）→ **拐点在 N=4** |
| **单会话代价** | N=4 时每会话纯 decode 12.38 → 10.16~10.55 tok/s（延迟劣化约 17~22%），换来 3.33× 总吞吐 |
| **容量（实测）** | **每卡硬上限 5 个 session**（第 6 个失败）；**4 路 × 3894 token 上下文已验证全部成功** → 方案目标有 1 个 session 余量 |
| **每会话成本** | **64.4–70.7 MB/卡**，在 `session_init` 时按满上下文**一次性预分配**，与实际用掉多少上下文无关 |
| **权重** | ✅ **共享，不 per-session 复制**（否则第 2 个 session 就会 OOM） |
| **KV 隔离** | ✅ **隔离已证**：两个不同 prompt 各 12 步 decode 逐步交错，序列逐 token 一致 |
| **模型** | ✅ **不用重新转换。** 单 KV group；**每卡 5 session 的上限与上下文大小无关**，降 ctx 重导**不能**提高并发数（§4.4）。**2026-09-17 已换成 8192 上下文导出**（判据是 `rope_cos_cache` 形状 `F16 [1, 4, 1, 8192, 16]`，按 `CTX=8192` 启动）：5013 / 6853 token 的 prompt 都能从**中段**取回暗号（§9.9.4）。旧导出是 4096，对它传 8192 只会被静默降级 |
| **主要改造** | ✅ 4 处全局状态已全部下沉到 `Conversation`；并发靠 **`StageContext::run_mutex`（每卡一把锁）**，持锁覆盖整个 `session_run` |
| **正确性** | ✅ N=1/2/4 共 7 路 token id 与单会话路径**逐字节一致**；单会话路径本身零回归（3 组黄金 token 全中）。**交互式多会话同样逐 token 等价**（1539/3078），N=4 时 12 个输入块无穿插、四路各分到 3 轮 |
| **服务化（M5）** | ✅ **已完成**：`--serve` 帧协议（独立 fd）+ `python3` 网关（标准库，无依赖）→ OpenAI 兼容 `/v1/chat/completions`（SSE + 非流式）/ `/v1/models` / `/health` / `/v1/pool`。**HTTP 层 N=4 聚合 36.44 tok/s = 3.33×**（连量 4 轮 3.32~3.38；官方 server 同条件 1.03×）；会话粘性 + 跨轮 KV 复用实测 prefill 23 vs 75 tok（§9.7） |
| **多用户接入（边缘服务器）** | ✅ **已实现**：身份显式化（`conversation_id`/`X-Conversation-Id`/`user`）+ **只排队不抢占** + `POST /v1/conversations/close`（客户端主动交还）+ 空闲回收 `IDLE_TTL` + `GET /v1/pool` 可观测；`NSESSION` 以内谁问谁拿，超过排队、超 `QUEUE_TIMEOUT` 返 503（§9.8）。**⚠️ 已在桩后端验完，板上真后端还没跑过多人** |
| **工具调用（Phase 1）** | ✅ **已实现并真机验收**（`c1fd763`）：格式**取自模型自己的 `tokenizer.chat_template`**（**XML，不是 Hermes JSON**），工具目录在**网关侧**拼——**后端二进制一行没动**。板上 3/3 探针解出可解析调用、参数名 ⊆ 声明的 properties、`required` 一个不缺；用模型自己那次调用的历史回显 **热=32 vs 冷=630**；`check_template.py` 与真 Jinja2 **逐字节对拍**（板卡 Python 3.12.3 无 jinja2 也跑）。**桩上全绿、板上第一遍就抓到一条真 FAIL**（回显助手轮丢 `tool_calls` → 前缀断 → 整段重算），修后经变异测试证明检查非空转（§9.9） |
| **工作量** | P0/P1/P2/P3(交互式)/R11/M5/**Phase 1 工具调用**已完成；**M5 的交付件已全部入库**（`cpp/main.cc` 的帧协议 + `examples/multicard/serve/`） |
| **剩余风险** | `unaligned tcache chunk` 堆损坏（30 分钟稳定性 0 命中 + **ASan 四阶段 0 报告且阳性对照成立** → 概率**下调**，但未关闭：ASan 会改变堆布局、且没做满 30 分钟 soak）；5 上限的成因未知（非阻塞）；**SDK 内部 TSan 报告未定性**（卡级锁管不到 SDK 自己的传输线程；其中 4 条的写侧是我方 `input_callback`，写的是 SDK 给的缓冲，§9.5 / R6）；**网关无鉴权（R12）**——**这条在多人场景下从"不设防"升级为"排队策略可被单个客户端作废"（§9.8）**、**账本失配会静默退化（R13，只慢不错）** |
| **版本管理** | 代码 commit `d59a239`/`6347f57`/`446cbdf`/P3/`9d2d167`(R11)/`6d0666a`(M5，tag `m5-serve`) + 各自 tag；回归对照物是独立保存的源码/二进制快照 + 黄金 token 文件；下发靠 **md5 + BuildID 双校验**（§8.4）；**M5 的交付件已全部入库**：帧协议在 `cpp/main.cc`，网关与脚本在 `examples/multicard/serve/`（§9.7） |
| **下一步** | ① 定性 SDK 内部的 TSan 报告（需原厂/SDK 源码）——**优先那 4 条写侧在我方 `input_callback` 的**；② ~~ASan 跑一轮~~ → **已跑（§9.3）**；要把 R4 彻底关闭需在 ASan 下跑满 30 分钟 soak；③ ~~R11 卡锁~~ → **已验并补锁（§9.6）**；④ ~~HTTP 前端服务化（M5）~~ → **已完成（§9.7）**；⑤ ~~接 workflow agent 前补测 `tool` 角色渲染~~ → **已完成：Phase 1 工具调用（§9.9）**；⑥ ~~决定网关的入库位置~~ → **已定：移进仓库 `examples/multicard/serve/`（§9.7）**；⑦ 每卡卡级统计拆到会话维度；⑧ **把多用户特性下发到板卡并跑一次真后端多人演示**（§9.8 的桩验还不能替代它）；⑨ 若真要把端口暴露给不受信的人，先加鉴权（否则 §9.8 的排队策略形同虚设） |

**关键数据速查**

```
卡内天花板        5 session/卡
每会话 KV         8.05 MB/node = 64.4 MB/卡
4 会话总开销      ~283 MB/卡（与上下文填满与否无关）
满上下文并发      4 × 3894 token ✅
单会话纯 decode   12.38 tok/s（80.8 ms/tok），每卡 25.5% 利用率
并发聚合吞吐      N=1 10.73 → N=2 19.78 → N=3 28.63 → N=4 35.74 → N=5 34.94 tok/s
                  （同 workload 口径：每会话 4 轮 × 64 token）
并发增益          N=4 相对 N=1 为 3.33×（扩展效率 83%）
30 分钟稳定       6 轮 N=4 全 rc=0，41.26→41.62 tok/s 无衰减，首末轮 token 逐字节一致
交互式多会话      N=4 + 12 行输入 → 12 个 [s<i>] 块无穿插，四路各 3 轮，37.70 tok/s
TSan              两轮共 10 份报告：完成轮 52 race + 40 unlock，52/52 落在 SDK 的 16 MB
                  FreeListAllocator arena、一侧恒为 output_callback；单会话 0 竞争
                  控制组（ignore_noninstrumented_modules=1）→ 并发盘的报告全压到 0
                  **但要如实写：3/52 的写侧是我方 input_callback（写 SDK 给的缓冲）**
                  条数不稳定（同 workload 实测 14/10/12），稳定的是位置与形态
ASan              四阶段 0 报告（AS-1 黄金逐字节 / AS-2 = R11 T2 同 workload、4 次清 KV
                  且四路 dump 与 Release T2 dump 逐字节一致 / AS-3 52 块无穿插 / AS-4 无泄漏）
                  **阳性对照：堆越界、use-after-free、LSan 三项都报出来了** → 零报告是有意义的零
清 KV 卡锁        两处 worker 已补（§9.6）；补锁前后 token 逐字节一致（四路各 4812 行）
                  T3 soak 8 次清 KV、T2 post 4 次清 KV，均无穿插无崩溃
                  **TSan 补轮：TS-C3 清 4 次 @ 3765/4096、TS-D3 四路各清 1 次，无新竞争形态**
模型上下文        4096（固化，不可运行时调大）
HTTP 层并发伸缩   N=1 10.96 → N=2 20.21 → N=4 36.44 tok/s（3.33×；连量 4 轮 3.32~3.38，单请求延迟 8.8→10.5s）
                  对照官方 rkllm3-server 同板同条件：10.63 / 10.98 / 10.98（1.03×）
KV 复用（HTTP）   续聊 prefill 12 vs 全量 154 tok（开思考 0.08）/ 16 vs 158（关思考 0.10）
                  粘性轮 23 vs 75 tok，sent=118 < full=366、base=248
```

---

*文档版本：v1.13（2026-09-17）—— **Phase 1 工具调用 + 8192 上下文，真机验收**（§9.9）：工具调用的格式**不查文档、取自模型自己的 `tokenizer.chat_template`**（**XML，不是 Hermes JSON**），工具目录在**网关侧**拼，**后端二进制一行没动**（板上仍是 `b13bbdb` 的 `3355dc4f`）。新增板卡侧 `toolcall_check.py`——它问的全是**桩后端永远答不了**的问题；`check_template.py` 拿真 Jinja2 与网关拼出来的工具前言**逐字节对拍**。**桩上全绿、板上第一遍就抓到一条真 FAIL**（回显助手轮丢了 `tool_calls` → 前缀断 → 整段重算 469/469），修后用 A/B/C 变异测试证明检查不是空转。模型换成 **8192 导出**（判据 `rope_cos_cache [1,4,1,8192,16]`），5013 / 6853 token 的 prompt 都能从**中段**取回暗号。`serve/` 下 10 个文件与工作区**逐字节相同**——"板上跑的不是仓库 HEAD 副本"那段历史到此结束。顺带修掉 `restart_gateway.sh` 的 `pkill` 自匹配自杀*
*文档版本：v1.12（2026-09-16）—— **交叉编译 + 上板全套件验收**：把 v1.10 那个"未做 aarch64 交叉编译、未上板"的边界补掉。后端 `3355dc4f`（源码 `e0c49989`）+ 网关 `7a06e5ca` 一起下发，`run_all_board_tests.sh` **连跑两轮全绿**（FAIL 0 / 非零 rc 0 / traceback 0），伸缩 3.41× 与 3.38×（对照官方 server 1.03×）；§9.8 的身份/排队/`close`/`/v1/pool` **第一次在真后端上跑到**；`usage` 新口径在板上拿到非零 `cached_tokens`。**R14 在板上复现并验证修好**（`--bucket-size 64` + 跨分块的 prompt：改动前 `embed buffer too small` + `prefill failed`，改动后正常作答）——这是本批唯一一条板上"改动前 vs 改动后"对照，因为默认 128 下那条 bug 根本不触发。同时修掉一处**记录过期**：§8.1 网关那行的"当前 HEAD"在 `1dde346` 之后就错了*
*文档版本：v1.11（2026-09-16）—— **把构建服务器上那份未提交的 prefill 分块修复移植进仓库**（新增 R14）：运行时分块按模型的 max dynamic seq len（128）切，不看 `--bucket-size`，所以分块大小必须从 output tensor 反推，否则记账错位报 `embed buffer too small`。**默认 128 下不触发**，这就是它一直没被撞到的原因，也是它在一个检出上躺了很久没进仓库的原因*
*文档版本：v1.10（2026-09-16）—— **服务侧收尾加固**：新增 `REJECT` 帧（上下文将满改为"拒掉这一轮"，不再自动清 KV 把本轮跑完——那样模型会基于一段没有开头的对话给出一个自信的错答案）、会话线程退出清扫（不清扫则钉给死会话的活占满队列额度 → 输入线程与工作线程互等）、`max_new`/`session` 越界校验（不丢载荷会让帧流**永久**错位）、清 KV 收敛为单一实现；**并记录一次由执行发现的竞态**（`dispatcher_push` 等完队列空间后未重判目标会话存活 → 那条请求永远没有回帧，网关白等 1800s），以及本轮"把服务侧路径搬到主机上执行"的验证结果与它的边界（§9.7）。⚠️ 本轮**未做交叉编译、未上板**，板卡上跑的仍是旧二进制*
*文档版本：v1.9（2026-09-15）—— **多用户接入（把板卡当边缘服务器）**：新增 §9.8（身份显式化 / **去掉抢占改排队** / **`POST /v1/conversations/close` 主动交还** / 空闲回收 `IDLE_TTL` / `GET /v1/pool` 可观测（槽位带原始 `key`）/ 排队超时 503 / CORS 放行 `X-Conversation-Id`；`demo_multiuser.py` 终端客户端、`--close` 与 `--no-id` 两个对照）；**修订 §9.7 坑 4 的修法描述**（原文写"挑不出干净的才允许窃取"，窃取已删除，改为排队）；联动 §10（新增"多用户接入"行、R12 风险升级为"排队策略可被单个客户端作废"、下一步加两条）。**`close` 是删掉抢占之后才暴露出来的缺口**：抢占时代"不再提问的对话占着会话"无所谓，排队之后它会**永久占住**到 `IDLE_TTL`——所以"排队"要有意义就必须给客户端一条说"我走了"的路。**这个改动会改变所有现有调用方的行为**（`serve_http_test.py` 当场就被自己堵死了：它原先每问一次就换一段新对话），修法是让套件像真实客户端那样显式命名 + 用完 `close`，并在 `IDLE_TTL=0`（**关掉回收**）+ 4 会话下连跑两遍全绿来证明它靠的是自己 `close`。本轮验证**全部在本地桩后端**完成，**板卡上的真后端多人演示尚未跑过***；补 `README.md` 的多用户章节与 `DEMO.md` 第 6 步*
*v1.8（2026-09-15）—— **补演示层**：网页版 4 对话框演示（`demo_4chat.html`，网关自己用 `GET /demo` 发出；`OPTIONS` 预检 + `Access-Control-Allow-Origin/Expose-Headers`）、命令行演示 `demo_4session.py`、演示手册 `DEMO.md`、页面逻辑检查 `page_check.js`；§9.7 补"演示层产物"与 CORS 这条已知限制，并记录**加这一层之后重跑全量套件仍 FAIL 0、同轮伸缩 3.37×**（证明多出来的一层没有吃掉并发）*
*v1.7（2026-09-15）—— **修正 M5 的伸缩结论**：新增坑 4（会话租约算错 → 并发被偶发串行化；修复前同构建量到过 3.40× 也量到过 1.00×），因此 §0/§5/§6/§7(R13)/§9.7/§10 里的 **3.40× 全部改为修复后连量 4 轮的 3.33×（3.32~3.38）**，并写明"单轮数字不可作为结论"；补齐坑 4 的回归用例（修复前代码上复现失败）与租约落点这一诊断手段；顺带修掉 `run_all_board_tests.sh` 汇总块自引用 `grep` 的报错*
*v1.6（2026-09-15）—— **M5 服务化**（§9.7 新增：帧协议 / 网关设计 / HTTP 层伸缩 vs 官方 1.03× / 三个"只慢不错"的坑 / 不可复现性 / 未验证项）；**网关与板端脚本已入库 `examples/multicard/serve/`**（原先只在 `rt_work/` 而它要下发到板卡，违反了 §8.1 红线——按"要落到板卡上的东西属于仓库"纠正，路径全部参数化，入库后重新下发并重跑全量套件）；联动 §0(第 9 条) / §5(M5 指针) / §6(M5) / §7(R12/R13) / §8.1(快照表 + 红线复盘) / §10*
*v1.5（2026-09-15）—— **ASan 四阶段 + 阳性对照**（§9.3 新增）；**TSan 补轮**重写 §9.5（R11 清 KV 路径已覆盖 / 控制组 / 聚合统计 / **首轮"我方零竞争"结论的收窄**）；联动 §0(第 8 条) / §5(待确认项 + P2 欠账) / §7(R4/R6) / §8.1 / §9.4 / §10*
*v1.4（2026-09-15）—— R11 补锁记录：更新 §5(待确认项关闭) / §7(R11) / §8.1 / §9.6(新增) / §10*
*v1.3（2026-09-14）—— 依据板卡实测更新 §0(第 8 条) / §5(P3 完成记录 + P2 欠账状态) / §6(M3~M5) / §7(R4/R6/R10/R11) / §8.1 / §9.1~§9.5 / §10*
*基线：`examples/multicard/cpp/main.cc` @ `446cbdf`，md5 `ef943467048fd5f985a89e1ec725f248`；P3 后 `ff85aa377f91eb3b5e543dfa8754e4c6`（commit `075a7a4`）；**R11 补锁后 `c1fcc366958c34941b99c46c8f22cd62`（commit `9d2d167`，tag `r11-kvlock`）**；**M5 后 `108706c0c96c231f1632c489fc5963b3`（commit `6d0666a`，tag `m5-serve`）***
*v1.2（2026-09-14）—— §5(P2 完成记录) / §6 / §8.1 / §8.5 / §9*
*v1.0 中以下估算已被实测推翻，勿再引用：每卡 ~128 MB/会话（实测 64.4–70.7）、可用 8192 上下文（实测 4096）、「降 kvcache_len 提高并发」退路（无效）*
*v1.1 中以下预期需修正：N=2/3/4 理论值 24/36/48 tok/s 未达到（实测同口径 19.78/28.63/35.74），缺口来自 prefill 气泡与卡级串行排队，非实现缺陷——N=1 的每会话纯 decode 12.38 tok/s 与 P0 基线完全吻合可作为旁证*

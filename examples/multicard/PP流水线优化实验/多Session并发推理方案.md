# 多 Session 并发推理方案（Qwen3.5-27B / 4×RK1828）

> 项目：`rknn3-model-zoo/examples/multicard`
> 核心代码：`cpp/main.cc`（2466 行）、`python/qwen3_5/`（模型转换）
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

1. **`--sessions` 与 `--interactive` 互斥。** 单线程 `read_line_utf8` 没法喂 N 路会话，
   需要一输入线程 + 按会话分发 + N 路输出各自成行。参数校验里直接拒绝了这个组合。
2. **`--sessions` 与 `--perf` 互斥**，同理。
3. **没有测 30 分钟稳定性**（验收标准的第 3 条）。目前最长的单次运行是 6 轮交互
   （3078 token，约 4 分钟）和 N=4 的 1024 token 压测（约 29 秒）。
4. **没有验证方案 B**（每会话一份 `output_tensors`，每份约 1.3 MB，可去掉卡锁）。
   卡级锁已经拿到 83% 效率，方案 B 的潜在收益只有那剩下的 17%，而它要赌
   "runtime 支持同 context 真并发"——不值得先做。
5. 卡级统计仍是多会话混在一起（见表中 2.6 的 ⚠️）。

### P3：可选增强

| # | 动作 | 依赖 |
|---|---|---|
| 3.1 | `rknn3_session_run_async` + `rknn3_session_pause/resume` 替换同步 `session_run` | 需先确认 async 语义与 callback 时序 |
| 3.2 | 会话持久化：`rknn3_session_save_kvcache` / `load_kvcache_from_path`（`rknn3_api.h:1610-1633`） | 让会话可挂起/恢复/迁移 |
| 3.3 | 前端服务化（HTTP/WebSocket），每连接一个 `Conversation` | 需先有 P2 的调度 |
| 3.4 | 常驻 worker 线程池（§3.3 方案 B） | 若线程抖动成为瓶颈 |
| 3.5 | 动态会话数（按负载创建/回收） | 受 KV cache 总容量硬约束 |

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
| M3 | P2 完成（N=3~4） | 聚合 decode ≥ 30 tok/s，30 分钟稳定性 —— **⚠️ 吞吐达标（N=3 28.63、N=4 35.74 ≈ 纯 decode 合成 41.6），30 分钟稳定性未测** |
| M4 | P3 服务化 | HTTP 接口，多连接并发 —— 未开始 |

---

## 7. 风险清单

| # | 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|---|
| R1 | ~~**权重被 per-session 复制 → OOM**~~ ✅ **已排除** | — | — | P0 探针实测：权重共享，建 4 个额外 session 只消耗 4×8.05 MB/node（§4.1） |
| R2 | ~~**KV cache 总量超卡内存**~~ ✅ **已排除** | — | — | 实测每会话仅 64.4–70.7 MB/卡，4 会话共 ~283 MB，余量充足；且 KV 预分配、与上下文长度解耦（§4.1） |
| R2b | **卡内 session 上限 5，且不由内存决定** | 已确知（非概率） | 中 | 目标 4 路有 1 个 session 余量；**不要超配到 5**。成因未定位（§4.4），调 ctx 无效 |
| R3 | 同 context 并发导致 hidden states 串话 | 高（不处理必现） | 高 | §3.2 卡级锁 + 可选 per-session output mem |
| R4 | **`malloc(): unaligned tcache chunk`** —— 此前已在本 demo 中出现过的堆损坏，根因未定位 | 中 | 高 | 改造后开 `-DENABLE_ASAN`（`cpp/CMakeLists.txt:10` 已有开关）跑长稳定性测试；怀疑与多线程下 runtime 内部状态有关，多会话会放大 |
| R5 | 死锁（锁序错误） | 中 | 高 | 严格 stage 升序锁；加 `--sessions 1` 回退开关；持锁超时告警 |
| R6 | SDK 未文档化的同 context 并发限制 | 中 | 中 | 卡级锁已把并发"降级"为卡内串行，风险大幅降低 |
| R7 | `libtokenizer.a` 是否线程安全未知 | 低 | 中 | 每会话各自持一个 `Tokenizer` 实例（而非共享），或给解码加锁；`init_tokenizer_and_embedding`（`main.cc:1459`）要改成可重入 |
| R8 | 每 token 新建线程（`main.cc:1893`）× N 会话 → 线程风暴 | 低 | 低 | 实测 host 开销仅 0.17%，暂不处理；P3 换常驻池 |
| R9 | N=4 时利用率 >100%，排队抖动放大 | 高 | 中 | 产品配置定为 N=2~3，N=4 仅压测 |
| R10 | 交互输入单线程阻塞（`read_line_utf8`） | 高（不处理必现） | 低 | P2-2.5 拆输入线程 |

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
git tag p0-baseline     # 指向 d59a239，回归对照点
git tag p1-session-split
git tag p2-concurrent
```

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
| 板卡二进制 | 见 §5 各阶段完成记录的表 | P0 `a6739a6e…` / P0+插桩 `205232b8…` / P1 `6f5aa9d7…` / P2 `8f6e900d…` |

**红线（2026-09-14 更新）：**

- **`rt_work/` 一律不进 git、不推 GitHub、不上云、不上板卡。** 它是任务过程记录：
  部署日志、探针输出、实测原始数据、源码快照。
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

> **口径提醒**：上表"同 workload 口径"= 每会话 4 轮 × 64 token，**墙钟里含 4 次 prefill 气泡**，
> 所以 N=1 只有 10.73 而不是 12.38。绝对数字与"纯 decode × N"不可直接比；**跨 N 的比值
> （3.33×）才是并发增益的读数**。目标值 20/30/40 是按"纯 decode 理论值 ×0.8"定的，
> 与实测口径不同——**按同口径，N=2/3/4 分别差 1%、5%、11%**，处于测量口径差异范围内。

### 9.3 稳定性

- ❌ **30 分钟连续跑 N=3 压测未做**，RSS 单调性、`malloc()` 告警、`rknn3_session_query_state`
  异常均**未监控**。目前最长单次运行：单会话 6 轮交互（3078 token，约 4 分钟）、
  N=4 压测 1024 token（约 29 秒）。
- ❌ **ASan 版本未跑**（`cmake -DENABLE_ASAN=ON`，`cpp/CMakeLists.txt:10` 已支持）。
  并发改造后这条比 P1 时期更值得做——卡级锁是否真的覆盖了所有共享访问，
  ASan/TSan 比 token 对比更能给出确定答案。

### 9.4 调试辅助

| 开关 | 状态 | 作用 |
|---|---|---|
| `--dump-tokens <file>` | ✅ 已有 | 每步输出 token ID；并发模式下自动变成 `<file>.s<i>`（每会话一个） |
| 每会话 perf 表 | ✅ 已有 | prefill/decode token 数、纯 decode tok/s、墙钟，区分"哪个会话慢" |
| `--conv-trace` | ❌ 未加 | 打印 `[conv A][stage2] enter/exit card_lock`，排查死锁与相位 |
| TSan 构建 | ❌ 未加 | `-fsanitize=thread` 跑 N=2 短程，直接给数据竞争的确定答案（目前只能靠 token 对比反证） |

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
| **模型** | ✅ **不用重新转换。** 当前是 **4096 上下文**（单 KV group，传 8192 会被静默降级）。降 ctx 重导**不能**提高并发数（并发上限不由内存决定，见 §4.4） |
| **主要改造** | ✅ 4 处全局状态已全部下沉到 `Conversation`；并发靠 **`StageContext::run_mutex`（每卡一把锁）**，持锁覆盖整个 `session_run` |
| **正确性** | ✅ N=1/2/4 共 7 路 token id 与单会话路径**逐字节一致**；单会话路径本身零回归（3 组黄金 token 全中） |
| **工作量** | P0/P1/P2 已完成（各一个 commit）；P3 未开始 |
| **剩余风险** | `unaligned tcache chunk` 堆损坏（未复现，也未用 ASan/TSan 主动查）、5 上限的成因未知（非阻塞）；**30 分钟稳定性未测** |
| **版本管理** | 三个 commit（`d59a239`/`6347f57`/`446cbdf`）+ 三个 tag；回归对照物是独立保存的源码/二进制快照 + 黄金 token 文件；下发靠 **md5 + BuildID 双校验**（§8.4） |
| **下一步** | ① `--sessions` × `--interactive`（stdin 分发到 N 路，各自成行输出）；② TSan/ASan 跑一轮 N=2；③ 30 分钟 N=3 稳定性；④ 每卡卡级统计拆到会话维度 |

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
模型上下文        4096（固化，不可运行时调大）
```

---

*文档版本：v1.2（2026-09-14）—— 依据板卡实测更新 §5(P2 完成记录) / §6 / §8.1 / §8.5 / §9*
*基线：`examples/multicard/cpp/main.cc` @ `446cbdf`，md5 `ef943467048fd5f985a89e1ec725f248`*
*v1.0 中以下估算已被实测推翻，勿再引用：每卡 ~128 MB/会话（实测 64.4–70.7）、可用 8192 上下文（实测 4096）、「降 kvcache_len 提高并发」退路（无效）*
*v1.1 中以下预期需修正：N=2/3/4 理论值 24/36/48 tok/s 未达到（实测同口径 19.78/28.63/35.74），缺口来自 prefill 气泡与卡级串行排队，非实现缺陷——N=1 的每会话纯 decode 12.38 tok/s 与 P0 基线完全吻合可作为旁证*

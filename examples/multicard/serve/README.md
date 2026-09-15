# 板端 OpenAI 兼容服务（多会话）

给工作流 Agent 用的 HTTP 接口，跑在板卡上，后端是 `cpp/main.cc --serve` 的**多会话并发执行器**。

> **要给人演示**（而不是自己用）：直接看 **[`DEMO.md`](DEMO.md)**——那是给第一次接触项目的人写的
> 完整操作流程，含每一步"观众该看到什么"和排障表。

**设计说明、实测数据与全部已知限制**在 `../PP流水线优化实验/多Session并发推理方案.md` 的 §9.7，
这里只讲怎么用。一句话版本：官方 `rkllm3-server` 的 OpenAI 接口是现成的，但按设计
「推理执行排队串行」（同板实测 1.03×）；这份门面复用自己的执行器，同板同条件 **3.33×**（连量 4 轮 3.32~3.38）。

## 组成

| 文件 | 作用 |
|---|---|
| **`DEMO.md`** | **演示手册**：给第一次接触项目的人，从零到跑通一次 4 路并发演示的完整流程 |
| **`demo_4session.py`** | 4 路并发的"看得见"演示：4 行实时进度条一起往前爬（**展示时用这个**） |
| `rkllm_gateway.py` | 网关本体（python3 **标准库**，无第三方依赖）。父进程，自己拉起后端子进程 |
| `start_gateway.sh` | 启动网关（含后端子进程）；参数走环境变量 |
| `restart_gateway.sh` | 杀掉重启（含两个测试陷阱的注释，别改成命令行一行流） |
| `build_serve.sh` | 在构建机上交叉编译出 `rknn_multicard_demo.serve`（Release） |
| `run_all_board_tests.sh` | 板上跑完全部验收测试，失败项集中列在最后 |
| `check_template.py` | 网关渲染的 prompt 与 `main.cc` 模板**逐字节**比对（见下） |
| `serve_http_test.py` | 接口面：health / models / 非流式 / SSE / 多轮 KV 复用 / 并发 |
| `sticky_check.py` | 粘性复用的**语义**正确性（暗号靠 KV 存活 + prefill 计数） |
| `nothink_check.py` | 关思考（`enable_thinking=false`）下的粘性复用回归 |
| `http_scaling.py` | HTTP 层并发伸缩（自研门面存在的唯一理由） |
| `determinism_probe.py` | 记录"贪婪解码不可逐字节复现"这个事实，供验收标准参考 |
| `fake_backend.py` | 假后端，本地无板卡时跑网关回归（Windows 上用它，`pass_fds` 不可用） |

## 起服务

```sh
# 1) 构建（在 x86 构建机上，交叉编译）
bash build_serve.sh                       # 产物默认落在仓库根 rknn_multicard_demo.serve

# 2) 把 rkllm_gateway.py / start_gateway.sh / restart_gateway.sh / rknn_multicard_demo.serve
#    放到板卡的同一个目录（下称 $GATEWAY_DIR）

# 3) 启动（模型加载 ~240s）
cd "$GATEWAY_DIR"
INSTALL_DIR=<安装目录>/rknn_multicard_demo \
MODEL_DIR=<模型目录> \
./start_gateway.sh
```

环境变量（都有默认值，见脚本头部注释）：`INSTALL_DIR`（含后端二进制与 `lib/`）、
`MODEL_DIR`、`GATEWAY_DIR`、`B`（后端二进制名）、`NSESSION`（默认 4）、`PORT`（默认 8080）、
`NP`（每轮 max_new_tokens 默认值）、`HOST`（默认 `0.0.0.0`）、`LOG`。

## 接口

- `GET /health` → `{"status":"ok","model":...,"sessions":N,"uptime_s":...}`
- `GET /v1/models` → OpenAI 格式的模型列表
- `POST /v1/chat/completions` → 兼容 OpenAI；`stream: true` 走 SSE，`stream: false` 一次性返回

```sh
# 非流式
curl -s http://<板卡>:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.5-27b",
  "messages": [{"role":"user","content":"1+1=?"}],
  "max_tokens": 64 }'

# 流式 + 关思考（软开关：网关会给每条 user 贴上 /no_think，并把 <think> 段摘掉）
curl -N -s http://<板卡>:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.5-27b",
  "messages": [{"role":"user","content":"1+1=?"}],
  "stream": true,
  "chat_template_kw": {"enable_thinking": false} }'
```

两类扩展：

- **多轮会话粘性**：`messages` 里带上完整历史即可，网关按历史自动把同一段对话钉在同一个会话上，
  跨轮只给后端发**差异部分**（KV 复用）。想强制不复用就换一段全新的对话（首条 user 内容不同）。
- **`X-KV-Reuse` 响应头**：`session=<i>; reset=<0|1>; sent=<n>; base=<n>; full=<n>`——
  复用有没有生效只看这里。**答案正确但慢 3~4 倍**这类问题，正确性测试抓不到，全靠这个头。

## 已知限制（完整清单见方案文档 §9.7）

- **没有鉴权**。`HOST=0.0.0.0` 时局域网内谁能连上谁就能用满全部算力；Agent 跑在板卡本机时
  建议 `HOST=127.0.0.1`。要对外提供服务请自行加反向代理鉴权。
- `tool` 角色的渲染**未测**（`render_messages` 只处理 system/user/assistant）——接 function
  calling 之前要先补这个测试。
- `finish_reason` 是**反推值**（`decode_tok >= max_new_tokens` → `length`，否则 `stop`）。
- 开思考（默认）时推理过程**原样透传**，不单独成字段（协议里没有 `reasoning_content`）。
- 没有 `--no-sticky` 回退开关。
- 同一 prompt 重复请求**不保证逐字节相同**（贪婪解码在本栈上不可复现），验收标准必须是语义级的。

## 测试

```sh
# 板上（网关已起）
./run_all_board_tests.sh http://127.0.0.1:8080      # 结果落 board_tests.log，末尾有 FAIL 汇总

# 本地（无板卡，用假后端；Windows 上 pass_fds 不可用，靠 --frames-stdout）
python3 rkllm_gateway.py --selftest
python3 fake_backend.py --frames-stdout &
python3 serve_http_test.py http://127.0.0.1:8080 quick
python3 check_template.py
```

`check_template.py` 守的是一条**必须守住的不变量**：服务模式下 prompt 由网关渲染、后端逐字节透传，
所以两边渲染必须**逐字节一致**，否则同一段对话在 `--interactive` 与服务两条前端上的行为会悄悄分叉。

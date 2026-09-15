#!/bin/bash
# 板卡上启动 OpenAI 网关。网关是父进程，自己拉起 rknn_multicard_demo --serve 作为
# 子进程（用管道把帧通道 fd 交给它），所以只有一个进程要管。
#
# 为什么不用官方 rkllm3-server：它的 OpenAI 接口是现成的，但文档 §4.5.2 写明
# 「推理执行排队串行」，实测 --n-session 1/2/4 聚合都是 ~11 tok/s（1.03x）——slot
# 只提供隔离、不提供并行。我们自己的执行器在同一硬件上是 3.33x，所以 HTTP 自己写。
#
# 命令行沿用验收时用的那一套（taskset f0 + LD_LIBRARY_PATH=./lib +
# --core-mask 0xff --stage-count 4 --bucket-size 128 --ctx-size 4096），不要改动，
# 否则实测数字不再可比。多会话方案的实测记录见
# `../PP流水线优化实验/多Session并发推理方案.md` §9.7（含与官方 server 的对照）。
#
# 环境变量：
#   INSTALL_DIR  安装目录（含 rknn_multicard_demo 与 lib/）；默认取仓库里的默认安装路径
#   MODEL_DIR    模型目录（*.rknn / *.weight / *.tokenizer.gguf / *.embed.bin / *.safetensors）
#   GATEWAY_DIR  本脚本与网关所在目录；默认 = 脚本自己所在目录
#   B            后端二进制名；默认 ./rknn_multicard_demo.serve（由 build_serve.sh 产出）
#   NSESSION     会话数 = **同时活跃的对话数上限**（板卡硬上限实测 5）；默认 4
#   IDLE_TTL     一段对话静默超过这么多秒就把它占的会话收回给排队者；默认 300（0=不回收）
#   QUEUE_TIMEOUT 取不到会话时最多排队等这么多秒，超了返回 503；默认 600（要 > IDLE_TTL）
#   PORT         监听端口；默认 8080
#   NP           每轮 max_new_tokens 的进程默认值；默认 512
#   HOST         监听地址；默认 0.0.0.0 = 局域网可达
#   LOG          后端日志路径；默认 $GATEWAY_DIR/gateway_backend.log
#
# 关于 HOST：默认 0.0.0.0 是为了让工作流 Agent 能从别的机器/容器连进来（官方 server
# 也是这么暴露的）。如果 Agent 就跑在板卡本机，把它设成 127.0.0.1 更稳妥——这个网关
# 没有任何鉴权，谁能连上谁就能用满 4 张卡的算力。
#
# 用法（板上）：
#   INSTALL_DIR=.../install-Qwen/rk3588_linux_aarch64/rknn_multicard_demo \
#   MODEL_DIR=.../Qwen3.5-27B ./start_gateway.sh
set -u
GATEWAY_DIR=${GATEWAY_DIR:-$(cd "$(dirname "$0")" && pwd)}
# 默认按仓库根目录的相对位置找：仓库里 `build-linux.sh -d multicard` 的安装产物。
REPO_ROOT=${REPO_ROOT:-$(cd "$GATEWAY_DIR/../../.." && pwd)}
INSTALL_DIR=${INSTALL_DIR:-$REPO_ROOT/install-Qwen/rk3588_linux_aarch64/rknn_multicard_demo}
MODEL_DIR=${MODEL_DIR:-$REPO_ROOT/Qwen3.5-27B}
B=${B:-./rknn_multicard_demo.serve}
NSESSION=${NSESSION:-4}
IDLE_TTL=${IDLE_TTL:-300}
QUEUE_TIMEOUT=${QUEUE_TIMEOUT:-600}
PORT=${PORT:-8080}
NP=${NP:-512}
HOST=${HOST:-0.0.0.0}
LOG=${LOG:-$GATEWAY_DIR/gateway_backend.log}

cd "$INSTALL_DIR" || { echo "cannot cd $INSTALL_DIR (set INSTALL_DIR)"; exit 1; }
[ -d "$MODEL_DIR" ] || { echo "no such model dir: $MODEL_DIR (set MODEL_DIR)"; exit 1; }
export LD_LIBRARY_PATH=./lib

exec taskset f0 python3 "$GATEWAY_DIR/rkllm_gateway.py" \
  --host "$HOST" --port "$PORT" \
  --sessions "$NSESSION" --verbose \
  --idle-ttl "$IDLE_TTL" --queue-timeout "$QUEUE_TIMEOUT" \
  --backend-log "$LOG" \
  -- "$B" \
     --model "$MODEL_DIR/Qwen3.5-27B-llm_seg0.rknn" \
     --weight "$MODEL_DIR/Qwen3.5-27B-llm_seg0.weight" \
     --vocab "$MODEL_DIR/Qwen3.5-27B-llm.tokenizer.gguf" \
     --embed "$MODEL_DIR/Qwen3.5-27B-llm.embed.bin" \
     --ctx-size 4096 --core-mask 0xff --stage-count 4 --bucket-size 128 \
     --rope-tensor "$MODEL_DIR/Qwen3.5-27B-llm_seg0.safetensors" \
     --sessions "$NSESSION" --serve -n "$NP"

#!/bin/bash
# 重启网关（含后端子进程）。模型加载 ~240s，之后才能收请求。
#
# 为什么单独写成脚本，而不是在命令行里串一串：`pkill -f rkllm_gateway.py` 会**匹配到
# 执行它的那条命令本身**——远端 shell 的 cmdline 里就含这个字符串，于是 pkill 把
# 自己所在的 shell 一起杀了，后面的 rm / 重启全都没执行。现场表现极具误导性：
# 进程没了、日志还是上一次的、看起来像"网关自己崩了"。用 [.] 让模式匹配不到自己
# （正则 [.] 只匹配字面点，而命令行里的文本 'rkllm_gateway[.]py' 不匹配该正则）。
#
# 环境变量：GATEWAY_DIR（默认 = 脚本自己所在目录）、LOG。其余转发给 start_gateway.sh。
# 用法（板上）：INSTALL_DIR=... MODEL_DIR=... ./restart_gateway.sh
set -u
GATEWAY_DIR=${GATEWAY_DIR:-$(cd "$(dirname "$0")" && pwd)}
LOG=${LOG:-$GATEWAY_DIR/gateway.log}
pkill -f 'rkllm_gateway[.]py' && sleep 2
pkill -f 'rknn_multicard_demo' && sleep 3
rm -f "$LOG" "$GATEWAY_DIR/gateway_backend.log"
cd "$GATEWAY_DIR" || exit 1
setsid nohup ./start_gateway.sh < /dev/null > "$LOG" 2>&1 &
echo "gateway restarting (model load ~240s); watch $LOG"

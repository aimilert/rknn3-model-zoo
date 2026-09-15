#!/bin/bash
# 板卡上一次性跑完全部验收测试，结果落盘到 board_tests.log。
#
# 为什么要有这么个脚本：这些测试要么很慢（一次模型加载 ~240s，一轮并发几分钟），要么
# 必须**按顺序**跑（http_scaling 会把 4 个会话都弄脏，正好是后面粘性测试需要的初态）。
# 手工一条条敲既慢又容易漏测项；一条命令全跑完、失败项集中列在最后。
#
# 前置：网关已在跑（./start_gateway.sh 或 ./restart_gateway.sh）。
# 用法：./run_all_board_tests.sh [base_url]
#   环境变量：GATEWAY_DIR（默认 = 脚本自己所在目录）、LOG（默认 $GATEWAY_DIR/board_tests.log）
set -u
BASE=${1:-http://127.0.0.1:8080}
GATEWAY_DIR=${GATEWAY_DIR:-$(cd "$(dirname "$0")" && pwd)}
R=${LOG:-$GATEWAY_DIR/board_tests.log}
cd "$GATEWAY_DIR" || exit 1
: > "$R"

run() {
  echo "########## $* ##########" >> "$R"
  local t0 t1
  t0=$(date +%s)
  "$@" >> "$R" 2>&1
  echo "  -> rc=$? 用时 $(( $(date +%s) - t0 ))s" >> "$R"
  echo >> "$R"
}

run python3 check_template.py
run python3 serve_http_test.py "$BASE"
run python3 sticky_check.py "$BASE"
run python3 nothink_check.py "$BASE"
# 伸缩测量放最后：它会把 4 个会话都写脏
run python3 http_scaling.py "$BASE" 96 1,2,4

# 注意：先把统计结果抓进变量，最后才 >> 追加。直接在这里 `grep "$R"` 又 `>> "$R"`
# 会让 grep 的输入文件同时是输出文件（实测报 "input file is also the output"）。
NFAIL=$(grep -c '\[FAIL\]' "$R" || true)
FAILLINES=$(grep -n '\[FAIL\]' "$R" || true)
{
  echo "===== 汇总 ====="
  echo "FAIL 行数：$NFAIL"
  if [ -n "$FAILLINES" ]; then echo "$FAILLINES"; else echo "（无失败项）"; fi
} >> "$R"
echo "done; see $R"

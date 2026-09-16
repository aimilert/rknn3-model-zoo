#!/bin/bash
# 在 Linux x86 构建机上交叉编译 --serve 版本的 demo（**Release，不带任何 sanitizer**）。
#
# 为什么要单独写一个 Release 脚本：sanitizer 构建是 Debug + -fsanitize=thread，
# 出来的二进制慢一个量级，用它测吞吐等于白测。板上验收过的 3.33x 是 Release 的数，
# 网关这边的聚合吞吐必须和它同口径才能比。
#
# 为什么先 rm 掉 Release 构建目录：CMake 缓存里带着上次 configure 的编译选项，
# CXXFLAGS 覆盖不掉缓存（被这个坑过）。宁可多花几分钟全量重编。
#
# 环境变量：REPO_DIR（仓库根目录，默认 = 脚本上两级再上一级推断）、OUT（产物另存路径）。
# 用法：bash build_serve.sh
#
# 为什么必须 `set -e`（2026-09-16 修）：这个脚本结尾会把 $B 拷成 rknn_multicard_demo.serve，
# 而 start_gateway.sh 用的就是这个文件。以前没有 set -e，编译器报错之后脚本照跑到底：
# `ls -l $B` 列的是**上一轮的旧二进制**（构建失败时它还在原地），`cout`/`grep` 也照样
# 打印数字，最后 `cp` 把旧二进制覆盖到 .serve 上，并打出一行 `SERVE BUILD DONE`。
# 也就是说"构建失败"和"构建成功"的输出长得一模一样，而部署出去的是旧代码。
# 光靠 set -e 还不够——它管不到"命令返回 0 但结果是错的"，所以下面的产物检查改成
# 显式断言（时间戳必须晚于本次构建开始、必须没有 sanitizer 依赖），不再只打印不断言。
set -eu
set -x
HERE=$(cd "$(dirname "$0")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd "$HERE/../../.." && pwd)}
cd "$REPO_DIR" || exit 1
echo "=== main.cc md5 ==="
md5sum examples/multicard/cpp/main.cc

# 记下"本次构建开始"的时刻：产物必须比它新，否则说明这次根本没重新链接成功。
STAMP=$(mktemp)
trap 'rm -f "$STAMP"' EXIT

rm -rf build/build_rknn_multicard_demo_rk3588_linux_aarch64_Release
unset CFLAGS CXXFLAGS LDFLAGS
export GCC_COMPILER=aarch64-linux-gnu
./build-linux.sh -t rk3588 -a aarch64 -d multicard -b Release

B=install/rk3588_linux_aarch64/rknn_multicard_demo/rknn_multicard_demo
echo "=== 产物 ==="
ls -l "$B"
md5sum "$B"

echo "=== 检查 1/3：产物是本次构建新生成的 ==="
if [ ! -f "$B" ] || [ ! "$B" -nt "$STAMP" ]; then
  echo "**构建没有产出新的 $B —— 编译器应该已经报错了，往上翻。**" >&2
  exit 1
fi

echo "=== 检查 2/3：动态依赖不得出现 libasan/libtsan ==="
NEEDED=$(aarch64-linux-gnu-readelf -d "$B" | grep -i 'NEEDED' || true)
echo "$NEEDED"
if echo "$NEEDED" | grep -qiE 'lib(asan|tsan|ubsan)'; then
  echo "**产物链上了 sanitizer 运行时；吞吐数字会失真，不能用。**" >&2
  exit 1
fi

echo "=== 检查 3/3：用法行带上了本次的改动（--serve-fd / REJECT）==="
for pat in '--serve-fd' 'REJECT'; do
  n=$(strings "$B" | grep -c -- "$pat" || true)
  echo "  '$pat' 出现 $n 次"
  if [ "$n" -eq 0 ]; then
    echo "**产物里找不到 '$pat'：装的不是这一次的 main.cc。**" >&2
    exit 1
  fi
done

echo "=== 另存一份（名字与 start_gateway.sh 的默认 B 一致）==="
OUT=${OUT:-$REPO_DIR/rknn_multicard_demo.serve}
cp "$B" "$OUT"
md5sum "$OUT"
echo "=== SERVE BUILD DONE ==="

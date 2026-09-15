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
set -x
HERE=$(cd "$(dirname "$0")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd "$HERE/../../.." && pwd)}
cd "$REPO_DIR" || exit 1
echo "=== main.cc md5 ==="
md5sum examples/multicard/cpp/main.cc

rm -rf build/build_rknn_multicard_demo_rk3588_linux_aarch64_Release
unset CFLAGS CXXFLAGS LDFLAGS
export GCC_COMPILER=aarch64-linux-gnu
./build-linux.sh -t rk3588 -a aarch64 -d multicard -b Release

B=install/rk3588_linux_aarch64/rknn_multicard_demo/rknn_multicard_demo
echo "=== 产物 ==="
ls -l "$B"
md5sum "$B"
echo "=== 动态依赖（不得出现 libasan/libtsan）==="
aarch64-linux-gnu-readelf -d "$B" | grep -i 'NEEDED'
echo "=== 确认带上了新用法行 ==="
strings "$B" | grep -c -- '--serve-fd'
echo "=== 另存一份（名字与 start_gateway.sh 的默认 B 一致）==="
OUT=${OUT:-$REPO_DIR/rknn_multicard_demo.serve}
cp "$B" "$OUT"
md5sum "$OUT"
echo "=== SERVE BUILD DONE ==="

#!/bin/bash
# 在**板卡上原生编译** --serve 版 demo。产出与 `build_serve.sh`（交叉编译）是同一件东西。
#
# 为什么要有它：租的交叉编译机到期之后，板卡成了唯一的构建机——本仓库 2026-09-17 那次
# 改动（REJECT 判据 + 帧上限不再打死输入线程）就是这么编出来的，二进制 md5 也因此和
# 交叉编译那版对不上。它不需要 ARM 工具链、也不需要完整 SDK 源码树，只要四样：
#   · main.cc（工作区那份原件，逐字节拷过来；**别在板上改代码**，板上没有 git）
#   · 三个头：Tokenizer.h / float16.h / rknn3_api.h，以及 nlohmann/json.hpp
#   · libtokenizer.a（静态链进去）
#   · 链接期找得到的 librknn3_api.so，运行时靠 rpath `$ORIGIN/lib` 找
#
# 优化选项抄的是工程的 Release 口径（`tokenizer/CMakeLists.txt` 里 CMAKE_CXX_FLAGS_RELEASE
# = `-O2 -s`）。**不要在这里自作主张换 -O3 之类**：这套 demo 的吞吐数字（10.97 / 20.22 /
# 37.13 tok/s 那一组）都是 Release 口径，换了优化级别就不可比了。
#
# 环境变量：
#   NATIVE_DIR   放 main.cc / include/ / json/ / libtokenizer.a 的目录；默认 = 脚本自己所在目录
#   INSTALL_DIR  装了 lib/librknn3_api.so 的 SDK 目录（也是产物的默认安装位置）
#   OUT          产物路径；默认 $INSTALL_DIR/rknn_multicard_demo.serve
#   KEEP_BAK     1 = 装之前把同名旧产物另存成 .bak_<md5>（默认 1）
#
# 用法（板上）：
#   INSTALL_DIR=.../install-Qwen/rk3588_linux_aarch64/rknn_multicard_demo \
#   NATIVE_DIR=.../native_build bash build_serve_native.sh
#
# 与 build_serve.sh 同样的道理，这里也必须 `set -e` 且**对产物做断言**：只打印不判断的话，
# "编译失败"和"编译成功"的输出长得一模一样，而装上去的是上一轮的旧二进制。
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
NATIVE_DIR=${NATIVE_DIR:-$HERE}
INSTALL_DIR=${INSTALL_DIR:?set INSTALL_DIR to the SDK dir that has lib/librknn3_api.so}
OUT=${OUT:-$INSTALL_DIR/rknn_multicard_demo.serve}
KEEP_BAK=${KEEP_BAK:-1}
SRC=$NATIVE_DIR/main.cc

[ -f "$SRC" ] || { echo "**找不到 $SRC**（NATIVE_DIR=$NATIVE_DIR）" >&2; exit 1; }
[ -f "$NATIVE_DIR/libtokenizer.a" ] || { echo "**找不到 $NATIVE_DIR/libtokenizer.a**" >&2; exit 1; }
[ -f "$INSTALL_DIR/lib/librknn3_api.so" ] || {
  echo "**$INSTALL_DIR/lib/librknn3_api.so 不在**：INSTALL_DIR 指错了？" >&2; exit 1; }

STAMP=$(mktemp)
trap 'rm -f "$STAMP"' EXIT
TMP=$NATIVE_DIR/out/rknn_multicard_demo.native.$$
mkdir -p "$NATIVE_DIR/out"
trap 'rm -f "$STAMP" "$TMP"' EXIT

echo "=== main.cc md5 ==="
md5sum "$SRC"

echo "=== 编译（-O2 -s，工程 Release 口径）==="
g++ -std=c++11 -O2 -s -pthread -o "$TMP" "$SRC" \
    -I"$NATIVE_DIR/include" -I"$NATIVE_DIR/json" \
    -L"$INSTALL_DIR/lib" -lrknn3_api \
    "$NATIVE_DIR/libtokenizer.a" -ldl \
    -Wl,-rpath,'$ORIGIN/lib'

echo "=== 检查 1/3：产物是本次构建新生成的 ==="
if [ ! -f "$TMP" ] || [ ! "$TMP" -nt "$STAMP" ]; then
  echo "**没有产出新的 $TMP —— 编译器应该已经报错了，往上翻。**" >&2
  exit 1
fi

echo "=== 检查 2/3：动态依赖不得出现 libasan/libtsan ==="
NEEDED=$(readelf -d "$TMP" | grep -i 'NEEDED' || true)
echo "$NEEDED"
if echo "$NEEDED" | grep -qiE 'lib(asan|tsan|ubsan)'; then
  echo "**产物链上了 sanitizer 运行时；吞吐数字会失真，不能用。**" >&2
  exit 1
fi

echo "=== 检查 3/3：用法行与本次的改动都在里面 ==="
for pat in '--serve-fd' 'REJECT' 'prompt too long'; do
  n=$(strings "$TMP" | grep -c -- "$pat" || true)
  echo "  '$pat' 出现 $n 次"
  if [ "$n" -eq 0 ]; then
    echo "**产物里找不到 '$pat'：装的不是这一份 main.cc。**" >&2
    exit 1
  fi
done

echo "=== 安装到 $OUT ==="
if [ -f "$OUT" ] && [ "$KEEP_BAK" = 1 ]; then
  BAK="$OUT.bak_$(md5sum "$OUT" | cut -c1-8)"
  cp "$OUT" "$BAK"
  echo "  旧产物另存：$BAK"
fi
cp "$TMP" "$OUT"
md5sum "$OUT"
echo "=== SERVE NATIVE BUILD DONE（重启网关才会用上：./restart_gateway.sh）==="

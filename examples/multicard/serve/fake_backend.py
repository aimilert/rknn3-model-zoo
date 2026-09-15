#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rkllm_multicard_demo --serve 的桩后端：只讲帧协议，不做推理。

存在的理由：板卡上一次模型加载 ~230-240s，而网关这半边（分帧、增量 UTF-8 解码、
会话池、模板渲染、并发写不交错）全都是纯 Python 逻辑，本地就能验。把桩后端接在
网关后面跑 `--selftest`，就能在**没有任何板卡参与**的情况下把这一半清零，剩下的
问题必然出在 C++ 侧或硬件上。

它刻意比真后端更"刻薄"：
  · DELTA 按固定字节数切，**故意把一个多字节字符切成两半**投递——网关必须用增量
    解码器才能拼回原字，逐帧 decode 会吐出替换字符；
  · 每个请求在独立线程里回答（带一点延迟），这样 N 路并发会同时往 stdout 写帧，
    网关那边如果没把 stdin 的帧写整体加锁，这里的解析器立刻会解析失败；
  · 头行解析不了一律 rc=2 退出，让失败显式暴露，而不是静默丢帧。

用法（由网关自动拉起，一般不手跑）：
    python3 fake_backend.py --sessions 2 [--serve-fd N]
"""
import argparse
import os
import sys
import threading
import time

REPLY = ("你好，我是一个跑在 4 片 RK1828 上的 Qwen3.5-27B，"
         "由多会话并发执行器驱动。这句话里混了中文、数字 12345 和 ASCII 混排。")
CTX_LIMIT = 4096


def emit(out, tag, head, payload=b""):
    """一帧一次性写完整（真后端用 g_output_mutex 做同样的事）。"""
    with emit.lock:
        out.write(("%s%s\n" % (tag, head)).encode("utf-8"))
        if payload:
            out.write(payload)
        out.flush()


emit.lock = threading.Lock()


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--sessions", type=int, default=2)
    ap.add_argument("--serve-fd", type=int, default=None)
    ap.add_argument("--default-n", type=int, default=512)
    ap.add_argument("--delay", type=float, default=0.30)
    # 让回复像真模型那样以 `<think>  </think>  ` 开头（关思考时的真实形态），用来端到端
    # 验网关的 ThinkStripper：它必须把这段摘掉，而且**标签被 7 字节切分切开时也要摘对**。
    ap.add_argument("--think-prefix", action="store_true")
    args, _unknown = ap.parse_known_args()

    if args.serve_fd is not None:
        out = os.fdopen(args.serve_fd, "wb")
    else:
        # 桩模式下网关把帧从我们的 stdout 读（Windows 上 pass_fds 不可用）
        out = sys.stdout.buffer

    emit(out, "READY ", "%d %d" % (args.sessions, args.default_n))
    sys.stderr.write("[stub] ready: sessions=%d default_n=%d\n" % (args.sessions, args.default_n))
    sys.stderr.flush()

    ctx = [0] * args.sessions
    ctx_lock = threading.Lock()
    workers = []

    def serve(rid, session, max_new, reset, prompt):
        s = session if 0 <= session < args.sessions else 0
        with ctx_lock:
            if reset and ctx[s] > 0:
                emit(out, "CLEAR ", "%d %d %d %d" % (rid, s, ctx[s], CTX_LIMIT))
                ctx[s] = 0
        prefill = max(1, len(prompt.encode("utf-8")) // 5)
        time.sleep(args.delay)
        # 切得要"不整齐"：把多字节字符也切开，逼网关用增量解码器
        reply = ("<think>  </think>  " + REPLY) if args.think_prefix else REPLY
        raw = reply[:max_new].encode("utf-8")
        step = 7
        for i in range(0, len(raw), step):
            emit(out, "DELTA ", "%d %d" % (rid, min(step, len(raw) - i)), raw[i:i + step])
        time.sleep(0.05)
        with ctx_lock:
            ctx[s] += prefill + len(raw)
            after = ctx[s]
        finish = "length" if len(REPLY) <= max_new else "stop"
        emit(out, "DONE ", "%d %d %s %d %d %.1f %.1f %d"
             % (rid, s, finish, prefill, min(len(REPLY), max_new),
                prefill * 50.0, min(len(REPLY), max_new) * 90.0, after))

    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            break
        line = line.rstrip(b"\r\n")
        if not line:
            continue
        if line == b"QUIT":
            break
        try:
            parts = line.split()
            if parts[0] != b"REQ" or len(parts) != 6:
                raise ValueError("bad header: %r" % (line,))
            rid = int(parts[1])
            session = int(parts[2])
            max_new = int(parts[3])
            reset = int(parts[4])
            n = int(parts[5])
            payload = sys.stdin.buffer.read(n) if n > 0 else b""
            if len(payload) != n:
                raise ValueError("short read: want %d got %d" % (n, len(payload)))
        except Exception as exc:
            # 头行解析不了 = 帧流错位了（多半是网关那边写 stdin 没整体加锁，
            # 两条线程的头行和载荷互相插了）。显式炸掉，不要静默丢帧。
            sys.stderr.write("[stub] FRAME STREAM CORRUPT: %r\n" % (exc,))
            sys.stderr.flush()
            emit(out, "ERR ", "0 - %d" % len(str(exc)), str(exc).encode("utf-8"))
            return 2
        th = threading.Thread(target=serve,
                             args=(rid, session, max_new, reset,
                                   payload.decode("utf-8", "replace")))
        th.start()
        workers.append(th)

    for th in workers:
        th.join()
    return 0


if __name__ == "__main__":
    sys.exit(main())

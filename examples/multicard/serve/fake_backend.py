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
    python3 fake_backend.py --sessions 2 [--serve-fd N] [--think-prefix] [--tool-call]
"""
import argparse
import os
import sys
import threading
import time

REPLY = ("你好，我是一个跑在 4 片 RK1828 上的 Qwen3.5-27B，"
         "由多会话并发执行器驱动。这句话里混了中文、数字 12345 和 ASCII 混排。")
# --tool-call 时的固定调用：一个字符串参数 + 一个数字参数。数字那个是刻意的——解析侧会
# 把它转成 Python 的 int，渲回 prompt 时如果不用原文，`3` 这轮还能对上、`true`/`1.50`
# 那种就对不上了（见 toolcalls.render_assistant_turn）。
TOOL_CALL = ("\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\n北京\n"
             "</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>")
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
    # 让桩也能造出"上下文装不下"那条路径（真后端在 prefill **之前**判，发 CLEAR+REJECT，
    # **不**标死会话）。0 = 不限。有它才能把网关那半边（软错误 / 不标死 / 作废粘性 / 不
    # 把这轮 prompt 记进 KV 账）在本地测到，否则那四条只有板卡上把上下文撑满才能验。
    ap.add_argument("--ctx-limit", type=int, default=0)
    # 让桩先回一个工具调用、拿到 tool 结果之后再正常回答，好把"请求 -> 注入工具说明 ->
    # 生成 -> 摘出调用 -> OpenAI 形态响应 -> 回灌 tool 结果 -> 再问一轮"整条回路在没有
    # 板卡的情况下走通。判据是 prompt 里有没有 <tool_response>（由对话本身决定，不需要
    # 桩自己记状态），所以也顺带验了网关把 tool 结果正确渲进了下一轮 prompt。
    ap.add_argument("--tool-call", action="store_true")
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
        prefill = max(1, len(prompt.encode("utf-8")) // 5)
        with ctx_lock:
            if reset and ctx[s] > 0:
                emit(out, "CLEAR ", "%d %d %d %d" % (rid, s, ctx[s], CTX_LIMIT))
                ctx[s] = 0
            # 上下文装不下就**先拒掉这一轮**，顺序与真后端一致（见 main.cc 的
            # context_almost_full 分支）：先 CLEAR（让网关作废这段对话的粘性记录），
            # 再 REJECT（会话还活着，只是这一轮不跑），最后把 KV 当已清。
            # 不这么做的话，网关只发增量，KV 一清模型就拿增量去拼一段没有开头的对话，
            # 然后以一个自信的错答案正常返回。
            was = ctx[s]
            reject = (args.ctx_limit > 0 and was > 0 and
                      was + prefill + 512 >= args.ctx_limit)
            if reject:
                emit(out, "CLEAR ", "%d %d %d %d" % (rid, s, was, args.ctx_limit))
                ctx[s] = 0
        if reject:
            msg = ("context limit reached: %d + 512 >= %d tokens; start a new "
                   "conversation or trim the history" % (was, args.ctx_limit))
            emit(out, "REJECT ", "%d %d %d" % (rid, s, len(msg)), msg.encode("utf-8"))
            return
        time.sleep(args.delay)
        # 切得要"不整齐"：把多字节字符也切开，逼网关用增量解码器
        reply = ("<think>  </think>  " + REPLY) if args.think_prefix else REPLY
        if args.tool_call and "<tool_response>" not in prompt:
            reply += TOOL_CALL
        raw = reply[:max_new].encode("utf-8")
        step = 7
        for i in range(0, len(raw), step):
            emit(out, "DELTA ", "%d %d" % (rid, min(step, len(raw) - i)), raw[i:i + step])
        time.sleep(0.05)
        with ctx_lock:
            ctx[s] += prefill + len(raw)
            after = ctx[s]
        # 判据按"发出去的字节是不是比整段回复短"来定。原版是
        # `finish = "length" if len(REPLY) <= max_new else "stop"`，两个错叠在一起：
        #   · **反了**——回复装得下（没被截断）反而被报成 `length`；
        #   · **单位混用**——`len(REPLY)` 是字符数，`max_new` 是 token 预算。
        # 后果不是"桩看起来不对"：本地全套回归里凡是拿 finish_reason 判"这一轮有没有
        # 被截断"的断言，在桩上都拿到反的答案（真板卡上是 `stop`，桩上是 `length`）。
        finish = "length" if len(raw) < len(reply.encode("utf-8")) else "stop"
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
        # max_new_tokens 超范围：真后端（main.cc）2026-09-16 起会在 prefill 前回 ERR——
        # 那个字段最终落到一个 `int` 上，不校验的话大数会静默截断成别的值（10**12 变成
        # 0 = "用默认值"）。桩跟着实现同一条契约，好让"网关有没有把它夹进范围"
        # （`_pick_max_tokens`）在本地也测得出来。**载荷必须已经读掉**再判——否则流错位。
        if max_new > 0x7fffffff:
            msg = b"max_new_tokens out of range: %d" % max_new
            emit(out, "ERR ", "%d %d %d" % (rid, session, len(msg)), msg)
            continue
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

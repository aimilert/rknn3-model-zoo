#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HTTP 层的并发伸缩测量——这是自研门面存在的**唯一理由**。

为什么必须从 HTTP 这一层量，而不是看后端日志：网关/HTTP/SSE 都有可能把并发吃掉
（比如把请求排成一队、或者持着全局锁发帧）。后端日志只证明推理执行器能重叠，不能
证明用户真正拿到的是并发。所以这里用 N 条线程同时打 /v1/chat/completions，量
「墙钟时间内产出的 token 总数」——这就是用户实际感受到的吞吐。

对照口径：官方 rkllm3-server 同样条件实测 N=1/2/4 是 10.63 / 10.98 / 10.98 tok/s
（1.03x，文档 §4.5.2 写明「推理执行排队串行」）；我们的执行器是 3.33x。
用同一份脚本、同一台板子、同样的 N，这两个数才可比。

用法：python3 http_scaling.py http://127.0.0.1:8080 [每路token数] [N列表,逗号分隔]
"""
import json
import sys
import threading
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
NP = int(sys.argv[2]) if len(sys.argv) > 2 else 96
NS = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["1", "2", "4"])]
TIMEOUT = 1800


def chat(prompt, max_tokens):
    body = {"model": "qwen3.5-27b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "stream": False}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        d = json.loads(r.read().decode("utf-8"))
    return time.time() - t, d["usage"]["completion_tokens"]


def run_batch(n, tag):
    res = [None] * n

    def worker(i):
        try:
            # 每路的首条 user 都不同 => 网关视为不同对话 => 落在不同 session 上。
            # 相同的 prompt 会让它们抢同一个 session，量出来的是排队不是并发。
            res[i] = chat("%s第 %d 路：请从 1 数到 200，只写数字。" % (tag, i), NP)
        except Exception as exc:                       # noqa: BLE001
            res[i] = (-1.0, "FAILED: %r" % (exc,))

    t0 = time.time()
    ths = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return time.time() - t0, res


print("=== HTTP 并发伸缩：%s（每路 max_tokens=%d）===" % (BASE, NP))
print("warmup ...")
try:
    dt, n = chat("热身。", 8)
    print("  warmup: %.1fs, %s tok" % (dt, n))
except Exception as exc:                               # noqa: BLE001
    print("  warmup 失败：%r" % (exc,))
    sys.exit(2)

base = None
rows = []
for n in NS:
    wall, res = run_batch(n, "伸缩")
    if any(not isinstance(r[1], int) for r in res):
        print("N=%d 有失败：%r" % (n, res))
        sys.exit(1)
    total = sum(r[1] for r in res)
    agg = total / wall if wall else 0.0
    lats = [r[0] for r in res]
    if base is None:
        base = agg
    rows.append((n, wall, total, agg, sum(lats) / len(lats), agg / base if base else 0))
    print("  N=%d: 墙钟 %.1fs, 合计 %d tok, 聚合 %.2f tok/s, 单请求均lat %.1fs, "
          "伸缩 %.2fx" % (n, wall, total, agg, sum(lats) / len(lats), agg / base))

print()
print("| N | 墙钟(s) | 合计tok | 聚合 tok/s | 伸缩比 | 单请求延迟(s) |")
print("|---|---|---|---|---|---|")
for n, wall, total, agg, lat, sp in rows:
    print("| %d | %.1f | %d | %.2f | %.2fx | %.1f |" % (n, wall, total, agg, sp, lat))
print()
print("对照（官方 rkllm3-server，同板同条件）：N=1/2/4 -> 10.63 / 10.98 / 10.98 tok/s，"
      "伸缩 1.03x")

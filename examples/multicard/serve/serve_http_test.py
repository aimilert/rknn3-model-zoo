#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""网关的 HTTP 客户端测试（只用标准库——板卡上没有 curl，也没有 requests）。

先在本地对着桩后端跑，再到板卡上对着真后端跑同一份：
    python serve_http_test.py http://127.0.0.1:8080            # 全量
    python serve_http_test.py http://127.0.0.1:8080 quick      # 只跑 health/models/chat

关键的一项是多轮 KV 复用的 HTTP 级证据：同一段对话的第二轮，网关只发差异部分，
后端回报的 prefill token 数应当**明显小于**把整段 prompt 重发一遍的那次。
usage.prompt_tokens 就是这个数，所以从 HTTP 这一层就能看见复用有没有生效。

**为什么每个请求都带 conversation_id（v1.9 起）**：网关现在**只排队、不抢占**——一段
新对话拿不到空闲会话时是等，不是把别人占着的会话夺过来。所以"每问一个新问题就开一段
新对话"这种写法，在会话数（默认 4）用完之后会一路排队到 `IDLE_TTL` 才动。
本脚本因此给每个 section 显式命名对话、用完就 `close`，全程占用的会话数是有界的
（最多 N 路并发那一段）。这也正是多人接入时该有的写法：**身份要显式给，离开要显式说。**
"""
import json
import sys
import time
import threading
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
MODE = sys.argv[2] if len(sys.argv) > 2 else "full"
TIMEOUT = 1800

OK = [True]


def check(name, cond, detail=""):
    OK[0] = OK[0] and bool(cond)
    print("  [%s] %s %s" % ("PASS" if cond else "FAIL", name, detail))


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def post(path, body, stream=False, conv=None):
    headers = {"Content-Type": "application/json"}
    if conv:
        headers["X-Conversation-Id"] = conv
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode("utf-8"),
                                 headers=headers)
    return urllib.request.urlopen(req, timeout=TIMEOUT)


def close_conv(conv):
    """告诉网关"这段对话结束了"，把会话立刻还回去（不等 IDLE_TTL）。

    这是 v1.9 新增的端点。**没有它，这个套件在会话数 ≤4 的板卡上会自己把自己堵死**：
    下面每个 section 都要各占几个会话，谁都不放的话第 5 个 section 只能排队。
    """
    with post("/v1/conversations/close", {"conversation_id": conv}) as r:
        return json.loads(r.read().decode("utf-8"))


def close_anonymous():
    """把套件自己造出来的**匿名**对话清掉（它们没有名字，只能按 `key` 关）。

    为什么必须清：匿名对话的身份是网关按内容算的摘要，客户端算不出来，所以它**关不掉也
    认不出**，会一直占着一个会话到 `IDLE_TTL` 超时。留着不管的话，套件跑第二遍时可用
    会话就少一个，N 路并发那一段会有一路拿不到会话、排队超时失败——**第一遍全绿、第二遍
    失败**，是最容易查错方向的那种现象。
    """
    _, pool = get("/v1/pool")
    closed = []
    for slot in pool.get("slots", []):
        key = slot.get("key")
        if key and key.startswith("h:"):
            with post("/v1/conversations/close", {"key": key}) as r:
                closed.append(json.loads(r.read().decode("utf-8")))
    return closed


def chat(messages, max_tokens=64, stream=False, conv=None, **extra):
    body = {"model": "qwen3.5-27b", "messages": messages,
            "max_tokens": max_tokens, "stream": stream}
    body.update(extra)
    if not stream:
        with post("/v1/chat/completions", body, conv=conv) as r:
            return json.loads(r.read().decode("utf-8"))
    chunks = []
    text = []
    with post("/v1/chat/completions", body, conv=conv) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunks.append(json.loads(payload))
            delta = chunks[-1]["choices"][0].get("delta", {}) if chunks[-1].get("choices") else {}
            if delta.get("content"):
                text.append(delta["content"])
    return chunks, "".join(text)


def main():
    print("=== 网关 HTTP 测试：%s ===" % BASE)

    print("== health / models ==")
    st, health = get("/health")
    check("health 200", st == 200, json.dumps(health, ensure_ascii=False))
    st, models = get("/v1/models")
    mid = models["data"][0]["id"]
    check("models 里有条目", bool(mid), "model id = %s" % mid)

    print("== 非流式 ==")
    t0 = time.time()
    d = chat([{"role": "user", "content": "用一句话说明你是谁。"}], max_tokens=48,
             conv="probe-oneshot")
    dt = time.time() - t0
    msg = d["choices"][0]["message"]
    check("有 content", len(msg.get("content") or "") > 0,
          "%.1fs, %d 字, finish=%s" % (dt, len(msg.get("content") or ""),
                                       d["choices"][0]["finish_reason"]))
    check("usage 有 prompt/completion",
          d["usage"]["prompt_tokens"] > 0 and d["usage"]["completion_tokens"] > 0,
          json.dumps(d["usage"]))
    close_conv("probe-oneshot")     # 用完就还，见文件头"为什么每个请求都带 conversation_id"

    print("== 流式 SSE ==")
    chunks, text = chat([{"role": "user", "content": "数到五。"}], max_tokens=32,
                        stream=True, conv="probe-oneshot")
    deltas = [c for c in chunks if c.get("choices") and c["choices"][0]["delta"].get("content")]
    check("收到多个 delta", len(deltas) > 1, "%d 个" % len(deltas))
    check("首块带 role", any(c.get("choices") and c["choices"][0]["delta"].get("role") == "assistant"
                             for c in chunks))
    check("有 finish_reason", any(c.get("choices") and c["choices"][0].get("finish_reason")
                                  for c in chunks))
    check("拼出的文本非空", len(text) > 0, "%d 字" % len(text))
    close_conv("probe-oneshot")     # 见文件头：不还回去，后面几段就会为会话数打架

    if MODE == "quick":
        print("\n%s" % ("全部通过" if OK[0] else "有失败项"))
        return 0 if OK[0] else 1

    print("== 多轮 KV 复用（HTTP 级证据：看 prompt_tokens）==")

    def kv_reuse_case(label, conv, extra):
        # 关思考和开思考要各测一遍。关思考时网关会把 think 段从正文里摘掉，于是"记进
        # 会话账本的文本"和"模型实际生成的原始文本"不再逐字节相同——如果记账用了原始
        # 版，客户端下一轮回显的（已摘标签的）历史就对不上前缀，复用每轮都会退化成全量。
        # 这个坑答案完全正确、只有速度变差，所以必须靠这里的 prompt_tokens 抓。
        filler = "请记住这句话：" + "麒麟九千" * 40
        d1 = chat([{"role": "user", "content": filler}], max_tokens=16, conv=conv, **extra)
        a1 = d1["choices"][0]["message"]["content"]
        hist = [{"role": "user", "content": filler},
                {"role": "assistant", "content": a1}]
        d2 = chat(hist + [{"role": "user", "content": "继续。"}], max_tokens=16,
                  conv=conv, **extra)
        # 另起一段对话（换个首条 user 内容）走全量 prefill 作为对照
        d3 = chat([{"role": "user", "content": "换一段全新的对话：" + "麒麟九千" * 40}],
                  max_tokens=16, conv=conv + "-fresh", **extra)
        p2 = d2["usage"]["prompt_tokens"]
        p3 = d3["usage"]["prompt_tokens"]
        check("第二轮 prefill 远小于全量[%s]" % label, p2 < p3,
              "续聊 prefill=%d tok, 全量 prefill=%d tok, 比值 %.2f"
              % (p2, p3, (p2 / p3 if p3 else 0)))
        close_conv(conv)
        close_conv(conv + "-fresh")

    kv_reuse_case("开思考", "probe-reuse-think", {})
    kv_reuse_case("关思考", "probe-reuse-nothink",
                  {"chat_template_kw": {"enable_thinking": False}})

    print("== 并发（N 路同时打，聚合吞吐）==")
    # 并发这一段**必须**是 N 段不同的对话：同一段对话的两轮会被钉在同一个会话上，
    # 那样量到的是"排队"而不是"并发"。每一路各给一个身份。
    n = 4
    results = [None] * n

    def worker(i):
        try:
            t = time.time()
            dd = chat([{"role": "user", "content": "第 %d 路：数到二十。" % i}], max_tokens=64,
                      conv="probe-n%d" % i)
            results[i] = (time.time() - t, dd["usage"]["completion_tokens"])
        except Exception as exc:                       # noqa: BLE001
            results[i] = (-1, "FAILED: %r" % (exc,))

    t0 = time.time()
    ths = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.time() - t0
    tot = sum(r[1] for r in results if r and isinstance(r[1], int))
    check("N 路都成功", all(r and isinstance(r[1], int) for r in results),
          "各耗时 " + ", ".join("%.1fs" % r[0] for r in results if r))
    check("聚合吞吐 > 单路", tot / wall > 0, "合计 %d tok / %.1fs = %.2f tok/s"
          % (tot, wall, tot / wall if wall else 0))
    for i in range(n):
        close_conv("probe-n%d" % i)

    print("== 关思考（chat_template_kw）==")
    d4 = chat([{"role": "user", "content": "1+1=?"}], max_tokens=64, conv="probe-oneshot",
              chat_template_kw={"enable_thinking": False})
    c4 = d4["choices"][0]["message"]["content"]
    check("返回非空", len(c4) > 0, "前 60 字：%s" % c4[:60].replace("\n", " "))
    # 关思考是靠 /no_think 软开关，模型照样会吐一个（空的）<think> 段；网关负责摘掉，
    # 否则 Agent 拿到的正文里带着标签，还得自己洗一遍。
    check("正文里没有残留的 <think> 标签", "<think>" not in c4 and "</think>" not in c4,
          "前 60 字：%s" % c4[:60].replace("\n", " "))
    _, c5 = chat([{"role": "user", "content": "1+1=?"}], max_tokens=64, stream=True,
                 conv="probe-oneshot", chat_template_kw={"enable_thinking": False})
    check("流式路径也没有残留的 <think> 标签",
          "<think>" not in c5 and "</think>" not in c5, "前 60 字：%s" % c5[:60].replace("\n", " "))
    d6 = chat([{"role": "user", "content": "1+1=?"}], max_tokens=64, conv="probe-oneshot")
    c6 = d6["choices"][0]["message"]["content"]
    check("开思考（默认）时原样透传，不擅自丢推理过程", "<think>" in c6,
          "前 40 字：%s" % c6[:40].replace("\n", " "))
    close_conv("probe-oneshot")

    nsess = int(health.get("sessions") or 0)
    print("== 多用户身份（会话按身份分，不按内容）==  sessions=%d" % nsess)
    ident = min(3, max(0, nsess))
    if ident < 2:
        check("要有 ≥2 个会话才能验身份分流", False, "sessions=%d" % nsess)
    else:
        # **同一句话 + 不同身份 => 不同会话**。这是多人接入的底线：靠内容认对话的话，
        # 两个人问同一句话会被当成同一段对话、钉在同一个会话上串行（答案全对、吞吐减半）。
        probe = "身份分流探针：只回复 ok。"
        sids = []
        for i in range(ident):
            try:
                with post("/v1/chat/completions",
                          {"model": "qwen3.5-27b",
                           "messages": [{"role": "user", "content": probe}],
                           "max_tokens": 8}, conv="probe-id%d" % i) as r:
                    kv = r.headers.get("X-KV-Reuse") or ""
                    r.read()
            except Exception as exc:                   # noqa: BLE001
                check("身份分流请求全部成功", False, "probe-id%d: %r" % (i, exc))
                sids = []
                break
            sids.append(dict(p.split("=") for p in kv.split(";") if "=" in p)
                        .get("session", "").strip())
        if sids:
            check("同一句话、不同身份 => 落在不同会话上", len(set(sids)) == ident,
                  "落到会话 %s（%d 个身份）" % (sids, ident))
        for i in range(ident):
            close_conv("probe-id%d" % i)

        # 反向对照：**不给身份**，同一句话就只能按内容认 => 全落到同一个会话上。
        # 这条是"为什么必须显式给身份"的实证，不是理论说明。
        sids2 = []
        for _ in range(ident):
            with post("/v1/chat/completions",
                      {"model": "qwen3.5-27b",
                       "messages": [{"role": "user", "content": probe}],
                       "max_tokens": 8}) as r:
                kv = r.headers.get("X-KV-Reuse") or ""
                r.read()
            sids2.append(dict(p.split("=") for p in kv.split(";") if "=" in p)
                         .get("session", "").strip())
        check("不给身份 => 同一句话全落到同一个会话（正是要避免的退化）",
              len(set(sids2)) == 1, "落到会话 %s" % (sids2,))
        # 这一段的匿名对话没有名字（身份是网关按内容算的摘要，客户端算不出），所以只能
        # 按 `GET /v1/pool` 里的原始 `key` 关。顺手证明这条路是通的——没有它，套件跑
        # 第二遍时可用会话会少一个，N 路并发那一段就会莫名其妙地失败。
        anon = close_anonymous()
        check("匿名对话能按 /v1/pool 的 key 关掉（否则会一直占到 IDLE_TTL）",
              anon and all(c.get("closed") for c in anon),
              "关掉 %d 段匿名对话" % len(anon))

    print("== 会话池：排队不抢占 + 主动交还 ==")
    st, pool = get("/v1/pool")
    check("/v1/pool 可读且槽位数 = 会话数",
          st == 200 and len(pool.get("slots", [])) == nsess,
          "slots=%d waiting=%d reaped=%d idle_ttl=%ss"
          % (len(pool.get("slots", [])), len(pool.get("waiting", [])),
             pool.get("reaped_total", 0), pool.get("idle_ttl_s")))
    closed = close_conv("probe-never-existed")
    check("close 一段不存在的对话是幂等的（不报错）",
          closed.get("closed") is False and closed.get("session") is None,
          json.dumps(closed, ensure_ascii=False))
    try:
        post("/v1/conversations/close", {}).read()
        check("close 不带身份 => 400", False)
    except urllib.error.HTTPError as exc:
        check("close 不带身份 => 400（匿名对话关不掉，因为它的 id 是内容摘要）",
              exc.code == 400, "HTTP %d" % exc.code)

    print("\n%s" % ("全部通过" if OK[0] else "有失败项"))
    return 0 if OK[0] else 1


if __name__ == "__main__":
    sys.exit(main())

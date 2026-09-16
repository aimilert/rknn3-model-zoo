#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""关思考（chat_template_kw.enable_thinking=false）下，粘性复用还成不成立？

为什么单独测这一条：关思考是靠给 user 消息尾巴贴 `/no_think` 实现的软开关（这条 GGUF
不吃官方 server 的 chat_template_kw，实测输出照样带 <think>）。而「贴哪条 user 消息」
会决定整段 prompt 的前缀稳不稳定：

  · 只贴**最后一条** user => 第一轮 u1 是"最后一条"（带标记），第二轮 u1 变成了历史
    消息（不带标记）=> 两轮渲染出来的 u1 不是同一个字符串 => 前缀判据不成立 =>
    每轮都全量重算。正确性没问题，但 Agent 场景下关思考是常用配置，缓存会静默全废。
  · 贴**所有** user 消息 => 每一轮渲染同一段历史都得到同样的字节 => 前缀稳定，
    复用照常生效。

这个脚本就是把这个判断钉死：同一段对话两轮都开 enable_thinking=false，看第二轮的
X-KV-Reuse 头里 base 是不是非空、sent 是不是远小于 full。

用法：python3 nothink_check.py http://127.0.0.1:8080
"""
import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
NEEDLE = "青柠味苏打水"
FAILS = []


def _think_body(text):
    """取出第一个 <think>…</think> 的内容；没有这个段就返回空串。

    关思考下这条 GGUF 仍会吐一对**空**标记（`<think>  </think>`），所以判据不能是
    "有没有标签"，只能是"段里有没有内容"。
    """
    start = text.find("<think>")
    if start < 0:
        return ""
    end = text.find("</think>", start)
    if end < 0:
        return text[start + len("<think>"):]     # 没闭合：剩下的都算段内
    return text[start + len("<think>"):end]


def check(name, cond, detail=""):
    print("  [%s] %s %s" % ("PASS" if cond else "FAIL", name, detail))
    if not cond:
        FAILS.append(name)


def chat(messages, conv, max_tokens=96, think=None):
    body = {"model": "qwen3.5-27b", "messages": messages,
            "max_tokens": max_tokens, "stream": False, "conversation_id": conv}
    if think is not None:
        body["chat_template_kw"] = {"enable_thinking": think}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read().decode("utf-8"))
        kv = r.headers.get("X-KV-Reuse")
    c = d["choices"][0]
    return c["message"]["content"], d["usage"]["prompt_tokens"], kv


def parse_kv(kv):
    if not kv:
        return {}
    out = {}
    for part in kv.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


print("=== 关思考下的粘性复用核对：%s ===" % BASE)
FIRST = "记住暗号：%s。收到请只回复两个字：收到。" % NEEDLE

print("== 1. 第一轮（enable_thinking=false，会话可能已脏 => 会 RESET）==")
a1, p1, kv1 = chat([{"role": "user", "content": FIRST}], "nothink-1", 96, False)
print("     prompt=%d tok, kv=[%s]" % (p1, kv1))
print("     答=%r" % a1[:80].replace("\n", " "))
# 这条断言 2026-09-16 改过：原来是 `"<think>" not in a1 or "</think>" in a1`，
# **恒真式**——模型吐一个成对的 <think>…</think> 就通过了，而"关思考没生效"恰恰长这样。
# 实测过它漏判：日志里下一轮模型确实吐了 `<think>Thinking Process: ...`，套件照样全绿。
# 关思考下这条 GGUF 仍会吐一对**空**标记（`<think>  </think>`，见 fake_backend 的
# --think-prefix 就是照它做的），所以正确的判据是：没有**非空**的 think 段。
check("关思考生效（没有非空 <think> 段）", _think_body(a1).strip() == "",
      "think 段内容=%r" % _think_body(a1)[:60])

hist = [{"role": "user", "content": FIRST}, {"role": "assistant", "content": a1}]
Q = {"role": "user", "content": "暗号是什么？只回答暗号本身。"}

print("== 2. 第二轮（同一 conversation_id，同样关思考）==")
a2, p2, kv2 = chat(hist + [Q], "nothink-1", 96, False)
kv2d = parse_kv(kv2)
print("     prompt=%d tok （整段；本轮只算的部分见 kv 头的 sent）, kv=[%s]" % (p2, kv2))
print("     答=%r" % a2[:80].replace("\n", " "))

print("== 3. 判定 ==")
check("第二轮仍然复用（base 非空且 sent < full）",
      bool(kv2d) and kv2d.get("reset") == "0" and int(kv2d.get("base", 0)) > 0
      and int(kv2d.get("sent", 1)) < int(kv2d.get("full", 0)),
      "kv=[%s]" % kv2)
check("暗号还在（复用没把历史弄丢）", NEEDLE in a2)

# 用完把会话交还：板卡上只有 4 个，而 run_all_board_tests.sh 是一串脚本连着跑，
# 留着不还的话后面的脚本（http_scaling 的 N 路并发）会没有可用会话。
body = json.dumps({"conversation_id": "nothink-1"}).encode("utf-8")
_req = urllib.request.Request(BASE + "/v1/conversations/close", data=body,
                              headers={"Content-Type": "application/json"})
with urllib.request.urlopen(_req, timeout=30) as _r:
    _r.read()

print("\n%s" % ("全部通过" if not FAILS else "失败项：%s" % ", ".join(FAILS)))
sys.exit(0 if not FAILS else 1)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""粘性 KV 复用的合取核对：既看「有没有复用」，也看「复用之后模型还记不记得」。

背景（为什么要这么绕）：
  网关记的 known[s] = 上一轮整段 prompt + 上一轮回复，下一轮只发差异部分。这里有个
  推理想不清楚的隐患——分词是左到右贪婪 BPE，**不是按前缀可分的**：上一轮回复的结尾
  和这一轮后缀在拼接处可能合并成别的 token。于是「KV 里的 token + 新后缀的 token」
  和「整段重新分词」在边界那一个 token 上可能不同。后果不是崩溃，是模型看到的上下文
  悄悄差一点，答出来的东西看着通顺。这种错只能靠实测。

判据为什么是语义的：
  实测这套栈**本来就不是逐位可复现**的（determinism_probe.py：同 session 同 prompt
  连打 3 次，回答有 2 种）。所以「两次回答逐字节相同」会把固有抖动误判成复用 bug。
  改用**信息存活**判据：第一轮给模型一个暗号，第二轮的提问故意不含暗号、也不含第一轮
  的回复——粘性路径下暗号只可能来自会话 KV。答得出 => KV 真的装住了那段历史且拼接
  正确；答不出 => 复用把上下文弄丢了，必须查。

判据也不要只看 usage.prompt_tokens：prefill 小可能是「复用生效」，也可能是「换到了
另一个空会话」。所以同时看 X-KV-Reuse 响应头（会话号 / 是否 RESET / 实发字节数）。

用法：python3 sticky_check.py http://127.0.0.1:8080
"""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
TIMEOUT = 1800
# 暗号要中性：第一版用了「紫色犀牛731」，模型把它读成历史敏感事件直接拒答，
# 实验就废了（踩过）。下面这个是纯无害的名词短语。
NEEDLE = "青柠味苏打水"
FAILS = []


def check(name, cond, detail=""):
    print("  [%s] %s %s" % ("PASS" if cond else "FAIL", name, detail))
    if not cond:
        FAILS.append(name)


def chat(messages, conv, max_tokens=96):
    body = {"model": "qwen3.5-27b", "messages": messages,
            "max_tokens": max_tokens, "stream": False, "conversation_id": conv}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        d = json.loads(r.read().decode("utf-8"))
        kv = r.headers.get("X-KV-Reuse")
    c = d["choices"][0]
    return (c["message"]["content"], d["usage"]["prompt_tokens"],
            c["finish_reason"], kv, time.time() - t0)


print("=== 粘性复用核对：%s ===" % BASE)

print("== 1. 第一轮：把暗号告诉模型 ==")
a1, p1, f1, kv1, t1 = chat([{"role": "user", "content":
                          "记住暗号：%s。收到请只回复两个字：收到。" % NEEDLE}],
                        "keep-1", 96)
print("     prefill=%d tok, finish=%s, kv=[%s]" % (p1, f1, kv1))
print("     答=%r" % a1[:100].replace("\n", " "))
check("第一轮回答非空", len(a1) > 0)
check("第一轮没被截断（否则第二轮结果不可解释）", f1 == "stop", "finish=%s" % f1)

hist = [{"role": "user", "content": "记住暗号：%s。收到请只回复两个字：收到。" % NEEDLE},
        {"role": "assistant", "content": a1}]
Q = {"role": "user", "content": "暗号是什么？只回答暗号本身，不要解释。"}

print("== 2. 第二轮走粘性复用（同一 conversation_id）==")
a_s, p_s, f_s, kv_s, t_s = chat(hist + [Q], "keep-1", 96)
print("     prefill=%d tok, finish=%s, kv=[%s]" % (p_s, f_s, kv_s))
print("     答=%r" % a_s[:100].replace("\n", " "))

print("== 3. 第二轮走全量 prefill（换 conversation_id => 无粘性记录）==")
a_f, p_f, f_f, kv_f, t_f = chat(hist + [Q], "keep-2", 96)
print("     prefill=%d tok, finish=%s, kv=[%s]" % (p_f, f_f, kv_f))
print("     答=%r" % a_f[:100].replace("\n", " "))

print("== 4. 判定 ==")
check("粘性那轮更快（少算了 prefill 的历史部分）", t_s < t_f,
      "%.1fs vs %.1fs" % (t_s, t_f))
check("粘性那轮确实只发了差异部分（看响应头 sent < full）",
      kv_s is not None and "sent=" in kv_s
      and int(kv_s.split("sent=")[1].split(";")[0]) < len(json.dumps(hist)) * 4,
      "kv=[%s]" % kv_s)
check("粘性那轮 prefill 明显更小", p_s < p_f / 2,
      "%d vs %d tok（%.2f）" % (p_s, p_f, p_s / p_f if p_f else 0))
check("粘性路径答出了暗号（KV 装住了历史且拼接正确）", NEEDLE in a_s)
check("全量路径答出了暗号（对照组本身没问题）", NEEDLE in a_f)

print("\n%s" % ("全部通过" if not FAILS else "失败项：%s" % ", ".join(FAILS)))
sys.exit(0 if not FAILS else 1)

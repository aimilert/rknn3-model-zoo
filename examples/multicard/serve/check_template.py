#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模板逐字节核对 + 粘性复用前提核对。

为什么要单独核这个：网关自己渲染 prompt（demo 内置模板只有「首轮/后续轮」两态，
表达不了 Agent 发来的任意角色历史），而渲染结果**差一个字符不会有任何报错**，
只会让模型收到的对话格式悄悄变味、输出质量下降。这类错误只能靠逐字节比对抓。

第二项（前缀性）是网关能不能只发差异部分的**前提**：第二轮拼出来的完整 prompt
必须以「上一轮的 prompt + 上一轮回复」开头，否则粘性复用会把模型带到一个
自相矛盾的上下文里。

第三项是工具调用（function calling）的渲染。**main.cc 里的 QWEN35_CHAT_TEMPLATE 没有
工具支持**，所以拿它当权威是错的——权威是模型自己的 `tokenizer.chat_template`。板卡上
没有 jinja2（实测 Python 3.12 无此包），渲不了，于是参考字节在本机用真 Jinja2 渲好后冻
进 `tool_golden.json`（生成器 rt_work/gen_golden.py，不进 git）。比对的仍然是模板的真
输出；只是把「渲染」这一步挪到了生成那一刻。
"""
import io
import json
import os
import sys

sys.path.insert(0, ".")
from rkllm_gateway import render_messages                        # noqa: E402
from toolcalls import selftest as toolcalls_selftest             # noqa: E402

# 与 examples/multicard/cpp/main.cc 的 QWEN35_CHAT_TEMPLATE 完全一致
SYS = ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. "
       "You are a helpful assistant.<|im_end|>\n")
UPRE = "<|im_start|>user\n"
UPOST = "<|im_end|>\n<|im_start|>assistant\n"

fails = []


def check(name, cond, detail=""):
    print("  [%s] %s %s" % ("PASS" if cond else "FAIL", name, detail))
    if not cond:
        fails.append(name)


print("== 与 main.cc 的模板逐字节比对 ==")
mine1 = render_messages([{"role": "user", "content": "你好"}])
want1 = SYS + UPRE + "你好" + UPOST
check("首轮（system + user）", mine1 == want1)
if mine1 != want1:
    print("     mine: %r" % mine1)
    print("     want: %r" % want1)

# 注意 UPOST 本身就以 "<|im_start|>assistant\n" 结尾（它就是 assistant 头），
# 所以 assistant 那一段只需接内容 + <|im_end|>，不要再补一个 assistant 头。
mine2 = render_messages([{"role": "user", "content": "A"},
                         {"role": "assistant", "content": "B"},
                         {"role": "user", "content": "C"}])
want2 = (SYS + UPRE + "A" + UPOST
         + "B" + "<|im_end|>\n"
         + UPRE + "C" + UPOST)
check("多轮（user/assistant/user）", mine2 == want2)
if mine2 != want2:
    print("     mine: %r" % mine2)
    print("     want: %r" % want2)

print("== 粘性复用的前提 ==")
prev_full = mine1
gen = "上一轮的回答"
follow = render_messages([{"role": "user", "content": "你好"},
                          {"role": "assistant", "content": gen},
                          {"role": "user", "content": "继续"}])
check("第二轮 prompt 以 上一轮prompt+回复 为前缀", follow.startswith(prev_full + gen),
      "suffix=%d 字 / full=%d 字" % (len(follow) - len(prev_full + gen), len(follow)))
check("差异部分非空", len(follow) > len(prev_full + gen))

print("== /no_think 贴在所有 user 上（跨轮渲染才稳定）==")
# 这条曾经写成"只贴最后一条 user"，并且测试也跟着这么断言——那是错的：第一轮 u1 是
# "最后一条"（带标记），第二轮 u1 变成历史消息（不带）=> 同一段历史两轮渲染的字节不同
# => 前缀判据失败 => 关思考时每轮全量重算（实测：第二轮 reset=1, base=0, sent=full）。
# 所以这里断言的不是"贴哪一条"，而是**同一个历史在两轮里渲染成同样的字节**。
m3 = render_messages([{"role": "user", "content": "A"}, {"role": "user", "content": "C"}],
                     enable_thinking=False)
check("每条 user 都带 /no_think", "A /no_think<|im_end|>" in m3 and "C /no_think<|im_end|>" in m3)
check("不传 enable_thinking 时不贴 /no_think",
      "/no_think" not in render_messages([{"role": "user", "content": "A"}])
      and "/no_think" not in render_messages([{"role": "user", "content": "A"}],
                                             enable_thinking=True))

print("== 关思考下第二轮 prompt 仍以第一轮为前缀（粘性复用前提）==")
u1 = {"role": "user", "content": "记住暗号：青柠味苏打水。"}
h1 = render_messages([u1], enable_thinking=False)
h2 = render_messages([u1, {"role": "assistant", "content": "收到"},
                      {"role": "user", "content": "暗号是什么？"}], enable_thinking=False)
check("关思考：第二轮以第一轮为前缀", h2.startswith(h1),
      "suffix=%d 字 / full=%d 字" % (len(h2) - len(h1), len(h2)))
check("关思考：差异部分非空", len(h2) > len(h1))

print("== 工具调用渲染：与模型自己的 chat_template 逐字节比对（金标准）==")
# 逐字节比对抓得到而人眼抓不到的：工具定义走 Jinja 的 tojson（**键排序**、中文转成
# \uXXXX、`<` `>` `&` `'` 全转义），助手轮第一条调用前的分隔是 \n\n 还是 \n，连续的
# tool 消息是合成一个 user 轮还是各成一轮——这些差一点都不会报错，只会让模型收到的
# 对话格式悄悄变味。三处曾经真的写错过，都是这一项抓出来的。
GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tool_golden.json")
with io.open(GOLDEN, encoding="utf-8") as fp:
    golden = json.load(fp)

WRAP = golden["wrap"]           # 模板给助手轮加的推理包装
WRAP_OFF = golden["wrap_off"]
ASSISTANT_HEAD = "<|im_start|>assistant\n"
for case in golden["cases"]:
    # 两处**刻意**的偏差（理由见 toolcalls.py 顶部，都是为了 KV 前缀对得上）：
    #   · add_generation_prompt：模板会补 `<think>\n`（关思考时是
    #     `<think>\n\n</think>\n\n`），我们只补 assistant 头，靠 /no_think 软开关 +
    #     回复侧摘除实现关思考。这里把那截尾巴**显式换掉**，偏差留在比对代码里。
    #   · 助手轮的推理包装我们不做——逐字替换掉，但**先数一遍**：替换 0 个还报 PASS
    #     就成了空转，那条用例根本没在测东西。
    want = case["want"]
    if not want.endswith(case["gen_tail"]):
        check(case["name"], False, "金标准不以 gen_tail 结尾（模板变了，重跑 gen_golden.py）")
        continue
    want = want[:-len(case["gen_tail"])] + ASSISTANT_HEAD
    n_wrap = want.count(WRAP)
    want = want.replace(WRAP, WRAP_OFF)
    got = render_messages(case["messages"], enable_thinking=case["enable_thinking"],
                          tools=case["tools"] or None)
    # /no_think 由 render_messages 自己贴，金标准里的 messages 是**没贴过**的原文，
    # 所以这里不预处理；no_think_marks 只用来核对"参考渲时确实贴了"。
    marks_ok = (" /no_think" in want) if case["no_think_marks"] else ("/no_think" not in want)
    ok = (got == want) and (n_wrap == case["wrappers"]) and marks_ok
    detail = ""
    if not ok:
        if n_wrap != case["wrappers"]:
            detail = "推理包装数 %d != 记录 %d" % (n_wrap, case["wrappers"])
        elif not marks_ok:
            detail = "/no_think 标记与记录不符"
        else:
            i = next((k for k, (a, b) in enumerate(zip(want, got)) if a != b), None)
            detail = ("第 %d 字节起不同\n     want=%r\n     got =%r" % (i, want[i:i + 40], got[i:i + 40])
                      if i is not None else "\n     want=%r\n     got =%r" % (want, got))
    check(case["name"], ok, detail)

print("")
# 回复侧的解析（含"拆分->渲染 == 模型原始输出"这条 KV 前缀的地基）。跟渲染一样是纯字符
# 串逻辑，不需要后端，板卡上也能跑——所以它必须进这个套件，不能只留在本机的临时脚本里。
toolcalls_selftest(check)

print("\n%s" % ("模板检查通过" if not fails else "失败项：%s" % ", ".join(fails)))
sys.exit(0 if not fails else 1)

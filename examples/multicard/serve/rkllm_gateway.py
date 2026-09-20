#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rkllm_multicard_demo（--serve）前面的 OpenAI 兼容网关。

为什么要有这一层：官方 rkllm3-server 的 OpenAI 接口是现成的，但文档 §4.5.2 写明
「推理执行排队串行」，实测 --n-session 1/2/4 的聚合都是 ~11 tok/s（1.03x）——
4 个 slot 只提供隔离、不提供并行。而我们自己的 demo 在同一硬件上是 3.33x
（10.73 -> 35.74 tok/s，见方案文档 §9.2）。所以 HTTP 这一层自己写，推理仍然走
我们自己验证过的多会话执行器。

架构（三个进程角色，全在板卡上）：
    workflow agent ──HTTP/OpenAI──▶ rkllm_gateway.py ──帧协议(stdin/fd)──▶ rkllm_multicard_demo --serve
                          (本文件)                         (一个进程, N 个会话)

网关做三件事：
  1. 把 OpenAI 的 messages 按 Qwen3.5 模板渲染成一个 prompt 字符串（渲染在网关侧做，
     因为 demo 内置的模板只有「首轮 / 后续轮」两态，表达不了任意角色的历史消息）；
  2. 会话粘性：把「同一段对话」钉在同一个 demo 会话上，并只在会话已有的 KV 之后
     补发差异部分，让多轮对话不必每轮重算整个上下文（长上下文下这是每轮上百秒的差别）；
  3. 把帧协议翻成 OpenAI 的 JSON / SSE。并发请求落到不同会话上，由 demo 的执行器
     跨阶段重叠，这才是 3.33x 的来源。

安全性上有一条不能省：**前缀校验**。粘性记录只是"猜测"，真正决定能不能只发差异部分的
是「本次完整 prompt 是否以该会话已知文本为前缀」。不满足就 RESET + 发全量。所以粘性记录
错了最多是慢，不会答出串味的结果。

用法（板卡上）：
    ./start_gateway.sh                       # 见同目录的启动脚本
    python3 rkllm_gateway.py --port 8080 --sessions 4 -- <demo argv...>
    python3 rkllm_gateway.py --selftest -- <demo argv...>   # 只验帧协议，不起 HTTP
"""
import argparse
import codecs
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# 工具调用的渲染/解析单独一个文件：格式细节多（XML 形态、tojson 的坑、连续 tool 消息
# 合并成一个 user 轮），而且它自己就能被逐字节对拍（check_template.py），不该混在
# 已经 1800 行的网关里。
from toolcalls import (ToolCallExtractor, normalize_tool_calls, render_assistant_turn,
                       render_tools_system, split_tool_calls, to_openai_tool_calls,
                       tool_response_block)
from toolcalls import selftest as toolcalls_selftest

MODEL_ID = "qwen3.5-27b"
BACKEND_READY_TIMEOUT = 900.0     # 模型加载约 230-240s，留足余量
REQUEST_TIMEOUT = 1800.0          # 单轮最长等待（长上下文 prefill 很慢）
DEFAULT_MAX_NEW_TOKENS = 512

# 会话不够时怎么办。板卡上会话的硬上限实测是 5 个（第 6 个 session_init 在 stage0 失败），
# 而且只能在启动时建，运行期建不了。所以**同时活跃的对话数**就是并发上限，第 N+1 段对话
# 只能等——等谁腾出来。
#
# 谁腾？一段对话静默超过 IDLE_TTL 就把它的会话收回。这两个默认值是**配对**的：
# 队列最长也就等到"最闲的那段对话静默满 TTL"，所以 QUEUE_TIMEOUT 要留得比 IDLE_TTL 大，
# 否则会出现"明明马上就轮到了，请求却先超时报 503"。
#
# IDLE_TTL 别调太小：10 个人活跃、只有 5 个会话时，每个人都在 TTL 后被踢掉、又排队等回来，
# 每一轮都要全量重算 prefill（多花 2~3 秒）。它必须明显大于"正常人相邻两轮之间的间隔"。
DEFAULT_IDLE_TTL = 300.0          # 秒；0 = 不回收（= 第 N+1 段对话永远排队）
DEFAULT_QUEUE_TIMEOUT = 600.0     # 秒；排队超过它就返回 503，不要让终端用户干等
# 上面那对默认值有个不对称，2026-09-18 被用户抓出来："NPU 明明没人用，输入问题后还是
# 在等"。**只有 IDLE_TTL 一条路决定谁腾会话**，而它被调得很大（300s）是为了保住 KV
# 复用——可这段静默期里 NPU 是**真闲着**，等的人却在空转。板上实测（`rt_work/
# page_repro.py` 的 queue 场景）：四段对话各自静默 70~90s，第 5 段对话干等 **226.0s**
# 才拿到会话，全程四个会话都是 idle。等待时长 = IDLE_TTL − 它自己静默了多久。
#
# 所以把"静默"分成两级：
#   · 没人等：按 IDLE_TTL（300s）——保住"多轮对话的第二轮直接命中 KV"这条主路径；
#   · 有人在等（请求进来发现一个空闲会话都没有，非等不可）：按 CONTEND_IDLE 收，
#     取**最闲的那一段**对话，只收一个。多花的是被收走那段对话下一轮的 prefill
#     （页面上这种两百来字节的对话不到 0.3s），换来的是让等的人从"分钟级"降到"秒级"。
# 这不违反"不抢占"：只收**没在跑**的会话，而且按静默时长挑受害者（不是按请求到达顺序），
# 行为可预期。CONTEND_IDLE 必须明显大于"同一段对话相邻两轮的真实间隔"，否则一个正常的
# 多轮用户每轮都在被收走、每轮都全量重算。
DEFAULT_CONTEND_IDLE = 15.0       # 秒；0 = 关掉这一级（退回"永远等到 IDLE_TTL"）

IM_START = "<|im_start|>"
IM_END = "<|im_end|>\n"
# 与 examples/multicard/cpp/main.cc 的 QWEN35_CHAT_TEMPLATE 保持一致：
#   system_prompt = "<|im_start|>system\n...<|im_end|>\n"
#   user_prefix   = "<|im_start|>user\n"
#   user_postfix  = "<|im_end|>\n<|im_start|>assistant\n"
QWEN35_DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

ASSISTANT_HEAD = IM_START + "assistant\n"

# 关思考时，模型自己的 chat_template 在 `add_generation_prompt` 里要补的那一截
# （`enable_thinking=false` 分支渲出来就是这对**空**标记）。
#
# 为什么非得补：` /no_think` 只是**软**提示。2026-09-18 板上实测，"要组织语言"的题
# （`介绍一下杭州` / `写一首关于秋天的诗`）在关思考下会先写一整段
# `<think>Thinking Process:…`，正文里甚至有一句 `* Constraint: /no_think (This means
# I should no…`——它看见了提示、权衡了一下，然后照开不误。256 token 一截，`</think>`
# 没机会吐出来，ThinkStripper 只能按"宁可露标签也不静默吞内容"把整段推理原样发给
# 客户端，屏幕上就是一堵英文推理墙（用户报的"网页端某一格固定出现思考模式"）。
#
# 补上它 = 模型从 `</think>` **后面**开始生成，结构上开不了 think 段。代价别当成顺手
# 加的一行：这段文本今天是被模型**生成**出来、再被网关摘掉的，改成预填后同样的回答
# 少生成 ~4 个 token（略快），但 prompt 长度与 `completion_tokens` 都变了——
# 2026-09-18 之前量到的吞吐数字**不能与之后的逐字节互比**。收益见 toolcalls.py 顶部：
# 那两处"刻意偏差"（助手轮的推理包装、生成尾巴）在关思考下就此消掉，`known`
# 这笔 KV 记账也从"近似"变成"精确"。
THINK_OFF = "<think>\n\n</think>\n\n"


# ===========================================================================
# 1. 后端进程与帧协议
# ===========================================================================

class BackendError(Exception):
    """后端给出的一次失败。

    `soft` 区分两种失败，因为对客户端的语义完全不同：
      · soft=False（默认）——后端出了问题，或者会话的驱动线程已经死了。503，
        客户端该重试；会话同时被标死（见 SessionPool.mark_dead）。
      · soft=True——**这一轮被后端拒了，但会话是好的**（目前只有"上下文装不下"，
        见 main.cc 的 REJECT 帧）。这是客户端要改请求的错误（历史太长），所以回 400
        而不是 503——503 会让客户端原样重试，而原样重试必然再次被拒。
    """

    def __init__(self, message, soft=False):
        Exception.__init__(self, message)
        self.soft = soft


class _Request(object):
    """一次在飞的请求。reader 线程按 rid 把帧丢进来。"""

    def __init__(self, rid):
        self.rid = rid
        self.raw = bytearray()
        # 增量解码器：TokenToPiece 可能把一个 UTF-8 字的一切两半（多字节字符被切成
        # 两个 token），逐帧 decode 会在切缝处吐出替换字符。非流式路径整段 decode，
        # 流式路径必须用增量解码器。
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.done = threading.Event()
        self.error = None
        self.session = -1
        self.finish_reason = None
        self.prefill_tokens = 0
        self.decode_tokens = 0
        self.prefill_ms = 0.0
        self.decode_ms = 0.0
        self.context_tokens = 0
        self.was_cleared = False        # 后端在本轮里清过 KV（上下文将满 / RESET）
        self.rejected = False           # 后端在 prefill 前拒了这一轮（会话没死，见 REJECT）
        self.on_delta = None            # 流式回调；在 reader 线程里被调用
        self.dead = False               # 后端进程没了
        self.kv = None                  # 会话池本轮的复用决策，用于 X-KV-Reuse 头
        # 「谁把这一段字节交给客户端」的账本。reader 线程（有帧就投）和 HTTP 线程
        # （接手时把等待期间攒下的补发）是两条线程，两边都要原子地决定这件事：
        # `delivered` 是 raw 里**已经交出去**的字节数，两边都在 deliver_lock 里推进它，
        # 于是每个字节恰好交一次（重复投递和整段重发都由此消失），且解码器按字节
        # 顺序被喂——跨帧切开的 UTF-8 字符才拼得回来。
        self.delivered = 0
        self.deliver_lock = threading.Lock()

    def text(self):
        return self.raw.decode("utf-8", "replace")


def _argv_int(argv, name):
    """从后端命令行里取一个整数选项（`--ctx-size 8192` 或 `--ctx-size=8192`）。

    要它是为了 `/health` 能报出 `ctx_size`：**窗口数上限是上下文长度决定的**
    （KV 按"每路满上下文"预分配，ctx 越大每路越贵、能开的窗口越少），页面不拿到
    这个数就只能写死一个，换一份导出就变成骗人。

    解析**真实 argv** 而不是另存一份配置：子进程真正收到的那个值才算数。
    取不到/不是整数就回 None——宁可让页面少显示一格，也不要编一个数出来。
    """
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            val = argv[i + 1]
        elif a.startswith(name + "="):
            val = a.split("=", 1)[1]
        else:
            continue
        try:
            return int(val)
        except ValueError:
            return None
    return None


# ===========================================================================
# 资源观测：RK3588 这边网关自己读，RK1828 那边问后端（STAT 帧）
# ===========================================================================
#
# 为什么分两半：**只有持有设备的那个进程能查 RK1828**。四张卡是 PCIe 设备（driver
# `rkep`），主机侧既没有 hwmon 也没有 thermal_zone，`rknn-smi` 在后端驻留时必然是
# `Failed to initialize rknnsmi`（2026-09-20 实测；那个"还能出数"的 `rknn-smi info -w`
# 是它在后端起来**之前**就抢住了设备）。所以每卡设备内存只能由后端用
# `RKNN3_QUERY_DEVICE_MEM_INFO` 采，经 STAT 帧送过来。
#
# 而 RK3588 的一切都在 procfs/sysfs 里，网关直接读——不必为它绕一圈帧协议。

STAT_MIN_INTERVAL = 0.9     # 两次 STAT 之间至少隔这么久（页面约 2s 轮一次，够用）
# 发出 STAT 后最多等这么久。后端是**就地答原子量**、不碰设备也不进队列，正常 <1ms
# （板上实测：同一台机器连着问 10 次，HTTP 全程 7~11ms）。但 2026-09-20 在板上量到
# **偶发**的答复会晚于 0.35s——那版把 wait 设成 0.35，结果 /v1/system 每隔几次就拿到
# 上一份 doc，busy_pct 由 0.0 变成 None，页面上就在数字和 "—" 之间闪。
# 所以放成 1.0：页面轮询是 2s，多等这 0.65s 谁也不影响（这条路径只服务 /v1/system，
# 不在聊天请求的路上）；而"等不到就返回旧的"这条兜底照旧在——后端真卡住时它仍然
# 按时返回、只把年龄标出来。
STAT_WAIT = 1.0

_THERMAL_BASE = "sys/class/thermal"


def _read_proc_stat(path):
    """{'cpu': (busy, total), 'cpu0': …, …}，单位 jiffies。

    busy = total − idle − iowait。iowait 算空闲：它等的是磁盘，不是 CPU 在干活。
    """
    out = {}
    try:
        with open(path, "r") as fh:
            for line in fh:
                if not line.startswith("cpu"):
                    break
                f = line.split()
                try:
                    vals = [int(x) for x in f[1:]]
                except ValueError:
                    continue
                if len(vals) < 5:
                    continue
                idle = vals[3] + vals[4]
                out[f[0]] = (sum(vals) - idle, sum(vals))
    except (OSError, ValueError):
        return {}
    return out


def _read_meminfo(path):
    want = {"MemTotal": "total", "MemFree": "free", "MemAvailable": "available",
            "Buffers": "buffers", "Cached": "cached", "SReclaimable": "sreclaimable",
            "SwapTotal": "swap_total", "SwapFree": "swap_free"}
    out = {}
    try:
        with open(path, "r") as fh:
            for line in fh:
                f = line.split()
                if len(f) >= 2:
                    key = want.get(f[0].rstrip(":"))
                    if key:
                        out[key] = int(f[1]) * 1024
    except (OSError, ValueError):
        return {}
    return out


def _read_thermals(base):
    """板上 7 路热区。**全是 RK3588 自己的**（含它自己那个 npu-thermal），
    RK1828 四张卡一个都不在里面——这一点别在页面上含糊掉。"""
    out = []
    try:
        names = sorted(os.listdir(base))
    except OSError:
        return out
    for name in names:
        if not name.startswith("thermal_zone"):
            continue
        d = os.path.join(base, name)
        try:
            with open(os.path.join(d, "type"), "r") as fh:
                label = fh.read().strip()
            with open(os.path.join(d, "temp"), "r") as fh:
                milli = int(fh.read().strip())
        except (OSError, ValueError):
            continue
        out.append({"zone": name, "label": label, "c": round(milli / 1000.0, 1)})
    return out


class HostSampler(object):
    """RK3588 的 CPU / 内存 / 温度。没有依赖，也不碰 NPU。

    CPU 占用率是**两次采样之差**（和 `top` 同口径）。所以第一次调用没有"上一次"，
    `pct` 是 None 而不是 0.0——"还没量到"和"占用 0%"在屏幕上必须是两句话，
    否则页面刚打开那一瞬间会显示一个假的 0%。

    `root` 是给测试用的：把 `/proc`、`/sys` 换成一个夹具目录，本机（Windows，没有
    /proc）就能把**这段解析与百分比算术**真跑一遍。板卡上传默认的 `/`。
    读不到文件时每一项都退化成空/None，不抛——观测端点不该因为读不到一个文件而 500。
    """

    def __init__(self, root="/"):
        self.root = root
        self._lock = threading.Lock()
        self._prev = None          # (monotonic, {key: (busy, total)})

    def _p(self, rel):
        return os.path.join(self.root, rel)

    def sample(self):
        now = time.monotonic()
        cur = _read_proc_stat(self._p("proc/stat"))
        with self._lock:
            prev = self._prev
            self._prev = (now, cur)

        def pct(key):
            if not prev or not cur or key not in cur or key not in prev[1]:
                return None
            pb, pt = prev[1][key]
            cb, ct = cur[key]
            d_total = ct - pt
            if d_total <= 0:
                return None
            cpu = (cb - pb) * 100.0 / d_total
            return round(max(0.0, min(100.0, cpu)), 1)

        mem = _read_meminfo(self._p("proc/meminfo"))
        mem_out = dict(mem)
        if mem.get("total"):
            # MemAvailable 才是"还能拿来用的"，MemFree 不含可回收的页缓存。用 MemFree
            # 会把占用率显示得偏高（板上 16 GB 的机器上差 1 GB 量级）。
            usable = mem.get("available", mem.get("free", 0))
            mem_out["used"] = max(0, mem["total"] - usable)
            mem_out["used_pct"] = round(mem_out["used"] * 100.0 / mem["total"], 1)

        load = []
        try:
            with open(self._p("proc/loadavg"), "r") as fh:
                load = [float(x) for x in fh.read().split()[:3]]
        except (OSError, ValueError):
            pass

        uptime = None
        try:
            with open(self._p("proc/uptime"), "r") as fh:
                uptime = float(fh.read().split()[0])
        except (OSError, ValueError):
            pass

        cores = sorted([k for k in cur if k != "cpu"],
                       key=lambda s: int(s[3:]) if s[3:].isdigit() else 0)
        return {
            "cpu": {
                "n": len(cores) or None,
                "pct": pct("cpu"),
                "per_core": [pct(k) for k in cores],
                "loadavg": load,
            },
            "mem": mem_out,
            "thermal": _read_thermals(self._p(_THERMAL_BASE)),
            "uptime_s": round(uptime, 1) if uptime is not None else None,
        }


class CardStats(object):
    """把后端 STAT 帧变成每张卡的"现在"。

    NPU 利用率 = **Δ忙时 / Δ墙上时间**，两个数都取后端自己的单调时钟（`now_us`），
    所以不受网关这边调度抖动影响。忙时由后端在卡级锁内累加，量的是"这张卡真在算"
    的时间，不是"有没有人在等"。

    第一次调用没有"上一次"，`busy_pct` 是 None——**不是 0%**。这和 HostSampler 是
    同一个道理：刚打开页面就显示"NPU 0%"，会让人以为板卡闲着。

    同一份 doc 被连着采两次（后端那边 `now_us` 没动）时：不推进窗口，并把上一次算出的
    百分比原样再给一次。**多个看页面的人同时轮询时每个请求都会走到这条路上**——窗口
    是"上一次真拿到新 doc 到现在"这一段，谁先问谁算，后面的人拿到同一个数（而不是 "—"，
    也不是一个被切短的窗口算出来的数）。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._prev = None          # (now_us, {card_name: busy_us})
        self._last_pct = {}        # card_name -> 上一次算出来的利用率（同一份 doc 直接复用它）

    def sample(self, doc):
        if not isinstance(doc, dict):
            return {"ok": False, "cards": [], "busy_pct": None}
        now_us = doc.get("now_us") or 0
        raw = doc.get("cards") or []
        busy = {}
        for c in raw:
            if isinstance(c, dict) and c.get("name"):
                busy[c["name"]] = c.get("busy_us") or 0
        with self._lock:
            prev = self._prev
            # ⚠️ 同一份 doc 连着被采两次（now_us 没动）时**不推进 _prev**，而且下面
            # 把上一次算出来的百分比原样再给一次。这道"不推进"是 2026-09-20 在板上
            # 量出来的：STAT 的答复偶尔会晚于 STAT_WAIT（0.35s 那版实测过 0.3s 还没到），
            # 超时的调用拿到的是**上一份** doc ⇒ Δt=0 ⇒ 算不出比值 ⇒ 页面在 0% 和 "—"
            # 之间闪。若这时把 _prev 挪到这份旧 doc 上，下一份真 doc 的 Δ 还只剩下
            # 后半段，等于把一段观测悄悄切掉。两个毛病一起修：拿不到新 doc 就
            # **既不动窗口、也照旧显示上一次的值**（值本身没变旧，它是同一份数据算的）。
            if prev is None or not prev[0] or now_us > prev[0]:
                self._prev = (now_us, busy)
                repeated = False
            else:
                repeated = True

        cards = []
        pcts = []
        for c in raw:
            if not isinstance(c, dict):
                continue
            name = c.get("name")
            mem_at = c.get("mem_at_us") or 0
            total = c.get("mem_total") or 0
            free = c.get("mem_free") or 0
            card = {
                "name": name,
                "ctx_len": c.get("ctx_len"),
                "run_calls": c.get("run_calls"),
                "mem_total": total,
                "mem_free": free,
                "mem_used": max(0, total - free) if total else None,
                "mem_used_pct": (round(max(0.0, total - free) * 100.0 / total, 1)
                                 if total else None),
                "node_num": c.get("node_num") or None,
                "node_min_free": c.get("node_min_free") or None,
                # 快照年龄。mem_at_us == 0 表示后端**一次都没采到**（查询失败），
                # 那和"采过但很旧"是两回事，得让页面分开说。
                "mem_age_s": (round((now_us - mem_at) / 1e6, 1)
                              if (mem_at and now_us >= mem_at) else None),
                "busy_pct": None,
            }
            if repeated:
                # 这次是同一份 doc：百分比没得算，但**数据本身没变旧**，把上一次算的
                # 原样给回去，别让页面在数字和 "—" 之间闪（"—" 要留给"真的没有值"）。
                with self._lock:
                    card["busy_pct"] = self._last_pct.get(name)
            else:
                p0 = prev[1].get(name) if prev else None
                p1 = busy.get(name)
                if p0 is not None and p1 is not None and p1 >= p0:
                    span = now_us - prev[0]
                    card["busy_pct"] = round(max(0.0, min(100.0,
                                                         (p1 - p0) * 100.0 / span)), 1)
                if card["busy_pct"] is not None:
                    with self._lock:
                        self._last_pct[name] = card["busy_pct"]
            if card["busy_pct"] is not None:
                pcts.append(card["busy_pct"])
            cards.append(card)
        return {
            "ok": True,
            "cards": cards,
            # 四张卡是一条流水线，同一时刻大致同样忙；取平均当"整机 NPU 利用率"。
            "busy_pct": round(sum(pcts) / len(pcts), 1) if pcts else None,
        }


class Backend(object):
    """demo 子进程 + 帧协议。所有方法都线程安全。"""

    def __init__(self, argv, log_path, frame_fd=None):
        self.argv = argv
        self.log_path = log_path
        self.ctx_size = _argv_int(argv, "--ctx-size")
        # None  → 正常路径：自己开管道，用 pass_fds 把写端交给子进程，并追加
        #         --serve-fd <号> 告诉它认哪个号（Linux 板卡上走这条）。
        # "stdout" → 帧从子进程的 stdout 读。这条是给**本地自检的桩后端**用的：
        #         pass_fds 在 Windows 上直接不支持（实测 AssertionError），
        #         没有它就没法在本地把网关这半边验掉，每次试错都要等板卡上 230s
        #         的模型加载。顺带它也覆盖了 C++ 侧"该 fd 打不开就退回 stdout"的情形。
        self.frame_fd = frame_fd
        self.proc = None
        self.sessions = 0
        self.default_max_new_tokens = DEFAULT_MAX_NEW_TOKENS
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()   # 帧写 stdin 必须整体持锁，见 submit()
        self._reqs = {}
        self._next_rid = 1
        self._ready = threading.Event()
        self._dead = False
        self.stopping = False                 # 我们自己发的 QUIT，用来区分"正常收工"与"后端死了"
        self._rfile = None
        self._reader = None
        self.on_clear = None                  # 回调 (session, ctx, limit)
        # 资源观测：最近一次的 STAT 答复 + 它到达的时刻。读线程写、HTTP 线程读，
        # 用一把 cond 同时当锁和"新的答复到了"的信号（见 request_stats）。
        self.stats = None
        self.stats_seq = 0
        self.stats_at = 0.0
        self._stats_cv = threading.Condition()
        self._stats_sent_at = 0.0

    # ---- 生命周期 ----

    def start(self):
        logf = open(self.log_path, "wb")
        if self.frame_fd == "stdout":
            # 桩模式：帧走子进程的 stdout，日志走 stderr→文件。
            self.proc = subprocess.Popen(list(self.argv), stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=logf)
            logf.close()
            self._rfile = self.proc.stdout
        else:
            r, w = os.pipe()
            # 子进程按 --serve-fd 认这个号：pass_fds 保留父进程里的 fd 号，所以直接把 w
            # 告诉它，不需要 dup2 硬凑成 3（那在多线程下不安全）。
            argv = list(self.argv) + ["--serve-fd", str(w)]
            self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=logf,
                                         stderr=logf, pass_fds=(w,))
            os.close(w)  # 父进程必须关掉写端，否则读端在子进程退出后也等不到 EOF
            logf.close()
            self._rfile = os.fdopen(r, "rb")
        self._reader = threading.Thread(target=self._read_loop, name="frame-reader")
        self._reader.daemon = True
        self._reader.start()
        if not self._ready.wait(BACKEND_READY_TIMEOUT):
            raise BackendError("backend did not send READY within %.0fs"
                               % BACKEND_READY_TIMEOUT)
        if self._dead:
            raise BackendError("backend exited during startup; see %s" % self.log_path)

    def alive(self):
        return (not self._dead) and self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                self.stopping = True
                self.proc.stdin.write(b"QUIT\n")
                self.proc.stdin.flush()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=20)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass

    # ---- 读帧 ----

    def _read_loop(self):
        try:
            while True:
                line = self._rfile.readline()
                if not line:
                    break
                line = line.rstrip(b"\r\n")
                if not line:
                    continue
                tag, _, rest = line.partition(b" ")
                if tag == b"READY":
                    f = rest.split()
                    self.sessions = int(f[0])
                    self.default_max_new_tokens = int(f[1])
                    self._ready.set()
                elif tag in (b"DELTA", b"ERR", b"REJECT"):
                    # DELTA 是 `DELTA <rid> <n>`，ERR/REJECT 多一个 session 字段：
                    # `ERR <rid> <session> <n>`、`REJECT <rid> <session> <n>`
                    # （写侧见 cpp/main.cc 的 serve_err / serve_reject）。
                    # **两个不能共用一套解析**：按 DELTA 切第一刀的话，ERR 的 n 会拿到
                    # "<session> <n>"，int() 抛异常 → 读线程死 → 所有在途请求被判
                    # "backend exited" → 之后每个请求都 503，直到重启网关。一次"某轮
                    # 失败"就能把 4 会话的服务整体打掉，所以这里按字段位置取：
                    # 两种帧的 rid 都是第一个字段、n 都是最后一个字段。
                    # （ERR 的 session 这里用不上：请求自己知道落在哪个会话上。）
                    f = rest.split()
                    n = int(f[-1])
                    payload = self._rfile.read(n) if n > 0 else b""
                    self._dispatch_payload(tag, int(f[0]), payload)
                elif tag == b"STAT":
                    # `STAT <n>` + JSON 载荷，没有 rid（它不是某个请求的答复）。
                    # ⚠️ 这一帧**必须在这里被消费掉**：不认它的 tag 就读不走那 n 个
                    # 字节，整条帧流从此错位——和 ERR 帧那次是同一个坑（§9.7 末）。
                    f = rest.split()
                    n = int(f[0]) if f else 0
                    payload = self._rfile.read(n) if n > 0 else b""
                    self._on_stats(payload)
                elif tag == b"DONE":
                    f = rest.split()
                    self._handle_done(f)
                elif tag == b"CLEAR":
                    f = rest.split()
                    rid, session, ctx, limit = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                    if self.on_clear:
                        self.on_clear(session, ctx, limit)
                    q = self._get(rid)
                    if q is not None:
                        q.was_cleared = True
        except Exception as exc:                       # noqa: BLE001 - 读线程不能把异常吞成静默死锁
            sys.stderr.write("[gateway] frame reader failed: %r\n" % (exc,))
        finally:
            self._dead = True
            self._ready.set()
            # 后端没了 = 这个网关从此刻起只能回 503（池子里再等也不会多出会话来）。
            # 必须吵一声：现场除了这一行，其余症状全是"客户端卡住/503"，看不出是谁先死的。
            # 自己发 QUIT 收工的那条路（stopping）不算故障，别误报。
            if not self.stopping:
                sys.stderr.write("[gateway] backend exited (frame stream ended); every "
                                 "request fails with 503 until the gateway is "
                                 "restarted\n")
                sys.stderr.flush()
            with self._lock:
                pending = list(self._reqs.values())
            for q in pending:
                q.dead = True
                q.error = q.error or "backend exited"
                q.done.set()

    def _on_stats(self, payload):
        """读线程收到一帧 STAT：解出 JSON 放进缓存，并叫醒等它的人。

        解析失败**只丢这一帧**，不往上抛：帧流里出现一个解不开的 STAT 不应该把读线程
        带走（读线程一死，所有在途请求都被判 "backend exited"，整个服务变 503）。
        """
        try:
            doc = json.loads(payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            sys.stderr.write("[gateway] bad STAT payload: %r\n" % (exc,))
            return
        with self._stats_cv:
            self.stats = doc
            self.stats_at = time.monotonic()
            self.stats_seq += 1
            self._stats_cv.notify_all()

    def request_stats(self, min_interval=STAT_MIN_INTERVAL, wait=STAT_WAIT):
        """要一份最新的后端资源快照。**永远不会抛异常，也永远不长时间阻塞。**

        两道节制：
        · `min_interval` 之内问过就直接给缓存——STAT 和 REQ 走同一条 stdin，
          没人在看页面时一个字节都不该发，有人在看时也不该比页面刷新还快；
        · 发出去之后最多等 `wait` 秒。后端是**就地答原子量**（不排队、不碰设备），
          正常 <1ms；等不到说明后端卡住了，那就返回上一次的缓存并让页面自己标旧
          ——**卡住的时候返回旧数比返回 500 有用得多**，看护恰恰要看这个。
        """
        with self._stats_cv:
            now = time.monotonic()
            if now - self._stats_sent_at < min_interval:
                return self.stats
            if self._dead or self.proc is None or self.proc.stdin is None:
                return self.stats
            want = self.stats_seq + 1
            self._stats_sent_at = now
            try:
                with self._write_lock:
                    self.proc.stdin.write(b"STAT\n")
                    self.proc.stdin.flush()
            except Exception:                              # noqa: BLE001
                # 写失败就是后端没了/管道断了。这条路径**不能抛**：/v1/system 是观测
                # 端点，它挂掉不该把看页面的人也带走。
                return self.stats
            deadline = now + wait
            while self.stats_seq < want:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                self._stats_cv.wait(left)
            return self.stats

    def _get(self, rid):
        with self._lock:
            return self._reqs.get(rid)

    def _dispatch_payload(self, tag, rid, payload):
        q = self._get(rid)
        if q is None:
            return          # 客户端已经放弃这个 rid 了，丢掉帧即可
        if tag == b"DELTA":
            # 与 HTTP 线程的"接手补发"互斥：要么这一段由我投递（delivered 随之推进），
            # 要么它还留在 raw 里等着被补发，两者只能发生一件。
            with q.deliver_lock:
                q.raw.extend(payload)
                if q.on_delta is None:
                    return          # 还没人接手（HTTP 线程尚未走到 attach）：先攒着
                chunk = q.decoder.decode(bytes(payload))
                q.delivered += len(payload)
                cb = q.on_delta
            if chunk:
                try:
                    cb(chunk)
                except Exception:
                    # 客户端断开了：不影响后端，继续把这一轮读完
                    q.on_delta = None
        elif tag == b"REJECT":
            # 后端在 prefill **之前**拒了这一轮（上下文装不下）：会话还活着，只是这一轮
            # 一个 token 都没跑。所以只失败这一条请求，**绝不能标死这个会话**。
            q.rejected = True
            q.error = payload.decode("utf-8", "replace")
            q.done.set()
        else:               # ERR
            q.error = payload.decode("utf-8", "replace")
            q.done.set()

    def _handle_done(self, f):
        if len(f) < 8:
            # 字段不齐的 DONE **不能静默 return**：这个请求从此既没有 DONE 也没有 ERR，
            # 会一直挂在 HTTP 线程的 q.done.wait(REQUEST_TIMEOUT) 上（默认 1800s），
            # 期间白占着一个会话租约——别的对话只能排队等它超时。按 rid 判它失败，
            # 让客户端立刻拿到一个能看懂的错误，同时读线程照常活着（不牵连其他请求）。
            try:
                rid = int(f[0])
            except (IndexError, ValueError):
                sys.stderr.write("[gateway] malformed DONE frame: %r\n" % (f,))
                return
            q = self._get(rid)
            if q is not None and not q.done.is_set():
                q.error = "malformed DONE frame from backend: %r" % (f,)
                q.done.set()
            return
        rid = int(f[0])
        q = self._get(rid)
        if q is None:
            return
        q.session = int(f[1])
        # 头行是 ASCII 字节串：必须 decode 成 str，否则它会被原样塞进 JSON 里，
        # json.dumps 对 bytes 直接抛 TypeError（非流式路径必崩）。
        q.finish_reason = f[2].decode("ascii", "replace")
        q.prefill_tokens = int(f[3])
        q.decode_tokens = int(f[4])
        q.prefill_ms = float(f[5])
        q.decode_ms = float(f[6])
        q.context_tokens = int(f[7])
        if q.on_delta is not None:
            tail = q.decoder.decode(b"", True)
            if tail:
                try:
                    q.on_delta(tail)
                except Exception:
                    q.on_delta = None
        q.done.set()

    # ---- 提交请求 ----

    def submit(self, prompt, max_new_tokens, session=-1, reset=False):
        if self._dead:
            raise BackendError("backend exited")
        payload = prompt.encode("utf-8")
        with self._lock:
            rid = self._next_rid
            self._next_rid += 1
            q = _Request(rid)
            self._reqs[rid] = q
        header = "REQ %d %d %d %d %d\n" % (rid, session, max_new_tokens,
                                           1 if reset else 0, len(payload))
        # 必须整体持锁一次性写出：头行与载荷是两条独立的 write，中间被另一条 HTTP
        # 线程插进它自己的头行，帧流立刻就错位了。
        with self._write_lock:
            try:
                self.proc.stdin.write(header.encode("ascii") + payload)
                self.proc.stdin.flush()
            except Exception as exc:
                with self._lock:
                    self._reqs.pop(rid, None)
                raise BackendError("failed to write request: %r" % (exc,))
        return q

    def wait(self, q, timeout=REQUEST_TIMEOUT):
        if not q.done.wait(timeout):
            raise BackendError("timeout after %.0fs waiting for request %d" % (timeout, q.rid))
        if q.error:
            raise BackendError(q.error, soft=q.rejected)
        return q

    def release_request(self, q):
        with self._lock:
            self._reqs.pop(q.rid, None)


# ===========================================================================
# 2. 会话池（粘性 + 前缀校验）
# ===========================================================================

_VERBOSE = False        # 由 main() 按 --verbose 置位。模块级，因为会话池拿不到 Gateway


def _note(fmt, *args):
    """会话池的事件（排队、回收）。

    为什么要单独一条日志：**多用户场景下"谁在等、谁被回收了"是唯一能看见调度的地方**。
    只盯着 tok/s 看不出排队——第 6 个人没拿到会话时他不会变慢，他是拿不到。
    """
    if _VERBOSE:
        sys.stderr.write("[pool] " + (fmt % args) + "\n")
        sys.stderr.flush()


def _key_label(key):
    """把会话身份缩短，好进日志。显式的 id 原样保留（`id:alice`），内容摘要截断。"""
    if key is None:
        return "(anon)"
    return key if len(key) <= 24 else key[:21] + "..."


class Lease(object):
    __slots__ = ("session", "sent_prompt", "base", "reset", "key", "waited")

    def __init__(self, session, sent_prompt, base, reset, key=None, waited=0.0):
        self.session = session
        self.sent_prompt = sent_prompt
        self.base = base            # acquire 时该会话已知的文本（reset 时为 ""）
        self.reset = reset
        self.key = key              # 这段对话的身份，release 时要靠它续期
        self.waited = waited        # 排队等了多久（0 = 没等）


class SessionPool(object):
    """把对话钉到会话上，并决定每轮能不能只发差异部分。

    这里的 `known[s]` 是"该会话的 KV 里确定已经装进去的文本"，三态：
      · 字符串 => 确定装着这些文本，能不能复用只看一条判据：本次完整 prompt 是否以它
        开头。判据不成立就 RESET 发全量——**正确性靠这条判据，不靠粘性记录**，所以
        映射错了最多慢一点，不会答出串味的上下文。
      · ""     => 确定是空的（刚启动、或上一轮自己在空会话上跑完）。
      · None   => 内容未知（清过 KV，而清完之后又被写进了什么，我们从帧序上分不出来）。
        必须 RESET。这一态是踩过坑才加的：见 release() 的注释。
    """

    def __init__(self, backend, idle_ttl=DEFAULT_IDLE_TTL,
                 queue_timeout=DEFAULT_QUEUE_TIMEOUT,
                 contend_idle=DEFAULT_CONTEND_IDLE):
        self.backend = backend
        self.n = backend.sessions
        self.cv = threading.Condition()
        self.known = [""] * self.n
        self.busy = [False] * self.n
        self.dead = [False] * self.n
        self.bound = {}          # conversation key -> session index
        self.last_used = {}      # conversation key -> 最后一次提问的 monotonic 时刻
        self.waiting = {}        # conversation key -> 进队列的 monotonic 时刻（仅用于观测）
        # 空闲回收：一段对话静默超过 idle_ttl 就把它占的会话收回给排队的人。
        # **这不是优化，是"排队"能成立的前提**：没有它，5 个会话被 5 段对话绑着时第 6 段
        # 对话永远等不到——网关根本不知道谁"说完了"（客户端不发结束消息，浏览器关了也
        # 没人通知）。0 或负数 = 不回收（那就等价于第 6 段对话永远排队）。
        self.idle_ttl = float(idle_ttl or 0)
        # 第二级：**已经有请求非等不可**（一个空闲会话都没有）时用的静默阈值。见文件头
        # DEFAULT_CONTEND_IDLE 的注释。0 = 关掉第二级。
        self.contend_idle = float(contend_idle or 0)
        self.queue_timeout = float(queue_timeout)
        self.reaped = 0          # 累计回收了几次（观测用）
        # bound 的上限：键是"system + 首条 user"的摘要或被回收之前攒下的对话。有了空闲
        # 回收，正常情况这张表只装活跃对话；这个上限只是长期运行的安全网——丢绑定只影响
        # 速度，不影响正确性。
        self.bound_cap = 4096
        backend.on_clear = self._on_clear

    def _on_clear(self, session, ctx, limit):
        # 后端把该会话的 KV 清了（RESET 或上下文将满）。粘性记录必须立刻作废，否则下一轮
        # 网关以为历史还在、只发差异部分，而模型那边什么都没有——答出来的东西看着通顺
        # 却串味，是最难查的一类错。
        #
        # 置 None（= 未知，下一轮必须 RESET）而不是 ""（= 确定是空的）：清 KV 只是这段
        # 历史的**起点**，紧接着这次请求自己的 prompt 就会被 prefill 进去。此刻这次
        # 请求的 release() 还没跑（它在 HTTP 线程里等 DONE），所以这个窗口里"KV 是空的"
        # 这句话是错的；等 release() 回来会用「本轮发出 + 本轮生成」写回确定值。
        with self.cv:
            if 0 <= session < self.n:
                self.known[session] = None
            self.cv.notify_all()

    def mark_dead(self, session):
        # 会话的推理失败后，demo 里那个驱动线程会退出，钉在它上面的请求永远不会被消费
        # （会一直躺在待办队列里占额度）。这里把它标死，不再往上派活。
        with self.cv:
            if 0 <= session < self.n:
                self.dead[session] = True
                self.busy[session] = False
            for key, s in list(self.bound.items()):
                if s == session:
                    del self.bound[key]
                    self.last_used.pop(key, None)
            self.cv.notify_all()

    def _reap_idle_locked(self, now, ttl=None, limit=None, why="静默"):
        """把静默超过 TTL 的对话占的会话收回。调用方必须已持 self.cv。

        只收**没在跑**的会话；正在跑的那一轮不能被打断（打断也没意义，它马上就还回来）。
        回收只解绑 + 作废 known，**不动 KV**——下一个拿到这个会话的对话若前缀对不上，
        `_make_lease` 自己会 RESET。所以回收本身不会有额外的清 KV 开销。

        `ttl=None` 用常规的 `idle_ttl`（没人排队时的行为）；给 `contend_idle` 就是
        "有人在等"那一级。`limit` 限制收几个——**抢手的那一级只收一个**，收多了等于把
        好几段对话的 KV 一起扔掉，只为了给一个请求腾地方。

        一轮里按**静默时长从长到短**收，不是按 dict 顺序：受害者必须是"最闲的那一段"，
        否则同一份脚本会有时抢到 A、有时抢到 B，行为不可预期（这正是当初决定"不抢占"
        的理由，现在用"按静默挑"把它解决掉）。
        """
        ttl = self.idle_ttl if ttl is None else ttl
        if ttl <= 0:
            return 0
        victims = sorted(((now - self.last_used.get(k, now), k, s)
                          for k, s in self.bound.items() if not self.busy[s]),
                         reverse=True)
        n = 0
        for idle, k, s in victims:
            if idle <= ttl:
                break                       # 已按静默倒序排好，后面只会更不闲
            if limit is not None and n >= limit:
                break
            del self.bound[k]
            self.last_used.pop(k, None)
            if 0 <= s < self.n:
                self.known[s] = None        # 里面的内容从此未知，下一轮必须 RESET
            self.reaped += 1
            n += 1
            _note("会话回收：%s %s %.1fs > %.1fs，s%d 交还给排队者"
                  % (_key_label(k), why, idle, ttl, s))
        return n

    def acquire(self, key, prompt, timeout=None):
        """返回 Lease；排队超过 timeout 仍拿不到就抛 BackendError。

        **不抢占**：所有会话都被别的对话绑着时，这里只等，不把谁的会话夺过来。
        抢夺看着"响应快"，代价却是被抢的那段对话下一轮前缀对不上、要全量重算 prefill
        （多花 2~3 秒），而且**谁被抢取决于请求到达顺序**——同一份脚本时好时坏。
        等，加上 `_reap_idle_locked` 把真正闲下来的会话交出来，行为才是可预期的。
        """
        if timeout is None:
            timeout = self.queue_timeout
        deadline = time.time() + timeout
        t_enter = time.time()
        queued = False
        with self.cv:
            while True:
                # 等不出结果的两件事，必须在**进队列之前**判掉。不判的代价不是"慢一点"，
                # 是请求在队列里空等满 queue_timeout 而屏幕上什么都不打印——现场看起来
                # 像"网关卡了"，真相在 `GET /v1/pool` 里一眼可见（四个 slot 全是 dead）。
                #   · 后端子进程没了：一个超大请求就能让它退出（见 main.cc 的
                #     kServeMaxPromptBytes 分支），池子里再等也不会凭空多出会话来；
                #   · 四个会话全 dead：dead 没有任何复活路径（见 mark_dead），等下去
                #     是纯粹的干等。
                # 2026-09-17 上板实测：一个 3.4 MB 的请求打死后端 → 四个会话全 dead →
                # 5 段对话（4 个终端 + 1 个 Claude Code）在这里排到 600s 超时才收到 503。
                if not self.backend.alive():
                    self.waiting.pop(key, None)
                    raise BackendError("backend exited; every request fails until the "
                                       "gateway is restarted")
                if all(self.dead):
                    self.waiting.pop(key, None)
                    raise BackendError("all %d sessions are dead; restart the gateway"
                                       % self.n)

                # 优先复用已经绑在这一段对话上的会话：只有它才可能已经有这段历史。
                cand = self.bound.get(key)
                if cand is not None and not self.dead[cand] and not self.busy[cand]:
                    if queued:
                        self.waiting.pop(key, None)
                        _note("排队结束：%s 等了 %.1fs，拿到 s%d"
                                              % (_key_label(key), time.time() - t_enter, cand))
                    return self._make_lease(cand, prompt, t_enter, key)
                if cand is not None and not self.dead[cand]:
                    pass        # 同一段对话的上一轮还在跑：等它，不能换会话
                else:
                    # 挑一个空闲会话。**"空闲"不等于"没主"**：会话可能正被另一段对话绑着
                    # （bound 里挂着），只是它此刻没在跑。把新对话派到那种会话上，两段对话
                    # 就会钉在同一个会话上——后端同一个会话一次只跑一条，于是它们只能互相
                    # 等，答案照样对、吞吐掉一半。所以候选里必须排除有主的。
                    now = time.time()
                    self._reap_idle_locked(now)
                    owned = set(self.bound.values())
                    free = [s for s in range(self.n)
                            if not self.dead[s] and not self.busy[s] and s not in owned]
                    # `idle_ttl <= 0` 是**总开关**（文档里的"0 = 不回收"）：第二级只是
                    # 对它的细化，不能绕过它。回归套件正是靠 `--idle-ttl 0` 来证明自己
                    # 真的在主动 close、而不是碰巧被回收救了——这里绕过就等于把那套证明
                    # 悄悄作废。
                    if not free and self.contend_idle > 0 and self.idle_ttl > 0:
                        # 一个空闲会话都没有 = 这个请求**非等不可**，才启用第二级阈值。
                        # 放在"确认没得用"之后判，是为了让常规路径（还有空闲会话）与
                        # 之前逐字节一样——那才是 KV 复用真正省下时间的地方。
                        # 只收一个：腾出够它用的地方就行，别顺手把好几段对话的 KV 扔掉。
                        self._reap_idle_locked(now, ttl=self.contend_idle, limit=1,
                                               why="静默（有人排队）")
                        owned = set(self.bound.values())
                        free = [s for s in range(self.n)
                                if not self.dead[s] and not self.busy[s] and s not in owned]
                    if free:
                        # 优先没装过东西的（省一次 RESET）：unknown（None）排在有内容的前面，
                        # 反正都要 RESET，清空的会话至少不用先扔垃圾。
                        best = min(free, key=lambda s: (self.known[s] is not None, s))
                        if queued:
                            self.waiting.pop(key, None)
                            _note("排队结束：%s 等了 %.1fs，拿到 s%d"
                                                  % (_key_label(key), time.time() - t_enter, best))
                        # key 是 None（请求里既没有 conversation_id 也没有 user，连一条
                        # user 消息都没有 => conversation_key 返回 None）= **无身份的请求，
                        # 不能记绑定**。记了的话所有匿名请求都会共用 bound[None] 这一个
                        # 条目，于是它们全被钉到同一个会话上互相等——两段素不相识的对话
                        # 排成一队，正是这个池子存在的意义反面。
                        if key is not None:
                            self.bound[key] = best
                            self.last_used[key] = now
                            if len(self.bound) > self.bound_cap:
                                # 安全网，正常有回收就走不到。丢绑定只让那段对话下次重新
                                # 挑会话，正确性不受影响；不丢就是一条随对话数线性增长的
                                # 慢泄漏。
                                oldest = next(iter(self.bound))
                                self.bound.pop(oldest)
                                self.last_used.pop(oldest, None)
                        return self._make_lease(best, prompt, t_enter, key)
                if time.time() >= deadline:
                    self.waiting.pop(key, None)
                    # 能排满 timeout 才走到这里，说明不是"别人占着没放"，而是**四个会话
                    # 全都在真的跑**（第二级回收只收没在跑的，所以它救不了这种情况）。
                    # 这条区分很重要：用户问"NPU 没人用为什么还要等"时，看一眼这个提示
                    # 和 `GET /v1/pool` 就能知道是排队还是真忙。
                    raise BackendError(
                        "no session available: all %d sessions are busy or bound to "
                        "other conversations (waited %.0fs; idle_ttl=%.0fs; "
                        "contend_idle=%.0fs). Retry later or reuse a conversation_id "
                        "that already has a session."
                        % (self.n, timeout, self.idle_ttl, self.contend_idle))
                if not queued:
                    queued = True
                    self.waiting[key] = t_enter
                    _note("进入排队：%s 等空闲会话（当前 %d/%d 被占，"
                                          "队列 %d 人）"
                                          % (_key_label(key), len(self.bound), self.n,
                                             len(self.waiting)))
                # 唤醒间隔取小值：等待期间会有别人 release / 到点该回收，靠它重查。
                self.cv.wait(min(0.5, max(0.05, deadline - time.time())))

    def close(self, key):
        """客户端主动说"这段对话结束了"：立刻把会话交还给池子。

        为什么必须有这个口子：网关**不知道一段对话什么时候结束**（客户端不发关闭消息），
        所以只能靠 `IDLE_TTL` 超时回收。对"来问一句就走"的客户端（脚本、Agent）这很糟：
        他一个人连着问 4 个不相干的问题，就占满 4 个会话、把后面所有人挡在队列里
        **整整一个 IDLE_TTL**。排队策略要想在真实使用里成立，就得有一条"我走了"的路。

        返回放掉的 session 号；`None` = 这段对话本来就没有会话（幂等，重复 close 不报错）。
        `busy` 时抛 `BackendError`（这一轮还在跑，不能抽走它脚下的会话；等它答完再来）。
        """
        if key is None:
            raise BackendError("no conversation id given")
        with self.cv:
            s = self.bound.get(key)
            if s is None:
                return None
            if self.busy[s]:
                raise BackendError("conversation is still generating; retry after it "
                                   "finishes")
            del self.bound[key]
            self.last_used.pop(key, None)
            if 0 <= s < self.n:
                # 内容从此未知：下一个拿到它的人必须 RESET，否则会把新 prompt 追加在
                # 上一段对话的 KV 后面（= 坑 1 那个"两份历史"）。**不能写 ""**。
                self.known[s] = None
            self.cv.notify_all()
            _note("会话交还：%s 放掉 s%d" % (_key_label(key), s))
            return s

    def snapshot(self):
        """给 `GET /v1/pool` 看的一眼状态：谁占着哪个会话、谁在等。

        为什么需要它：多用户场景下"第 6 个人在排队"这件事**从吞吐上是看不出来的**
        ——他没拿到会话，不是变慢了。终端用户和排障的人都得有个地方能看。
        """
        now = time.time()
        with self.cv:
            by_sess = {}
            for k, s in self.bound.items():
                by_sess[s] = k
            slots = []
            for s in range(self.n):
                k = by_sess.get(s)
                slots.append({
                    "session": s,
                    "state": ("dead" if self.dead[s] else
                              "busy" if self.busy[s] else
                              "idle" if k is not None else "free"),
                    # null = 没人占（不要写成字符串 "(anon)"，那看起来像"有个叫 anon 的
                    # 用户在占着"，正交性会丢）
                    "conversation": None if k is None else _key_label(k),
                    # 原始键。`conversation` 是给人看的（长了会截断），这个是给机器用的：
                    # 匿名对话（`h:...`）的键客户端算不出来，只能从这儿读出来，再用
                    # `POST /v1/conversations/close {"key": ...}` 把它放掉。没有这一条，
                    # 一段匿名对话会一直占到 IDLE_TTL 超时——**看得到却踢不掉**。
                    "key": k,
                    "idle_s": (round(now - self.last_used.get(k, now), 1)
                               if k is not None and not self.busy[s] else None),
                })
            waiting = [{"conversation": _key_label(k), "waited_s": round(now - t, 1)}
                       for k, t in sorted(self.waiting.items(), key=lambda kv: kv[1])]
            return {
                "sessions": self.n,
                "idle_ttl_s": self.idle_ttl,
                # 第二级阈值也报出来：只看 `/v1/pool` 的人得能算出"我这个请求最坏等几秒"。
                # 有它，`排队 226s` 这种现场不用再翻日志猜是哪一级没生效。
                "contend_idle_s": self.contend_idle,
                "queue_timeout_s": self.queue_timeout,
                "reaped_total": self.reaped,
                "slots": slots,
                "waiting": waiting,
            }

    def _make_lease(self, session, prompt, t_enter=None, key=None):
        self.busy[session] = True
        waited = max(0.0, time.time() - t_enter) if t_enter else 0.0
        base = self.known[session]
        if base and prompt.startswith(base):
            return Lease(session, prompt[len(base):], base, False, key=key, waited=waited)
        # 复用不了，两种情况必须分开：
        #   base 非空但前缀对不上 => 里面确实装着别的对话，必须 RESET，否则这次的全量
        #     prompt 会**追加**在旧上下文后面，模型看到两份历史；
        #   base 是 None（清过 KV，内容未知）=> 同样必须 RESET。这里不能写成
        #     `bool(base)`：None 和 "" 都是假值，会把"未知"当"空"处理，正好踩中
        #     上面那个追加的坑。
        return Lease(session, prompt, "", base is None or bool(base),
                     key=key, waited=waited)

    def discard(self, lease):
        """这一轮**根本没跑**（后端在 prefill 前拒了它）：只把会话还回去，不动 known。

        不能走 release()：它会把「本轮发出的 prompt」算进"KV 里确定装着的内容"，而这一
        轮一个 token 都没 prefill 进去。照 release 记账的话，下一轮网关会以为历史都在、
        只发差异部分——模型看到的是一段没有开头的对话，答出来通顺但串味，正是这套记账
        存在的意义反面。known 这里**一个字节都不动**，两种来源各自已经是对的：
          · 上下文装不下 —— 后端在拒之前先发了 CLEAR，`_on_clear` 已把它置成 None，
            下一轮必定 RESET 全量重发；
          · 载荷超帧上限 —— 后端**不发 CLEAR**（KV 一个字节都没动），known 保持原样
            正好描述着 KV 里真正装着的东西。
        """
        with self.cv:
            s = lease.session
            if 0 <= s < self.n:
                self.busy[s] = False
            if lease.key is not None and lease.key in self.bound:
                self.last_used[lease.key] = time.time()
            self.cv.notify_all()

    def release(self, lease, generated_text, was_cleared):
        """本轮结束后，把「KV 里现在确定装着什么」写回去。

        这里曾经写成 `if was_cleared: known[s] = ""`——那是**错的**，而且是那种只在
        会话变脏之后才现形的错：Reset 必然伴随一帧 CLEAR，于是每段对话只要第一轮需要
        RESET（会话里装着上一段对话），第二轮开始就永远拿不回粘性，每轮全量重算、
        KV 里还越堆越多份历史。清 KV 发生在**本轮 prefill 之前**（见 main.cc 的
        serve_clear -> clear_conversation_kv -> prefill 顺序），所以本轮结束时 KV 里
        确定装着的就是「本轮发出 + 本轮生成」；被清掉的旧 base 不能再算进去。

        `generated_text` 传进来的是**客户端会回显的版本**（调用方已摘掉 think 段）。
        于是有个已知的小偏差：KV 里其实还留着模型原始生成的 `<think>…</think>`，
        而 known 声称没有。方向是安全的——模型最多多看到一小段空的推理标记（那**本来
        就是它自己这一轮吐的**），不会少看到任何东西；反过来按原始版记账则会让关思考的
        每轮都判"前缀对不上"，缓存静默全废。
        """
        with self.cv:
            s = lease.session
            if 0 <= s < self.n:
                if was_cleared:
                    self.known[s] = lease.sent_prompt + generated_text
                else:
                    self.known[s] = lease.base + lease.sent_prompt + generated_text
                self.busy[s] = False
            # 续期：TTL 数的是"这段对话多久没提问了"，所以在这一轮**开始和结束**都刷新。
            # 只刷开始的话，一轮跑很久（长上下文几十秒）会被自己在跑到一半时判成空闲。
            if lease.key is not None and lease.key in self.bound:
                self.last_used[lease.key] = time.time()
            self.cv.notify_all()


# ===========================================================================
# 3. 聊天模板渲染
# ===========================================================================

def _content_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):       # OpenAI 允许 content 是分段数组
        out = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in (None, "text"):
                out.append(part.get("text") or "")
        return "".join(out)
    return ""


def render_messages(messages, enable_thinking=None, tools=None):
    """按 Qwen3.5 模板把 OpenAI 的 messages 渲染成一个 prompt 字符串。

    渲染放在网关侧而不是 demo 里：demo 的模板只有「首轮带 system / 后续轮不带」两态，
    表达不了任意角色的历史消息（Agent 会发 system + user + assistant + tool 的长列表）。
    """
    msgs = [m for m in (messages or []) if isinstance(m, dict)]
    if not msgs or msgs[0].get("role") != "system":
        msgs = [{"role": "system", "content": QWEN35_DEFAULT_SYSTEM}] + msgs

    # 关思考：Qwen3 系列用软开关 /no_think，贴成 user 消息末尾的一个词。
    # 实测官方 server 的 chat_template_kw.enable_thinking=false 在这条 GGUF 上不生效
    # （输出照样带 <think>），所以这里退回到模板层面的开关。
    #
    # 贴在**每一条** user 消息上，而不是只贴最后一条。这不是风格选择，是粘性复用的
    # 前提：只贴最后一条的话，第一轮 u1 是"最后一条"（带标记），第二轮 u1 变成历史
    # 消息（不带）=> 同一段历史两轮渲染出的字节不同 => 前缀判据失败 => 每轮全量重算。
    # 实测过这个退化（nothink_check.py）：关思考时第二轮 reset=1、base=0、sent=full，
    # 缓存全废但答案仍然是错的看不出来——属于"静默变慢"。全贴则每轮渲染同一段历史
    # 都是同样的字节，复用照常。第一条 user 在两种规则下都带标记，所以对首轮无影响。
    #
    # **工具返回那个 user 轮不贴**：它是模板合成的、不是用户说的话，贴上去属于往历史里
    # 塞模板没有的东西（渲染结果仍然逐轮稳定，所以复用不受影响）。
    no_think = (enable_thinking is False)

    parts = []
    i = 0
    while i < len(msgs):
        m = msgs[i]
        role = m.get("role") or "user"
        text = _content_text(m.get("content"))

        if role == "tool":
            # 模板把工具返回包在 <tool_response> 里，而且**连续多条合并进同一个 user
            # 轮**（一个回合里并行调了三个工具，就是三条 tool 消息）。这跟"每条渲染成
            # 一个 <|im_start|>tool 轮"完全不同——模板里根本没有 tool 角色轮，旧实现
            # 渲成 <|im_start|>tool 是错的，模型收到的是它没见过的形状。
            blocks = []
            while i < len(msgs) and (msgs[i].get("role") or "") == "tool":
                blocks.append(tool_response_block(_content_text(msgs[i].get("content"))))
                i += 1
            parts.append("%suser\n%s%s" % (IM_START, "\n".join(blocks), IM_END))
            continue

        if role == "system" and tools:
            text = render_tools_system(tools, text)
        elif role == "assistant":
            calls = normalize_tool_calls(m.get("tool_calls"))
            if calls:
                text = render_assistant_turn(text, calls)
        if no_think and role == "user":
            text = text + " /no_think"
        if no_think and role == "assistant":
            # 助手轮按模板的"推理包装"渲。关思考时这个包装就是那对空标记，**不含推理
            # 正文**，所以我们渲得起（思考开着时包装里要放模型当时的推理，那是渲染侧
            # 拿不到的东西，见 toolcalls.py 顶部的偏差 1）。
            parts.append("%s%s%s" % (ASSISTANT_HEAD + THINK_OFF, text, IM_END))
        else:
            parts.append("%s%s\n%s%s" % (IM_START, role, text, IM_END))
        i += 1
    # 生成位置同样补上：模型从这里接着写，写出来的就是正文本身。
    parts.append(ASSISTANT_HEAD + (THINK_OFF if no_think else ""))
    return "".join(parts)


def conversation_key(messages, explicit=None):
    """一段对话的身份。

    取"system + 第一条 user 消息"的摘要：它在一段对话的各轮之间稳定，不同对话之间
    基本不会撞。撞了也不会错——前缀校验会挡住复用，只是退回全量重算。
    """
    if explicit:
        return "id:" + str(explicit)
    msgs = [m for m in (messages or []) if isinstance(m, dict)]
    sys_text = ""
    first_user = ""
    for m in msgs:
        if m.get("role") == "system" and not sys_text:
            sys_text = _content_text(m.get("content"))
        elif m.get("role") == "user" and not first_user:
            first_user = _content_text(m.get("content"))
        if sys_text and first_user:
            break
    if not first_user:
        return None
    h = hashlib.sha1(("%s\x00%s" % (sys_text, first_user)).encode("utf-8"))
    return "h:" + h.hexdigest()[:16]


# ===========================================================================
# 4. HTTP / OpenAI
# ===========================================================================

# 演示页（多路对话框那个，窗口数由页面自己问 /health 决定）与网关同目录，用 GET /demo 直接打开。
# 为什么由网关自己发而不是另起一个静态服务器：**同源**，浏览器就不会有 CORS 的坑，
# 演示时也只有一个进程要管。CORS 头照样给（见下），是为了页面被存到本地用
# file:// 打开、或者被别的 Agent 前端引用时也能用。
VIEW_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_PAGE = "demo_4chat.html"

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    # 浏览器默认**不让 JS 读自定义响应头**，X-KV-Reuse 必须显式 expose 出来，
    # 否则页面上"复用到底有没有生效"这一栏永远是空的。
    "Access-Control-Expose-Headers": "X-KV-Reuse",
}


class Gateway(object):
    def __init__(self, backend, pool, verbose=False):
        self.backend = backend
        self.pool = pool
        self.verbose = verbose
        self.started_at = time.time()
        self.host = HostSampler()
        self.cards = CardStats()

    def log(self, fmt, *args):
        if self.verbose:
            sys.stderr.write("[gateway] " + (fmt % args) + "\n")
            sys.stderr.flush()

    def system_snapshot(self):
        """`/v1/system` 的内容：RK3588（本机读）+ RK1828（后端 STAT 帧）+ 网关自己。

        **两半的可信度不一样，所以不用同一个形状糊在一起**：`host` 是这一秒读到的，
        `cards` 是从后端来的、带着它的采样时刻；`backend_stats_age_s` 说后者有多旧。
        """
        doc = self.backend.request_stats()
        cards = self.cards.sample(doc)
        stats_age = (round(time.monotonic() - self.backend.stats_at, 1)
                     if self.backend.stats_at else None)
        pool = self.pool.snapshot()
        return {
            "ts": round(time.time(), 3),
            "host": self.host.sample(),
            "npu": cards,
            "backend": {
                "alive": self.backend.alive(),
                "sessions": self.backend.sessions,
                "ctx_size": self.backend.ctx_size,
                "stats_age_s": stats_age,
            },
            "gateway": {
                "busy": sum(1 for s in pool["slots"] if s["state"] == "busy"),
                "bound": sum(1 for s in pool["slots"] if s["state"] in ("busy", "idle")),
                "dead": sum(1 for s in pool["slots"] if s["state"] == "dead"),
                "waiting": len(pool["waiting"]),
                "uptime_s": round(time.time() - self.started_at, 1),
            },
        }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "rkllm-openai-gateway/1.0"
    # 读请求行的超时。默认（None = 不超时）+ ThreadingHTTPServer 是一个没有上限的组合：
    # 只连上、一个字节都不发的客户端会把那一个线程**永久**占住（每个连接一个线程，没有
    # 上限），几百个空连接就能把内存吃干。超时后按默认行为关连接、放掉线程。
    # 注意它只作用于"读下一次请求"，不作用于我们正在生成的那一轮——生成期间我们在写、
    # 不在读，长上下文一轮跑几百秒也不会被它打断。
    timeout = 60.0

    gateway = None          # 由 main() 注入

    def log_message(self, fmt, *args):
        gw = self.gateway
        if gw is not None and gw.verbose:
            sys.stderr.write("[http] %s - %s\n" % (self.address_string(), fmt % args))

    # ---- 工具 ----

    def _json(self, code, obj, extra_headers=None):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self._responded = True

    def _error(self, code, message, err_type="invalid_request_error"):
        self._json(code, {"error": {"message": message, "type": err_type, "code": code}})

    def _fail(self, code, message, err_type):
        """出错收尾。响应已经开始写（比如 SSE 头已发出）时**不能**再写一个 HTTP 响应，
        那只会往流里塞垃圾让客户端解析崩掉；直接断开连接即可。"""
        if self._responded:
            if self.gateway is not None:
                self.gateway.log("error after response started: %s", message)
            self.close_connection = True
            return
        self._error(code, message, err_type)

    def _chunk(self, data):
        self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")

    def _read_json(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        raw = self.rfile.read(n) if n > 0 else b""
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return None

    # ---- 路由 ----

    def do_OPTIONS(self):
        """CORS 预检。

        带 `Content-Type: application/json` 的 POST **不是**简单请求，浏览器会先发一个
        OPTIONS 问一句"能不能发"。不处理这一条，页面上每个请求都会在控制台里失败，
        而在 curl/python 那一侧完全看不出来——因为它们压根不发预检。
        """
        self._responded = False
        self.send_response(204)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        # X-Conversation-Id 必须列进来，否则浏览器里发这个头会被预检拦掉（curl/python
        # 不发预检，所以这个问题只在网页端现形，很容易查错方向）。
        self.send_header("Access-Control-Allow-Headers",
                         "Content-Type, X-Conversation-Id")
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.end_headers()
        self._responded = True

    def _static_demo(self):
        """把演示页发出去。

        页面从这里取就是同源的（地址栏和接口都是同一个 host:port），CORS 这一关天然过。
        头照样发，是为了页面被另存到本地、或被人挂到别的域名下时也能用。
        """
        path = os.path.join(VIEW_DIR, DEMO_PAGE)
        try:
            with open(path, "rb") as f:
                body = f.read()
        except (IOError, OSError):
            return self._error(404, "demo page missing: %s (应与网关放在同一目录)" % path)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self._responded = True

    def do_GET(self):
        self._responded = False
        path = self.path.split("?")[0]
        if path in ("/", "/demo", "/demo_4chat.html"):
            return self._static_demo()
        if path in ("/health", "/healthz"):
            pool = self.gateway.pool.snapshot()
            dead = sum(1 for s in pool["slots"] if s["state"] == "dead")
            backend_alive = self.gateway.backend.alive()
            return self._json(200, {"status": ("ok" if backend_alive and dead == 0 else
                                               "backend_exited" if not backend_alive else
                                               "degraded"),
                                    "model": MODEL_ID,
                                    "sessions": self.gateway.backend.sessions,
                                    # 后端真正拿到的 --ctx-size。页面要用它把"窗口数
                                    # 上限由上下文长度决定"这条关系讲给用户听
                                    # （KV 每路按满上下文预分配：32K 每路 406.7 MB/卡，
                                    # 每卡只够 2 路；8K 那份导出便宜得多）。取不到就是 null，
                                    # 页面会自己退化成不显示这一格。
                                    "ctx_size": self.gateway.backend.ctx_size,
                                    # 「还能不能干活」必须能从这一行看出来。现场最坑的一种
                                    # 状态是 status:"ok" + sessions:4 + 四个 slot 全 dead：
                                    # 后端子进程已经退出、每个请求都 503，而 /health 报的是
                                    # 满员。看护脚本本来就在轮询这里，别让它去学 /v1/pool。
                                    "backend_alive": backend_alive,
                                    "sessions_dead": dead,
                                    # 会话全被占 + 有人在等 = 到上限了。放进 /health 是因为
                                    # 监控/看护脚本本来就在轮询这个端点，不必再教它一个新路径。
                                    "sessions_busy": sum(1 for s in pool["slots"]
                                                         if s["state"] == "busy"),
                                    # busy + idle = 有主的（free 才是没人占；dead 的会话
                                    # 在 mark_dead 里已经被解绑了，所以不会是 dead+有主）
                                    "sessions_bound": sum(1 for s in pool["slots"]
                                                          if s["state"] in ("busy", "idle")),
                                    "waiting": len(pool["waiting"]),
                                    "uptime_s": round(time.time() - self.gateway.started_at, 1)})
        if path == "/v1/system":
            # 非 OpenAI 标准端点，纯观测：板卡此刻的 CPU / 内存 / 温度 / 每卡 NPU。
            # 页面每 2s 轮一次；**这个端点永远不会因为后端卡住而报错**——拿不到新数就
            # 返回上一份并带上年龄，因为"看板卡现在怎么样"正是它卡住时最需要的。
            try:
                return self._json(200, self.gateway.system_snapshot())
            except Exception as exc:                        # noqa: BLE001
                return self._error(500, "system snapshot failed: %r" % (exc,),
                                   err_type="server_error")
        if path == "/v1/pool":
            # 非 OpenAI 标准端点，纯观测：谁占着哪个会话、闲了多久、谁在排队。
            return self._json(200, self.gateway.pool.snapshot())
        if path == "/v1/models":
            return self._json(200, {"object": "list", "data": [{
                "id": MODEL_ID, "object": "model",
                "created": int(self.gateway.started_at), "owned_by": "local"}]})
        return self._error(404, "unknown path: %s" % path)

    def _close_conversation(self):
        """`POST /v1/conversations/close`：把这段对话占的会话立刻交还（不等 `IDLE_TTL`）。

        body 给 `conversation_id`（或 `user`），或只给 `X-Conversation-Id` 头——与
        `/v1/chat/completions` 同一套身份优先级。

        **匿名对话要用 `key` 关**（`{"key": "h:faa51b39..."}`，键从 `GET /v1/pool` 的
        `slots[].key` 读）：匿名请求的身份是网关按 system+首问算的内容摘要，客户端自己
        算不出来，不给它一条路的话那段对话会一直占到 `IDLE_TTL` 超时。**只给了身份没法
        关的那种情况**（既没 id 也没 key）回 400，把原因写清楚，别让人以为是网关坏了。
        """
        req = self._read_json()
        if req is None:
            req = {}
        raw_key = req.get("key")
        explicit = (req.get("conversation_id")
                    or self.headers.get("X-Conversation-Id")
                    or req.get("user"))
        if raw_key:
            key = str(raw_key)
        elif explicit:
            key = conversation_key([], explicit=explicit)
        else:
            return self._error(400, "conversation_id (or X-Conversation-Id / user) is "
                                    "required; anonymous conversations must be closed by "
                                    "their raw `key` from GET /v1/pool, because their id "
                                    "is derived from message content")
        try:
            session = self.gateway.pool.close(key)
        except BackendError as exc:
            return self._error(409, str(exc), "server_error")
        self.gateway.log("close: %s -> %s", explicit or key, session)
        return self._json(200, {"conversation_id": str(explicit) if explicit else None,
                                "key": key,
                                "closed": session is not None,
                                "session": session})

    def do_POST(self):
        self._responded = False
        path = self.path.split("?")[0]
        if path == "/v1/conversations/close":
            return self._close_conversation()
        if path != "/v1/chat/completions":
            return self._error(404, "unknown path: %s" % path)
        req = self._read_json()
        if req is None:
            return self._error(400, "body is not valid JSON")
        messages = req.get("messages")
        if not messages:
            return self._error(400, "messages is required")

        stream = bool(req.get("stream"))
        max_new = _pick_max_tokens(req)
        ctk = req.get("chat_template_kw") or {}
        enable_thinking = ctk.get("enable_thinking") if isinstance(ctk, dict) else None
        tools = _pick_tools(req)
        prompt = render_messages(messages, enable_thinking=enable_thinking, tools=tools)
        # 会话身份，优先级从高到低：
        #   1. 请求体的 conversation_id —— 本项目自己的扩展，语义最明确
        #   2. X-Conversation-Id 请求头 —— 同上，给不方便改 body 的客户端
        #   3. OpenAI 标准的 user 字段 —— 多用户场景直接写用户名就行，不用教调用方新东西
        #   4. 都没有 => 退回"按 system + 首条提问做摘要"
        # 第 4 条在多人场景下**是会撞的**：两个人问同一句话就是同一段对话，会被钉到同一个
        # 会话上串行（答案照对，吞吐掉一半）。所以多人接入时务必显式给前三者之一。
        key = conversation_key(messages, explicit=req.get("conversation_id")
                               or self.headers.get("X-Conversation-Id")
                               or req.get("user"))
        self.gateway.log("chat: %d messages, %d prompt bytes, stream=%s, max_new=%d, tools=%d",
                         len(messages), len(prompt.encode("utf-8")), stream, max_new,
                         len(tools))

        lease = None
        q = None
        try:
            lease = self.gateway.pool.acquire(key, prompt)
            q = self.gateway.backend.submit(prompt=lease.sent_prompt,
                                            max_new_tokens=max_new,
                                            session=lease.session,
                                            reset=lease.reset)
            q.kv = {"session": lease.session, "reset": lease.reset,
                    "sent": len(lease.sent_prompt.encode("utf-8")),
                    "base": len(lease.base.encode("utf-8")),
                    "full": len(prompt.encode("utf-8")),
                    "wait": lease.waited}
            self.gateway.log("lease: session=%d reset=%s sent=%d/%d bytes (base=%d)%s",
                             q.kv["session"], q.kv["reset"], q.kv["sent"], q.kv["full"],
                             q.kv["base"],
                             # 排过队就一定要打出来：**"慢"和"没轮到"在日志里必须分得开**，
                             # 否则事后只能靠 pool 快照的时间线去猜（见 CHANGELOG 09-18）。
                             (" 排队 %.1fs" % lease.waited) if lease.waited >= 0.05 else "")
            if stream:
                self._stream_response(req, q, lease, enable_thinking)
            else:
                self._blocking_response(req, q, enable_thinking)
        except BackendError as exc:
            if exc.soft:
                # 后端拒了这一轮（上下文装不下）：这是**客户端要改请求**的错误，不是
                # 服务端故障。回 503 的话客户端会原样重试，而原样重试必然再被拒一次。
                self._fail(400, str(exc), "invalid_request_error")
            else:
                self._fail(503, str(exc), "server_error")
        except Exception as exc:                        # noqa: BLE001
            self._fail(500, "%r" % (exc,), "server_error")
        finally:
            # 无论成功、失败还是客户端断开，都要把会话还回去；推理失败还要把会话标死。
            if lease is not None:
                if q is not None and q.rejected:
                    # 后端在 prefill 前拒了这一轮：会话**没死**。标死它就等于每撞一次
                    # 上下文超限就永久少一个会话。而且这一轮一个 token 都没进 KV，
                    # 记账也不能走 release（见 pool.discard）。
                    self.gateway.pool.discard(lease)
                else:
                    failed = q is None or q.error or q.dead
                    if failed:
                        self.gateway.pool.mark_dead(lease.session)
                        self.gateway.pool.release(lease, "", True)
                    else:
                        # 记账用的必须是**客户端会回显的那份文本**：下一轮 prompt 是这个
                        # 版本的拼接，前缀判据才有可能成立。用 q.text()（原始版）记的话，
                        # 关思考时每轮都会判"对不上"，缓存静默全废——而且答案是对的，
                        # 看不出来（nothink_check.py 抓的就是这个）。工具调用把这条规则
                        # 又抬高一档：带着 tool_calls 的助手轮在下一轮 prompt 里是
                        # 「正文 + <tool_call> XML」，所以记账也必须是渲染后的那一份，
                        # 而不是模型原始输出（见 assistant_continuation）。
                        self.gateway.pool.release(
                            lease, assistant_continuation(q, enable_thinking),
                            q.was_cleared)
            if q is not None:
                self.gateway.backend.release_request(q)

    # ---- 两种响应形态 ----

    def _stream_response(self, req, q, lease, enable_thinking=None):
        created = int(time.time())
        cid = "chatcmpl-%d-%d" % (created, q.rid)
        include_usage = bool((req.get("stream_options") or {}).get("include_usage"))
        # 关思考时把开头那个（空的）think 段摘掉，两种响应形态共用同一份状态机
        stripper = ThinkStripper() if enable_thinking is False else None
        extractor = ToolCallExtractor(reasoning=reply_reasoning(enable_thinking))

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        for k, v in kv_reuse_headers(q.kv).items():
            self.send_header(k, v)

        def sse(obj):
            self._chunk(("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n")
                        .encode("utf-8"))

        def delta_obj(content, finish=None):
            return {"id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": MODEL_ID,
                    "choices": [{"index": 0,
                                 "delta": ({"content": content} if content is not None else {}),
                                 "finish_reason": finish}]}

        self.end_headers()
        self._responded = True          # 头已发出：后面出错只能断连，不能再写响应
        # 首个 chunk 按惯例只带 role
        sse({"id": cid, "object": "chat.completion.chunk", "created": created,
             "model": MODEL_ID,
             "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})

        sent = [0]
        n_calls = [0]

        def tool_delta(call):
            """一条工具调用的 SSE 增量。

            `arguments` 一次性发完，不按 token 切片：客户端是按 `index` 把同一路调用的
            arguments 拼起来的，一次给全和分片给在协议上等价。分片要在网关侧多维护一份
            "这个 JSON 字符串切到哪了"的状态，换来的只是首字节早一点——不值得为它多一份
            会出错的逻辑。
            """
            idx = n_calls[0]
            n_calls[0] += 1
            return {"id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": MODEL_ID,
                    "choices": [{"index": 0,
                                 "delta": {"tool_calls": [{
                                     "index": idx,
                                     "id": "call_%d" % idx,
                                     "type": "function",
                                     "function": {
                                         "name": call["name"],
                                         "arguments": json.dumps(
                                             call.get("arguments") or {},
                                             ensure_ascii=False)}}]},
                                 "finish_reason": None}]}

        def emit(text):
            """统一出口：过 stripper（没开就是原样）再摘工具调用，各自非空才发。"""
            if stripper is not None:
                text = stripper.feed(text)
            if not text:
                return
            text, calls = extractor.feed(text)
            if text:
                sse(delta_obj(text))
                sent[0] += 1
            for call in calls:
                sse(tool_delta(call))

        def on_delta(text):
            emit(text)

        # 接手：与 reader 线程在 deliver_lock 里交接，保证"等待期间攒下的那一段"
        # 恰好交一次。两点都不能省：
        #   · 只补发 raw 里 delivered 之后的字节，不是整段 raw——整段重发正是
        #     "首段内容重复"的来源（reader 可能正好在 attach 的瞬间投过一段）；
        #   · 补发也走 q.decoder（增量解码器），因为跨帧切开的 UTF-8 字符要靠它拼回来；
        #     整段 decode 会在切缝处吐出替换字符，那个字符就**消失**了。
        with q.deliver_lock:
            q.on_delta = on_delta
            backlog = b""
            if q.delivered < len(q.raw):
                backlog = q.decoder.decode(bytes(q.raw[q.delivered:]))
                q.delivered = len(q.raw)
        if backlog:
            emit(backlog)
        q.done.wait(REQUEST_TIMEOUT)
        if not q.done.is_set():
            self.gateway.pool.mark_dead(lease.session)
            self._chunk(b"")            # 收尾：别让客户端一直挂着
            raise BackendError("timeout waiting for generation")

        if q.error:
            # HTTP 头已经发出去了，只能用 SSE 里的错误对象收尾：先报错，再按协议
            # 正常结束这个流（clients 普遍是看到 [DONE] 才算读完）。
            # 被拒（上下文装不下）时类型是客户端错误：流式下虽然已经回不了 400 状态码，
            # 但 type 字段至少要如实说这是请求的问题，客户端才好决定是重试还是改请求。
            sse({"error": {"message": q.error,
                           "type": "invalid_request_error" if q.rejected
                                   else "server_error"}})
            self._chunk(b"data: [DONE]\n\n")
            self._chunk(b"")
            raise BackendError(q.error, soft=q.rejected)

        # 收尾前先把 stripper 里还攒着的字节放出来（正常情况是空；被截断在 <think> 里时
        # 会连标签一起吐出来，见 flush() 的说明），否则客户端会少一段正文。走 emit 而不是
        # 直接发：这段尾巴里也可能装着一个工具调用。
        if stripper is not None:
            tail = stripper.flush()
            if tail:
                emit(tail)

        # 同理：extractor 里攒着的可能是半个 <tool_call>（被 max_tokens 截断），
        # flush 会连标签一起交回来当正文，不吞。
        leftover, _ = extractor.flush()
        if leftover:
            sse(delta_obj(leftover))
            sent[0] += 1

        # 有工具调用时 finish_reason 必须是 tool_calls（理由同非流式路径）
        sse(delta_obj(None, finish="tool_calls" if n_calls[0]
                      else (q.finish_reason or "stop")))
        if include_usage:
            sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                 "model": MODEL_ID, "choices": [],
                 "usage": self._usage(q)})
        self._chunk(b"data: [DONE]\n\n")
        self._chunk(b"")                # 终止 chunk
        self.gateway.log("stream done: rid=%d session=%d chunks=%d tool_calls=%d tokens=%d",
                         q.rid, q.session, sent[0], n_calls[0], q.decode_tokens)

    def _blocking_response(self, req, q, enable_thinking=None):
        q.done.wait(REQUEST_TIMEOUT)
        if not q.done.is_set():
            raise BackendError("timeout waiting for generation")
        if q.error:
            raise BackendError(q.error, soft=q.rejected)
        created = int(time.time())
        content, calls = split_reply(q, enable_thinking)
        message = {"role": "assistant", "content": content}
        finish = q.finish_reason or "stop"
        if calls:
            # 有工具调用时 finish_reason 必须是 tool_calls：客户端（以及各家 Agent 框架）
            # 就是靠它决定"该执行工具了"还是"这轮结束了"。照搬后端的 stop 会让 Agent
            # 以为模型已经答完，工具根本不会被调用。
            message["tool_calls"] = to_openai_tool_calls(calls)
            finish = "tool_calls"
        self._json(200, {
            "id": "chatcmpl-%d-%d" % (created, q.rid),
            "object": "chat.completion",
            "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "finish_reason": finish, "message": message}],
            "usage": self._usage(q),
        }, extra_headers=kv_reuse_headers(q.kv))

    def _usage(self, q):
        """OpenAI 口径的 usage。

        `prompt_tokens` 是**客户端发来的那段 prompt 的长度**，不是后端本轮真正算过
        的那部分。命中 KV 复用时后端只报增量（板上实测：整段 522 字节的 prompt，
        本轮只 prefill 了对应 84 字节的那部分；DONE 帧第 4 字段就是这个增量），
        照搬进 prompt_tokens 会让客户端以为上下文只有 21 tok——按它做上下文预算
        就**永远不会触发裁剪**，一路涨到后端自动清 KV，然后出现那种"只错一轮、
        不留痕"的答案。

        所以：prompt_tokens = 本轮结束时的上下文长度 − 本轮生成的 token 数；
        被复用的那部分放 prompt_tokens_details.cached_tokens（OpenAI 同名字段）。
        两个数都来自 DONE 帧（第 4 与第 8 字段），网关不需要自己分词。
        """
        ctx = q.context_tokens or (q.prefill_tokens + q.decode_tokens)
        prompt_tokens = max(0, ctx - q.decode_tokens)
        cached = max(0, prompt_tokens - q.prefill_tokens)
        return {"prompt_tokens": prompt_tokens,
                "completion_tokens": q.decode_tokens,
                "total_tokens": prompt_tokens + q.decode_tokens,
                # 恒存在（没错时是 0）：客户端读它时不必先判有没有这个键。
                "prompt_tokens_details": {"cached_tokens": cached}}


# 帧协议里 max_new_tokens 字段实际按 `int` 用（后端拿 `%llu` 读进来之后要落到
# `PendingInput::max_new_tokens` 这个 int）。后端 2026-09-16 起会**拒收**超出这个范围的
# 值（回 ERR），所以这里必须先卡住——否则一个客户端随口写 `"max_tokens": 1e12`
# 就能让后端回 ERR，而 ERR 的含义是"这个会话的驱动线程没了"，网关据此把会话标死：
# **四次离谱的 max_tokens 就能把整个服务打死**。这是客户端不该有能力做到的事。
MAX_NEW_TOKENS_CAP = 0x7fffffff


def _pick_max_tokens(req):
    for name in ("max_tokens", "max_completion_tokens", "n_predict", "max_new_tokens"):
        v = req.get(name)
        if isinstance(v, int) and v > 0:
            # **夹住而不是回 400**：`"max_tokens": 100000` 是客户端很常见的写法，意思是
            # "能吐多少吐多少"，不是"我要十万个 token"。真按 400 拒掉会把正常的 OpenAI
            # 风格客户端挡在门外。夹到上限之后语义仍然正确——后端本来就会在 EOS 或上下文
            # 耗尽时停，这个字段只是个上界。
            return min(v, MAX_NEW_TOKENS_CAP)
    return 0        # 0 = 让后端用它自己的默认值（--n-predict）


def _pick_tools(req):
    """OpenAI 的 `tools` / `tool_choice` -> 要注入 prompt 的工具列表（空 = 不注入）。

    注入方式是把工具定义写进 **system 消息**（Qwen3.5 模板就是这么定的），所以工具变了
    system 就变了，粘性复用的前缀判据会判"对不上"、这一轮全量重算。这没法避免——
    工具集本来就是这个对话的一部分。要注意的是**别让它每轮都变**：Agent 每轮传同一份
    tools，渲染出的字节就是同一份，复用照常。真正会踩的坑是客户端给 tools 里的
    dict 换了键顺序——渲染时键是排序的（toolcalls.jinja_tojson），顺序无关，没事。

    `tool_choice`：`"none"` 不注入；指定某个 function 就只注入它；其余（含
    `"auto"` / `"required"`）全注入。"required" 我们没法强制模型一定调用，就不假装
    能做到——把它当 auto 处理，比回一个 400 更接近客户端的意图。
    """
    raw = req.get("tools")
    if not isinstance(raw, list):
        return []
    tools = [t for t in raw if isinstance(t, dict)]
    if not tools:
        return []
    choice = req.get("tool_choice")
    if isinstance(choice, str) and choice.strip().lower() == "none":
        return []
    if isinstance(choice, dict):
        fn = choice.get("function")
        if not isinstance(fn, dict):
            fn = choice
        want = fn.get("name")
        if want:
            picked = [t for t in tools
                      if (t.get("function") or {}).get("name") == want]
            if picked:
                return picked
    return tools


class ThinkStripper(object):
    """把开头的 <think>…</think> 从正文里摘掉。**只在关思考时**使用。

    为什么需要：关思考是靠 /no_think 软开关实现的，模型照样会吐一个 `<think>` 段
    （实测通常是空的：`<think>  </think>  收到`）。原样返回的话 Agent 拿到的是带标签的
    字符串，还得自己洗一遍。这里在网关侧洗掉，两种响应形态用同一份逻辑。

    为什么开思考时不摘：那时的 <think> 段是**有内容的推理过程**，属于模型输出的一部分，
    擅自丢掉等于替用户做决定。要暴露的话应该另开一个字段（reasoning_content），
    这一版没做——所以开思考时原样透传，行为与不加这段代码时完全一致。

    流式为什么要缓冲：`</think>` 可能被切在两个 DELTA 帧之间（本来就是按 token 投递的），
    不能逐帧正则。于是维持一个小状态机，只在"还可能是在开头那个 think 段"的极短窗口里
    攒字节；一旦判定不是 think 段就立刻透传，不等。关思考时那段是空的，所以这个窗口
    通常只有几个 token 的延迟。
    """

    MARK_OPEN = "<think>"
    MARK_CLOSE = "</think>"
    MAX_HOLD = 1 << 16      # 攒这么多还没看到 </think> 就放弃摘除（见 body 分支）

    def __init__(self):
        # lead: 还在对 <think> 的开头 -> pre: 吃掉紧跟的空白 -> body: 找 </think>
        # -> post: 吃掉紧跟的空白 -> done: 剩下的一律透传
        self.state = "lead"
        self.buf = ""

    def feed(self, text):
        """送进一段（可能是半个字的）新文本，返回现在可以确定发出的部分。"""
        if not text:
            return ""
        self.buf += text
        out = []
        while self.buf:
            if self.state == "done":
                out.append(self.buf)
                self.buf = ""
                break
            if self.state == "lead":
                if self.buf.startswith(self.MARK_OPEN):
                    self.buf = self.buf[len(self.MARK_OPEN):]
                    self.state = "pre"
                    continue
                if self.MARK_OPEN.startswith(self.buf):
                    break                   # 还可能是 think 的开头：再等等
                self.state = "done"         # 不是 think 段：整段透传
                continue
            if self.state in ("pre", "post"):
                # 这两个状态是"吃掉紧随标签的空白"。必须做成**状态**而不是一次性
                # lstrip：按 token 投递时，标签和它后面的空格/换行常常落在两帧里，
                # 一次性 lstrip 那一刻缓冲区可能还是空的，空白就漏进正文了（踩过）。
                stripped = self.buf.lstrip()
                if not stripped:
                    self.buf = ""           # 全是空白：吃掉，继续等
                    break
                self.buf = stripped
                self.state = "body" if self.state == "pre" else "done"
                continue
            if self.state == "body":
                i = self.buf.find(self.MARK_CLOSE)
                if i < 0:
                    if len(self.buf) > self.MAX_HOLD:
                        # 关思考却吐了一大段推理（不该发生）。放弃摘除、原样透传：
                        # 宁可让用户看见标签，也不能把内存吃光。
                        self.state = "done"
                        out.append(self.MARK_OPEN + self.buf)
                        self.buf = ""
                        break
                    break                   # 推理段还没结束（正常情况下它是空的）
                self.buf = self.buf[i + len(self.MARK_CLOSE):]
                self.state = "post"
                continue
        return "".join(out)

    def flush(self):
        """收尾：把还攒着的字节交出来。

        卡在 pre/body（模型吐了 <think> 却没闭合，通常是 max_tokens 截断）时，把原始
        内容连同标签一起交出去，而不是丢弃——宁可让用户看见半个推理段，也不能把可能
        存在的正文吞掉。
        """
        out, self.buf = self.buf, ""
        if self.state in ("pre", "body"):    # 标签已经吃掉了，交还时补回去
            out = self.MARK_OPEN + out
        self.state = "done"
        return out


def strip_think(text, enable_thinking):
    """非流式路径：一次性把开头的 think 段摘掉。"""
    if enable_thinking is not False:
        return text
    s = ThinkStripper()
    return s.feed(text) + s.flush()


def reply_reasoning(enable_thinking):
    """扫工具调用时要不要跳过 `<think>` 段。

    开思考（默认，含客户端没明说）时要跳过：模型是"先推理、后调用"，但推理过程里
    **讨论**到 `<tool_call>` 字样并不等于它打算调用那个工具，抽出来执行就成了替模型
    做决定。关思考时不必——ThinkStripper 已经把开头那个（空的）think 段摘掉了。
    """
    return enable_thinking is not False


def split_reply(q, enable_thinking):
    """模型这一轮的回复 -> (给客户端的正文, 工具调用列表)。"""
    return split_tool_calls(strip_think(q.text(), enable_thinking),
                            reasoning=reply_reasoning(enable_thinking))


def assistant_continuation(q, enable_thinking):
    """这一轮的助手回复在**下一轮 prompt 里**长什么样（不含 assistant 头与 `<|im_end|>`）。

    响应路径和 KV 记账路径共用这一个函数，两边的字节因此天然一致——不一致的话前缀
    判据会失败、每轮全量重算，而那是"只慢不错"的静默退化。
    没有工具调用时它就是 `strip_think(...)` 本身（render_assistant_turn 的空列表分支
    原样返回正文），所以不带工具的老行为一个字节都没变。
    """
    content, calls = split_reply(q, enable_thinking)
    return render_assistant_turn(content, calls)


def kv_reuse_headers(kv):
    """把本轮的复用决策回报给客户端。

    为什么值得占一个响应头：**光看 usage.prompt_tokens 分不清「复用生效」和「换了个
    空会话」**——两种情况下 prefill 都可能等于全量。这里直接把会话号、是否 RESET、
    实际发出去多少字节摊开，出问题时不用猜。（也方便 Agent 侧观察缓存命中率。）

    `wait` 是**排队等了多久**（秒，0 = 没等）。它不是锦上添花：客户端看到的"慢"往往是
    这个数，而它此前在报文里**完全隐形**——页面上只看到 `1.57 tok/s`，看起来像模型慢，
    实际是四个会话都闲着、请求在队列里干等（2026-09-18 用户就是这么报上来的）。
    分母里带着它，速率就不是"解码速度"了，所以必须能单独看。
    """
    if not kv:
        return {}
    return {"X-KV-Reuse": "session=%d; reset=%d; sent=%d; base=%d; full=%d; wait=%.1f"
                          % (kv["session"], 1 if kv["reset"] else 0, kv["sent"],
                             kv["base"], kv["full"], kv.get("wait", 0.0))}


# ===========================================================================
# 5. 自检：只验帧协议，不起 HTTP
# ===========================================================================

def selftest_reject(backend, check):
    """REJECT 的回归：后端拒了一轮（上下文装不下），网关这半边该做什么。

    对应的缺陷：serve 模式下"上下文将满"原本是后端**自动清 KV 再把这一轮跑完**，而
    网关只发增量（`prompt[len(base):]`）→ 模型拿到的是一段没有开头的对话 → 以一个
    自信的错答案、finish_reason=stop 返回。不报错、不重试、只错一轮，是最难查的一类。

    改成后端发 CLEAR+REJECT 之后，网关这半边有三条义务：
      ① 只失败这一条请求，而且是**软错误**（HTTP 400，而不是 503——客户端要做的是改
         请求，原样重试必然再被拒一次）；
      ② 会话照常留在池子里（标死等于每撞一次超限就永久少一个会话，四次之后没会话可用）；
      ③ 粘性记录作废（下一轮 RESET 全量重发），**且不能把这一轮的 prompt 记进 KV 账**
         ——这一轮一个 token 都没 prefill 进去。

    单独成一个函数是因为它要**另起一个桩后端**（--ctx-limit 是桩专有的开关）。
    真后端上这段跑不了，调用处会 SKIP。
    """
    rb = Backend(list(backend.argv) + ["--ctx-limit", "500"],
                 backend.log_path + ".reject", frame_fd=backend.frame_fd)
    rb.start()
    try:
        rpool = SessionPool(rb, idle_ttl=0, queue_timeout=5)
        # 第一轮：ctx 还是 0，不该触发（真后端同理：空的会话没什么"装不下"的）
        l1 = rpool.acquire("reject-probe", render_messages(
            [{"role": "user", "content": "记住暗号：青柠味苏打水。"}]))
        q1 = rb.submit(l1.sent_prompt, 8, session=l1.session, reset=l1.reset)
        rb.wait(q1)
        rpool.release(l1, q1.text(), q1.was_cleared)
        sess = l1.session
        rb.release_request(q1)
        check("第一轮正常跑完（还没到上限）", q1.error is None and q1.decode_tokens > 0,
              "ctx=%d" % q1.context_tokens)

        # 第二轮：桩的 ctx 已经攒够，会回 CLEAR + REJECT
        l2 = rpool.acquire("reject-probe", render_messages(
            [{"role": "user", "content": "记住暗号：青柠味苏打水。"},
             {"role": "assistant", "content": q1.text()},
             {"role": "user", "content": "再说一遍暗号。"}]))
        q2 = rb.submit(l2.sent_prompt, 8, session=l2.session, reset=l2.reset)
        soft = None
        try:
            rb.wait(q2)
        except BackendError as exc:
            soft = exc
        check("被拒的那轮抛的是**软**错误（HTTP 层会回 400 而不是 503）",
              soft is not None and soft.soft and "context limit" in str(soft), "%r" % (soft,))
        check("这一轮确实没跑（没有 DONE 的计数）",
              q2.decode_tokens == 0 and q2.finish_reason is None)
        check("CLEAR 帧到了（粘性记录必须作废）", q2.was_cleared)

        # 模拟 HTTP 线程的 finally 那段处置（这里没有 HTTP 层，selftest 只到协议这一层）
        if q2.rejected:
            rpool.discard(l2)
        else:                                   # pragma: no cover - 上面的断言已覆盖
            rpool.release(l2, "", True)
        rb.release_request(q2)
        check("会话没被标死（还能继续用）", not rpool.dead[sess], "dead=%s" % (rpool.dead[sess],))
        check("会话已归还（不再占着租约）", not rpool.busy[sess])
        check("KV 账没被这一轮污染：known 是 None（未知），不是本轮 prompt",
              rpool.known[sess] is None, "known=%r" % (rpool.known[sess],))

        # 下一轮必须 RESET 全量重发（known=None 的语义），而不是拿增量去拼一段残缺历史
        l3 = rpool.acquire("reject-probe", render_messages(
            [{"role": "user", "content": "全新的一轮。"}]))
        check("被拒之后下一轮走 RESET 全量重发，不复用", l3.reset and l3.base == "",
              "reset=%s base=%r" % (l3.reset, l3.base[:20]))
        rpool.release(l3, "", True)
    finally:
        rb.stop()


class _AliveOnlyBackend(object):
    """只给 selftest_pool_liveness 用：池子对后端只要求 `sessions` / `on_clear` / `alive` 三样。

    这里敢用桩是因为被测的是**池子那条判据本身**（"等不出结果就别让人排"），它与帧协议、
    与哪台后端都无关；用真后端反而要额外起一个进程、白等 240s 加载模型。端到端那一半
    （真后端被打死之后，5 段对话不再一起空等到 600s）在板卡上验，见文档 §9.11。
    """

    def __init__(self, sessions=4):
        self.sessions = sessions
        self.on_clear = None
        self.is_alive = True

    def alive(self):
        return self.is_alive


def selftest_pool_liveness(check):
    """后端没了 / 会话全 dead：`acquire` 必须**立刻**报错，而不是让人排在队列里空等。

    对应的现场（2026-09-17 上板实测）：一个 3.4 MB 的请求让后端进程退出（见 main.cc 的
    kServeMaxPromptBytes 分支），四个会话全 dead；之后 4 个终端 + 1 个 Claude Code 一起
    进来排队，屏幕上什么都不打印，600s 之后才各收到一个 503。看相是"网关卡了"，真相是
    队列在等一个已经不存在的东西——所以这条判据必须在**进队列之前**生效。

    判据本身也有"写松了"的两种形态，两条反面对照就钉它们：
      · 写成恒失败 —— 后端活着、会话有空时必须照常拿到会话；
      · 两种情况糊成一句话 —— 错误信息要分得清"后端死了"还是"会话没了"，现场靠它
        决定下一步（前者只能重启网关，后者同样是，但成因完全不同）。
    """
    be = _AliveOnlyBackend(4)
    pool = SessionPool(be, idle_ttl=0, queue_timeout=5)

    def timed_acquire(key):
        t0 = time.time()
        try:
            pool.acquire(key, "hi")
            return None, time.time() - t0
        except BackendError as exc:
            return exc, time.time() - t0

    lease = pool.acquire("live", "hi")
    pool.release(lease, "hi", False)

    # ① 后端子进程没了
    be.is_alive = False
    err, dt = timed_acquire("backend-gone")
    check("后端没了 => 立刻报错，不在队列里空等到 queue_timeout",
          err is not None and not err.soft and dt < 1.0, "%.2fs %r" % (dt, err))
    check("错误分得清是「后端退出了」", err is not None and "backend" in str(err),
          "%r" % (err,))

    # ② 后端活着，但四个会话全 dead（dead 没有任何复活路径）
    be.is_alive = True
    for s in range(pool.n):
        pool.mark_dead(s)
    err, dt = timed_acquire("all-dead")
    check("四个会话全 dead => 立刻报错，不在队列里空等到 queue_timeout",
          err is not None and not err.soft and dt < 1.0, "%.2fs %r" % (dt, err))
    check("错误分得清是「会话没了」", err is not None and "dead" in str(err), "%r" % (err,))


def _fixture(root, rel, text):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


def selftest_system(check, backend=None):
    """资源观测那半边：/proc 解析 + CPU 百分比算术 + STAT 帧往返。

    为什么非要夹具目录：`HostSampler` 在**本机（Windows）根本没有 /proc**，只按真板卡
    验的话，这段解析和百分比算术要等到上了板子、盯着页面看才发现算错。`root=` 就是
    为这个留的——把 /proc、/sys 换成一个临时目录，本机就能把**算术**真跑一遍。

    CPU 占用率是两次采样之差，夹具就靠**换 jiffies** 造出确定的一段：百分比只跟差值
    有关，跟两次采样之间真实的墙钟无关，所以这里的期望值是精确的、不是"大约"。
    """
    root = tempfile.mkdtemp(prefix="sysfix-")
    try:
        # ---- /proc/stat：四个核各钉一件事 ----
        # cpu0 差值 100/100 = 100.0%（满核）；cpu1 差 0/100 = 0.0%（空核）；
        # cpu2 **这一段只涨 iowait**（iowait 必须算空闲，否则等磁盘会被报成 CPU 忙）
        #  ⚠️ 涨的必须是 iowait 那一格、不能是 idle 格：只动 idle 的话"把 iowait 当忙"
        #  这个错法在**两次采样的差值**上正好抵消（两个快照都多算了那 50），变异体
        #  会活下来（第一版就是这样，M1 逃掉了）。
        # cpu3 差值**为负**（时钟回摆/计数器重置那种形态）必须夹到 0.0，不能吐负数。
        _fixture(root, "proc/stat", """\
cpu  1000 0 200 7000 0 0 0 0 0 0
cpu0 400 0 100 3000 0 0 0 0 0 0
cpu1 300 0 50 2000 0 0 0 0 0 0
cpu2 200 0 25 1500 50 0 0 0 0 0
cpu3 100 0 25 500 0 0 0 0 0 0
intr 0
""")
        # MemAvailable 6/16 GB ⇒ 用掉 62.5%。若误用 MemFree（2 GB）会算成 87.5%，
        # 所以这个期望值同时钉住了"取的是哪一格"。
        _fixture(root, "proc/meminfo", """\
MemTotal:       16000000 kB
MemFree:         2000000 kB
MemAvailable:    6000000 kB
Buffers:          100000 kB
Cached:          3000000 kB
SReclaimable:     200000 kB
SwapTotal:       4194300 kB
SwapFree:        4194300 kB
""")
        _fixture(root, "proc/loadavg", "0.50 0.40 0.35 2/1234 4567\n")
        _fixture(root, "proc/uptime", "12345.67 12345.67\n")
        # 温度：三个好区 + 一个 temp 是垃圾 + 一个连 type 都没有。后两个必须被跳过
        # 而不是让整段报错（真机上某个区偶尔读不出来不能拖垮整个面板）。
        _fixture(root, "sys/class/thermal/cooling_device0/type", "processor\n")
        _fixture(root, "sys/class/thermal/thermal_zone0/type", "soc-thermal\n")
        _fixture(root, "sys/class/thermal/thermal_zone0/temp", "45000\n")
        _fixture(root, "sys/class/thermal/thermal_zone1/type", "bigcore0-thermal\n")
        _fixture(root, "sys/class/thermal/thermal_zone1/temp", "51234\n")
        _fixture(root, "sys/class/thermal/thermal_zone2/type", "npu-thermal\n")
        _fixture(root, "sys/class/thermal/thermal_zone2/temp", "48000\n")
        _fixture(root, "sys/class/thermal/thermal_zone3/type", "gpu-thermal\n")
        _fixture(root, "sys/class/thermal/thermal_zone3/temp", "not-a-number\n")
        _fixture(root, "sys/class/thermal/thermal_zone4/type", "missing-temp\n")

        print("== HostSampler：第一次采样只该有「静态量」，百分比必须是 None 而不是 0 ==")
        hs = HostSampler(root=root)
        a = hs.sample()
        check("核数认得出（cpu0..cpu3）", a["cpu"]["n"] == 4, "n=%r" % (a["cpu"]["n"],))
        check("第一次 CPU%% 是 None（没上一次可减，不是 0%）",
              a["cpu"]["pct"] is None, "= %r" % (a["cpu"]["pct"],))
        check("第一次每核也都是 None",
              a["cpu"]["per_core"] == [None] * 4, "= %r" % (a["cpu"]["per_core"],))
        check("静态量第一次就有", a["mem"]["used_pct"] == 62.5 and len(a["thermal"]) == 3,
              "used_pct=%r 热区=%d 个" % (a["mem"]["used_pct"], len(a["thermal"])))
        check("loadavg 三个数", a["cpu"]["loadavg"] == [0.5, 0.4, 0.35],
              "= %r" % (a["cpu"]["loadavg"],))
        check("uptime 保留一位", a["uptime_s"] == 12345.7, "= %r" % (a["uptime_s"],))

        # ---- 第二次采样：换一份 jiffies，差值就是上面设计好的那几组 ----
        # 注意每一行都要自己算一遍：百分比只跟**差值**有关，写错一格就会得到一个
        # "看着挺合理"的数（第一版把 cpu 行的 idle 写成了 7975，于是 25/100 变成了
        # 2.5/100，自检报 FAIL —— 夹具写错和代码写错在这里是同一个现象）。
        _fixture(root, "proc/stat", """\
cpu  1025 0 200 7075 0 0 0 0 0 0
cpu0 500 0 100 3000 0 0 0 0 0 0
cpu1 300 0 50 2100 0 0 0 0 0 0
cpu2 200 0 25 1500 150 0 0 0 0 0
cpu3 90 0 20 600 0 0 0 0 0 0
intr 0
""")
        b = hs.sample()
        print("== 第二次采样：CPU 百分比 ==")
        check("整机 25/100 => 25.0%", b["cpu"]["pct"] == 25.0, "= %r" % (b["cpu"]["pct"],))
        check("cpu0 满核 => 100.0%", b["cpu"]["per_core"][0] == 100.0,
              "= %r" % (b["cpu"]["per_core"][0],))
        check("cpu1 差值为 0 => 0.0%", b["cpu"]["per_core"][1] == 0.0,
              "= %r" % (b["cpu"]["per_core"][1],))
        check("cpu2 只涨 iowait => 0.0%（等磁盘不算 CPU 在干活）",
              b["cpu"]["per_core"][2] == 0.0, "= %r" % (b["cpu"]["per_core"][2],))
        check("cpu3 差值为负 => 夹到 0.0%（不吐负数）",
              b["cpu"]["per_core"][3] == 0.0, "= %r" % (b["cpu"]["per_core"][3],))
        check("热区按 zone 号排序、label 与摄氏度都对",
              [(t["zone"], t["label"], t["c"]) for t in b["thermal"]] ==
              [("thermal_zone0", "soc-thermal", 45.0),
               ("thermal_zone1", "bigcore0-thermal", 51.2),
               ("thermal_zone2", "npu-thermal", 48.0)],
              "%r" % (b["thermal"],))

        print("== 读不到就退化，不抛（观测端点不该因为一个文件读不到就 500）==")
        empty = HostSampler(root=os.path.join(root, "does-not-exist")).sample()
        check("没有 /proc 也不抛，核数是 None（**不是 0**——0 核是假话）",
              empty["cpu"]["n"] is None and empty["cpu"]["pct"] is None,
              "n=%r pct=%r" % (empty["cpu"]["n"], empty["cpu"]["pct"]))
        check("内存退化成空而不是 0 占用",
              empty["mem"].get("total", 0) == 0 and "used_pct" not in empty["mem"],
              "total=%r" % (empty["mem"].get("total"),))
        check("热区退化成空表", empty["thermal"] == [], "= %r" % (empty["thermal"],))
        check("uptime 退化成 None", empty["uptime_s"] is None, "= %r" % (empty["uptime_s"],))

        print("== CardStats：Δ忙时 / Δ墙钟 ==")
        def card(name, busy_us, mem_at_us, **kw):
            d = {"name": name, "busy_us": busy_us, "run_calls": 7, "mem_at_us": mem_at_us,
                 "mem_total": 1000, "mem_free": 400, "node_num": 8,
                 "node_min_free": 50, "ctx_len": 8192}
            d.update(kw)
            return d

        cs = CardStats()
        first = cs.sample({"now_us": 1000000,
                           "cards": [card("stage0", 100000, 900000),
                                     card("stage1", 100000, 0)]})
        check("第一次 busy_pct 是 None（不是 0%）",
              first["busy_pct"] is None and
              all(c["busy_pct"] is None for c in first["cards"]),
              "= %r" % (first["busy_pct"],))
        check("整卡占用 = (total-free)/total", first["cards"][0]["mem_used"] == 600 and
              first["cards"][0]["mem_used_pct"] == 60.0,
              "%r/%r" % (first["cards"][0]["mem_used"], first["cards"][0]["mem_used_pct"]))
        check("mem_age_s 按后端自己的时钟算（0.1s）",
              first["cards"][0]["mem_age_s"] == 0.1,
              "= %r" % (first["cards"][0]["mem_age_s"],))
        check("mem_at_us==0 => mem_age_s 是 None（一次都没采到，不是「很旧」）",
              first["cards"][1]["mem_age_s"] is None,
              "= %r" % (first["cards"][1]["mem_age_s"],))

        second = cs.sample({"now_us": 2000000,           # 距上一次 1.0s
                            "cards": [card("stage0", 600000, 1900000),
                                      card("stage1", 1100000, 1900000),
                                      card("stage2", 5, 1900000)]})
        check("stage0 Δ0.5s / 1.0s => 50.0%", second["cards"][0]["busy_pct"] == 50.0,
              "= %r" % (second["cards"][0]["busy_pct"],))
        check("stage1 Δ1.0s / 1.0s => 100.0%", second["cards"][1]["busy_pct"] == 100.0,
              "= %r" % (second["cards"][1]["busy_pct"],))
        check("上一帧没出现过的卡 => None（不拿 0 冒充基线）",
              second["cards"][2]["busy_pct"] is None,
              "= %r" % (second["cards"][2]["busy_pct"],))
        check("整机取的是**有值的那几张**的平均（不是拿 None 当 0 摊进去）",
              second["busy_pct"] == 75.0, "= %r" % (second["busy_pct"],))

        third = cs.sample({"now_us": 2500000,            # 距上一次只有 0.5s
                           "cards": [card("stage0", 1600000, 2400000),
                                     card("stage1", 1100000, 2400000)]})
        check("Δ忙时 > Δ墙钟（采样点被压近）=> 夹到 100.0%，不吐 200%",
              third["cards"][0]["busy_pct"] == 100.0,
              "= %r" % (third["cards"][0]["busy_pct"],))
        check("忙时倒退（不该发生但别崩）=> 该卡给 None、其余照算",
              cs.sample({"now_us": 3000000,
                         "cards": [card("stage0", 1, 2900000)]})["cards"][0]["busy_pct"]
              is None, "")

        # 同一份 doc 连着采两次：后端那边 now_us 没动（拿到的是缓存 / 答复超时）。
        # 这时候**不许**把窗口推过去、也不许把上一次的数抹成 "—"。
        # 板上就是这么闪的（2026-09-20）：busy_pct 在 0.0 和 None 之间来回跳。
        cs2 = CardStats()
        cs2.sample({"now_us": 1000000, "cards": [card("stage0", 0, 0)]})
        ready = cs2.sample({"now_us": 2000000, "cards": [card("stage0", 500000, 0)]})
        check("（前置）两次不同 doc 才算出 50.0%", ready["cards"][0]["busy_pct"] == 50.0,
              "= %r" % (ready["cards"][0]["busy_pct"],))
        again = cs2.sample({"now_us": 2000000, "cards": [card("stage0", 500000, 0)]})
        check("同一份 doc 再来一次：给回上一次的数，不是 None",
              again["cards"][0]["busy_pct"] == 50.0,
              "= %r" % (again["cards"][0]["busy_pct"],))
        # 关键：窗口没被这次"重复"吃掉——下一份真 doc 的 Δ 是从**上一次真 doc**起算的。
        # 若把 _prev 推到了那份旧 doc 上，这里会算成 100%（Δ0.5s / Δ0.5s）而不是 50%。
        span = cs2.sample({"now_us": 3000000, "cards": [card("stage0", 1000000, 0)]})
        check("重复采样不吃掉窗口：下一份 doc 仍从上一份**真** doc 起算（50%，不是 100%）",
              span["cards"][0]["busy_pct"] == 50.0,
              "= %r" % (span["cards"][0]["busy_pct"],))

        print("== 后端 STAT 帧的解包与缓存 ==")
        bad_input = CardStats().sample(None)
        check("后端还没答过（stats=None）=> ok=False，不抛",
              bad_input == {"ok": False, "cards": [], "busy_pct": None},
              "= %r" % (bad_input,))

        be = Backend(["true"], os.path.join(root, "unused.log"), frame_fd="stdout")
        be._on_stats(b'{"now_us": 1, "cards": [{"name": "s", "busy_us": 2}]}')
        check("好载荷进缓存、序号 +1", be.stats_seq == 1 and
              be.stats["cards"][0]["name"] == "s", "seq=%d" % be.stats_seq)
        be._on_stats(b"{ this is not json")
        check("坏载荷只丢这一帧，读线程不许死（stats_seq 不动、stats 不脏）",
              be.stats_seq == 1, "seq=%d" % be.stats_seq)
        check("没起进程时 request_stats 返回缓存且不抛",
              be.request_stats(min_interval=0) is be.stats, "")
        try:
            got = be.request_stats(min_interval=0)
            raised = False
        except Exception:                                   # noqa: BLE001
            got, raised = None, True
        check("观测端点永不抛异常", not raised, "= %r" % (got,))

        if backend is not None and any("fake_backend" in str(a) for a in backend.argv):
            print("== STAT 帧真往返（桩后端；真正要钉住的是**帧流没被这帧撑错位**）==")
            seq0 = backend.stats_seq
            live = backend.request_stats(min_interval=0, wait=2.0)
            seq1 = backend.stats_seq
            ok_live = isinstance(live, dict) and live.get("cards")
            check("桩答得出 STAT", bool(ok_live),
                  "%d 张卡" % (len(live.get("cards") or [])) if ok_live else repr(live))
            if ok_live:
                live2 = backend.request_stats(min_interval=0, wait=2.0)
                seq2 = backend.stats_seq
                # 每要一次就恰好该多一条答复。载荷没被读走的话，读线程下一帧看到的
                # 是一段 JSON 当帧头，从此再也认不出 STAT ⇒ 序号就停在这儿了。
                # 判"两张卡数一样"是不够的：错位之后页面拿到的是**同一份缓存**，
                # 卡数一样、内容也一样旧。
                check("每要一次恰好多一条答复（载荷被完整消费，没留下半帧）",
                      seq1 == seq0 + 1 and seq2 == seq1 + 1,
                      "seq %d -> %d -> %d" % (seq0, seq1, seq2))
                check("第二次是**新**快照，不是把上一次的缓存还回来",
                      isinstance(live2, dict) and (live2.get("now_us") or 0) >
                      (live.get("now_us") or 0),
                      "now_us %r -> %r" % (live.get("now_us"), (live2 or {}).get("now_us")))
                c0 = (live.get("cards") or [{}])[0]
                check("每张卡该带来的字段都在",
                      all(k in c0 for k in ("name", "busy_us", "run_calls", "mem_at_us",
                                            "mem_total", "mem_free", "node_num",
                                            "node_min_free", "ctx_len")),
                      "keys=%s" % sorted(c0.keys()))
                # 上面几条都一样、但流已经错位一帧时，最先炸的是**下一个普通请求**。
                # 所以最后再跑一轮真问答：这帧要是把流撑歪了，这里必然不通过。
                q = backend.submit(render_messages([{"role": "user",
                                                     "content": "答一个字。"}]),
                                   16, session=0, reset=True)
                # `wait()` 成功时**返回 q**、失败时抛，所以别写成 `err = wait(q)` 再判
                # None——那永远是假的（第一版就这么写的，于是这一项无脑 FAIL）。
                try:
                    backend.wait(q, timeout=REQUEST_TIMEOUT)
                    err = None
                except BackendError as exc:
                    err = exc
                text = q.text()
                backend.release_request(q)
                check("STAT 之后普通请求照常（帧流对齐）",
                      err is None and len(text) > 0,
                      "%r / %d 字" % (err, len(text)))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def selftest(backend, pool):
    """验证 C++ 侧的帧协议。比"直接上 HTTP"分段定位得清楚：协议不通时不用怀疑 HTTP。"""
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print("  [%s] %s %s" % ("PASS" if cond else "FAIL", name, detail))

    print("== READY ==")
    check("sessions", backend.sessions == pool.n, "sessions=%d" % backend.sessions)
    check("default max_new_tokens", backend.default_max_new_tokens > 0,
          "= %d" % backend.default_max_new_tokens)

    print("== max_tokens 夹到后端能接受的范围（否则一个离谱的值能打死一个会话）==")
    check("不传 => 0（用后端默认值）", _pick_max_tokens({}) == 0,
          "= %d" % _pick_max_tokens({}))
    check("正常值原样透传", _pick_max_tokens({"max_tokens": 2000}) == 2000,
          "= %d" % _pick_max_tokens({"max_tokens": 2000}))
    check("超大值被夹到上限（不是原样发出去）",
          _pick_max_tokens({"max_tokens": 10 ** 12}) == MAX_NEW_TOKENS_CAP,
          "10**12 -> %d" % _pick_max_tokens({"max_tokens": 10 ** 12}))
    check("上限本身不变", _pick_max_tokens({"max_tokens": MAX_NEW_TOKENS_CAP}) == MAX_NEW_TOKENS_CAP,
          "= %d" % MAX_NEW_TOKENS_CAP)

    print("== 单请求（非流式）==")
    p1 = render_messages([{"role": "user", "content": "用一句话说明你是谁。"}])
    t0 = time.time()
    q = backend.submit(p1, 32, session=-1, reset=False)
    backend.wait(q)
    dt = time.time() - t0
    check("有回复", len(q.text()) > 0, "%.1fs, %d 字, session=%d, finish=%s"
          % (dt, len(q.text()), q.session, q.finish_reason))
    check("有 decode 计数", q.decode_tokens > 0,
          "prefill=%d decode=%d ctx=%d" % (q.prefill_tokens, q.decode_tokens, q.context_tokens))
    s1 = q.session
    backend.release_request(q)

    print("== 流式（on_delta 回调）==")
    chunks = []
    q = backend.submit(render_messages([{"role": "user", "content": "数到三。"}]), 24,
                       session=s1, reset=True)
    q.on_delta = lambda t: chunks.append(t)
    backend.wait(q)
    joined = "".join(chunks)
    check("收到多个 delta", len(chunks) > 1, "%d 个 delta" % len(chunks))
    check("delta 拼起来 == 整段", joined == q.text(),
          "delta=%d 字, full=%d 字" % (len(joined), len(q.text())))
    backend.release_request(q)

    print("== 会话粘性：只发差异部分应该更快 ==")
    # 先在一个会话里铺一段上下文，然后分别用「全量 + RESET」和「只发差异」跑同一轮，
    # 比 prefill token 数——这是复用生效最直接的证据。
    base_msgs = [{"role": "user", "content": "请记住这句话：" + "麒麟九千" * 60}]
    q = backend.submit(render_messages(base_msgs), 8, session=s1, reset=True)
    backend.wait(q)
    first_ans = q.text()
    backend.release_request(q)
    known = render_messages(base_msgs) + first_ans

    follow = base_msgs + [{"role": "assistant", "content": first_ans},
                          {"role": "user", "content": "继续。"}]
    full = render_messages(follow)
    suffix = full[len(known):]
    check("差异部分确实短得多", 0 < len(suffix) < len(full),
          "full=%d 字, suffix=%d 字" % (len(full), len(suffix)))

    q2 = backend.submit(suffix, 8, session=s1, reset=False)
    backend.wait(q2)
    backend.release_request(q2)
    q3 = backend.submit(full, 8, session=s1, reset=True)
    backend.wait(q3)
    backend.release_request(q3)
    check("复用比全量重算 prefill 少", q2.prefill_tokens < q3.prefill_tokens,
          "复用 prefill=%d tok, 全量 prefill=%d tok"
          % (q2.prefill_tokens, q3.prefill_tokens))

    print("== 关思考时摘掉开头那个（空的）think 段 ==")
    # 这是纯字符串逻辑，不该等板卡——本地一次跑完。逐字符投递那一组是关键：DELTA 是按
    # token 投的，标签完全可能被切成两帧（"<thi" + "nk>"），逐帧正则一定会漏。
    think_cases = [
        ("<think>  </think>  收到", "收到"),
        ("<think>\n</think>\n7 8 9", "7 8 9"),
        ("收到", "收到"),
        ("<think>推理没闭合", "<think>推理没闭合"),   # 被截断：原样交还，不吞正文
        ("<thi", "<thi"),
    ]
    for raw, want in think_cases:
        got1 = strip_think(raw, False)
        s = ThinkStripper()
        got2 = "".join(s.feed(c) for c in raw) + s.flush()
        check("摘 think: %r -> %r" % (raw, want), got1 == want and got2 == want,
              "一次投=%r, 逐字投=%r" % (got1, got2))
    check("开思考时原样透传（不擅自丢推理过程）",
          strip_think("<think>推理</think>答案", True) == "<think>推理</think>答案"
          and strip_think("<think>推理</think>答案", None) == "<think>推理</think>答案")

    # 工具调用的解析也是纯字符串逻辑，同样不必等板卡。放在这里而不是这里重写一遍：
    # 实现和测试同住 toolcalls.py，check_template.py 也调同一个（两处都不需要后端）。
    toolcalls_selftest(check)

    print("== 会话池：RESET+CLEAR 之后仍要能复用（回归）==")
    # 这条是补的：原实现里 release() 在 was_cleared 时把 known 清成空串，于是每段对话
    # 只要第一轮需要 RESET（会话里装着上一段对话），后面每一轮都退化成全量重算，还会把
    # 历史重复堆进 KV。它在"每个对话都拿到干净会话"的测试里完全看不出来，只在会话变脏
    # 之后现形——所以必须专门造出"所有会话都脏"的局面来测。
    def pool_turn(key, prompt, n=8):
        lease = pool.acquire(key, prompt)
        qq = backend.submit(lease.sent_prompt, n, session=lease.session, reset=lease.reset)
        backend.wait(qq)
        pool.release(lease, qq.text(), qq.was_cleared)
        backend.release_request(qq)
        return lease, qq

    def arrange(known, bindings=None):
        """把会话池摆到一个**确定**的初态，并返回原状态。

        为什么要这么写：acquire 会优先挑"没主、没装过东西"的会话，所以池子的历史状态
        会改变它走哪条分支——靠前面的 section 顺手留下的状态来构造场景，断言随时会因为
        上游改动而失去意义（本次就撞上了：加了"优先挑没主的会话"之后，原断言里"所有
        会话都脏"这个前提不成立了）。显式摆状态，测试才是在测它声称要测的东西。
        """
        saved = (list(pool.known), dict(pool.bound), dict(pool.last_used),
                 list(pool.busy), pool.idle_ttl)
        pool.known = list(known)
        pool.bound = dict(bindings or {})
        # last_used 必须一起清：空闲回收（reap）是按它判定的，留着上一段的时刻会让
        # "某段对话静默了多久"变成一个随执行顺序变化的量，断言就失去意义了。
        # 清空之后 reap 里的 `last_used.get(k, now)` 会给所有 key 默认"刚刚才用过"，
        # 于是只有测试显式往前拨过的那些 key 会被回收——这正是我们要的确定性。
        pool.last_used = {}
        return saved

    def restore(saved):
        (pool.known, pool.bound, pool.last_used,
         pool.busy, pool.idle_ttl) = (list(saved[0]), dict(saved[1]), dict(saved[2]),
                                      list(saved[3]), saved[4])

    # 初态：每个会话都"清过 KV、内容未知"（实测里这是常态——每次 RESET 都伴随 CLEAR），
    # 且没有任何绑定。
    saved = arrange([None] * pool.n)
    m1 = [{"role": "user", "content": "请记住这句话：" + "麒麟九千" * 60}]
    lease_a, q_a = pool_turn("reuse-after-reset", render_messages(m1))
    check("会话都脏了 => 这一轮走的是 RESET", lease_a.reset)
    check("后端回了 CLEAR（本轮 KV 被清过）", q_a.was_cleared,
          "ctx=%d" % q_a.context_tokens)
    m2 = m1 + [{"role": "assistant", "content": q_a.text()},
               {"role": "user", "content": "继续。"}]
    p2 = render_messages(m2)
    lease_b, q_b = pool_turn("reuse-after-reset", p2)
    check("RESET 之后第二轮仍然只发差异部分", bool(lease_b.base) and not lease_b.reset
          and len(lease_b.sent_prompt) < len(p2),
          "sent=%d / full=%d 字, base=%d 字, reset=%s"
          % (len(lease_b.sent_prompt), len(p2), len(lease_b.base), lease_b.reset))
    check("复用那轮 prefill 确实更少", q_b.prefill_tokens < q_a.prefill_tokens,
          "%d vs %d tok" % (q_b.prefill_tokens, q_a.prefill_tokens))

    print("== 会话池：新对话不许被派到别人占着的会话上（否则两段对话只能互相等）==")
    # 这条是实测撞出来的：http_scaling 在同一构建上量到过 3.40x，也量到过 1.90x / 1.00x。
    # 根因是 acquire 挑空闲会话时只看 busy、不看 bound——"空闲"的会话可能正被另一段
    # 对话绑着（只是此刻没在跑），新对话被派过去之后两段对话共用一个会话，后端同一会话
    # 一次只跑一条，于是它们**只能轮流跑**：答案全对、并发掉一半，而且退化成不成全看
    # 请求到达顺序（谁先抢到谁的绑定就赢），所以是偶发。
    # 构造要点：让被绑着的那些会话 known 都是 None（清过 KV 的常态——实测现场就是这样），
    # 否则 min() 的备选项排序会"顺手"避开它，掩盖这个 bug。
    restore(saved)
    arrange([None] * pool.n, {"owner-x": 0})
    lease_y, _ = pool_turn("owner-y", render_messages([{"role": "user", "content": "乙"}]))
    check("新对话没有落到 owner-x 占着的会话上（s0）",
          lease_y.session != 0 or pool.n == 1,
          "新对话拿到 s%d（共 %d 个会话）" % (lease_y.session, pool.n))
    # 上限路径：所有会话都有主时，新对话**排队等**，不许把谁的会话夺过来。
    # 这条以前断言的是相反的（"仍能派活且旧主人被解绑"）——那时的取舍是"响应快"，
    # 代价是被抢的那段对话下一轮前缀对不上、要全量重算，而且谁被抢**取决于请求到达
    # 顺序**。现在改成可预期的排队：等 + 空闲回收（见下一条）。
    arrange([None] * pool.n, dict(("fill-%d" % i, i) for i in range(pool.n)))
    before = dict(pool.bound)
    try:
        pool.acquire("intruder", render_messages([{"role": "user", "content": "来抢"}]),
                     timeout=0.3)
        check("会话全都有主时排队等待（不抢占）", False, "竟然拿到了会话")
    except BackendError as exc:
        check("会话全都有主时排队等待（不抢占）",
              dict(pool.bound) == before and "no session available" in str(exc),
              "超时报错且绑定未变（%d 个会话全都有主）" % pool.n)
    check("排队失败后不留残留", "intruder" not in pool.waiting and "intruder" not in pool.bound)

    # 空闲回收：一段对话静默超过 TTL，它的会话要交出来——**这是"排队"能结束的唯一机制**，
    # 没有它，第 N+1 段对话永远等不到（网关不知道谁"说完了"，客户端不发结束消息、浏览器
    # 关了也没人通知）。构造：所有会话都有主 + 显式把 fill-0 的"最后提问时刻"往前拨。
    saved_ttl, saved_reaped = pool.idle_ttl, pool.reaped
    pool.idle_ttl = 0.2
    pool.last_used["fill-0"] = time.time() - 10.0
    # 这里直接 acquire 而不是 pool_turn：要断言的 reset/known 是 **acquire 那一刻**的状态，
    # pool_turn 会顺手 release 掉，把 known 又写成确定值，断言就落空了。
    lease_w = pool.acquire("waiter", render_messages([{"role": "user", "content": "轮到我了吗"}]))
    check("静默超时的对话被回收，排队的立刻拿到会话",
          pool.reaped == saved_reaped + 1 and "fill-0" not in pool.bound
          and lease_w.waited < 1.0,
          "reaped %d->%d，waiter 等了 %.2fs 拿到 s%d"
          % (saved_reaped, pool.reaped, lease_w.waited, lease_w.session))
    check("回收后该会话内容未知 => 强制 RESET（否则新对话的全量 prompt 会追加在旧上下文后）",
          lease_w.reset is True and pool.known[lease_w.session] is None,
          "known=%r reset=%s" % (pool.known[lease_w.session], lease_w.reset))
    pool.release(lease_w, "", True)
    # 回收只影响**闲着的**那段对话。正在跑的那一轮不能被收走：收走也没用（它马上
    # 就还回来），只会让它在跑到一半时被判成"空闲"、正跑着的 KV 被标成"内容未知"。
    s1_busy = pool.bound["fill-1"]
    pool.last_used["fill-1"] = time.time() - 10.0
    pool.busy[s1_busy] = True
    n_reaped = pool.reaped
    with pool.cv:
        pool._reap_idle_locked(time.time())
    pool.busy[s1_busy] = False
    check("正在跑的会话不被回收",
          pool.reaped == n_reaped and "fill-1" in pool.bound,
          "fill-1 在 s%d 上跑着，仍在 bound 里" % s1_busy)
    pool.idle_ttl = saved_ttl

    print("== 会话池：有人在等时按 contend_idle 收（NPU 空着不许让人干等）==")
    # 用户 2026-09-18 报的："明明 NPU 没人使用，但是我输入问题后还是在等待"。
    # 机制：**只有 idle_ttl 一条路决定谁腾会话**，而它必须调得很大（300s）才保得住多轮
    # 对话的 KV 复用。于是四段对话把四个会话绑住后，第 5 段对话要等到某一段静默满 300s；
    # 那段时间里四个会话全是 idle、NPU 一个 token 都没在算。板上实测等了 226.0s。
    # 修法：**已经有请求非等不可**（进来发现一个空闲会话都没有）时，改用 contend_idle
    # 这一级：只收**最闲的**那一段，且只收一个。没人排队时这一级完全不生效——常规路径
    # 逐字节不变，这也是下面第一条断言要钉的。
    saved_ttl3, saved_contend, saved_reaped3 = (pool.idle_ttl, pool.contend_idle,
                                                pool.reaped)
    pool.idle_ttl, pool.contend_idle = 999.0, 1.0
    # ① 常规路径：还有空闲会话可用时，**不许**用第二级阈值把别人踢掉。
    #    这里 fill-0 已经静默 30s（远超 contend_idle=1.0），但只要还有空会话，它就该
    #    原地留着——它的 KV 是下一轮复用的本钱，为了"顺手腾地方"扔掉就是白花 prefill。
    if pool.n >= 2:
        arrange([None] * pool.n, dict(("fill-%d" % i, i) for i in range(pool.n - 1)))
        pool.last_used["fill-0"] = time.time() - 30.0
        lease_free = pool.acquire("newbie", render_messages([{"role": "user", "content": "甲"}]))
        check("还有空闲会话时，第二级回收不生效（常规路径不变）",
              pool.reaped == saved_reaped3 and "fill-0" in pool.bound,
              "reaped %d->%d，fill-0 %s"
              % (saved_reaped3, pool.reaped,
                 "仍在 bound 里" if "fill-0" in pool.bound else "被踢了"))
        pool.release(lease_free, "", True)

        # ② 抢手时：一个空闲会话都没有 => 立刻收**最闲的那一段**，而且只收一个。
        #    fill-0 静默 5s、fill-1 静默 60s：受害者必须是 fill-1。按静默挑而不是按
        #    dict 顺序，行为才可预期（当初决定"不抢占"就是怕"谁被抢看到达顺序"）。
        arrange([None] * pool.n, dict(("fill-%d" % i, i) for i in range(pool.n)))
        pool.last_used["fill-0"] = time.time() - 5.0
        pool.last_used["fill-1"] = time.time() - 60.0
        # fill-2（如果有）正在跑：再闲也不许收——收走没用（它马上就还回来），还会让
        # 正跑着的那一轮的 KV 被标成"内容未知"。
        s_busy = pool.bound.get("fill-2")
        if s_busy is not None:
            pool.busy[s_busy] = True
            pool.last_used["fill-2"] = time.time() - 120.0
        t0q = time.time()
        lease_c = pool.acquire("late-comer", render_messages([{"role": "user", "content": "乙"}]))
        took = time.time() - t0q
        check("一个空闲会话都没有时，立刻收最闲的那一段（不再等到 idle_ttl）",
              lease_c.waited < 1.0 and "fill-1" not in pool.bound
              and pool.reaped == saved_reaped3 + 1,
              "等了 %.2fs，reaped %d->%d，拿到 s%d"
              % (lease_c.waited, saved_reaped3, pool.reaped, lease_c.session))
        check("收的是**最闲的**那一段（fill-1 静默 60s > fill-0 的 5s）",
              "fill-0" in pool.bound, "bound=%s" % sorted(pool.bound))
        check("只收一个（多收等于把几段对话的 KV 一起扔掉）", pool.reaped == saved_reaped3 + 1)
        if s_busy is not None:
            check("正在跑的会话即使在抢手时也不被收", "fill-2" in pool.bound,
                  "fill-2 在 s%d 上跑着" % s_busy)
            pool.busy[s_busy] = False
        check("收走的那段对话回来必须 RESET（内容已归别人）",
              lease_c.reset is True and pool.known[lease_c.session] is None,
              "known=%r reset=%s" % (pool.known[lease_c.session], lease_c.reset))
        pool.release(lease_c, "", True)
    else:
        print("  [SKIP] 只有 %d 个会话，跳过" % pool.n)

    # ③ contend_idle=0 = 关掉这一级，退回老行为（排到 idle_ttl 为止）。留这个开关是
    #    为了让"到底该不该抢"这件事可以被现场实验推翻，而不是写死在代码里。
    if pool.n >= 2:
        pool.contend_idle = 0.0
        arrange([None] * pool.n, dict(("fill-%d" % i, i) for i in range(pool.n)))
        pool.last_used["fill-1"] = time.time() - 600.0
        try:
            pool.acquire("waiter-2", render_messages([{"role": "user", "content": "丙"}]),
                         timeout=0.3)
            check("contend_idle=0 时退回老行为（只排队，不回收）", False, "竟然拿到了会话")
        except BackendError:
            check("contend_idle=0 时退回老行为（只排队，不回收）",
                  "fill-1" in pool.bound and pool.reaped == saved_reaped3 + 1,
                  "超时报错，绑定未变")
    pool.idle_ttl, pool.contend_idle, pool.reaped = (saved_ttl3, saved_contend,
                                                     saved_reaped3)

    # 无身份的请求（key=None）**不许记绑定**：记了的话 bound[None] 只有一条，所有匿名
    # 请求都会读到它，于是素不相识的几段对话被钉到同一个会话上互相等——正好是这个池子
    # 存在的意义的反面。这条是差点被我改丢的：原来的代码有 `if key is not None`，
    # 重写排队逻辑时漏了，而漏了它**不会报任何错**，只是并发静默退化。
    arrange([None] * pool.n)
    lease_n1 = pool.acquire(None, render_messages([{"role": "user", "content": "匿名甲"}]))
    pool.release(lease_n1, "", True)
    lease_n2 = pool.acquire(None, render_messages([{"role": "user", "content": "匿名乙"}]))
    pool.release(lease_n2, "", True)
    check("无身份的请求不记绑定（否则所有匿名请求会共用一个会话）",
          None not in pool.bound and lease_n1.session != lease_n2.session,
          "bound 里没有 None；两次匿名请求落在 s%d / s%d"
          % (lease_n1.session, lease_n2.session))

    # 主动交还（`POST /v1/conversations/close`）：客户端说"我走了"，会话立刻回到池子里，
    # 不用等 IDLE_TTL。这是"排队"在真实使用里能成立的前提之一——否则一个连着问 4 个
    # 不相干问题的脚本会把 4 个会话全占住，后面的人等满一个 TTL。
    arrange([None] * pool.n)
    lease_c = pool.acquire("alice", render_messages([{"role": "user", "content": "甲"}]))
    pool.release(lease_c, "答完了", True)
    s_alice = lease_c.session
    check("交还前：这段对话占着会话", pool.bound.get("alice") == s_alice,
          "alice 在 s%d" % s_alice)
    got = pool.close("alice")
    check("close 放掉了它占的会话", got == s_alice and "alice" not in pool.bound
          and pool.last_used.get("alice") is None,
          "关闭返回 s%s，bound=%s" % (got, dict(pool.bound)))
    check("交还后内容标成未知（下一个拿到的人必须 RESET，不能把新 prompt 追加在旧 KV 后）",
          pool.known[s_alice] is None, "known[s%d]=%r" % (s_alice, pool.known[s_alice]))
    check("重复 close 是幂等的（返回 None，不报错）", pool.close("alice") is None)
    # 交还之后，**新的**对话应该能立刻拿到这个会话，而不是排队等 TTL
    saved_ttl2 = pool.idle_ttl
    pool.idle_ttl = 999.0          # 关掉回收：这里要证明的是 close 起了作用，不是 TTL 起了作用
    pool.busy = [False] * pool.n   # 其余会话都装成"忙"，只留 s_alice 一个是空的
    for s in range(pool.n):
        if s != s_alice:
            pool.busy[s] = True
    t0c = time.time()
    lease_d = pool.acquire("bob", render_messages([{"role": "user", "content": "乙"}]))
    check("交还出来的会话能被新对话立刻拿到（不用等 TTL）",
          lease_d.session == s_alice and lease_d.waited < 0.5,
          "bob 等了 %.2fs 拿到 s%d" % (lease_d.waited, lease_d.session))
    # 正在跑的那一轮**不许**被抽走会话：那会让它脚下的 KV 被下一个人 RESET 掉。
    pool.busy[lease_d.session] = True
    try:
        pool.close("bob")
        busy_refused = False
    except BackendError:
        busy_refused = True
    pool.busy[lease_d.session] = False
    check("正在生成时 close 被拒（不能抽走跑着的会话）",
          busy_refused and pool.bound.get("bob") == lease_d.session)
    pool.release(lease_d, "", True)
    pool.idle_ttl = saved_ttl2
    restore(saved)

    print("== RESET 语义 ==")
    q = backend.submit(render_messages([{"role": "user", "content": "你好"}]), 8,
                       session=s1, reset=True)
    backend.wait(q)
    after = q.context_tokens
    backend.release_request(q)
    check("RESET 后上下文从头累计", after < 200, "本轮结束 ctx=%d" % after)

    print("== 并发：N 个会话同时跑 ==")
    if backend.sessions >= 2:
        n = min(backend.sessions, 4)
        res = [None] * n
        t0 = time.time()

        def worker(i):
            r = backend.submit(render_messages([{"role": "user", "content": "数到二十。"}]),
                               64, session=i, reset=True)
            backend.wait(r)
            res[i] = (r.decode_tokens, r.decode_ms)
            backend.release_request(r)

        ths = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        wall = time.time() - t0
        tot = sum(x[0] for x in res if x)
        check("N 路都出了 token", all(x and x[0] > 0 for x in res),
              "合计 %d tok / %.1fs = %.2f tok/s（N=%d）" % (tot, wall, tot / wall, n))
    else:
        print("  [SKIP] 只有 %d 个会话，跳过并发项" % backend.sessions)

    print("== REJECT：后端拒了一轮，会话**不能**被标死 ==")
    if any("fake_backend" in str(a) for a in backend.argv):
        selftest_reject(backend, check)
    else:
        # 真后端没有 --ctx-limit，造不出这个帧序；真板上要把一段对话撑到 4096 token 才
        # 触发，那不叫自检、那叫压测。真后端那半个只能在板卡上手动验（把上下文跑满）。
        print("  [SKIP] 这一段要桩后端（--ctx-limit 是桩专有的开关）")

    print("== 后端死了 / 会话全 dead：不能让人排在队列里空等 ==")
    selftest_pool_liveness(check)

    print("== 资源观测：/proc 解析、CPU 百分比算术、STAT 帧 ==")
    selftest_system(check, backend)

    print("\n%s" % ("自检全部通过" if ok else "自检有失败项，见上面的 FAIL"))
    return 0 if ok else 1


# ===========================================================================
# 6. main
# ===========================================================================

def main():
    global _VERBOSE
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--sessions", type=int, default=4,
                    help="期望的会话数。与后端 READY 报的对不上时**只警告、不退出**——"
                         "池子大小取自后端报的数（它才是真正知道几个会话可用的一方），"
                         "这个参数只是让你早点看见「参数和实际对不上」，真不一致也照样能服务")
    ap.add_argument("--backend-log", default="gateway_backend.log")
    ap.add_argument("--idle-ttl", type=float, default=DEFAULT_IDLE_TTL,
                    help="一段对话静默超过这么多秒就把它占的会话收回给排队者"
                         "（0 = 不回收，等于第 N+1 段对话永远排队）。默认 %.0f"
                         % DEFAULT_IDLE_TTL)
    ap.add_argument("--contend-idle", type=float, default=DEFAULT_CONTEND_IDLE,
                    help="**已经有请求非等不可**（一个空闲会话都没有）时用的静默阈值："
                         "静默超过这么多秒的对话所占的会话立刻收回给等的人，只收最闲的"
                         "那一个。没人排队时仍按 --idle-ttl。默认 %.0f；0 = 关掉这一级"
                         "（退回「一定要等到 --idle-ttl」）"
                         % DEFAULT_CONTEND_IDLE)
    ap.add_argument("--queue-timeout", type=float, default=DEFAULT_QUEUE_TIMEOUT,
                    help="取不到会话时最多排队等这么多秒，超了返回 503。"
                         "默认 %.0f；应大于 --idle-ttl，否则会出现"
                         "「马上就要轮到了，请求却先超时」"
                         % DEFAULT_QUEUE_TIMEOUT)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="只验帧协议（不起 HTTP），用于把 C++ 侧和 HTTP 侧分开定位")
    ap.add_argument("--frames-stdout", action="store_true",
                    help="仅本地自检：帧从后端的 stdout 读（配桩后端 fake_backend.py）。"
                         "pass_fds 在 Windows 上不可用，板卡上用不到这个开关")
    argv = sys.argv[1:]
    if "--" not in argv:
        ap.error("backend command must follow '--', e.g. "
                 "-- python3 ... -- <demo argv>")
    cut = argv.index("--")
    args = ap.parse_args(argv[:cut])
    demo_argv = argv[cut + 1:]
    if not demo_argv:
        ap.error("empty backend command after '--'")
    _VERBOSE = bool(args.verbose)

    backend = Backend(demo_argv, args.backend_log,
                      frame_fd="stdout" if args.frames_stdout else None)
    print("[gateway] starting backend, model load takes ~230-240s ...", flush=True)
    try:
        backend.start()
    except BackendError as exc:
        print("[gateway] backend start failed: %s" % exc, file=sys.stderr)
        return 2
    print("[gateway] backend ready: sessions=%d default_max_new_tokens=%d"
          % (backend.sessions, backend.default_max_new_tokens), flush=True)
    if args.sessions and backend.sessions != args.sessions:
        print("[gateway] warning: --sessions %d but backend reports %d"
              % (args.sessions, backend.sessions), file=sys.stderr)

    pool = SessionPool(backend, idle_ttl=args.idle_ttl,
                       queue_timeout=args.queue_timeout,
                       contend_idle=args.contend_idle)
    print("[gateway] 会话调度: 最多 %d 段对话同时活跃，多的排队；"
          "静默 %.0fs 回收（**有人在等时**降到 %.0fs，只收最闲的一段），"
          "排队 %.0fs 未轮到则返回 503"
          % (backend.sessions, pool.idle_ttl, pool.contend_idle,
             pool.queue_timeout), flush=True)
    try:
        if args.selftest:
            return selftest(backend, pool)

        Handler.gateway = Gateway(backend, pool, verbose=args.verbose)
        httpd = ThreadingHTTPServer((args.host, args.port), Handler)
        httpd.daemon_threads = True
        print("[gateway] listening on http://%s:%d  (sessions=%d, model=%s)"
              % (args.host, args.port, backend.sessions, MODEL_ID), flush=True)
        # 别再写死"4 个对话框"：页面 2026-09-20 起按 /health 的 sessions 决定开几个窗口
        # （32768 那份导出只够 2 路），屏幕上看的是几个就是几个。ctx_size 也一并印出来，
        # 因为它才是"为什么只有这么几个窗口"的答案。
        print("[gateway] 演示页（窗口数 = 本后端的会话数，当前 %d 个；上下文 %s）: "
              "http://%s:%d/demo"
              % (backend.sessions,
                 backend.ctx_size if backend.ctx_size is not None else "未在命令行给出",
                 args.host, args.port), flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()
    finally:
        backend.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

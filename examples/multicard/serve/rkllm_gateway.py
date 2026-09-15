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
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = "qwen3.5-27b"
BACKEND_READY_TIMEOUT = 900.0     # 模型加载约 230-240s，留足余量
REQUEST_TIMEOUT = 1800.0          # 单轮最长等待（长上下文 prefill 很慢）
DEFAULT_MAX_NEW_TOKENS = 512

IM_START = "<|im_start|>"
IM_END = "<|im_end|>\n"
# 与 examples/multicard/cpp/main.cc 的 QWEN35_CHAT_TEMPLATE 保持一致：
#   system_prompt = "<|im_start|>system\n...<|im_end|>\n"
#   user_prefix   = "<|im_start|>user\n"
#   user_postfix  = "<|im_end|>\n<|im_start|>assistant\n"
QWEN35_DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


# ===========================================================================
# 1. 后端进程与帧协议
# ===========================================================================

class BackendError(Exception):
    pass


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
        self.was_cleared = False        # 后端在本轮里清过 KV（上下文将满）
        self.on_delta = None            # 流式回调；在 reader 线程里被调用
        self.dead = False               # 后端进程没了
        self.kv = None                  # 会话池本轮的复用决策，用于 X-KV-Reuse 头

    def text(self):
        return self.raw.decode("utf-8", "replace")


class Backend(object):
    """demo 子进程 + 帧协议。所有方法都线程安全。"""

    def __init__(self, argv, log_path, frame_fd=None):
        self.argv = argv
        self.log_path = log_path
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
        self._rfile = None
        self._reader = None
        self.on_clear = None                  # 回调 (session, ctx, limit)

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
                elif tag in (b"DELTA", b"ERR"):
                    rid_s, _, n_s = rest.partition(b" ")
                    n = int(n_s)
                    payload = self._rfile.read(n) if n > 0 else b""
                    self._dispatch_payload(tag, int(rid_s), payload)
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
            with self._lock:
                pending = list(self._reqs.values())
            for q in pending:
                q.dead = True
                q.error = q.error or "backend exited"
                q.done.set()

    def _get(self, rid):
        with self._lock:
            return self._reqs.get(rid)

    def _dispatch_payload(self, tag, rid, payload):
        q = self._get(rid)
        if q is None:
            return          # 客户端已经放弃这个 rid 了，丢掉帧即可
        if tag == b"DELTA":
            q.raw.extend(payload)
            if q.on_delta is not None:
                chunk = q.decoder.decode(bytes(payload))
                if chunk:
                    try:
                        q.on_delta(chunk)
                    except Exception:
                        # 客户端断开了：不影响后端，继续把这一轮读完
                        q.on_delta = None
        else:               # ERR
            q.error = payload.decode("utf-8", "replace")
            q.done.set()

    def _handle_done(self, f):
        if len(f) < 8:
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
            raise BackendError(q.error)
        return q

    def release_request(self, q):
        with self._lock:
            self._reqs.pop(q.rid, None)


# ===========================================================================
# 2. 会话池（粘性 + 前缀校验）
# ===========================================================================

class Lease(object):
    __slots__ = ("session", "sent_prompt", "base", "reset")

    def __init__(self, session, sent_prompt, base, reset):
        self.session = session
        self.sent_prompt = sent_prompt
        self.base = base            # acquire 时该会话已知的文本（reset 时为 ""）
        self.reset = reset


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

    def __init__(self, backend):
        self.backend = backend
        self.n = backend.sessions
        self.cv = threading.Condition()
        self.known = [""] * self.n
        self.busy = [False] * self.n
        self.dead = [False] * self.n
        self.bound = {}          # conversation key -> session index
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
            self.cv.notify_all()

    def acquire(self, key, prompt, timeout=REQUEST_TIMEOUT):
        """返回 Lease；拿不到（全忙/全死）抛 BackendError。"""
        deadline = time.time() + timeout
        with self.cv:
            while True:
                # 优先复用已经绑在这一段对话上的会话：只有它才可能已经有这段历史。
                cand = self.bound.get(key)
                if cand is not None and not self.dead[cand] and not self.busy[cand]:
                    return self._make_lease(cand, prompt)
                if cand is not None and not self.dead[cand]:
                    pass        # 同一段对话的上一轮还在跑：等它，不能换会话
                else:
                    # 挑一个空闲会话：优先没装过东西的（省一次 RESET）。unknown（None）
                    # 排在有内容的前面：反正都要 RESET，清空的会话至少不用先扔垃圾。
                    free = [s for s in range(self.n)
                            if not self.dead[s] and not self.busy[s]]
                    if free:
                        best = min(free, key=lambda s: (self.known[s] is not None,
                                                       s))
                        if key is not None:
                            self.bound[key] = best
                        return self._make_lease(best, prompt)
                if time.time() >= deadline:
                    raise BackendError("no session available within %.0fs "
                                       "(all %d busy or dead)" % (timeout, self.n))
                self.cv.wait(min(1.0, max(0.05, deadline - time.time())))

    def _make_lease(self, session, prompt):
        self.busy[session] = True
        base = self.known[session]
        if base and prompt.startswith(base):
            return Lease(session, prompt[len(base):], base, False)
        # 复用不了，两种情况必须分开：
        #   base 非空但前缀对不上 => 里面确实装着别的对话，必须 RESET，否则这次的全量
        #     prompt 会**追加**在旧上下文后面，模型看到两份历史；
        #   base 是 None（清过 KV，内容未知）=> 同样必须 RESET。这里不能写成
        #     `bool(base)`：None 和 "" 都是假值，会把"未知"当"空"处理，正好踩中
        #     上面那个追加的坑。
        return Lease(session, prompt, "", base is None or bool(base))

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


def render_messages(messages, enable_thinking=None):
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
    no_think = (enable_thinking is False)

    parts = []
    for m in msgs:
        role = m.get("role") or "user"
        text = _content_text(m.get("content"))
        if role == "tool":
            # Qwen 系模板把工具返回包在 <tool_response> 里。这一支没有端到端验证过
            # （见文件末尾"未验证项"），先按通用角色块渲染，不要指望 tool 调用能跑通。
            text = "<tool_response>\n%s\n</tool_response>" % text
        if no_think and role == "user":
            text = text + " /no_think"
        parts.append("%s%s\n%s%s" % (IM_START, role, text, IM_END))
    parts.append("%sassistant\n" % IM_START)
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

class Gateway(object):
    def __init__(self, backend, pool, verbose=False):
        self.backend = backend
        self.pool = pool
        self.verbose = verbose
        self.started_at = time.time()

    def log(self, fmt, *args):
        if self.verbose:
            sys.stderr.write("[gateway] " + (fmt % args) + "\n")
            sys.stderr.flush()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "rkllm-openai-gateway/1.0"

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

    def do_GET(self):
        self._responded = False
        path = self.path.split("?")[0]
        if path in ("/health", "/healthz"):
            return self._json(200, {"status": "ok",
                                    "model": MODEL_ID,
                                    "sessions": self.gateway.backend.sessions,
                                    "uptime_s": round(time.time() - self.gateway.started_at, 1)})
        if path == "/v1/models":
            return self._json(200, {"object": "list", "data": [{
                "id": MODEL_ID, "object": "model",
                "created": int(self.gateway.started_at), "owned_by": "local"}]})
        return self._error(404, "unknown path: %s" % path)

    def do_POST(self):
        self._responded = False
        path = self.path.split("?")[0]
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
        prompt = render_messages(messages, enable_thinking=enable_thinking)
        key = conversation_key(messages, explicit=req.get("conversation_id")
                               or self.headers.get("X-Conversation-Id"))
        self.gateway.log("chat: %d messages, %d prompt bytes, stream=%s, max_new=%d",
                         len(messages), len(prompt.encode("utf-8")), stream, max_new)

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
                    "full": len(prompt.encode("utf-8"))}
            self.gateway.log("lease: session=%d reset=%s sent=%d/%d bytes (base=%d)",
                             q.kv["session"], q.kv["reset"], q.kv["sent"], q.kv["full"],
                             q.kv["base"])
            if stream:
                self._stream_response(req, q, lease, enable_thinking)
            else:
                self._blocking_response(req, q, enable_thinking)
        except BackendError as exc:
            self._fail(503, str(exc), "server_error")
        except Exception as exc:                        # noqa: BLE001
            self._fail(500, "%r" % (exc,), "server_error")
        finally:
            # 无论成功、失败还是客户端断开，都要把会话还回去；推理失败还要把会话标死。
            if lease is not None:
                failed = q is None or q.error or q.dead
                if failed:
                    self.gateway.pool.mark_dead(lease.session)
                    self.gateway.pool.release(lease, "", True)
                else:
                    # 记账用的必须是**客户端会回显的那份文本**，也就是摘掉 think 段之后的
                    # 版本：下一轮 prompt 是这个版本的拼接，前缀判据才有可能成立。用
                    # q.text()（原始版）记的话，关思考时每轮都会判"对不上"，缓存静默全废
                    # ——而且答案是对的，看不出来（nothink_check.py 抓的就是这个）。
                    self.gateway.pool.release(
                        lease, strip_think(q.text(), enable_thinking), q.was_cleared)
            if q is not None:
                self.gateway.backend.release_request(q)

    # ---- 两种响应形态 ----

    def _stream_response(self, req, q, lease, enable_thinking=None):
        created = int(time.time())
        cid = "chatcmpl-%d-%d" % (created, q.rid)
        include_usage = bool((req.get("stream_options") or {}).get("include_usage"))
        # 关思考时把开头那个（空的）think 段摘掉，两种响应形态共用同一份状态机
        stripper = ThinkStripper() if enable_thinking is False else None

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
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

        def emit(text):
            """统一出口：过一遍 stripper（没开 stripper 就是原样），非空才发。"""
            if stripper is not None:
                text = stripper.feed(text)
            if text:
                sse(delta_obj(text))
                sent[0] += 1

        def on_delta(text):
            emit(text)

        q.on_delta = on_delta
        # 已经攒在 q.raw 里的那部分（等待期间的 token）也要补发
        if q.raw:
            emit(q.text())
        q.done.wait(REQUEST_TIMEOUT)
        if not q.done.is_set():
            self.gateway.pool.mark_dead(lease.session)
            self._chunk(b"")            # 收尾：别让客户端一直挂着
            raise BackendError("timeout waiting for generation")

        if q.error:
            # HTTP 头已经发出去了，只能用 SSE 里的错误对象收尾：先报错，再按协议
            # 正常结束这个流（clients 普遍是看到 [DONE] 才算读完）。
            sse({"error": {"message": q.error, "type": "server_error"}})
            self._chunk(b"data: [DONE]\n\n")
            self._chunk(b"")
            raise BackendError(q.error)

        # 收尾前先把 stripper 里还攒着的字节放出来（正常情况是空；被截断在 <think> 里时
        # 会连标签一起吐出来，见 flush() 的说明），否则客户端会少一段正文。
        if stripper is not None:
            tail = stripper.flush()
            if tail:
                sse(delta_obj(tail))
                sent[0] += 1

        sse(delta_obj(None, finish=q.finish_reason or "stop"))
        if include_usage:
            sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                 "model": MODEL_ID, "choices": [],
                 "usage": self._usage(q)})
        self._chunk(b"data: [DONE]\n\n")
        self._chunk(b"")                # 终止 chunk
        self.gateway.log("stream done: rid=%d session=%d chunks=%d tokens=%d",
                         q.rid, q.session, sent[0], q.decode_tokens)

    def _blocking_response(self, req, q, enable_thinking=None):
        q.done.wait(REQUEST_TIMEOUT)
        if not q.done.is_set():
            raise BackendError("timeout waiting for generation")
        if q.error:
            raise BackendError(q.error)
        created = int(time.time())
        self._json(200, {
            "id": "chatcmpl-%d-%d" % (created, q.rid),
            "object": "chat.completion",
            "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "finish_reason": q.finish_reason or "stop",
                         "message": {"role": "assistant",
                                     "content": strip_think(q.text(), enable_thinking)}}],
            "usage": self._usage(q),
        }, extra_headers=kv_reuse_headers(q.kv))

    def _usage(self, q):
        return {"prompt_tokens": q.prefill_tokens,
                "completion_tokens": q.decode_tokens,
                "total_tokens": q.prefill_tokens + q.decode_tokens}


def _pick_max_tokens(req):
    for name in ("max_tokens", "max_completion_tokens", "n_predict", "max_new_tokens"):
        v = req.get(name)
        if isinstance(v, int) and v > 0:
            return v
    return 0        # 0 = 让后端用它自己的默认值（--n-predict）


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


def kv_reuse_headers(kv):
    """把本轮的复用决策回报给客户端。

    为什么值得占一个响应头：**光看 usage.prompt_tokens 分不清「复用生效」和「换了个
    空会话」**——两种情况下 prefill 都可能等于全量。这里直接把会话号、是否 RESET、
    实际发出去多少字节摊开，出问题时不用猜。（也方便 Agent 侧观察缓存命中率。）
    """
    if not kv:
        return {}
    return {"X-KV-Reuse": "session=%d; reset=%d; sent=%d; base=%d; full=%d"
                          % (kv["session"], 1 if kv["reset"] else 0, kv["sent"],
                             kv["base"], kv["full"])}


# ===========================================================================
# 5. 自检：只验帧协议，不起 HTTP
# ===========================================================================

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

    for k in ("dirty-a", "dirty-b"):       # 先把每个会话说脏（首轮不需要 RESET）
        pool_turn(k, render_messages([{"role": "user", "content": "占位 " + k}]))
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

    print("\n%s" % ("自检全部通过" if ok else "自检有失败项，见上面的 FAIL"))
    return 0 if ok else 1


# ===========================================================================
# 6. main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--sessions", type=int, default=4,
                    help="期望的会话数；与后端 READY 报的对不上就退出")
    ap.add_argument("--backend-log", default="gateway_backend.log")
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

    pool = SessionPool(backend)
    try:
        if args.selftest:
            return selftest(backend, pool)

        Handler.gateway = Gateway(backend, pool, verbose=args.verbose)
        httpd = ThreadingHTTPServer((args.host, args.port), Handler)
        httpd.daemon_threads = True
        print("[gateway] listening on http://%s:%d  (sessions=%d, model=%s)"
              % (args.host, args.port, backend.sessions, MODEL_ID), flush=True)
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

// demo_4chat.html 的自动化检查：用**桩 DOM**把页面里那段真实 script 跑起来，打到真板卡上。
//
// 为什么不另写一份等价代码来测：那样测的是"我以为页面在做什么"，不是页面本身。
// 这里跑的就是 `<script>` 里的原文（正则抽出来原样喂给 vm），只有 DOM 是假的。
//
// 它守的几条都是**踩过坑才加的**：
//   1. 四个默认问题必须互不相同  —— 相同的话网关会把它们钉到同一个会话上，变成排队
//   2. 流结束后必须清掉 active   —— render() 曾经把刚算好的 tok/s 统计行覆盖掉
//   3. 四路必须落在四个不同会话 —— 退化成串行时，屏幕上看不出来，只有会话号能戳穿
//   4. 统计行必须带会话号        —— 拿不到 X-KV-Reuse 时（缺 Expose-Headers）会静默变空
//
// 用法（**必须在 serve/ 目录下跑**，它按相对路径读 demo_4chat.html）：
//   node page_check.js http://<板卡IP>:8080
//   MAXTOK=64 node page_check.js http://<板卡IP>:8080    # 只改本次演示的每轮长度，不动页面默认值
//
// 需要 node 与一台正在跑网关的板卡；不进 run_all_board_tests.sh（那套是纯 python3 的）。
const fs = require("fs");
const vm = require("vm");

const BASE = process.argv[2] || "http://127.0.0.1:8080";

if (!fs.existsSync("demo_4chat.html")) {
  console.log("FAIL: 当前目录没有 demo_4chat.html —— 请在 serve/ 目录下运行");
  process.exit(1);
}

// ---------- 极简 DOM 桩 ----------
function makeEl(tag) {
  const el = {
    tagName: tag, className: "", value: "", checked: false, disabled: false,
    style: {}, children: [], _text: "", _html: "", _byRole: {}, _byClass: {},
    _listeners: {},
    classList: {
      _s: new Set(),
      toggle(c, on) { if (on === undefined) { on = !this._s.has(c); } on ? this._s.add(c) : this._s.delete(c); },
      add(c) { this._s.add(c); }, remove(c) { this._s.delete(c); },
      contains(c) { return this._s.has(c); },
    },
    appendChild(c) { this.children.push(c); c.parentNode = this; return c; },
    addEventListener(ev, fn) { (this._listeners[ev] = this._listeners[ev] || []).push(fn); },
    querySelector(sel) {
      if (sel[0] === "[") {
        const m = /\[data-role=["']?([\w-]+)["']?\]/.exec(sel);
        return m ? (this._byRole[m[1]] || null) : null;
      }
      if (sel[0] === ".") { return this._byClass[sel.slice(1)] || null; }
      return null;
    },
    get textContent() { return this._text; },
    set textContent(v) { this._text = String(v); },
    get innerHTML() { return this._html; },
    set innerHTML(v) {
      this._html = String(v);
      // innerHTML 里的 data-role="x" 和 class="dot" 建成子元素，供 querySelector 找到
      for (const m of v.matchAll(/data-role=["']?([\w-]+)["']?/g)) {
        if (!this._byRole[m[1]]) { this._byRole[m[1]] = makeEl("input"); }
      }
      for (const m of v.matchAll(/class=["']([\w\s-]+)["']/g)) {
        for (const c of m[1].split(/\s+/)) {
          if (c && !this._byClass[c]) { this._byClass[c] = makeEl("span"); }
        }
      }
    },
  };
  return el;
}

const byId = {};
const document = {
  getElementById(id) { return (byId[id] = byId[id] || makeEl("div")); },
  createElement(tag) { return makeEl(tag); },
};

// ---------- 抽页面里的 script 原样跑 ----------
const html = fs.readFileSync("demo_4chat.html", "utf8");
let script = /<script>([\s\S]*?)<\/script>/.exec(html)[1];
if (process.env.MAXTOK) {
  const before = script;
  script = script.replace(/const MAXTOK = \d+;/, "const MAXTOK = " + process.env.MAXTOK + ";");
  if (script === before) { console.log("FAIL: 没能替换 MAXTOK"); process.exit(1); }
  console.log("（本次 MAXTOK=" + process.env.MAXTOK + "）");
}
const ctx = {
  document, performance, TextDecoder, console, JSON, Promise, Math, Date,
  setTimeout, Error,
  // 页面里的 API 是 ""（同源相对路径），浏览器会自动按页面来源补全；node 不会，
  // 所以这里替它补——**只是替浏览器补全这一步，请求本身没做任何改写**。
  fetch: (url, opts) => fetch(url.startsWith("http") ? url : BASE + url, opts),
};
ctx.globalThis = ctx;
// 页面脚本没导出任何东西，补一行把内部函数交出来供测试调用
vm.createContext(ctx);
vm.runInContext(script + "\n;globalThis.__t = { panels: panels, send: send, "
               + "streamChat: streamChat, updateGlobal: updateGlobal };", ctx);

const { panels, send } = ctx.__t;
console.log("buildPanels 建出 %d 个面板", panels.length);
if (panels.length !== 4) { console.log("FAIL: 面板数不是 4"); process.exit(1); }

const qs = panels.map(p => p.input.value);
console.log("四个默认问题:", JSON.stringify(qs, null, 0));
if (new Set(qs).size !== 4) {
  // 首问相同的话，网关会把它们钉到同一个会话上 → 变成排队而不是并发
  console.log("FAIL: 四个默认问题必须互不相同（否则会抢同一个会话）");
  process.exit(1);
}

(async () => {
  const t0 = Date.now();
  const rs = await Promise.all(panels.map(p => send(p)));
  const wall = (Date.now() - t0) / 1000;

  let bad = 0;
  rs.forEach((r, i) => {
    const stat = panels[i].stat.textContent;
    const dot = panels[i].dot.classList.contains("on");
    const active = panels[i].el.classList.contains("active");
    console.log("\n--- 面板 %d ---", i + 1);
    console.log("  返回:", r ? (r.tok + " tok / " + r.el.toFixed(1) + "s = "
                                 + r.rate.toFixed(2) + " tok/s") : "null");
    console.log("  统计行:", stat);
    console.log("  会话:", panels[i].sess.textContent || "(空)");
    console.log("  答案前 40 字:", JSON.stringify((r ? r.text : "").slice(0, 40)));
    if (!r) { console.log("  FAIL: 这一路没返回"); bad++; return; }
    if (!/tok\/s/.test(stat)) { console.log("  FAIL: 统计行里没有 tok/s（render 把统计行覆盖了？）"); bad++; }
    if (!/会话/.test(panels[i].sess.textContent)) { console.log("  FAIL: 没拿到会话号"); bad++; }
    if (dot || active) { console.log("  FAIL: 流结束了但状态还是 active"); bad++; }
    if (!r.exact) { console.log("  WARN: usage 没到，token 数是数块数出来的"); }
  });

  const sess = panels.map(p => p.kvSession);
  console.log("\n四路落到的会话:", JSON.stringify(sess));
  if (new Set(sess).size !== 4) {
    console.log("FAIL: 四路没有落在四个不同会话上（会退化成串行！）");
    bad++;
  }
  const total = rs.reduce((a, r) => a + (r ? r.tok : 0), 0);
  console.log("合计 " + total + " tok / 墙钟 " + wall.toFixed(1) + "s → 聚合 "
              + (total / wall).toFixed(2) + " tok/s");
  console.log("\n统计行渲染后的全局栏:", byId["gstats"].textContent || byId["gstats"].innerHTML);
  console.log(bad ? "\n===== 有 " + bad + " 项 FAIL =====" : "\n===== 全部通过 =====");
  process.exit(bad ? 1 : 0);
})().catch(e => { console.error("炸了:", e); process.exit(2); });

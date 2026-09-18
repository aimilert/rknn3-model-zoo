// demo_4chat.html 的自动化检查：用**桩 DOM**把页面里那段真实 script 跑起来，打到真板卡上。
//
// 为什么不另写一份等价代码来测：那样测的是"我以为页面在做什么"，不是页面本身。
// 这里跑的就是 `<script>` 里的原文（正则抽出来原样喂给 vm），只有 DOM 是假的。
//
// 它守的几条都是**踩过坑才加的**：
//   1. 四路必须各自带**显式身份**（`X-Conversation-Id: web-1..4`）—— 靠"首问不同"认对话
//      是会撞的，而且撞了之后答案全对、只是变慢，屏幕上根本看不出来
//   2. 每一路占到的会话，在 `/v1/pool` 里必须记着**它自己的**名字 —— 光看会话号还不够：
//      身份没送出去的话会话号照样是四个不同值，但占的是四段**匿名**对话
//   3. 关页面 / 点「清空」都必须把会话**真的还回去**（两条不同的代码：sendBeacon 与 fetch）
//      —— 页面占着 4 个会话不放，会把后来的客户端（终端、第二个浏览器）全堵在队列里，
//      直到 IDLE_TTL 超时
//   4. 四个默认问题不许有重复     —— 现在并发不靠它了，这是演示素材的抄写检查
//   5. 流结束后必须清掉 active    —— render() 曾经把刚算好的 tok/s 统计行覆盖掉
//   6. 四路必须落在四个不同会话   —— 退化成串行时，屏幕上看不出来，只有会话号能戳穿
//   7. 统计行必须带会话号         —— 拿不到 X-KV-Reuse 时（缺 Expose-Headers）会静默变空
//   8. 页面要的 id 必须真实存在   —— 桩不再给不存在的 id 现造元素，拼错的 id 会像浏览器一样
//     返回 null（见 byId 那段注释）
//   9. 面板头的性能小字必须写出来 —— `data-role="perf"` 拼错、元素被删、setPerf 忘了调，
//     这三种坏法页面都照跑照出答案，只是那格永远空着（2026-09-18 加）
//  10. 新一轮开始时那格必须是空的 —— 挂着**上一轮**的首字延迟/速率，在最该判断"到底开始
//     没有"的那一帧上给的是旧数字，比空着更坏（2026-09-18 加）
//  11. 汇总那一句的两种"平均"不许混 —— 各路速率和 Σ(tok_i/el_i) 与单路基线**同口径**，
//     伸缩比只能用它；墙钟摊薄 Σtok/墙钟 在四路长短不齐时会被空转的尾巴摊小。用户
//     2026-09-18 报过这次真事：一路跑 135.7s、三路约 16s，屏幕上给"聚合 15.43 tok/s"，
//     看着跟单路基线 10.6 是一路的。下面那组 fixture 拿那几个真实数钉死两个数都要算对
//  12. 被摊薄/排过队必须当场说破 —— 稀释时不报"平均只有 1.37/4 路在跑"、排队时不标
//     "伸缩比不可引用"，这两种数就会被当成并发性能读走。**数没错，是没人知道它是什么**
//
// 用法（**必须在 serve/ 目录下跑**，它按相对路径读 demo_4chat.html）：
//   node page_check.js http://<板卡IP>:8080
//   MAXTOK=64 node page_check.js http://<板卡IP>:8080    # 只改本次演示的每轮长度，不动页面默认值
//
// 需要 node。后端**不一定是板卡**：对着本地桩后端（`fake_backend.py`，起法见 README.md 测试
// 一节）就能整跑，2026-09-18 实测全绿——改完这一页先在本地过一遍，省一次板卡往返。桩验不
// 到的是真后端的**长度与内容**行为（截断、吐不吐 <think>），那些还得上板。
// 不进 run_all_board_tests.sh（那套是纯 python3 的）。
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

// ---------- 抽页面里的 script 原样跑 ----------
const html = fs.readFileSync("demo_4chat.html", "utf8");

// 页面里**静态声明**的 id 全部先建出来（下面 getElementById 只认这些）。
// 为什么不是"要什么就现造一个"（2026-09-16 修）：桩以前是
// `getElementById(id) { return (byId[id] = byId[id] || makeEl("div")); }` —— 任何 id
// 都能凭空拿到一个元素。于是页面把 `getElementById("gstat")` 拼错成少一个 s，桩给一个
// 空 div，检查照跑照绿；真浏览器里拿到的是 null，下一行取属性就 TypeError，整页白屏。
// 也就是说**页面上的 id 拼写错误是这套检查唯一测不出来的东西**，而它正是最容易犯的错
// （改页面时顺手改了 id、忘了改 JS）。现在只认 HTML 里真实存在的 id，要别的就返回 null
// ——和浏览器一样，拼错立刻暴露。
const byId = {};
for (const m of html.matchAll(/\bid="([\w-]+)"/g)) {
  if (!byId[m[1]]) { byId[m[1]] = makeEl("div"); }
}
const missingIds = [];
const document = {
  getElementById(id) {
    if (byId[id]) { return byId[id]; }
    missingIds.push(id);
    console.log("  FAIL: 页面 getElementById(" + JSON.stringify(id)
                + ") —— HTML 里没有这个 id（拼错了？浏览器这里会返回 null 并抛 TypeError）");
    return null;
  },
  createElement(tag) { return makeEl(tag); },
};

let script = /<script>([\s\S]*?)<\/script>/.exec(html)[1];

// MAXTOK=64 node page_check.js … —— 只改**本次演示**的每轮长度，不动页面默认值。
//
// 页面把输出上限做成输入框之后（原来写死成源码里的 `const MAXTOK = 256;`），这里不再去
// 改页面源码里的常量，而是**像用户一样把值交进去**：页面启动时先读 localStorage 里记住的
// 上次设定（demo_4chat.html 的 initMaxtok），桩只要往那个 store 里放一个值，页面就会自己
// 把它填进输入框、并带着它发请求——和真浏览器走的是同一条路径，也就不存在"改源码改出来的
// 行为跟用户点出来的不一样"这种事。
const LS_KEY = "demo4chat.maxtok";
const lsStore = {};
if (process.env.MAXTOK) {
  lsStore[LS_KEY] = process.env.MAXTOK;
  console.log("（本次 MAXTOK=" + process.env.MAXTOK + "）");
}
// 记下每次请求带的身份头：验证"四路确实是以四段不同对话的身份发出去的"，不能只看结果。
const sentIds = [];
// 也把请求体记下来。"输出上限"从源码常量变成输入框之后，"输入框里的值有没有真的被
// 带出去"就成了一件**看不出来**的事：页面照跑、答案照出，只是长度还是老的默认值。
// 光看结果分不出"模型自己停的"和"上限没送出去"，所以直接查请求体。
const sentBodies = [];
const abs = (url) => (url.startsWith("http") ? url : BASE + url);
const winListeners = {};
const ctx = {
  document, performance, TextDecoder, console, JSON, Promise, Math, Date,
  setTimeout, Error,
  // 页面用 setInterval 在"还没拿到会话"的那段时间里每 200ms 刷新一次统计行（2026-09-18
  // 加：那一刻页面上本来只有一个转圈，看着像卡死）。桩里必须是**真的**定时器，不能是
  // 空函数——空函数会让踩醒间隔的那条路在桩里悄悄走不通，绿了也说明不了页面没问题
  // （这次就是先漏了它，页面在桩里直接报 `setInterval is not defined`，四路全灭）。
  setInterval, clearInterval,
  // 页面里的 API 是 ""（同源相对路径），浏览器会自动按页面来源补全；node 不会，
  // 所以这里替它补——**只是替浏览器补全这一步，请求本身没做任何改写**。
  fetch: (url, opts) => {
    const h = (opts && opts.headers) || {};
    sentIds.push(h["X-Conversation-Id"] || null);
    try { sentBodies.push(JSON.parse(opts.body)); } catch (e) { sentBodies.push(null); }
    return fetch(abs(url), opts);
  },
  // 页面用 localStorage 记住上次的输出上限。桩给一个**真的小 store**，而不是一对空函数：
  // 空函数会让"设了值 → 刷新还在"这条路径在桩里悄悄走不通，绿了也说明不了页面没问题。
  localStorage: {
    getItem(k) { return Object.prototype.hasOwnProperty.call(lsStore, k) ? lsStore[k] : null; },
    setItem(k, v) { lsStore[k] = String(v); },
    removeItem(k) { delete lsStore[k]; },
  },
  // 页面用到的浏览器全局。桩的职责是**把环境补齐**，页面代码不该为了能在桩里跑而变形。
  window: {
    addEventListener(ev, fn) { (winListeners[ev] = winListeners[ev] || []).push(fn); },
  },
  // sendBeacon 在桩里**真的把那句话发出去**（浏览器里也是这么发的）。要是这里摆个空函数，
  // "关页面有没有把会话还回去"就只能靠嘴说了——而这恰恰是最容易泄漏的一条路。
  navigator: {
    sendBeacon(url, body) {
      fetch(abs(url), { method: "POST", body: body,
                        headers: { "Content-Type": "text/plain;charset=UTF-8" } })
        .catch(() => {});
      return true;
    },
  },
};
ctx.globalThis = ctx;
// 页面脚本没导出任何东西，补一行把内部函数交出来供测试调用
vm.createContext(ctx);
vm.runInContext(script + "\n;globalThis.__t = { panels: panels, send: send, "
               + "streamChat: streamChat, updateGlobal: updateGlobal, "
               + "summarize: summarize, runAll: runAll, "
               + "lastRound: function () { return lastRound; }, "
               // 基线速率是页面里一个会被改的变量，"交出当时的值"没有意义——
               // 要的是**读得到当下的值**，所以交一个闭包而不是快照。
               + "baselineRate: function () { return lastSingleRate; } };", ctx);

const { panels, send, summarize } = ctx.__t;

// ---------- 汇总那一句的算术（纯函数，不联网） ----------
//
// 2026-09-18 用户报的："合计 2095 tok / 墙钟 135.7s → 聚合 15.43 tok/s"，看着跟单路基线
// 10.6 是一路的，于是"四路并发没什么用"就是那么读出来的。**问题不在数，在口径**：
// 那 135.7s 里只有前 18 秒是四路在跑，后面 120 秒是**一路**在长生成（1602 tok），
// 把"一路的吞吐"摊进了"四路并发"的分母。所以这里钉死两个数**都算出来、且不混淆**，
// 并且钉死**伸缩比只许用各路速率和**（口径与基线一致）。
//
// 这几个数是从用户那次真实运行抄下来的，不是编的：改了公式而忘了改断言，这一节会报 FAIL。
const near = (got, want, tol) => Math.abs(got - want) <= (tol === undefined ? 0.011 : tol);
const grab = (html, re) => { const m = html.match(re); return m ? parseFloat(m[1]) : NaN; };

function fixture(name, rs, wall, baseline) {
  const html = summarize(rs, wall, baseline);
  console.log("  [%s] %s", name, html.replace(/<[^>]*>/g, ""));
  return html;
}

{
  let bad = 0;
  const chk = (ok, msg) => { if (!ok) { console.log("  FAIL: " + msg); bad++; } };

  // ① 用户那次真事的数字：四路 153/182/1602/158 tok，墙钟 135.7s
  const long = [{ tok: 153, el: 15.6, waited: 0 }, { tok: 182, el: 18.1, waited: 0 },
                { tok: 1602, el: 135.7, waited: 0 }, { tok: 158, el: 16.3, waited: 0 }];
  const h1 = fixture("长短不齐", long, 135.7, 10.63);
  chk(near(grab(h1, /速率和 <b>([\d.]+)/), 41.36), "各路速率和应为 41.36 tok/s（实测值）");
  chk(near(grab(h1, /墙钟摊薄 ([\d.]+)/), 15.44), "墙钟摊薄应为 15.44 tok/s（实测值）");
  chk(/1\.37\/4 路在跑/.test(h1), "应报出「平均只有 1.37/4 路在跑」（稀释诊断）");
  chk(near(grab(h1, /伸缩 ([\d.]+)×/), 3.89), "伸缩比应为 3.89× = 41.36/10.63（用速率和，不是摊薄值）");
  // 摊薄值除以基线只有 1.45×，看着像"四路并发毫无用处"——那正是用户报的读数，
  // 所以这条不许只查"等于 3.89"，还要查它**大于**摊薄口径算出来的值。
  chk(grab(h1, /伸缩 ([\d.]+)×/) > 2,
      "伸缩比不许用摊薄值算（否则得到 1.45×，屏幕上就是「并发没用」）");
  // 这两条是"口径不许混"的正面断言：主数必须是速率和，摊薄值必须戴着它的帽子出现
  chk(h1.indexOf("各路速率和") < h1.indexOf("墙钟摊薄"), "速率和必须排在摊薄值前面（它是主数）");
  chk(/别当并发性能看/.test(h1), "摊薄时有义务说明「这不是并发性能」");
  chk(/合计 2095 tok/.test(h1), "合计 token 数应为 2095");

  // ② 四路等长：两个数本来就该接近，这时摊薄值是可以看的，且不许报稀释
  const even = [{ tok: 150, el: 15, waited: 0 }, { tok: 150, el: 15, waited: 0 },
                { tok: 150, el: 15, waited: 0 }, { tok: 150, el: 15, waited: 0 }];
  const h2 = fixture("四路等长", even, 15.1, null);
  chk(near(grab(h2, /速率和 <b>([\d.]+)/), 40.00, 0.005), "速率和应为 40.00 tok/s");
  chk(near(grab(h2, /墙钟摊薄 ([\d.]+)/), 39.74), "摊薄应为 39.74 tok/s");
  chk(!/长短不齐/.test(h2), "四路等长时不该报稀释警告（假警报会让人不再信这条）");
  chk(/二者等价/.test(h2), "等长时应说明两个数等价");
  chk(/先点「①」/.test(h2), "没有基线时应提示去量基线，不许默默不显示伸缩比");

  // ③ 排过队：伸缩比必须当场标成不可引用（排队混在墙钟里，数看着像"并发退化"）
  const q = [{ tok: 153, el: 15.6, waited: 0 }, { tok: 182, el: 18.1, waited: 40.2 },
             { tok: 1602, el: 135.7, waited: 0 }, { tok: 158, el: 16.3, waited: 0 }];
  const h3 = fixture("某路排队", q, 135.7, 10.63);
  chk(/有 1 路排过队（最多 40\.2s），上面的数含伸缩比都不可引用/.test(h3),
      "应报出排队并声明伸缩比不可引用");
  chk(/伸缩 3\.89×/.test(h3), "排队时数照样显示，只是标明不可引用");
  chk(h3.indexOf("伸缩 3.89×") < h3.indexOf("排过队"),
      "排队警告必须排在伸缩比之后（它撤的就是刚印出来的那个数）");
  chk(!/排过队/.test(h1), "没人排队时不许报排队（假警报会把真警报淹掉）");

  if (bad) {
    console.log("FAIL: 汇总算术有 %d 条不对（上面每一条都对着真实运行数钉过）", bad);
    process.exit(1);
  }
  console.log("  汇总算术 3 组 fixture 全过\n");
}

// 统计行是**HTML**（速率要加粗，见页面里的 setStat），桩又把 _text 与 _html 分开存
// （和浏览器一样），所以读 `stat.textContent` 会拿到空串。2026-09-18 修：页面把 setStat
// 从 textContent 改成 innerHTML 时，这里读的字段没跟着改，`/tok\/s/` 那条断言就会
// 在页面完全正常的情况下报 FAIL——**空跑也像失败**，比不检查更费时间。
// 这个取值器两边都认，改哪一边都不会再假装报错。
const statOf = (el) => el.textContent || el.innerHTML;
console.log("buildPanels 建出 %d 个面板", panels.length);
if (panels.length !== 4) { console.log("FAIL: 面板数不是 4"); process.exit(1); }

const qs = panels.map(p => p.input.value);
console.log("四个默认问题:", JSON.stringify(qs, null, 0));
if (new Set(qs).size !== 4) {
  // 四路是不是四段对话，现在由显式身份保证，**不再依赖问题不同**——所以这不是
  // 并发问题，是素材问题：这四问是挑过的演示默认值（见 demo_4chat.html 的注释），
  // 短了会四路参差、长了会顶到 max_tokens，重复说明改的时候抄错了。
  console.log("FAIL: 四个默认问题有重复（不影响并发，但这四个默认值是挑过的，请对齐页面注释）");
  process.exit(1);
}

(async () => {
  // 等 `id:web-*` 全部从会话池里消失（交还是异步发出的，给最多 4 秒）。
  // 返回**还占着的那几个**：空数组才算干净，非空就直接进 FAIL 信息里。
  async function waitReleased() {
    let held = [];
    for (let k = 0; k < 40; k++) {
      const p = await (await fetch(BASE + "/v1/pool")).json();
      held = (p.slots || []).map(s => s.key).filter(x => x && x.startsWith("id:web-"));
      if (!held.length) { return []; }
      await new Promise(r => setTimeout(r, 100));
    }
    return held;
  }

  // 走**用户点「②」的那条路**（runAll），不是自己 `panels.map(send)` 一遍。
  // 差别就在汇总行：自己发的话，页面里 summarize + setSum 那两句压根不执行，
  // 于是"四路那一句怎么算的"完全没有检查覆盖 —— 2026-09-18 改口径时就是这么发现的。
  // 代价是拿不到 runAll 的返回值，得问它要（lastRound）。
  await ctx.__t.runAll();
  const round = ctx.__t.lastRound();
  if (!round) { console.log("FAIL: runAll 没留下这一轮的结果"); process.exit(1); }
  const rs = round.rs, wall = round.wall;

  let bad = 0;
  rs.forEach((r, i) => {
    const stat = statOf(panels[i].stat);
    const dot = panels[i].dot.classList.contains("on");
    const active = panels[i].el.classList.contains("active");
    console.log("\n--- 面板 %d ---", i + 1);
    console.log("  返回:", r ? (r.tok + " tok / " + r.el.toFixed(1) + "s = "
                                 + r.rate.toFixed(2) + " tok/s") : "null");
    console.log("  统计行:", stat);
    console.log("  性能小字:", statOf(panels[i].perf) || "(空)");
    console.log("  会话:", panels[i].sess.textContent || "(空)");
    console.log("  答案前 40 字:", JSON.stringify((r ? r.text : "").slice(0, 40)));
    if (!r) { console.log("  FAIL: 这一路没返回"); bad++; return; }
    if (!/tok\/s/.test(stat)) { console.log("  FAIL: 统计行里没有 tok/s（render 把统计行覆盖了？）"); bad++; }
    // 面板头右边那格性能小字（首字延迟 + tok/s）。守的是**结构性**的坏法：`data-role` 拼错、
    // 元素在 innerHTML 里被删掉、setPerf 忘了调 —— 这三种情况下页面照跑、答案照出，只是那格
    // 永远空着，看演示的人根本不会发现少了个数。
    const perf = statOf(panels[i].perf);
    if (!/首字/.test(perf) || !/tok\/s/.test(perf)) {
      console.log("  FAIL: 面板头的性能小字没写出来（首字延迟/tok/s）：" + JSON.stringify(perf));
      bad++;
    }
    if (!/会话/.test(panels[i].sess.textContent)) { console.log("  FAIL: 没拿到会话号"); bad++; }
    if (dot || active) { console.log("  FAIL: 流结束了但状态还是 active"); bad++; }
    if (!r.exact) { console.log("  WARN: usage 没到，token 数是数块数出来的"); }
  });

  // 身份**有没有真的送出去**。这一条是最要紧的：四路匿名时，会话号照样是四个不同值
  // （按首问内容各分一个），四路也照样并发——但占的是四段**匿名**对话，认不出也关不掉，
  // 页面就这么把四个会话全扣在手里，后来的客户端一个都进不来。这个坑真踩过。
  console.log("\n四路带出去的身份:", JSON.stringify(sentIds));
  const wantIds = ["web-1", "web-2", "web-3", "web-4"];
  if (JSON.stringify(sentIds) !== JSON.stringify(wantIds)) {
    console.log("FAIL: 四路带出去的身份不是 " + JSON.stringify(wantIds)
                + "（匿名对话会把会话扣住不放，后面的客户端全被堵住）");
    // 匿名对话**关不掉名字**（它没有名字），所以这一项失败时池子里会留下几个清不掉的
    // 会话，后面每一次重跑都会撞上它们。把清理办法直接写在这里，省得下一个人从头查。
    console.log("      清理：读 /v1/pool 的 slots[].key（形如 h:...），"
                + "逐个 POST /v1/conversations/close {\"key\": ...}；"
                + "或者用 IDLE_TTL 不为 0 的网关重跑");
    bad++;
  }

  // 输出上限**必须真的跟着请求走**。这一条守的是"输入框接了、但没接线"：页面照跑、
  // 答案照出、四路照样并发，只是长度还停在老的默认值上——从结果上完全看不出来
  //（"模型自己停的"和"上限压根没送出去"长得一模一样）。
  const wantMax = Number(byId["maxtok"].value);
  console.log("\n四路带出去的 max_tokens:",
              JSON.stringify(sentBodies.map(b => (b ? b.max_tokens : null))),
              "（输入框里是 " + wantMax + "）");
  if (!(wantMax > 0)) {
    console.log("FAIL: 输入框里的输出上限不是正数：" + JSON.stringify(byId["maxtok"].value));
    bad++;
  } else if (sentBodies.some(b => !b || b.max_tokens !== wantMax)) {
    console.log("FAIL: 有请求带出去的 max_tokens 与输入框不符（该是 " + wantMax
                + "）——输入框没接线？");
    bad++;
  }

  const sess = panels.map(p => p.kvSession);
  console.log("\n四路落到的会话:", JSON.stringify(sess));
  if (new Set(sess).size !== 4) {
    console.log("FAIL: 四路没有落在四个不同会话上（会退化成串行！）");
    bad++;
  }
  // 会话池里那四个会话，是不是**记着各自的名字**（光有会话号不够：身份没送出去时
  // 会话号也是四个不同值，但池子里躺着的是四个 `h:...` 摘要）。
  const pool = await (await fetch(BASE + "/v1/pool")).json();
  const holder = {};
  (pool.slots || []).forEach(s => { holder[s.session] = s.key; });
  console.log("跑完四路后 /v1/pool 的归属:",
              JSON.stringify((pool.slots || []).map(s => s.session + "=" + s.key)));
  panels.forEach((p, i) => {
    const got = holder[p.kvSession];
    if (got !== "id:web-" + (i + 1)) {
      console.log("  FAIL: 面板 %d 占着 s%s，但池子里记的是 %s（该是 id:web-%d）",
                  i + 1, p.kvSession, JSON.stringify(got), i + 1);
      bad++;
    }
  });

  // 汇总行：印**页面上真写出来的那一句**（#gsum），再和这里独立算的一遍对一下。
  // 以前这里只是自己重算一个"聚合 tok/s"印出来——那是**检查自己的口径**，页面把它改了
  // 这里也照样印老口径，读日志的人会以为页面还是老样子。现在这里读的是页面产物。
  // 对不上就 FAIL：说明「②」那条路没走 summarize（或者走了别的公式）。
  const okr = rs.filter(Boolean);
  const gsum = byId["gsum"] ? statOf(byId["gsum"]) : "";
  // 比的是**各路速率和**，不是整句：墙钟这一项两边算不出同一个值（页面用 performance.now()，
  // 这里用 Date.now()，还差着一次 await 的往返），拿整句比等于在测时钟抖动。
  // 速率和只用到 tok 与 el —— 那是页面自己量出来、这里直接读回来的同几个对象，应该逐位相等。
  const want = summarize(okr, wall, ctx.__t.baselineRate());
  const soloOf = (s) => { const m = s.match(/速率和 <b>([\d.]+)/); return m ? parseFloat(m[1]) : NaN; };
  console.log("页面汇总行:", gsum.replace(/<[^>]*>/g, "") || "(空)");
  if (!(Math.abs(soloOf(gsum) - soloOf(want)) < 0.005)) {
    console.log("FAIL: 页面汇总行里的速率和与独立算的对不上（%s vs %s）——「②」没走 summarize？",
                soloOf(gsum), soloOf(want));
    bad++;
  }
  if (!/速率和/.test(gsum)) {
    console.log("FAIL: 汇总行里没有「速率和」——伸缩比的分母口径必须写在一句话里");
    bad++;
  }
  console.log("\n统计行渲染后的全局栏:", byId["gstats"].textContent || byId["gstats"].innerHTML);
  // 从这里开始验"页面走了以后，会话有没有还回去"。两条路，代码不是同一段：
  // 关页面走 sendBeacon（beforeunload），点「清空」走 closeConv（fetch）。
  console.log("");

  // 路一：关页面。这是**最容易出事**的一种——演示完顺手关掉标签页，四个会话就被一个
  // 已经消失的页面扣着，直到 IDLE_TTL（默认 300 秒）超时。后来的人全堵在队列里，
  // 而且没有任何地方能看到"是谁占着"（页面都没了）。
  winListeners["beforeunload"].forEach(function (fn) { fn(); });
  const afterUnload = await waitReleased();
  console.log("关页面之后还占着的会话:", JSON.stringify(afterUnload));
  if (afterUnload.length) {
    console.log("FAIL: 关页面没把会话还回去（后来的人要等 IDLE_TTL 超时）");
    bad++;
  }

  // 路二：点「清空」= 这四段对话不要了。再占一次会话来验，否则上一步已经清干净了，
  // 这一条会变成"空跑也算过"。
  await send(panels[0]);
  document.getElementById("clear")._listeners["click"][0]();
  const afterClear = await waitReleased();
  console.log("点「清空」之后还占着的会话:", JSON.stringify(afterClear));
  if (afterClear.length) {
    console.log("FAIL: 清空了但会话没还回去（后来的人要等 IDLE_TTL 超时）");
    bad++;
  }

  // ---- 排队那段等待，页面上必须"会说话" ----
  // 场景：四个会话都被别的对话占着，网关让这一路在队列里等。**这段时间里 fetch 还没
  // resolve**（响应头是拿到会话之后才发的），所以整段时间页面上一个字都不会多——用户
  // 2026-09-18 报的"NPU 明明没人用，输入问题后还是在等"就长这样：屏幕上只有一个转圈，
  // 看不出是模型慢、是排队、还是板卡坏了。
  //
  // 造法不靠真排队（那要占满会话、还得掐时间）：直接把 fetch 换成一个**慢 350ms + 头里
  // 带 wait=42.0** 的假响应。350ms > 页面的 200ms 刷新间隔，所以至少刷一次；头里的
  // wait 是**权威的那个数**（网关自己记的排队时长），比页面 tick 出来的更准。
  {
    const realFetch = ctx.fetch;
    let midWait = null;
    let midPerf = null;
    ctx.fetch = function () {
      return new Promise(function (resolve) {
        setTimeout(function () {
          // fetch 还没回，此刻页面处在"等会话"状态——正是要抓的这一帧
          midWait = statOf(panels[0].stat);
          midPerf = statOf(panels[0].perf);
          resolve({
            ok: true,
            headers: { get: (k) => (k === "X-KV-Reuse"
              ? "session=0; reset=1; sent=204; base=0; full=204; wait=42.0" : null) },
            body: { getReader: () => {
              const chunks = ["data: {\"choices\":[{\"delta\":{\"content\":\"杭\"}}]}\n\n",
                              "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],"
                              + "\"usage\":{\"completion_tokens\":1}}\n\n",
                              "data: [DONE]\n\n"];
              let i = 0;
              return { read: () => Promise.resolve(
                i < chunks.length ? { done: false, value: new TextEncoder().encode(chunks[i++]) }
                                  : { done: true }) };
            } },
          });
        }, 350);
      });
    };
    try {
      // 上一步「清空」已经把输入框清掉了，而 send() 对着空输入框会直接 return null
      // （不发请求）——不填这一下，这条检查就会变成"空跑也算过"。
      panels[0].input.value = "排队那一格";
      await send(panels[0]);
    } finally {
      ctx.fetch = realFetch;
    }
    const finalStat = statOf(panels[0].stat);
    console.log("\n等待期间的统计行:", midWait);
    console.log("拿到会话后的统计行:", finalStat);
    if (!/等待空闲会话/.test(midWait || "")) {
      console.log("FAIL: 等会话期间统计行没说话（屏幕上只剩转圈，看着像卡死）");
      bad++;
    }
    // 新的一轮开始时，面板头上那格**必须是空的**。留着上一轮的首字延迟/速率，
    // 在最该判断"到底开始了没有"的那一帧上给的是上一轮的数 —— 比空着更坏。
    console.log("等待期间的性能小字:", JSON.stringify(midPerf || ""));
    if (midPerf) {
      console.log("FAIL: 新一轮开始了，面板头上还挂着上一轮的性能数字：" + JSON.stringify(midPerf));
      bad++;
    }
    if (!/排队 42\.0s/.test(finalStat)) {
      console.log("FAIL: 统计行没把排队时长摊出来（tok/s 会把排队算进分母，看着像模型慢）");
      bad++;
    }
  }

  // 页面要过、但 HTML 里不存在的 id。上面每次命中都会打一行 FAIL，这里补一句汇总，
  // 顺带把"拼错的到底是哪个"钉在最显眼的位置（页面拼错 id 时多半会紧接着 TypeError，
  // 堆栈会把前面那几行冲掉）。
  if (missingIds.length) {
    console.log("\n页面问过的、HTML 里不存在的 id: " + JSON.stringify(missingIds));
    bad += missingIds.length;
  }

  console.log(bad ? "\n===== 有 " + bad + " 项 FAIL =====" : "\n===== 全部通过 =====");
  process.exit(bad ? 1 : 0);
})().catch(e => { console.error("炸了:", e); process.exit(2); });

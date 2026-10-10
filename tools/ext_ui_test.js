/*
 * 扩展 UI/行为测试（真实 Chrome，独立配置目录）：
 *  A. 免刷新注入：先开登录页 → 再装扩展 → 点弹窗「登录」→ 脚本自动注入并立即执行
 *  B. 取消条件：程序性 focus 不取消（模拟关闭翻译弹窗）；真实点击表单才取消
 *  C. 弹窗 UI：登录/保存/测试滑块按钮存在；密码显示/隐藏切换
 *  D. 倒计时为 1.5s 低时延
 * 用法: node ext_ui_test.js [chromeExe]
 */
'use strict';
const { spawn } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const OUT = path.join(ROOT, 'output');
const PORT = 9338;
const PROFILE = path.join(os.tmpdir(), 'sufe-ext-uitest-profile');
const CHROME = process.argv[2] || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const DUMMY = { username: '98765432100', password: 'DummyPass-For-UiTest', enabled: true };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

class CDP {
  constructor(wsUrl) { this.wsUrl = wsUrl; this.id = 0; this.pending = new Map(); }
  async connect() {
    this.ws = new WebSocket(this.wsUrl);
    await new Promise((res, rej) => { this.ws.onopen = res; this.ws.onerror = rej; });
    this.ws.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.id && this.pending.has(m.id)) {
        const { res, rej } = this.pending.get(m.id);
        this.pending.delete(m.id);
        m.error ? rej(new Error('CDP error: ' + JSON.stringify(m.error))) : res(m.result);
      }
    };
  }
  send(method, params = {}) {
    const id = ++this.id;
    return new Promise((res, rej) => { this.pending.set(id, { res, rej }); this.ws.send(JSON.stringify({ id, method, params })); });
  }
  async eval(expression) {
    const r = await this.send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (r.exceptionDetails) throw new Error('eval: ' + ((r.exceptionDetails.exception && r.exceptionDetails.exception.description) || r.exceptionDetails.text));
    return r.result.value;
  }
  close() { try { this.ws.close(); } catch (e) { /* ignore */ } }
}

const jget = async (p) => (await fetch(`http://127.0.0.1:${PORT}${p}`)).json();

async function openPage(browser, url) {
  const { targetId } = await browser.send('Target.createTarget', { url });
  for (let i = 0; i < 15; i++) {
    const t = (await jget('/json/list')).find((x) => x.id === targetId);
    if (t && t.webSocketDebuggerUrl) {
      const c = new CDP(t.webSocketDebuggerUrl);
      await c.connect();
      return { c, targetId };
    }
    await sleep(300);
  }
  throw new Error('目标无法调试: ' + url);
}

async function verifyExtId(browser, id) {
  try {
    const { c, targetId } = await openPage(browser, `chrome-extension://${id}/popup.html`);
    const ok = await c.eval('typeof chrome !== "undefined" && typeof chrome.storage === "object" && location.href.indexOf("' + id + '") >= 0');
    c.close();
    await browser.send('Target.closeTarget', { targetId });
    return ok === true;
  } catch (e) { return false; }
}

const READ_WIDGET = `(() => {
  const w = document.getElementById('sufe-auto-slider-widget');
  if (!w) return '(no-widget)';
  const body = w.querySelector('.ssw-body');
  const log = w.querySelector('.ssw-log');
  return 'BODY: ' + (body ? body.textContent.trim() : '') + '\\nLOG: ' + (log ? log.textContent.trim().replace(/\\n+/g, ' | ') : '');
})()`;

async function waitWidget(page, regexStr, timeoutMs = 25000, gapMs = 300) {
  const n = Math.ceil(timeoutMs / gapMs);
  for (let i = 0; i < n; i++) {
    let txt = '';
    try { txt = await page.eval(READ_WIDGET); } catch (e) { /* 页面导航中 */ }
    if (new RegExp(regexStr).test(txt)) return txt;
    await sleep(gapMs);
  }
  return null;
}

async function main() {
  fs.mkdirSync(OUT, { recursive: true });
  const results = [];
  const check = (name, ok, detail = '') => { results.push({ name, ok, detail }); console.log((ok ? '[PASS] ' : '[FAIL] ') + name + (detail ? '  | ' + detail : '')); };

  console.log('== 启动测试 Chrome（先不装扩展）==');
  const child = spawn(CHROME, [`--user-data-dir=${PROFILE}`, `--remote-debugging-port=${PORT}`,
    '--no-first-run', '--no-default-browser-check', '--disable-features=ChromeWhatsNewUI,PrivacySandboxSettings4',
    'about:blank'], { stdio: 'ignore' });

  let browser;
  try {
    let ver = null;
    for (let i = 0; i < 30; i++) { try { ver = await jget('/json/version'); break; } catch (e) { await sleep(1000); } }
    if (!ver) throw new Error('调试端口未就绪');
    browser = new CDP(ver.webSocketDebuggerUrl);
    await browser.connect();

    // 先开登录页（模拟：页面先开着，之后才装扩展）
    const { c: page, targetId: pageTid } = await openPage(browser, 'https://login.sufe.edu.cn/login/');
    await page.send('Page.enable');
    let formReady = false;
    for (let i = 0; i < 40; i++) { // 等 SPA 渲染出登录表单（placeholder 不在 innerText，改判 input 存在）
      try { formReady = await page.eval('!!document.querySelector("input") && document.body.innerText.indexOf("密码登录") >= 0'); } catch (e) { formReady = false; }
      if (formReady) break;
      await sleep(500);
    }
    check('登录页已渲染（装扩展前）', formReady === true);

    // 装扩展
    const r = await browser.send('Extensions.loadUnpacked', { path: ROOT });
    const extId = r.id;
    console.log('扩展 ID:', extId);

    // A1. 已开页面上没有内容脚本（未刷新）→ 不应出现浮窗
    const noWidget = await page.eval('!!document.getElementById("sufe-auto-slider-widget")');
    check('A1 免刷新场景成立：旧页面暂无浮窗', noWidget === false);

    // C. 弹窗 UI + 密码切换（先等 popup 页与 popup.js 就绪）
    const { c: popup, targetId: popupTid } = await openPage(browser, `chrome-extension://${extId}/popup.html`);
    for (let i = 0; i < 25; i++) {
      let ready = false;
      try { ready = await popup.eval('document.readyState === "complete" && !!document.getElementById("login")'); } catch (e) { ready = false; }
      if (ready) break;
      await sleep(300);
    }
    await sleep(400);
    const ui = await popup.eval(`JSON.stringify({
      login: !!document.getElementById('login'),
      save: !!document.getElementById('save'),
      test: !!document.getElementById('test'),
      toggle: !!document.getElementById('togglePwd'),
    })`);
    const u = JSON.parse(ui);
    check('C1 弹窗按钮齐全（登录/保存/测试滑块/密码切换）', u.login && u.save && u.test && u.toggle, ui);
    const tog = await popup.eval(`(() => {
      const p = document.getElementById('password'); const t = document.getElementById('togglePwd');
      p.value = 'abc'; t.click(); const a = p.type; const ta = t.textContent;
      t.click(); const b = p.type; const tb = t.textContent;
      return a + '/' + b + '/' + ta + '/' + tb;
    })()`);
    check('C2 密码显示/隐藏切换', tog === 'text/password/隐藏/显示', String(tog));
    await popup.eval('chrome.storage.local.set(' + JSON.stringify({ sufeAutoSliderCfg: DUMMY }) + ').then(()=>1)');

    // A2. 激活登录页标签，点真实「登录」按钮 → 走注入路径
    await browser.send('Target.activateTarget', { targetId: pageTid });
    await sleep(500);
    await popup.eval(`document.getElementById('login').click()`);
    await sleep(2500);
    const popStatus = await popup.eval('document.getElementById("status").textContent');
    const injected = await page.eval('!!document.getElementById("sufe-auto-slider-widget")');
    check('A2 点「登录」后脚本已注入（无需刷新）', injected === true, '弹窗状态: ' + popStatus);
    const forced = await page.eval(`(() => {
      const w = document.getElementById('sufe-auto-slider-widget');
      return w ? w.querySelector('.ssw-body').textContent : '';
    })()`);
    check('D1 注入即执行（跳过倒计时，显示登录进度而非 X 秒倒计时）', !/\d\.\d 秒后自动登录/.test(forced), String(forced).slice(0, 60));
    const solvedTxt = await waitWidget(page, '滑块已解锁|学号不存在|学号或密码错误', 30000, 500);
    check('A3 注入后的完整流程跑通（滑块解锁/凭据被拒）', !!solvedTxt, solvedTxt ? solvedTxt.split('\n')[0] : '未观察到');

    // B. 取消条件：重新加载登录页（此时走自动注入路径）
    await page.send('Page.navigate', { url: 'https://login.sufe.edu.cn/login/' });
    let cd = await waitWidget(page, '秒后自动登录', 25000, 200);
    check('D2 出现倒计时（1.5s 低时延）', !!cd, cd ? cd.split('\n')[0] : '未观察到');
    if (cd) {
      const shown = (cd.match(/([\d.]+) 秒后自动登录/) || [])[1];
      check('D3 倒计时起点为 1.5 秒', shown === '1.5' || shown === '1.0', '显示 ' + shown + ' 秒');
    }
    // B1: 程序性 focus（模拟关闭翻译弹窗后输入框重新获得焦点）→ 不应取消
    await page.eval(`(() => { const el = document.querySelector('input.f-account, input[name=username], input[type=text]'); if (el) el.focus(); return true; })()`);
    const afterFocus = await waitWidget(page, '滑块已解锁|学号不存在|学号或密码|滑块解锁中|获取登录配置', 12000, 300);
    const cancelled1 = await page.eval('document.body.innerText.indexOf("已取消自动登录") >= 0');
    check('B1 程序性 focus 不再取消自动登录', cancelled1 === false, afterFocus ? String(afterFocus).split('\n')[0] : (cancelled1 ? '被取消' : '流程未见'));

    // B2: 真实点击表单 → 应取消（先找一个"点下去必定命中输入框"的坐标）
    await page.send('Page.navigate', { url: 'https://login.sufe.edu.cn/login/' });
    cd = await waitWidget(page, '秒后自动登录', 25000, 150);
    let pt = null;
    if (cd) {
      try {
        pt = await page.eval(`(() => {
          const els = [...document.querySelectorAll('input')];
          for (const el of els) {
            const r = el.getBoundingClientRect();
            if (r.width < 60 || r.height < 20 || r.top < 0 || r.bottom > innerHeight || r.left < 0 || r.right > innerWidth) continue;
            const cands = [[r.x + r.width / 2, r.y + r.height / 2], [r.x + r.width / 2, r.y + r.height * 0.3], [r.left + 12, r.y + r.height / 2]];
            for (const [x, y] of cands) {
              const top = document.elementFromPoint(x, y);
              if (top && (top === el || el.contains(top) || top.contains(el))) return { x: x, y: y, ph: el.placeholder || el.name || el.type };
            }
          }
          return null;
        })()`);
      } catch (e) { /* ignore */ }
    }
    if (cd && pt) {
      const stillCounting = await page.eval(`(() => { const w = document.getElementById('sufe-auto-slider-widget'); return w ? w.textContent.indexOf('秒后自动登录') >= 0 : false; })()`);
      await page.send('Input.dispatchMouseEvent', { type: 'mousePressed', x: pt.x, y: pt.y, button: 'left', clickCount: 1 });
      await page.send('Input.dispatchMouseEvent', { type: 'mouseReleased', x: pt.x, y: pt.y, button: 'left', clickCount: 1 });
      const cancelled2 = await waitWidget(page, '已取消自动登录', 4000, 200);
      check('B2 真实点击表单仍会取消（保留手动接管）', !!cancelled2,
        '点击时倒计时中=' + stillCounting + '; 命中=(' + Math.round(pt.x) + ',' + Math.round(pt.y) + ') ' + pt.ph);
    } else {
      check('B2 真实点击表单仍会取消（保留手动接管）', false, '倒计时或可点坐标未取到 (cd=' + !!cd + ' pt=' + !!pt + ')');
    }

    // B3: 真实打字（trusted input 事件）→ 应取消
    await page.send('Page.navigate', { url: 'https://login.sufe.edu.cn/login/' });
    cd = await waitWidget(page, '秒后自动登录', 25000, 150);
    if (cd) {
      try {
        await page.eval(`(() => { const el = document.querySelector('input.f-account, input[name=username], input[type=text], input');
          if (el) el.focus(); return true; })()`);
        await page.send('Input.insertText', { text: 'x' });
      } catch (e) { /* ignore */ }
      const cancelled3 = await waitWidget(page, '已取消自动登录', 4000, 200);
      check('B3 真实打字输入仍会取消（保留手动接管）', !!cancelled3);
    } else {
      check('B3 真实打字输入仍会取消（保留手动接管）', false, '未捕捉到倒计时');
    }

    console.log('== 汇总 ==');
    const fails = results.filter((x) => !x.ok);
    console.log(results.length - fails.length + '/' + results.length + ' 项通过');
    for (const f of fails) console.log('失败:', f.name, f.detail);
    page.close(); popup.close();
  } finally {
    if (browser) { try { await browser.send('Browser.close'); } catch (e) { try { child.kill(); } catch (e2) { /* ignore */ } } }
    else { try { child.kill(); } catch (e2) { /* ignore */ } }
    await sleep(1200);
    console.log('测试 Chrome 已关闭');
  }
}

main().catch((e) => { console.error('测试失败:', e.message); process.exit(1); });

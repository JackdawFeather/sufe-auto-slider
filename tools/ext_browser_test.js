/*
 * 扩展真实环境测试：以开发者模式加载本仓库扩展启动独立 Chrome 实例（临时配置目录），
 * 预置假学号配置，打开登录页，观察扩展浮窗全流程（倒计时→CNN 解滑块→doLogin）。
 * 用法: node ext_browser_test.js [chromeExe]
 * 产物: output/ext_test_widget.png（浮窗截图）
 *
 * 扩展 ID 获取：优先 CDP Extensions.loadUnpacked（精准）；否则扫描 service worker 并逐个验证。
 */
'use strict';
const { spawn } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const OUT = path.join(ROOT, 'output');
const PORT = 9333;
const PROFILE = path.join(os.tmpdir(), 'sufe-ext-test-profile');
const CHROME = process.argv[2] || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const DUMMY = { username: '98765432100', password: 'DummyPass-For-ExtTest', enabled: true };

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

async function openExtPage(browser, url) {
  const { targetId } = await browser.send('Target.createTarget', { url });
  for (let i = 0; i < 10; i++) {
    const t = (await jget('/json/list')).find((x) => x.id === targetId);
    if (t && t.webSocketDebuggerUrl) {
      const c = new CDP(t.webSocketDebuggerUrl);
      await c.connect();
      return { c, targetId };
    }
    await sleep(300);
  }
  throw new Error('扩展页无法建立调试连接: ' + url);
}

async function verifyExtId(browser, id) {
  try {
    const { c, targetId } = await openExtPage(browser, `chrome-extension://${id}/popup.html`);
    const ok = await c.eval('typeof chrome !== "undefined" && typeof chrome.storage === "object" && location.href.indexOf("' + id + '") >= 0');
    c.close();
    await browser.send('Target.closeTarget', { targetId });
    return ok === true;
  } catch (e) { return false; }
}

async function main() {
  fs.mkdirSync(OUT, { recursive: true });
  console.log('== 启动测试 Chrome（独立配置目录）==');
  console.log('profile:', PROFILE);
  const child = spawn(CHROME, [
    `--user-data-dir=${PROFILE}`,
    `--remote-debugging-port=${PORT}`,
    '--no-first-run', '--no-default-browser-check',
    '--disable-features=ChromeWhatsNewUI,PrivacySandboxSettings4',
    'about:blank',
  ], { stdio: 'ignore' });

  let browser;
  try {
    let ver = null;
    for (let i = 0; i < 30; i++) {
      try { ver = await jget('/json/version'); break; } catch (e) { await sleep(1000); }
    }
    if (!ver) throw new Error('Chrome 调试端口未就绪');
    console.log('Chrome:', ver.Browser);
    browser = new CDP(ver.webSocketDebuggerUrl);
    await browser.connect();

    // 1. 加载扩展 + 获取准确 ID
    let extId = null;
    try {
      const r = await browser.send('Extensions.loadUnpacked', { path: ROOT });
      extId = r.id;
      console.log('Extensions.loadUnpacked → 扩展 ID:', extId);
    } catch (e) {
      console.log('Extensions 域不可用（', String(e.message).slice(0, 90), '），回退扫描 SW');
      await browser.send('Target.createTarget', { url: 'about:blank' });
    }
    if (!extId) {
      const cands = new Set();
      for (let i = 0; i < 15; i++) {
        for (const t of await jget('/json/list')) {
          if (t.type === 'service_worker' && /\/background\.js$/.test(t.url || '')) cands.add(t.url.split('/')[2]);
        }
        if (cands.size) break;
        await sleep(1000);
      }
      for (const id of cands) {
        if (await verifyExtId(browser, id)) { extId = id; break; }
      }
      if (!extId) throw new Error('无法定位扩展 ID（候选: ' + [...cands].join(',') + '）');
      console.log('扫描得到扩展 ID:', extId);
    }
    if (!(await verifyExtId(browser, extId))) throw new Error('扩展页 chrome.storage 校验失败: ' + extId);

    // 2. 预置假配置
    const { c: popup, targetId: popupTid } = await openExtPage(browser, `chrome-extension://${extId}/popup.html`);
    const setr = await popup.eval('chrome.storage.local.set(' + JSON.stringify({ sufeAutoSliderCfg: DUMMY }) + ').then(() => "cfg-set")');
    console.log('预置配置:', setr);
    popup.close();
    await browser.send('Target.closeTarget', { targetId: popupTid });
    if (setr !== 'cfg-set') throw new Error('预置配置失败');

    // 3. 打开登录页并观察浮窗
    const { c: page } = await openExtPage(browser, 'https://login.sufe.edu.cn/login/');
    await page.send('Page.enable');
    console.log('== 观察浮窗流程（最多 80 秒）==');
    const readWidget = () => page.eval(`(() => {
      const w = document.getElementById('sufe-auto-slider-widget');
      if (!w) return '(no-widget)';
      const body = w.querySelector('.ssw-body');
      const log = w.querySelector('.ssw-log');
      return 'BODY: ' + (body ? body.textContent.trim() : '') + '\\nLOG: ' + (log ? log.textContent.trim().replace(/\\n+/g, ' | ') : '');
    })()`);

    const timeline = [];
    let terminal = null;
    for (let i = 0; i < 40; i++) {
      await sleep(2000);
      let txt;
      try { txt = await readWidget(); } catch (e) { txt = '(eval失败: ' + String(e.message).slice(0, 80) + ')'; }
      const first = txt.split('\n')[0];
      if (!timeline.length || timeline[timeline.length - 1].first !== first) {
        timeline.push({ t: i * 2, first, full: txt });
        console.log('[' + (i * 2) + 's] ' + txt.replace(/\n/g, '  ||  '));
      }
      if (/学号不存在|密码错误|连续\s*\d+\s*轮未通过/.test(txt)) { terminal = txt; break; }
    }

    // 4. 截图
    try {
      const shot = await page.send('Page.captureScreenshot', { format: 'png' });
      fs.writeFileSync(path.join(OUT, 'ext_test_widget.png'), Buffer.from(shot.data, 'base64'));
      console.log('截图已保存: output/ext_test_widget.png');
    } catch (e) { console.log('截图失败:', e.message); }

    // 5. 读扩展统计
    let stats = null;
    try {
      const { c: p2, targetId: p2t } = await openExtPage(browser, `chrome-extension://${extId}/popup.html`);
      stats = await p2.eval('chrome.storage.local.get("sufeAutoSliderStats").then(o => JSON.stringify(o.sufeAutoSliderStats || null))');
      p2.close();
      await browser.send('Target.closeTarget', { targetId: p2t });
    } catch (e) { stats = '读取失败: ' + String(e.message).slice(0, 80); }

    console.log('== 流程时间线 ==');
    for (const t of timeline) console.log('+' + t.t + 's: ' + t.first);
    const solved = timeline.some((t) => /滑块已解锁/.test(t.full));
    console.log('== 结论 ==');
    console.log('滑块在真实扩展环境解锁:', solved ? '是' : '未观察到');
    console.log('扩展统计:', stats);
    console.log('终态:', terminal ? terminal.split('\n')[0] : '(超时未见终态)');
    page.close();
  } finally {
    if (browser) { try { await browser.send('Browser.close'); } catch (e) { try { child.kill(); } catch (e2) { /* ignore */ } } }
    else { try { child.kill(); } catch (e2) { /* ignore */ } }
    await sleep(1500);
    console.log('测试 Chrome 已关闭（临时配置目录: ' + PROFILE + '）');
  }
}

main().catch((e) => { console.error('测试失败:', e.message); process.exit(1); });

/*
 * v4.1.0 域名扩展冒烟测试（零依赖 CDP，Node >= 22）
 * 用法：node lab/cdp_smoke.js [--skip-real-e2e]
 *
 * 步骤：
 *  1) 起独立 Chrome（临时 profile、随机调试端口），Extensions.loadUnpacked 加载仓库根目录扩展
 *  2) 回归：真 login.sufe.edu.cn/login → 浮窗出现（点击触发卡片，未配置含「去设置」）
 *  3) 新域匹配：Fetch 拦截替身页 @ newapi-sufe-edu-cn-s.webvpn.sufe.edu.cn/login → 浮窗出现
 *  4) 点击触发干跑全链路：注入测试配置后，点击替身页原生「登录」按钮 → 全部 /esc-sso API 由 mock 应答
 *     → 走到 SSO10001；断言页面原生点击处理器被拦截、接口全部同源、空输入框时回填扩展凭据
 *  4b) 凭据策略：预填页面输入框（模拟浏览器填充）→ 登录用页面值、不覆盖
 *  5) 真实回归（默认执行，--skip-real-e2e 跳过）：真登录页点击原生「登录」按钮 + 假学号全链路 → 期待 SSO10001
 *  6) auto 模式回归：切 triggerMode=auto → 替身页出现倒计时卡片
 *  7) vcode 模式（mock）：queryAllValid 返回 displayVcode=1 → 拉图 OCR → 首次 doLogin 返回 SSO10023
 *     检验换图重试 → 再次返回 SSO10001；断言不调用滑块接口、vcode 均为 4 位小写字母数字
 *
 * 退出码：0 全部通过；1 有断言失败；2 致命错误。截图输出到 lab/output/。
 */
'use strict';
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawn } = require('child_process');

const REPO = path.resolve(__dirname, '..');
const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const PORT = 9333 + Math.floor(Math.random() * 60);
const FAKE_HOST = 'newapi-sufe-edu-cn-s.webvpn.sufe.edu.cn';
const SKIP_REAL = process.argv.includes('--skip-real-e2e');
const PROFILE = path.join(os.tmpdir(), 'sufe-smoke-' + Date.now());
const OUT = path.join(REPO, 'lab', 'output');

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const log = (...a) => console.log('[SMOKE]', ...a);
const results = [];
function check(name, ok, extra) {
  results.push({ name, ok: !!ok });
  log((ok ? 'PASS' : 'FAIL') + ' - ' + name + (extra ? ' | ' + extra : ''));
}

class CDP {
  constructor(ws) {
    this.ws = ws; this.id = 0; this.pending = new Map(); this.handlers = new Map();
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id != null && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        msg.error ? reject(new Error('CDP ' + JSON.stringify(msg.error))) : resolve(msg.result);
      } else if (msg.method) {
        for (const h of (this.handlers.get(msg.method) || [])) {
          try { h(msg.params, msg.sessionId); } catch (e) { log('handler 异常:', e.message); }
        }
      }
    };
  }
  on(method, fn) { if (!this.handlers.has(method)) this.handlers.set(method, []); this.handlers.get(method).push(fn); }
  cmd(method, params = {}, sessionId) {
    return new Promise((resolve, reject) => {
      const id = ++this.id;
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify(sessionId ? { id, method, params, sessionId } : { id, method, params }));
    });
  }
  async eval(sessionId, expression, awaitPromise = false) {
    const r = await this.cmd('Runtime.evaluate', { expression, returnByValue: true, awaitPromise }, sessionId);
    if (r.exceptionDetails) {
      const d = r.exceptionDetails.exception && r.exceptionDetails.exception.description;
      throw new Error('eval 异常: ' + (d || JSON.stringify(r.exceptionDetails)));
    }
    return r.result && r.result.value;
  }
  async waitFor(sessionId, expr, timeoutMs, intervalMs = 400) {
    const t0 = Date.now();
    let last;
    while (Date.now() - t0 < timeoutMs) {
      try { last = await this.eval(sessionId, expr); if (last) return last; } catch (e) { /* 导航/上下文重建中 */ }
      await sleep(intervalMs);
    }
    return last;
  }
}

function connect(wsUrl) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl);
    const t = setTimeout(() => reject(new Error('ws 连接超时')), 10000);
    ws.onopen = () => { clearTimeout(t); resolve(new CDP(ws)); };
    ws.onerror = () => { clearTimeout(t); reject(new Error('ws 连接失败: ' + wsUrl)); };
  });
}

async function main() {
  if (!fs.existsSync(CHROME)) throw new Error('未找到 Chrome: ' + CHROME);
  fs.mkdirSync(OUT, { recursive: true });

  log('获取真实 queryAllValid（作为 mock 数据源）…');
  const qaText = await (await fetch('https://login.sufe.edu.cn/esc-sso/api/v3/auth/queryAllValid')).text();
  const qa = JSON.parse(qaText);
  if (!qa.data || !qa.data.param || !qa.data.param.publicKey) throw new Error('queryAllValid 结构异常');

  log('启动 Chrome @', PORT, '| profile:', PROFILE);
  const child = spawn(CHROME, [
    '--user-data-dir=' + PROFILE,
    '--remote-debugging-port=' + PORT,
    '--no-first-run', '--no-default-browser-check', '--disable-sync',
    '--window-size=1280,900',
    '--host-resolver-rules=MAP ' + FAKE_HOST + ' 127.0.0.1',
    'about:blank',
  ], { stdio: 'ignore' });

  let browser;
  try {
    let ver = null;
    for (let i = 0; i < 60 && !ver; i++) {
      try { ver = await (await fetch('http://127.0.0.1:' + PORT + '/json/version')).json(); } catch (e) { await sleep(500); }
    }
    if (!ver) throw new Error('Chrome 调试端口未就绪');
    browser = await connect(ver.webSocketDebuggerUrl);

    // 1) 加载扩展（manifest 无效会让这里直接报错）
    const ext = await browser.cmd('Extensions.loadUnpacked', { path: REPO });
    check('扩展加载（manifest 有效，含新匹配模式）', !!ext.id, 'id=' + ext.id);

    const newPage = async (url) => {
      const t = await browser.cmd('Target.createTarget', { url });
      const a = await browser.cmd('Target.attachToTarget', { targetId: t.targetId, flatten: true });
      return { targetId: t.targetId, sid: a.sessionId };
    };
    const shot = async (targetId, sid, name) => {
      try {
        await browser.cmd('Target.activateTarget', { targetId });
        await sleep(300);
        const r = await browser.cmd('Page.captureScreenshot', { format: 'png' }, sid);
        fs.writeFileSync(path.join(OUT, name), Buffer.from(r.data, 'base64'));
      } catch (e) { log('截图失败:', name, e.message); }
    };
    const WIDGET = "(document.getElementById('sufe-auto-slider-widget')?.textContent || '')";
    const dumpWidget = async (sid) => log('  当前浮窗内容:', JSON.stringify(await browser.eval(sid, WIDGET).catch(() => '(取不到)')));

    // 2) 回归：真实登录页（新 profile 无配置 → 未配置提示卡片）
    const p1 = await newPage('about:blank');
    await browser.cmd('Page.navigate', { url: 'https://login.sufe.edu.cn/login' }, p1.sid);
    const t1 = await browser.waitFor(p1.sid, WIDGET + ".includes('点击页面上的「登录」按钮') ? " + WIDGET + " : ''", 30000);
    if (!t1) await dumpWidget(p1.sid);
    check('回归：login.sufe.edu.cn/login 注入并出现点击触发卡片（未配置含「去设置」）', !!t1 && t1.includes('去设置'), t1 && t1.replace(/\n/g, ' ').slice(0, 90));
    await shot(p1.targetId, p1.sid, 'smoke-1-login-page.png');

    // 3) 新域匹配：Fetch 拦截替身页
    const fakeHtml = '<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>统一认证（替身）</title></head><body>' +
      '<form><div class="f-account"><input class="f-account" name="username" placeholder="学号/工号"></div>' +
      '<div class="f-pwd"><input class="f-pwd" type="password" placeholder="密码"></div>' +
      '<button class="login-btn pc-login" type="button"><span class="btn-text">登录</span></button></form>' +
      '<script>window.__PAGE_CLICKED__ = false; document.querySelector("button.pc-login").addEventListener("click", function () { window.__PAGE_CLICKED__ = true; });</script>' +
      '</body></html>';
    const FAKE_URL = 'https://' + FAKE_HOST + '/login';
    const p2 = await newPage('about:blank');
    const intercepted = [];
    const doLoginBodies = [];
    let vcodeMode = false;          // 步骤 7 打开：queryAllValid 返回 displayVcode=1
    let randcodeCalls = 0;
    let initCallsVcode = 0;
    const doLoginVcodes = [];
    // 图片验证码 mock 图：取一张真实样本
    const vcodeDir = path.join(REPO, 'lab', 'samples', 'vcode');
    let randcodeB64 = '';
    try {
      const jpgs = fs.readdirSync(vcodeDir).filter((f) => f.endsWith('.jpg')).sort();
      randcodeB64 = fs.readFileSync(path.join(vcodeDir, jpgs[0])).toString('base64');
    } catch (e) { /* 无样本时 mock 返回空图，让识别失败暴露 */ }
    const bgB64 = fs.readFileSync(path.join(REPO, 'samples', '0000', 'bg.png')).toString('base64');
    const b64 = (s) => Buffer.from(s, 'utf8').toString('base64');
    const jsonRes = (obj) => ({ code: 200, headers: [{ name: 'Content-Type', value: 'application/json; charset=utf-8' }], body: b64(JSON.stringify(obj)) });
    const htmlRes = (html) => ({ code: 200, headers: [{ name: 'Content-Type', value: 'text/html; charset=utf-8' }], body: b64(html) });
    browser.on('Fetch.requestPaused', (p, sid) => {
      if (sid !== p2.sid) return;
      const url = p.request.url;
      intercepted.push(url);
      let res;
      if (url.includes('/api/v3/auth/queryAllValid')) {
        if (vcodeMode) {
          const qa2 = JSON.parse(JSON.stringify(qa));
          qa2.data.login.webLocalAuth.displayVcode = '1';
          res = jsonRes(qa2);
        } else res = jsonRes(qa);
      } else if (url.includes('/api/v1/image/getRandcode')) {
        randcodeCalls++;
        res = { code: 200, headers: [{ name: 'Content-Type', value: 'image/jpeg' }], body: randcodeB64 };
      } else if (url.includes('/api/v3/sliderCaptcha/init')) {
        if (vcodeMode) initCallsVcode++;
        res = jsonRes({ code: '0', data: { sourceImage: bgB64, newImage: '', Y: 120, token: 'smoke-token' } });
      } else if (url.includes('/api/v3/sliderCaptcha/check')) res = jsonRes({ code: '0', data: 'SMOKE-VCODE' });
      else if (url.includes('/api/v3/auth/doLogin')) {
        let body = {};
        try { body = JSON.parse(p.request.postData || '{}'); } catch (e) { /* ignore */ }
        doLoginBodies.push(body);
        if (vcodeMode) {
          doLoginVcodes.push(body.dataField && body.dataField.vcode);
          // 第一次故意返回 SSO10023（验证码错误）检验换图重试
          res = jsonRes(doLoginVcodes.length === 1
            ? { code: 'SSO10023', msg: '图片验证码错误' }
            : { code: 'SSO10001', msg: '学号不存在' });
        } else {
          res = jsonRes({ code: 'SSO10001', msg: '学号不存在' });
        }
      } else if (url.endsWith('/favicon.ico')) res = { code: 404, headers: [], body: '' };
      else res = htmlRes(fakeHtml);
      browser.cmd('Fetch.fulfillRequest', { requestId: p.requestId, responseCode: res.code, responseHeaders: res.headers, body: res.body }, sid).catch(() => {});
    });
    await browser.cmd('Fetch.enable', { patterns: [{ urlPattern: 'https://' + FAKE_HOST + '/*', requestStage: 'Request' }] }, p2.sid);
    await browser.cmd('Page.navigate', { url: FAKE_URL }, p2.sid);
    const t2 = await browser.waitFor(p2.sid, WIDGET + ".includes('点击页面上的「登录」按钮') ? " + WIDGET + " : ''", 20000);
    if (!t2) await dumpWidget(p2.sid);
    check('新域匹配：' + FAKE_HOST + '/login 注入并出现点击触发卡片', !!t2 && t2.includes('去设置'), t2 && t2.replace(/\n/g, ' ').slice(0, 90));
    await shot(p2.targetId, p2.sid, 'smoke-2-webvpn-fake.png');

    // 4) 注入测试配置（不写 triggerMode → 验证默认 click 模式）→ 替身页点击触发干跑全链路
    const pop = await newPage('chrome-extension://' + ext.id + '/popup.html');
    await browser.waitFor(pop.sid, "document.readyState === 'complete' ? 'ok' : ''", 15000);
    const setRes = await browser.eval(pop.sid,
      "chrome.storage.local.set({sufeAutoSliderCfg:{username:'smoke-test',password:'x',enabled:true}}).then(() => 'ok')", true);
    check('注入测试配置', setRes === 'ok');
    await browser.cmd('Target.closeTarget', { targetId: pop.targetId });

    await browser.cmd('Page.navigate', { url: FAKE_URL }, p2.sid);
    const ccard = await browser.waitFor(p2.sid, WIDGET + ".includes('点击页面') ? 'yes' : ''", 15000);
    if (!ccard) await dumpWidget(p2.sid);
    check('替身页出现「点击页面登录按钮」卡片（默认 click 模式）', !!ccard);
    await browser.eval(p2.sid, "document.querySelector('button.pc-login').click(); 'clicked'");
    const t3 = await browser.waitFor(p2.sid, WIDGET + ".includes('学号不存在') ? " + WIDGET + " : ''", 30000);
    if (!t3) await dumpWidget(p2.sid);
    check('替身页点击触发 → 干跑到 SSO10001', !!t3, t3 && t3.replace(/\n/g, ' ').slice(-90));
    const pageHandlerRan = await browser.eval(p2.sid, 'window.__PAGE_CLICKED__ === true');
    check('页面原生点击处理器被拦截（stopImmediatePropagation）', !pageHandlerRan);
    const u1 = (doLoginBodies[doLoginBodies.length - 1] || {}).dataField || {};
    const filledV = await browser.eval(p2.sid, "(document.querySelector('input.f-account')||{}).value || ''");
    check('空输入框 → 用扩展凭据登录并回填登录框', u1.username === 'smoke-test' && filledV === 'smoke-test',
      'username=' + u1.username + ' 框值=' + filledV);
    await shot(p2.targetId, p2.sid, 'smoke-3-mock-e2e-final.png');

    const apiCalls = intercepted.filter((u) => u.includes('/esc-sso/'));
    const allSameOrigin = apiCalls.length >= 4 && apiCalls.every((u) => u.startsWith('https://' + FAKE_HOST + '/esc-sso/'));
    check('接口请求全部同源指向代理域名（' + apiCalls.length + ' 个）', allSameOrigin,
      apiCalls.map((u) => u.replace('https://' + FAKE_HOST, '')).join(' '));

    // 4b) 凭据策略：预填页面输入框（模拟浏览器已填充）→ 登录用页面值、不覆盖
    await browser.cmd('Page.navigate', { url: FAKE_URL }, p2.sid);
    await browser.waitFor(p2.sid, WIDGET + ".includes('点击页面上的「登录」按钮') ? 'yes' : ''", 15000);
    await browser.eval(p2.sid, "(() => { document.querySelector('input.f-account').value = 'page-user'; document.querySelector('input.f-pwd').value = 'page-pass'; return 'set'; })()");
    await browser.eval(p2.sid, "document.querySelector('button.pc-login').click(); 'clicked'");
    const t3b = await browser.waitFor(p2.sid, WIDGET + ".includes('学号不存在') ? 'yes' : ''", 30000);
    const vals = await browser.eval(p2.sid,
      "((document.querySelector('input.f-account')||{}).value||'?') + '/' + ((document.querySelector('input.f-pwd')||{}).value||'?')");
    const u2 = (doLoginBodies[doLoginBodies.length - 1] || {}).dataField || {};
    check('页面已有值 → 直接用页面凭据且不覆盖', !!t3b && u2.username === 'page-user' && vals === 'page-user/page-pass',
      'username=' + u2.username + ' 框值=' + vals);

    // 5) 真实登录页全链路（假学号）：点击真实原生「登录」按钮触发 — 默认执行
    if (!SKIP_REAL) {
      await browser.cmd('Page.navigate', { url: 'https://login.sufe.edu.cn/login' }, p1.sid);
      const cc1 = await browser.waitFor(p1.sid, WIDGET + ".includes('点击页面') ? 'yes' : ''", 40000);
      if (!cc1) await dumpWidget(p1.sid);
      check('真实登录页出现「点击页面登录按钮」卡片', !!cc1);
      const hasBtn = await browser.eval(p1.sid, "!!document.querySelector('button.pc-login')");
      check('真实登录页存在原生登录按钮 button.pc-login', hasBtn);
      await browser.eval(p1.sid, "document.querySelector('button.pc-login').click(); 'clicked'");
      const t4 = await browser.waitFor(p1.sid, WIDGET + ".includes('学号不存在') ? " + WIDGET + " : ''", 60000);
      if (!t4) await dumpWidget(p1.sid);
      check('真实登录页点击触发全链路：滑块通过后 doLogin 返回 SSO10001（假学号）', !!t4, t4 && t4.replace(/\n/g, ' ').slice(-90));
      await shot(p1.targetId, p1.sid, 'smoke-4-real-e2e-final.png');
    }

    // 6) auto 模式回归：切 triggerMode=auto → 替身页倒计时卡片
    const pop2 = await newPage('chrome-extension://' + ext.id + '/popup.html');
    await browser.waitFor(pop2.sid, "document.readyState === 'complete' ? 'ok' : ''", 15000);
    await browser.eval(pop2.sid,
      "chrome.storage.local.get('sufeAutoSliderCfg').then(o => { const c = o.sufeAutoSliderCfg; c.triggerMode = 'auto'; return chrome.storage.local.set({sufeAutoSliderCfg: c}); }).then(() => 'ok')", true);
    await browser.cmd('Target.closeTarget', { targetId: pop2.targetId });
    await browser.cmd('Page.navigate', { url: FAKE_URL }, p2.sid);
    const cdAuto = await browser.waitFor(p2.sid, WIDGET + ".includes('秒后自动登录') ? 'yes' : ''", 10000);
    check('auto 模式回归：替身页出现倒计时卡片', !!cdAuto);

    // 7) 图片验证码模式（mock）：displayVcode=1 → 拉图 OCR → 首次 SSO10023 换图重试 → SSO10001
    //    先把触发方式切回 click（步骤 6 改成了 auto）
    const pop3 = await newPage('chrome-extension://' + ext.id + '/popup.html');
    await browser.waitFor(pop3.sid, "document.readyState === 'complete' ? 'ok' : ''", 15000);
    await browser.eval(pop3.sid,
      "chrome.storage.local.get('sufeAutoSliderCfg').then(o => { const c = o.sufeAutoSliderCfg; c.triggerMode = 'click'; return chrome.storage.local.set({sufeAutoSliderCfg: c}); }).then(() => 'ok')", true);
    await browser.cmd('Target.closeTarget', { targetId: pop3.targetId });
    vcodeMode = true;
    await browser.cmd('Page.navigate', { url: FAKE_URL }, p2.sid);
    const cdV = await browser.waitFor(p2.sid, WIDGET + ".includes('点击页面上的「登录」按钮') ? 'yes' : ''", 15000);
    if (!cdV) await dumpWidget(p2.sid);
    await browser.eval(p2.sid, "document.querySelector('button.pc-login').click(); 'clicked'");
    const t5 = await browser.waitFor(p2.sid, WIDGET + ".includes('学号不存在') ? 'yes' : ''", 60000);
    if (!t5) await dumpWidget(p2.sid);
    const vcodesOk = doLoginVcodes.length >= 2 && doLoginVcodes.every((v) => /^[0-9a-z]{4}$/.test(String(v)));
    check('vcode 模式：OCR 提交 + SSO10023 换图重试 → SSO10001（且不调滑块接口）',
      !!cdV && !!t5 && randcodeCalls >= 2 && vcodesOk && initCallsVcode === 0,
      'randcode=' + randcodeCalls + ' doLogin=' + doLoginVcodes.length +
      ' vcodes=' + JSON.stringify(doLoginVcodes) + ' init(vcode)=' + initCallsVcode);
    await shot(p2.targetId, p2.sid, 'smoke-5-vcode-mode.png');
  } finally {
    try { if (browser) await browser.cmd('Browser.close'); } catch (e) { /* ignore */ }
    await sleep(1500);
    try { fs.rmSync(PROFILE, { recursive: true, force: true, maxRetries: 8, retryDelay: 400 }); }
    catch (e) { log('profile 清理失败（可手动删）：', PROFILE); }
    try { child.kill(); } catch (e) { /* ignore */ }
  }

  const failed = results.filter((r) => !r.ok);
  console.log('\n[SMOKE] ===== ' + (results.length - failed.length) + '/' + results.length + ' 通过 =====');
  for (const r of results) console.log(' ', r.ok ? 'PASS' : 'FAIL', '-', r.name);
  process.exit(failed.length ? 1 : 0);
}

main().catch((e) => { console.error('[SMOKE] 致命错误:', e); process.exit(2); });

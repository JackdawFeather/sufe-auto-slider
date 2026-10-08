/*
 * 上财自动滑块 · 内容脚本
 * 检测到上财统一认证登录页后：倒计时 → 自动「滑块盲试 + 登录」→ 跳转。
 * 策略沿用生产环境长期验证的「候选 X 盲试 + 多轮重试 + 失败退避」：
 *   候选 X = 100/98/102，每个 token 最多试 3 个，最多 12 轮，退避 1.5s/4s/8s。
 * 仅向 login.sufe.edu.cn 提交数据，密码 RSA(PKCS#1 v1.5) 加密后传输。
 */
(async function () {
  'use strict';
  if (window.__SUFE_AUTO_SLIDER__) return;
  window.__SUFE_AUTO_SLIDER__ = true;

  // ── 常量（与生产脚本一致，一般无需修改） ────────────────────────────
  const BASE = 'https://login.sufe.edu.cn/esc-sso';
  const X_CANDIDATES = [100, 98, 102];
  const MAX_X_PER_ROUND = 3;
  const MAX_ROUNDS = 12;
  const AUTO_DELAY_MS = 4000;      // 检测到登录页后的倒计时，给用户留取消窗口
  const LOOP_GUARD_MS = 20000;     // 自动登录成功后 20s 内不再自动启动（防死循环）
  const CFG_KEY = 'sufeAutoSliderCfg';
  const STATE_KEY = 'sufeAutoSliderState';

  // ── 小工具 ──────────────────────────────────────────────────────────
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  function backoff(rnd) { return rnd >= 8 ? 8000 : rnd >= 5 ? 4000 : 1500; }

  async function getCfg() {
    const o = await chrome.storage.local.get(CFG_KEY);
    return o[CFG_KEY] || null;
  }
  async function getState() {
    const o = await chrome.storage.local.get(STATE_KEY);
    return o[STATE_KEY] || {};
  }

  async function api(path, opts = {}) {
    const res = await fetch(BASE + path, {
      method: opts.method || 'GET',
      headers: Object.assign(
        { Accept: 'application/json' },
        opts.body ? { 'Content-Type': 'application/json' } : {}
      ),
      body: opts.body ? JSON.stringify(opts.body) : undefined,
      credentials: 'same-origin',
    });
    const text = await res.text();
    try { return JSON.parse(text); } catch (e) {
      throw new Error('接口返回非 JSON（HTTP ' + res.status + '）：' + text.slice(0, 100));
    }
  }

  // ── 滑块与登录核心（移植自生产脚本） ────────────────────────────────
  async function sliderRound() {
    const init = await api('/api/v3/sliderCaptcha/init', { method: 'POST', body: {} });
    const d = init && init.data;
    if (!d || !d.token) throw new Error('滑块 init 失败：' + ((init && init.msg) || '未知错误'));
    const y = d.Y, token = d.token;
    let lastMsg = (init && init.msg) || '';
    for (const x of X_CANDIDATES.slice(0, MAX_X_PER_ROUND)) {
      const r = await api(
        '/api/v3/sliderCaptcha/check?X=' + encodeURIComponent(x) +
        '&Y=' + encodeURIComponent(y) + '&token=' + encodeURIComponent(token)
      );
      if (String(r.code) === '0') return { vcode: r.data, x: x, y: y };
      lastMsg = r.msg || lastMsg;
    }
    return { vcode: null, lastMsg: lastMsg };
  }

  async function testSlider(rounds) {
    const log = [];
    for (let rnd = 1; rnd <= rounds; rnd++) {
      try {
        const v = await sliderRound();
        if (v.vcode) return { ok: true, detail: '第 ' + rnd + ' 轮通过（X=' + v.x + ' Y=' + v.y + '）' };
        log.push('第 ' + rnd + ' 轮未通过（' + (v.lastMsg || '无信息') + '）');
      } catch (e) { log.push('第 ' + rnd + ' 轮异常：' + e.message); }
      if (rnd < rounds) await sleep(backoff(rnd));
    }
    return { ok: false, detail: log.join('\n') };
  }

  function extractRedirect(res) {
    const d = res && res.data;
    if (typeof d === 'string') {
      const m = d.match(/https?:\/\/[^\s"'<>]+/);
      if (m) return m[0];
    }
    if (d && typeof d === 'object') {
      for (const k of ['redirectUri', 'redirect', 'redirectUrl', 'url', 'target', 'forward']) {
        const v = d[k];
        if (typeof v === 'string' && /^https?:\/\//.test(v)) return v;
      }
    }
    return null;
  }

  async function autoLogin(cfg, ui, runId) {
    const checkStop = () => { if (runId !== currentRunId) throw new Error('已停止'); };

    ui.status('获取登录配置…');
    const cfgRes = await api('/api/v3/auth/queryAllValid');
    const param = cfgRes && cfgRes.data && cfgRes.data.param;
    const ticket = cfgRes && cfgRes.data && cfgRes.data.config && cfgRes.data.config.loginTicket;
    if (!param || !param.publicKey || !ticket) throw new Error('获取登录配置失败（接口结构可能已变化）');
    checkStop();

    const encPw = SufeCrypto.encryptPassword(cfg.password, param.publicKey);

    for (let rnd = 1; rnd <= MAX_ROUNDS; rnd++) {
      checkStop();
      ui.status('滑块解锁中（第 ' + rnd + '/' + MAX_ROUNDS + ' 轮）…');
      let v;
      try { v = await sliderRound(); }
      catch (e) { ui.note('init 异常：' + e.message); await sleep(backoff(rnd)); continue; }
      if (!v.vcode) {
        ui.note('第 ' + rnd + ' 轮未通过（' + (v.lastMsg || '无信息') + '）');
        if (rnd < MAX_ROUNDS) await sleep(backoff(rnd));
        continue;
      }
      checkStop();
      ui.status('滑块已解锁（X=' + v.x + '），正在登录…');
      const res = await api('/api/v3/auth/doLogin', {
        method: 'POST',
        body: {
          authType: 'webLocalAuth',
          dataField: {
            username: cfg.username,
            password: encPw,
            vcode: v.vcode,
            publicKeyId: param.publicKeyId,
          },
          loginTicket: ticket,
          redirectUri: '',
        },
      });
      checkStop();
      if (String(res.code) === '0') {
        await chrome.storage.local.set({ [STATE_KEY]: { lastAutoLoginAt: Date.now() } });
        return res;
      }
      if (res.code === 'SSO10002') {
        const err = new Error('学号或密码错误（SSO10002），请在扩展设置中更正');
        err.stopRetry = true;
        throw err;
      }
      ui.note('doLogin 未通过：code=' + (res.code || '?') + ' msg=' + (res.msg || ''));
      await sleep(1500);
    }
    throw new Error('连续 ' + MAX_ROUNDS + ' 轮未通过——疑似风控收紧期，请稍后再试或手动登录');
  }

  // ── 状态浮窗 ────────────────────────────────────────────────────────
  const WIDGET_ID = 'sufe-auto-slider-widget';
  let root = null, bodyEl = null, logEl = null, actionsEl = null;
  const logLines = [];

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }

  function ensureWidget() {
    if (root && root.isConnected) return;
    root = el('div'); root.id = WIDGET_ID;
    const title = el('div', 'ssw-title', '上财自动滑块');
    const closeTop = el('span', 'ssw-close', '×');
    closeTop.title = '隐藏浮窗（刷新页面后重新出现）';
    closeTop.addEventListener('click', hideWidget);
    title.appendChild(closeTop);
    bodyEl = el('div', 'ssw-body');
    logEl = el('div', 'ssw-log');
    actionsEl = el('div', 'ssw-actions');
    root.append(title, bodyEl, logEl, actionsEl);
    document.documentElement.appendChild(root);
  }

  function hideWidget() { if (root) root.remove(); root = null; }

  function mkBtn(text, onClick, primary) {
    const b = el('button', 'ssw-btn' + (primary ? ' ssw-btn-primary' : ''), text);
    b.addEventListener('click', onClick);
    return b;
  }

  function renderStatus(text) {
    ensureWidget();
    bodyEl.textContent = '';
    bodyEl.appendChild(el('div', 'ssw-line', text));
    actionsEl.textContent = '';
    actionsEl.appendChild(mkBtn('停止', stopRun));
  }

  function renderNote(text) {
    ensureWidget();
    logLines.push(text);
    if (logLines.length > 30) logLines.shift();
    logEl.textContent = '';
    for (const line of logLines.slice(-8)) logEl.appendChild(el('div', 'ssw-logline', line));
    logEl.scrollTop = logEl.scrollHeight;
  }

  function renderCard(mainText, buttons) {
    ensureWidget();
    bodyEl.textContent = '';
    bodyEl.appendChild(el('div', 'ssw-line', mainText));
    logEl.textContent = '';
    actionsEl.textContent = '';
    for (const b of buttons) actionsEl.appendChild(b);
  }

  function renderError(err) {
    const msg = err && err.message ? err.message : String(err);
    const buttons = err && err.stopRetry
      ? [mkBtn('去设置', openSettings, true), mkBtn('关闭', hideWidget)]
      : [mkBtn('重试', () => startFlow(), true), mkBtn('去设置', openSettings), mkBtn('关闭', hideWidget)];
    renderCard('✗ ' + msg, buttons);
  }

  function openSettings() {
    try { chrome.runtime.sendMessage({ type: 'openSettings' }); }
    catch (e) { /* 扩展上下文失效时忽略 */ }
  }

  // ── 流程控制 ────────────────────────────────────────────────────────
  let currentRunId = 0;
  let countdownTimer = null;
  let state = 'idle'; // idle | pending | running

  function stopRun() {
    currentRunId++;
    if (countdownTimer) { clearInterval(countdownTimer); countdownTimer = null; }
    state = 'idle';
  }

  async function startFlow() {
    stopRun();
    const myRunId = ++currentRunId;
    state = 'running';
    const ui = { status: renderStatus, note: renderNote };
    try {
      const cfg = await getCfg();
      if (!cfg || !cfg.username || !cfg.password) {
        const err = new Error('尚未配置学号/密码，请先在扩展设置中填写');
        err.stopRetry = true;
        throw err;
      }
      const res = await autoLogin(cfg, ui, myRunId);
      if (myRunId !== currentRunId) return;
      state = 'idle';
      const target = extractRedirect(res);
      renderCard('✓ 登录成功，正在跳转…', [mkBtn('立即跳转', () => {
        location.href = target || location.href;
      })]);
      setTimeout(() => {
        if (myRunId !== currentRunId) return;
        if (target) location.href = target; else location.reload();
      }, 1200);
    } catch (err) {
      if (myRunId !== currentRunId) return;
      state = 'idle';
      if (err && err.message === '已停止') return;
      renderError(err);
    }
  }

  async function startCountdown(cfg) {
    state = 'pending';
    let remain = Math.round(AUTO_DELAY_MS / 1000);
    const card = () => renderCard(
      '检测到上财登录页，' + remain + ' 秒后自动登录（' + cfg.username + '）',
      [
        mkBtn('立即登录', () => startFlow(), true),
        mkBtn('取消', () => { stopRun(); renderManual(); }),
      ]
    );
    card();
    countdownTimer = setInterval(() => {
      remain--;
      if (remain <= 0) { clearInterval(countdownTimer); countdownTimer = null; startFlow(); return; }
      card();
    }, 1000);
  }

  function renderManual() {
    renderCard('已取消自动登录。可点击按钮手动登录。', [
      mkBtn('立即登录', () => startFlow(), true),
      mkBtn('关闭', hideWidget),
    ]);
  }

  // 用户开始在页面表单里输入 → 取消倒计时，转为手动卡片
  function onUserInteract(e) {
    if (state !== 'pending') return;
    if (!(e.target instanceof Element)) return;
    if (e.target.closest('#' + WIDGET_ID)) return;
    if (e.target.closest('input, button, form')) {
      stopRun();
      renderManual();
    }
  }

  // ── 检测登录页（SPA 渲染需要等待） ─────────────────────────────────
  async function waitForForm(timeoutMs) {
    const t0 = Date.now();
    while (Date.now() - t0 < timeoutMs) {
      const acc = document.querySelector('input.f-account, .f-account input, input[name="username"]');
      const pwd = document.querySelector('input.f-pwd, .f-pwd input, input[type="password"]');
      if (acc && pwd) return { acc: acc, pwd: pwd };
      await sleep(300);
    }
    return null;
  }

  // ── 消息：来自 popup 的「测试滑块」 ─────────────────────────────────
  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg && msg.type === 'SUFE_TEST_SLIDER') {
      testSlider(3).then(sendResponse);
      return true;
    }
  });

  // ── 主入口 ──────────────────────────────────────────────────────────
  const cfg = await getCfg();
  const form = await waitForForm(20000);
  if (!form) return; // 不是登录页（或页面已跳转/渲染失败），静默退出

  document.addEventListener('focusin', onUserInteract, true);
  document.addEventListener('input', onUserInteract, true);

  const st = await getState();
  if (st.lastAutoLoginAt && Date.now() - st.lastAutoLoginAt < LOOP_GUARD_MS) {
    renderCard('刚刚已完成自动登录，但页面仍停留在登录页。可能跳转异常，请手动登录或稍后重试。', [
      mkBtn('重新尝试', () => startFlow(), true),
      mkBtn('关闭', hideWidget),
    ]);
    return;
  }

  if (!cfg || !cfg.username || !cfg.password) {
    renderCard('上财自动滑块：尚未配置学号/密码，无法自动登录。', [
      mkBtn('去设置', openSettings, true),
      mkBtn('关闭', hideWidget),
    ]);
    return;
  }

  if (cfg.enabled === false) {
    renderCard('上财自动滑块：自动登录已停用。', [
      mkBtn('本次登录', () => startFlow(), true),
      mkBtn('启用', async () => {
        const c = await getCfg();
        c.enabled = true;
        await chrome.storage.local.set({ [CFG_KEY]: c });
        startCountdown(c);
      }),
      mkBtn('关闭', hideWidget),
    ]);
    return;
  }

  startCountdown(cfg);
})();

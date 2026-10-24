/*
 * 上财自动滑块 · 内容脚本
 * 检测到上财统一认证登录页后，自动「缺口定位 + 登录」→ 跳转。
 * 触发方式两种（弹窗可切换）：
 *   click（默认）：点击页面上的「登录」按钮（或表单回车）时接管——拦截页面原生流程，
 *                 由扩展自动过滑块并登录；
 *   auto：检测到登录页后倒计时 2 秒自动登录（点击/打字可取消）。
 * 凭据以页面输入框为准：已有值（浏览器密码管理器填充/手动输入）直接采用、不覆盖；
 * 为空则把扩展保存的学号密码自动填入登录框后再登录。
 * 验证码模式两种（弹窗「验证码方式」切换）：
 *   vcode（默认）：图片验证码（getRandcode → 纯 JS OCR → 随 doLogin 提交，错码自动换图重试）
 *   slider：拖动拼图（init → 分割/CNN 定位 → check）
 *
 * 滑块策略（整图 CNN + 邻域梯队）：
 *   背景图整图 → sufe-cnn.js（纯 JS 前向）预测「可通过窗口中心」X；先试 0/±1/±2，
 *   再用经典分割兜底（缺口=平坦纯白连通块）与分割邻域，最后才是历史盲试候选。
 *   服务端容差实测：可通过窗口恒为 5 个连续整数（中心 = 缺口左缘-3），故 0/±1/±2 已覆盖全部余量。
 *   失败不会作废 token，同一 token 内可连续校验。
 * 接口与数据提交均限于当前登录页自身域名（同源）：login.sufe.edu.cn、*.webvpn.sufe.edu.cn
 * 等代理域名下，请求经代理转发，密码 RSA(PKCS#1 v1.5) 加密后传输。
 */
(async function () {
  'use strict';
  if (window.__SUFE_AUTO_SLIDER__) return;
  window.__SUFE_AUTO_SLIDER__ = true;

  // ── 常量（一般无需修改） ────────────────────────────────────────────
  // 同源接口基址：在 login.sufe.edu.cn 上即 https://login.sufe.edu.cn/esc-sso；
  // WebVPN 代理域名（*.webvpn.sufe.edu.cn）下同样以页面自身域名走代理
  const BASE = location.origin + '/esc-sso';
  const MAX_ROUNDS = 6;
  const VCODE_MAX_ROUNDS = 6;      // 图片验证码模式：提交轮数（实测单次约 75%）
  const AUTO_DELAY_MS = 2000;      // auto 模式：检测到登录页后的倒计时（保留短暂取消窗口）
  const LOOP_GUARD_MS = 20000;     // 自动登录成功后 20s 内不再自动启动（防死循环）
  const CFG_KEY = 'sufeAutoSliderCfg';
  const STATE_KEY = 'sufeAutoSliderState';
  const STATS_KEY = 'sufeAutoSliderStats';
  const BLIND_X = [100, 98, 102];  // 仅在识别完全失败时兜底
  const ACC_SEL = 'input.f-account, .f-account input, input[name="username"]';
  const PWD_SEL = 'input.f-pwd, .f-pwd input, input[type="password"]';

  // ── 小工具 ──────────────────────────────────────────────────────────
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  function backoff(rnd) { return rnd >= 4 ? 3000 : 1200; }

  async function getCfg() {
    const o = await chrome.storage.local.get(CFG_KEY);
    return o[CFG_KEY] || null;
  }
  async function getState() {
    const o = await chrome.storage.local.get(STATE_KEY);
    return o[STATE_KEY] || {};
  }
  async function recordStat(firstTry, X, conf) {
    try {
      const o = await chrome.storage.local.get(STATS_KEY);
      const s = o[STATS_KEY] || { n: 0, firstTry: 0 };
      s.n++; if (firstTry) s.firstTry++;
      s.lastX = X;
      s.lastConf = (conf == null) ? null : Math.round(conf * 1000) / 1000;
      await chrome.storage.local.set({ [STATS_KEY]: s });
    } catch (e) { /* 统计失败不影响流程 */ }
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

  // ── 缺口定位：经典分割严格优先（~3ms，缺口模板尺寸/面积恒定），CNN 仅作兜底 ──
  async function decodeSource(b64) {
    const bin = atob(b64);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    const bmp = await createImageBitmap(new Blob([arr], { type: 'image/png' }));
    const cv = document.createElement('canvas');
    cv.width = bmp.width; cv.height = bmp.height;
    const ctx = cv.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(bmp, 0, 0);
    const id = ctx.getImageData(0, 0, cv.width, cv.height);
    if (bmp.close) bmp.close();
    return { px: id.data, w: cv.width, h: cv.height };
  }

  // 候选偏移梯队：x, x+1, x-1, x+2, x-2
  // （可通过窗口恒为 5 个连续整数、半径 ±2，±2 已覆盖全部余量）
  function offsets(x, m) {
    if (x == null || !isFinite(x)) return [];
    const x0 = Math.round(x);
    if (x0 < 0 || x0 > 600) return [];
    const out = [];
    const push = (v, mm) => {
      if (v >= 0 && v <= 600 && !out.some((o) => o.x === v)) out.push({ x: v, m: mm });
    };
    push(x0, m);
    push(x0 + 1, m + '+1'); push(x0 - 1, m + '-1');
    push(x0 + 2, m + '+2'); push(x0 - 2, m + '-2');
    return out;
  }

  async function decodeAndSegment(d) {
    const img = await decodeSource(d.sourceImage);
    let strict = null;
    try { strict = SufeCNN.segmentStrict(img.px, img.w, img.h); } catch (e) { strict = null; }
    return { img: img, strict: strict, cnn: null, looseX: null };
  }

  // 懒执行 CNN（较慢 ~1-2s，只在需要时跑一次）
  function runCnn(det) {
    if (!det || !det.img || det.cnn) return det ? det.cnn : null;
    try { det.cnn = SufeCNN.locate(det.img.px, det.img.w, det.img.h); } catch (e) { det.cnn = null; }
    return det.cnn;
  }

  function runLooseSeg(det) {
    if (!det || !det.img || det.looseX != null) return det ? det.looseX : null;
    try { det.looseX = SufeCNN.segmentX(det.img.px, det.img.w, det.img.h); } catch (e) { det.looseX = null; }
    return det.looseX;
  }

  // 候选流（懒执行）：
  //   ① 严格分割可信 → seg 梯队（通常首探测即命中，本地仅 ~2ms）
  //   ② 分割不可信 → 宽松分割梯队（便宜，2ms）→ 仍需时再跑 CNN（~1-2s）
  //   ③ ① 全灭 → 同样补「宽松分割 → CNN」（说明渲染与假设不符）
  //   ④ 盲试兜底
  function* candidateStream(det) {
    const seen = new Set();
    const pick = function* (list) {
      for (const c of list) {
        if (seen.has(c.x)) continue;
        seen.add(c.x);
        yield c;
      }
    };
    const cheapThenCnn = function* () {
      const loose = runLooseSeg(det);
      if (loose != null) yield* pick(offsets(loose, 'seg'));
      const cnn = runCnn(det);
      if (cnn) yield* pick(offsets(cnn.x, 'cnn'));
    };
    if (det && det.strict) {
      yield* pick(offsets(det.strict.x, 'seg'));
      yield* cheapThenCnn();
    } else {
      yield* cheapThenCnn();
    }
    for (const b of BLIND_X) yield* pick([{ x: b, m: 'blind' }]);
  }

  // ── 滑块与登录核心 ──────────────────────────────────────────────────
  async function sliderRound(ui) {
    const init = await api('/api/v3/sliderCaptcha/init', { method: 'POST', body: {} });
    const d = init && init.data;
    if (!d || !d.token) throw new Error('滑块 init 失败：' + ((init && init.msg) || '未知错误'));
    const y = d.Y, token = d.token;
    let det = null;
    try { det = await decodeAndSegment(d); }
    catch (e) { if (ui) ui.note('背景图解码失败，退回盲试：' + e.message); }

    const segShown = () => (det ? (det.strict ? det.strict.x : det.looseX) : null);
    const cnnShown = () => (det && det.cnn ? Math.round(det.cnn.x) : null);
    const confShown = () => (det && det.cnn ? det.cnn.conf : 0);

    let lastMsg = (init && init.msg) || '';
    let first = true;
    for (const c of candidateStream(det)) {
      const r = await api(
        '/api/v3/sliderCaptcha/check?X=' + encodeURIComponent(c.x) +
        '&Y=' + encodeURIComponent(y) + '&token=' + encodeURIComponent(token)
      );
      const wasFirst = first;
      first = false;
      if (String(r.code) === '0') {
        await recordStat(wasFirst, c.x, confShown());
        return {
          vcode: r.data, x: c.x, y: y, method: c.m, firstTry: wasFirst,
          cnnX: cnnShown(), conf: confShown(), segX: segShown(),
        };
      }
      lastMsg = r.msg || lastMsg;
    }
    return {
      vcode: null, lastMsg: lastMsg, method: 'none',
      cnnX: cnnShown(), conf: confShown(), segX: segShown(),
    };
  }

  async function testSlider(rounds) {
    const log = [];
    for (let rnd = 1; rnd <= rounds; rnd++) {
      try {
        const v = await sliderRound();
        if (v.vcode) {
          return {
            ok: true,
            detail: '第 ' + rnd + ' 轮通过（X=' + v.x + ' Y=' + v.y + ' 方式=' + v.method +
              (v.cnnX != null ? ' CNN=' + v.cnnX + '/conf' + v.conf.toFixed(2) : '') +
              (v.segX != null ? ' 分割=' + v.segX : '') + '）',
          };
        }
        log.push('第 ' + rnd + ' 轮未通过（' + (v.lastMsg || '无信息') +
          (v.cnnX != null ? ' CNN=' + v.cnnX : '') +
          (v.segX != null ? ' 分割=' + v.segX : '') + '）');
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

  async function submitLogin(cfg, param, ticket, vcode) {
    const encPw = SufeCrypto.encryptPassword(cfg.password, param.publicKey);
    return api('/api/v3/auth/doLogin', {
      method: 'POST',
      body: {
        authType: 'webLocalAuth',
        dataField: {
          username: cfg.username,
          password: encPw,
          vcode: vcode,
          publicKeyId: param.publicKeyId,
        },
        loginTicket: ticket,
        redirectUri: '',
      },
    });
  }

  async function fetchRandcode() {
    const res = await fetch(BASE + '/api/v1/image/getRandcode?' + Date.now(), {
      credentials: 'same-origin',
      cache: 'no-store',
    });
    if (!res.ok) throw new Error('下载验证码图片失败（HTTP ' + res.status + '）');
    const bmp = await createImageBitmap(await res.blob());
    const cv = document.createElement('canvas');
    cv.width = bmp.width; cv.height = bmp.height;
    const ctx = cv.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(bmp, 0, 0);
    const id = ctx.getImageData(0, 0, cv.width, cv.height);
    if (bmp.close) bmp.close();
    return id;
  }

  // 图片验证码模式：拉图 → 纯 JS OCR → 登录；SSO10023（验证码错误）换图重试
  // 置信度只用于日志展示（实测「低置信先换图」在 E[T] 上不划算，算完直接提交）
  async function vcodeLogin(cfg, param, ticket, ui, runId) {
    const checkStop = () => { if (runId !== currentRunId) throw new Error('已停止'); };
    for (let rnd = 1; rnd <= VCODE_MAX_ROUNDS; rnd++) {
      checkStop();
      ui.status('图片验证码识别中（第 ' + rnd + '/' + VCODE_MAX_ROUNDS + ' 轮）…');
      let rec = null;
      try {
        const img = await fetchRandcode();
        rec = SufeVcode.recognize(img.data, img.width, img.height);
      } catch (e) {
        ui.note('验证码识别失败：' + e.message);
        await sleep(600);
        continue;
      }
      if (!rec || !rec.text || rec.text.length < 3) {
        ui.note('识别结果异常（' + ((rec && rec.text) || '空') + '），换图重试');
        await sleep(500);
        continue;
      }
      ui.note('验证码识别为 ' + rec.text + '（' + Math.round(rec.conf * 100) + '%）');
      checkStop();
      ui.status('正在登录…');
      const res = await submitLogin(cfg, param, ticket, rec.text);
      checkStop();
      if (String(res.code) === '0') {
        await chrome.storage.local.set({ [STATE_KEY]: { lastAutoLoginAt: Date.now() } });
        return res;
      }
      if (String(res.code) === 'SSO10023') {
        ui.note('验证码未通过，换图重试');
        await sleep(500);
        continue;
      }
      if (res.code === 'SSO10002' || res.code === 'SSO10001') {
        const err = new Error(res.code === 'SSO10001'
          ? '学号不存在（SSO10001），请在扩展设置中更正'
          : '学号或密码错误（SSO10002），请在扩展设置中更正');
        err.stopRetry = true;
        throw err;
      }
      ui.note('doLogin 未通过：code=' + (res.code || '?') + ' msg=' + (res.msg || ''));
      await sleep(1200);
    }
    throw new Error('图片验证码连续 ' + VCODE_MAX_ROUNDS + ' 轮未通过，可在扩展设置中改用「滑块」或稍后重试');
  }

  async function autoLogin(cfg, ui, runId) {
    const checkStop = () => { if (runId !== currentRunId) throw new Error('已停止'); };

    ui.status('获取登录配置…');
    const cfgRes = await api('/api/v3/auth/queryAllValid');
    const param = cfgRes && cfgRes.data && cfgRes.data.param;
    const ticket = cfgRes && cfgRes.data && cfgRes.data.config && cfgRes.data.config.loginTicket;
    if (!param || !param.publicKey || !ticket) throw new Error('获取登录配置失败（接口结构可能已变化）');
    checkStop();

    const useSlider = cfg && cfg.captchaMode === 'slider';
    if (!useSlider) {
      return await vcodeLogin(cfg, param, ticket, ui, runId);
    }

    for (let rnd = 1; rnd <= MAX_ROUNDS; rnd++) {
      checkStop();
      ui.status('滑块解锁中（第 ' + rnd + '/' + MAX_ROUNDS + ' 轮）…');
      let v;
      try { v = await sliderRound(ui); }
      catch (e) { ui.note('init 异常：' + e.message); await sleep(backoff(rnd)); continue; }
      if (!v.vcode) {
        ui.note('第 ' + rnd + ' 轮未通过（' + (v.lastMsg || '无信息') +
          (v.cnnX != null ? ' CNN=' + v.cnnX : '') +
          (v.segX != null ? ' 分割=' + v.segX : '') + '）');
        if (rnd < MAX_ROUNDS) await sleep(backoff(rnd));
        continue;
      }
      checkStop();
      ui.status('滑块已解锁（X=' + v.x + ' 方式=' + v.method + '），正在登录…');
      const res = await submitLogin(cfg, param, ticket, v.vcode);
      checkStop();
      if (String(res.code) === '0') {
        await chrome.storage.local.set({ [STATE_KEY]: { lastAutoLoginAt: Date.now() } });
        return res;
      }
      if (res.code === 'SSO10002' || res.code === 'SSO10001') {
        const err = new Error(res.code === 'SSO10001'
          ? '学号不存在（SSO10001），请在扩展设置中更正'
          : '学号或密码错误（SSO10002），请在扩展设置中更正');
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
    // click 模式下给一条退路：不再拦截页面登录，改用页面自身流程（滑块手动拖）
    if (clickTriggerEnabled && !clickTriggerDisabled) {
      buttons.splice(buttons.length - 1, 0, mkBtn('改用页面登录', disableClickTrigger));
    }
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

  // ── 凭据解析：以页面输入框为准 ──────────────────────────────────────
  function setFieldValue(input, value) {
    try {
      const desc = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(input), 'value');
      if (desc && desc.set) desc.set.call(input, value); else input.value = value;
    } catch (e) { input.value = value; }
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
  }

  function resolveCredentials(cfg, ui) {
    const acc = document.querySelector(ACC_SEL);
    const pwd = document.querySelector(PWD_SEL);
    const accV = acc && acc.value ? acc.value.trim() : '';
    const pwdV = pwd ? pwd.value : '';
    if (accV && pwdV) {
      if (ui) ui.note('用页面输入框凭据（' + accV + '）');
      return { username: accV, password: pwdV };
    }
    if (acc && pwd && cfg.username && cfg.password) {
      setFieldValue(acc, cfg.username);
      setFieldValue(pwd, cfg.password);
      if (ui) ui.note('登录框为空，已填入扩展保存的凭据');
    }
    return { username: cfg.username || '', password: cfg.password || '' };
  }

  async function startFlow() {
    stopRun();
    const myRunId = ++currentRunId;
    state = 'running';
    const ui = { status: renderStatus, note: renderNote };
    try {
      const cfg = (await getCfg()) || {};
      const creds = resolveCredentials(cfg, ui);
      if (!creds.username || !creds.password) {
        const err = new Error('登录框为空，扩展也未保存学号/密码');
        err.stopRetry = true;
        throw err;
      }
      const res = await autoLogin(Object.assign({}, cfg, creds), ui, myRunId);
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
      '检测到登录页：' + remain + ' 秒后自动登录' + (cfg.username ? '（' + cfg.username + '）' : ''),
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

  // 用户开始在页面表单里输入 → 取消倒计时，转为手动卡片。
  // 只认 pointerdown/input：关闭浏览器翻译弹窗等场景只会令输入框重新获得焦点（focusin），
  // 若无 pointerdown 也取消，会被误判为手动操作。
  function onUserInteract(e) {
    if (state !== 'pending') return;
    if (!(e.target instanceof Element)) return;
    if (e.target.closest('#' + WIDGET_ID)) return;
    if (e.target.closest('input, button, form')) {
      stopRun();
      renderManual();
    }
  }

  // ── 触发方式：click = 点击页面「登录」按钮时接管；auto = 倒计时自动 ──
  // 页面原生登录按钮实测为 button.pc-login（含隐藏的短信登录面板同款按钮）；
  // 在 capture 阶段拦截点击/回车/表单提交，阻止页面自己的滑块流程，改走扩展自动流程。
  const LOGIN_BTN_SEL = 'button.pc-login, button.login-btn';
  const LOGIN_INPUT_SEL = ACC_SEL + ', ' + PWD_SEL;
  let clickTriggerEnabled = false;
  let clickTriggerDisabled = false;

  function installClickTrigger() {
    clickTriggerEnabled = true;
    const takeOver = (e) => {
      if (clickTriggerDisabled) return;
      e.preventDefault();
      e.stopImmediatePropagation();
      if (state === 'running') return;
      startFlow();
    };
    const onClick = (e) => {
      if (!(e.target instanceof Element) || e.target.closest('#' + WIDGET_ID)) return;
      if (!e.target.closest(LOGIN_BTN_SEL)) return;
      takeOver(e);
    };
    const onSubmit = (e) => {
      if (e.target instanceof Element && e.target.closest('#' + WIDGET_ID)) return;
      takeOver(e);
    };
    const onKey = (e) => {
      if (e.key !== 'Enter') return;
      if (!(e.target instanceof Element) || e.target.closest('#' + WIDGET_ID)) return;
      if (!e.target.closest(LOGIN_INPUT_SEL)) return;
      takeOver(e);
    };
    document.addEventListener('click', onClick, true);
    document.addEventListener('submit', onSubmit, true);
    document.addEventListener('keydown', onKey, true);
  }

  function disableClickTrigger() {
    clickTriggerDisabled = true;
    renderCard('已改用页面原生登录（本页不再接管）', [
      mkBtn('恢复自动接管', () => { clickTriggerDisabled = false; startFlow(); }, true),
      mkBtn('关闭', hideWidget),
    ]);
  }

  function startEntry(c) {
    c = c || {};
    if ((c.triggerMode || 'click') === 'auto') {
      document.addEventListener('pointerdown', onUserInteract, true);
      document.addEventListener('input', onUserInteract, true);
      startCountdown(c);
      return;
    }
    installClickTrigger();
    const buttons = [mkBtn('立即登录', () => startFlow(), true)];
    if (!c.username || !c.password) buttons.push(mkBtn('去设置', openSettings));
    buttons.push(mkBtn('关闭', hideWidget));
    renderCard('点击页面上的「登录」按钮，自动完成验证并登录', buttons);
  }

  // ── 检测登录页（SPA 渲染需要等待） ─────────────────────────────────
  async function waitForForm(timeoutMs) {
    const t0 = Date.now();
    while (Date.now() - t0 < timeoutMs) {
      const acc = document.querySelector(ACC_SEL);
      const pwd = document.querySelector(PWD_SEL);
      if (acc && pwd) return { acc: acc, pwd: pwd };
      await sleep(300);
    }
    return null;
  }

  // ── 消息：来自 popup 的「测试滑块」「立即登录」 ─────────────────────
  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg && msg.type === 'SUFE_TEST_SLIDER') {
      testSlider(3).then(sendResponse);
      return true;
    }
    if (msg && msg.type === 'SUFE_RUN_NOW') {
      startFlow();
      sendResponse({ ok: true });
      return false;
    }
    if (msg && msg.type === 'SUFE_STATS') {
      chrome.storage.local.get(STATS_KEY).then((o) => sendResponse(o[STATS_KEY] || null));
      return true;
    }
  });

  // ── 主入口 ──────────────────────────────────────────────────────────
  const cfg = await getCfg();
  const form = await waitForForm(20000);
  if (!form) return; // 不是登录页（或页面已跳转/渲染失败），静默退出

  // popup「登录」免刷新注入路径携带此标记：跳过倒计时与各项拦截，立即开始
  if (window.__SUFE_RUN_NOW__) {
    delete window.__SUFE_RUN_NOW__;
    startFlow();
    return;
  }

  // 闭环自检触发：在任意支持的登录页 URL 后加 #sufe-test-slider
  // （如 login.sufe.edu.cn/login/#sufe-test-slider）；只跑滑块验证（不提交账号密码），结果显示在浮窗中。
  if (location.hash.indexOf('sufe-test-slider') >= 0) {
    const r = await testSlider(3);
    renderCard(r.ok ? '✓ ' + r.detail : '✗ ' + r.detail, [mkBtn('关闭', hideWidget)]);
    return;
  }

  const st = await getState();
  if (st.lastAutoLoginAt && Date.now() - st.lastAutoLoginAt < LOOP_GUARD_MS) {
    renderCard('刚刚已完成自动登录，但页面仍停留在登录页。可能跳转异常，请手动登录或稍后重试。', [
      mkBtn('重新尝试', () => startFlow(), true),
      mkBtn('关闭', hideWidget),
    ]);
    return;
  }

  if (cfg && cfg.enabled === false) {
    renderCard('上财自动滑块：自动登录已停用。', [
      mkBtn('本次登录', () => startFlow(), true),
      mkBtn('启用', async () => {
        const c = (await getCfg()) || {};
        c.enabled = true;
        await chrome.storage.local.set({ [CFG_KEY]: c });
        startEntry(c);
      }),
      mkBtn('关闭', hideWidget),
    ]);
    return;
  }

  startEntry(cfg || {});
})();

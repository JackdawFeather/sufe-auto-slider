/* 浏览器实测脚本（在 login.sufe.edu.cn 页面上下文中执行 evaluate_script）
 * 前置: 页面同源环境需已注入 SUFE_CNN_WEIGHTS 与 SufeCNN（sufe-cnn-weights.js + sufe-cnn.js）。
 * 行为: 连续 N 轮真实验证码: init → canvas 解码 → JS-CNN 定位 → 网格探测 → 记录命中。
 * 返回: JSON 字符串（每轮 pred/conf/命中探测次数）。
 */
(async () => {
  const N = (typeof window !== 'undefined' && window.__SUFE_TEST_N) ? window.__SUFE_TEST_N : 10;
  const BASE = '/esc-sso';
  const GRID = [0, 2, -2, 4, -4, 6, -6, 1, -1, 3, -3, 5, -5, 7, -7];
  const sleep = (ms) => new Promise((s) => setTimeout(s, ms));

  async function api(p, isPost) {
    const r = await fetch(BASE + p, {
      method: isPost ? 'POST' : 'GET',
      headers: Object.assign({ Accept: 'application/json' }, isPost ? { 'Content-Type': 'application/json' } : {}),
      body: isPost ? '{}' : undefined,
      credentials: 'same-origin',
    });
    return r.json();
  }

  function decodePng(b64) {
    return new Promise((res, rej) => {
      const img = new Image();
      img.onload = () => {
        const c = document.createElement('canvas');
        c.width = img.naturalWidth; c.height = img.naturalHeight;
        const ctx = c.getContext('2d', { willReadFrequently: true });
        ctx.drawImage(img, 0, 0);
        res({ data: ctx.getImageData(0, 0, c.width, c.height).data, w: c.width, h: c.height });
      };
      img.onerror = rej;
      img.src = 'data:image/png;base64,' + b64;
    });
  }

  window.__SUFE_TEST_RESULTS = window.__SUFE_TEST_RESULTS || [];
  const out = window.__SUFE_TEST_RESULTS;
  const baseIdx = out.length;
  for (let i = 1; i <= N; i++) {
    try {
      const d = (await api('/api/v3/sliderCaptcha/init', true)).data;
      const im = await decodePng(d.sourceImage);
      const r = SufeCNN.locate(im.data, im.w, im.h);
      const base = Math.round(r.x);
      let hit = null;
      let tries = 0;
      for (const off of GRID) {
        const x = base + off;
        if (x < 0) continue;
        tries++;
        const rr = await api('/api/v3/sliderCaptcha/check?X=' + x + '&Y=' + d.Y + '&token=' + d.token);
        if (String(rr.code) === '0') { hit = { x: x, tries: tries }; break; }
      }
      out.push({ i: baseIdx + i, w: im.w, pred: Math.round(r.x * 10) / 10, conf: Math.round(r.conf * 1000) / 1000, hit: hit });
    } catch (e) {
      out.push({ i: baseIdx + i, error: String(e) });
    }
    if (i < N) await sleep(400);
  }
  const pass = out.filter((o) => o.hit).length;
  return JSON.stringify({ pass: pass, total: out.length, rows: out.slice(baseIdx) });
})();

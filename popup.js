const CFG_KEY = 'sufeAutoSliderCfg';

// login.sufe.edu.cn 任意路径，或任意 *.sufe.edu.cn 的 /login* 页面（含 WebVPN 代理域名）
function isLoginPageUrl(url) {
  if (!url) return false;
  if (url.startsWith('https://login.sufe.edu.cn/')) return true;
  let u;
  try { u = new URL(url); } catch (e) { return false; }
  return u.protocol === 'https:' &&
    /(^|\.)sufe\.edu\.cn$/.test(u.hostname) &&
    u.pathname.startsWith('/login');
}

const $username = document.getElementById('username');
const $password = document.getElementById('password');
const $togglePwd = document.getElementById('togglePwd');
const $enabled = document.getElementById('enabled');
const $trigger = document.getElementById('trigger');
const $save = document.getElementById('save');
const $login = document.getElementById('login');
const $test = document.getElementById('test');
const $status = document.getElementById('status');
const $stats = document.getElementById('stats');

function setStatus(text, isErr) {
  $status.textContent = text;
  $status.className = isErr ? 'err' : '';
}

chrome.storage.local.get(CFG_KEY).then((o) => {
  const cfg = o[CFG_KEY] || {};
  $username.value = cfg.username || '';
  $password.value = cfg.password || '';
  $enabled.checked = cfg.enabled !== false;
  $trigger.value = cfg.triggerMode === 'auto' ? 'auto' : 'click';
});

chrome.storage.local.get('sufeAutoSliderStats').then((o) => {
  const s = o.sufeAutoSliderStats;
  if (s && s.n) {
    $stats.textContent = `历史滑块统计：${s.n} 次成功，${s.firstTry} 次首次命中（${Math.round(s.firstTry / s.n * 100)}%）`;
  }
});

$togglePwd.addEventListener('click', () => {
  const show = $password.type === 'password';
  $password.type = show ? 'text' : 'password';
  $togglePwd.textContent = show ? '隐藏' : '显示';
  $togglePwd.title = show ? '隐藏密码' : '显示密码';
});

$save.addEventListener('click', async () => {
  const cfg = {
    username: $username.value.trim(),
    password: $password.value,
    enabled: $enabled.checked,
    triggerMode: $trigger.value,
  };
  if (!cfg.username || !cfg.password) {
    setStatus('请填写学号和密码。', true);
    return;
  }
  await chrome.storage.local.set({ [CFG_KEY]: cfg });
  setStatus('已保存 ✓ 刷新上财登录页即可生效。');
});

async function activeLoginTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab || !tab.url || !isLoginPageUrl(tab.url)) return null;
  return tab;
}

$login.addEventListener('click', async () => {
  const tab = await activeLoginTab();
  if (!tab) {
    setStatus('请先在当前标签页打开上财登录页（login.sufe.edu.cn 或 *.sufe.edu.cn/login），再点「登录」。', true);
    return;
  }
  setStatus('正在登录…（结果见页面右上角浮窗）');
  try {
    await chrome.tabs.sendMessage(tab.id, { type: 'SUFE_RUN_NOW' });
    setStatus('已通知页面开始登录，请看页面浮窗 ✓');
  } catch (e) {
    // 页面还没有注入过脚本（例如刚安装扩展）：直接注入并带上“立即执行”标记
    try {
      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: () => { window.__SUFE_RUN_NOW__ = true; },
      });
      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files: ['sufe-crypto.js', 'sufe-cnn-weights.js', 'sufe-cnn.js', 'content.js'],
      });
      setStatus('已注入脚本并开始登录，请看页面浮窗 ✓');
    } catch (e2) {
      setStatus('注入失败：' + (e2 && e2.message ? e2.message : e2) + '\n请刷新登录页后重试。', true);
    }
  }
});

$test.addEventListener('click', async () => {
  const tab = await activeLoginTab();
  if (!tab) {
    setStatus('请先打开上财登录页（login.sufe.edu.cn 或 *.sufe.edu.cn/login）标签页，再点击测试。', true);
    return;
  }
  setStatus('测试中（最多 3 轮，约 10 秒）…');
  try {
    const res = await chrome.tabs.sendMessage(tab.id, { type: 'SUFE_TEST_SLIDER' });
    if (res && res.ok) {
      setStatus('✓ 滑块可正常通过：' + res.detail);
    } else {
      setStatus('✗ 未通过（多为风控收紧期，稍后再试）：\n' + (res ? res.detail : '无响应'), true);
    }
  } catch (e) {
    setStatus('测试失败：' + (e && e.message ? e.message : e) + '\n（请确认当前标签页是登录页且已刷新）', true);
  }
});

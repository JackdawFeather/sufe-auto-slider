const CFG_KEY = 'sufeAutoSliderCfg';

const $username = document.getElementById('username');
const $password = document.getElementById('password');
const $togglePwd = document.getElementById('togglePwd');
const $enabled = document.getElementById('enabled');
const $login = document.getElementById('login');
const $save = document.getElementById('save');
const $test = document.getElementById('test');
const $status = document.getElementById('status');

function setStatus(text, isErr) {
  $status.textContent = text;
  $status.className = isErr ? 'err' : '';
}

chrome.storage.local.get(CFG_KEY).then((o) => {
  const cfg = o[CFG_KEY] || {};
  $username.value = cfg.username || '';
  $password.value = cfg.password || '';
  $enabled.checked = cfg.enabled !== false;
});

$save.addEventListener('click', async () => {
  const cfg = {
    username: $username.value.trim(),
    password: $password.value,
    enabled: $enabled.checked,
  };
  if (!cfg.username || !cfg.password) {
    setStatus('请填写学号和密码。', true);
    return;
  }
  await chrome.storage.local.set({ [CFG_KEY]: cfg });
  setStatus('已保存 ✓ 刷新上财登录页即可自动登录。');
});

$togglePwd.addEventListener('click', () => {
  const show = $password.type === 'password';
  $password.type = show ? 'text' : 'password';
  $togglePwd.textContent = show ? '隐藏' : '显示';
  $togglePwd.title = show ? '隐藏密码' : '显示密码';
});

async function getLoginTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab || !tab.id || !tab.url || !tab.url.startsWith('https://login.sufe.edu.cn/')) {
    setStatus('请先打开上财登录页（login.sufe.edu.cn）标签页，再点击。', true);
    return null;
  }
  return tab;
}

$login.addEventListener('click', async () => {
  const tab = await getLoginTab();
  if (!tab) return;
  setStatus('正在触发登录…');
  try {
    await chrome.tabs.sendMessage(tab.id, { type: 'SUFE_RUN_NOW' });
    setStatus('✓ 已触发登录（页面右上角浮窗显示进度）');
  } catch (e) {
    // 页面里还没有扩展脚本（例如刚安装扩展、页面未刷新）→ 动态注入并立即执行，免刷新
    try {
      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: () => { window.__SUFE_RUN_NOW__ = true; },
      });
      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files: ['sufe-crypto.js', 'sufe-cnn-weights.js', 'sufe-cnn.js', 'content.js'],
      });
      setStatus('✓ 已注入脚本并触发登录（页面右上角浮窗显示进度）');
    } catch (e2) {
      setStatus('触发失败：' + (e2 && e2.message ? e2.message : e2), true);
    }
  }
});

$test.addEventListener('click', async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab || !tab.url || !tab.url.startsWith('https://login.sufe.edu.cn/')) {
    setStatus('请先打开上财登录页（login.sufe.edu.cn）标签页，再点击测试。', true);
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

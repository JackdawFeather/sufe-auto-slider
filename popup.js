const CFG_KEY = 'sufeAutoSliderCfg';

const $username = document.getElementById('username');
const $password = document.getElementById('password');
const $enabled = document.getElementById('enabled');
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

// 上财自动滑块 · 后台服务线程：仅负责代为打开设置页
chrome.runtime.onMessage.addListener((msg) => {
  if (msg && msg.type === 'openSettings') {
    chrome.tabs.create({ url: chrome.runtime.getURL('popup.html') });
  }
});

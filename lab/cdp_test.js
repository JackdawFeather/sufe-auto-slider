/* CDP closed-loop test: read the extension widget text from the login page target. */
const EXT_WIDGET = `(function(){
  var w = document.getElementById('sufe-auto-slider-widget');
  return w ? w.innerText : '';
})()`;

async function targets() {
  const r = await fetch('http://127.0.0.1:9222/json/list');
  return r.json();
}

function rpc(ws, id, method, params) {
  return new Promise((res, rej) => {
    const msg = JSON.stringify({ id, method, params: params || {} });
    const h = (ev) => {
      const d = JSON.parse(ev.data);
      if (d.id === id) { ws.removeEventListener('message', h); d.error ? rej(new Error(JSON.stringify(d.error))) : res(d.result); }
    };
    ws.addEventListener('message', h);
    ws.send(msg);
  });
}

async function main() {
  const deadline = Date.now() + 90000;
  let list = [];
  while (Date.now() < deadline) {
    try { list = await targets(); break; } catch (e) { await new Promise(r => setTimeout(r, 1500)); }
  }
  if (!list.length) { console.error('no CDP targets'); process.exit(1); }
  const page = list.find(t => t.type === 'page' && /login\.sufe\.edu\.cn/.test(t.url));
  if (!page) { console.error('login page target not found:', list.map(t => t.url)); process.exit(1); }
  console.log('target:', page.url);
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  let id = 1;
  await rpc(ws, id++, 'Runtime.enable');
  const t0 = Date.now();
  while (Date.now() - t0 < 120000) {
    const r = await rpc(ws, id++, 'Runtime.evaluate', { expression: EXT_WIDGET, returnByValue: true });
    const txt = (r.result && r.result.value) || '';
    if (txt && (/通过/.test(txt) || /✗/.test(txt) || /未通过/.test(txt))) {
      console.log('--- widget text ---');
      console.log(txt);
      console.log('--- end ---');
      ws.close();
      process.exit(/✓/.test(txt) ? 0 : 2);
    }
    await new Promise(r => setTimeout(r, 2500));
  }
  console.error('timeout waiting for widget result');
  ws.close();
  process.exit(3);
}

main();

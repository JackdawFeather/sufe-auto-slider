/* CDP debug: capture console + exceptions from the login page target for 25s. */
async function main() {
  const list = await (await fetch('http://127.0.0.1:9222/json/list')).json();
  const page = list.find(t => t.type === 'page' && /login\.sufe\.edu\.cn/.test(t.url));
  if (!page) { console.error('no page target'); process.exit(1); }
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  let id = 1;
  const send = (method, params) => new Promise((res) => {
    const my = id++;
    const h = (ev) => { const d = JSON.parse(ev.data); if (d.id === my) { ws.removeEventListener('message', h); res(d.result); } };
    ws.addEventListener('message', h);
    ws.send(JSON.stringify({ id: my, method, params: params || {} }));
  });
  ws.addEventListener('message', (ev) => {
    const d = JSON.parse(ev.data);
    if (d.method === 'Runtime.consoleAPICalled') {
      console.log('[console.' + d.params.type + ']', d.params.args.map(a => a.value || a.description || '').join(' ').slice(0, 300));
    }
    if (d.method === 'Runtime.exceptionThrown') {
      const e = d.params.exceptionDetails;
      console.log('[EXCEPTION]', (e.exception && e.exception.description || e.text || '').slice(0, 500));
    }
    if (d.method === 'Log.entryAdded') {
      console.log('[log.' + d.params.entry.level + ']', (d.params.entry.text || '').slice(0, 300), (d.params.entry.url || '').slice(0, 80));
    }
  });
  await send('Runtime.enable');
  await send('Log.enable');
  const r = await send('Runtime.evaluate', {
    expression: `JSON.stringify({widget: !!document.getElementById('sufe-auto-slider-widget'), inputs: document.querySelectorAll('input').length, url: location.href, flag: typeof window.__SUFE_AUTO_SLIDER__})`,
    returnByValue: true,
  });
  console.log('state:', r.result.value);
  await new Promise(res => setTimeout(res, 25000));
  const r2 = await send('Runtime.evaluate', {
    expression: `(document.getElementById('sufe-auto-slider-widget')||{}).innerText || 'NO WIDGET'`,
    returnByValue: true,
  });
  console.log('widget now:', JSON.stringify(r2.result.value));
  ws.close();
  process.exit(0);
}
main();

/* 浏览器闭环（新模型）：把出货版 sufe-cnn.js + 权重注入真实登录页，走「init → locate → 梯队 check」。 */
const fs = require('fs');

const EXT = '../';
const driver = `
(async function(){
  try {
    var BASE='https://login.sufe.edu.cn/esc-sso';
    var api=async function(p,o){var r=await fetch(BASE+p,o);return r.json();};
    var init=await api('/api/v3/sliderCaptcha/init',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
    var d=init.data;
    if(!d||!d.token){document.title='CLOSEDLOOP ERR init '+init.msg;return;}
    var bin=atob(d.sourceImage);var arr=new Uint8Array(bin.length);
    for(var i=0;i<bin.length;i++)arr[i]=bin.charCodeAt(i);
    var bmp=await createImageBitmap(new Blob([arr],{type:'image/png'}));
    var cv=document.createElement('canvas');cv.width=bmp.width;cv.height=bmp.height;
    var ctx=cv.getContext('2d',{willReadFrequently:true});ctx.drawImage(bmp,0,0);
    var im=ctx.getImageData(0,0,cv.width,cv.height);
    var t0=Date.now();
    var loc=SufeCNN.locate(im.data,cv.width,cv.height);
    var inferMs=Date.now()-t0;
    var segX=SufeCNN.segmentX(im.data,cv.width,cv.height);
    var cx=Math.round(loc.x);
    var ladder=[['cnn',cx],['cnn+1',cx+1],['cnn-1',cx-1],['cnn+2',cx+2],['cnn-2',cx-2]];
    if(segX!=null){ladder.push(['seg',segX],['seg+1',segX+1],['seg-1',segX-1]);}
    var done=null,used=0;
    for(var k=0;k<ladder.length;k++){
      if(ladder[k][1]<0)continue;
      used++;
      var r=await api('/api/v3/sliderCaptcha/check?X='+ladder[k][1]+'&Y='+d.Y+'&token='+encodeURIComponent(d.token));
      if(String(r.code)==='0'){done=ladder[k];break;}
    }
    document.title='CLOSEDLOOP '+(done?done[0]+' X='+done[1]:'FAIL')
      +' probes='+used+' cnnX='+cx+' conf='+loc.conf.toFixed(2)+' segX='+segX
      +' W='+cv.width+' infer='+inferMs+'ms';
  } catch(e){ document.title='CLOSEDLOOP ERR '+e.message; }
})();
`;

async function main() {
  const trials = parseInt(process.argv[2] || '5', 10);
  const list = await (await fetch('http://127.0.0.1:9222/json/list')).json();
  const page = list.find(t => t.type === 'page' && /login\.sufe\.edu\.cn/.test(t.url));
  if (!page) { console.error('no login target on :9222'); process.exit(1); }
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise(r => { ws.onopen = r; });
  let id = 1;
  const send = (m, p) => new Promise(res => {
    const my = id++;
    const h = ev => { const d = JSON.parse(ev.data); if (d.id === my) { ws.removeEventListener('message', h); res(d.result); } };
    ws.addEventListener('message', h);
    ws.send(JSON.stringify({ id: my, method: m, params: p || {} }));
  });
  const code = fs.readFileSync(EXT + 'sufe-cnn-weights.js', 'utf8') + '\n' +
    fs.readFileSync(EXT + 'sufe-cnn.js', 'utf8') + '\n' + driver;
  let pass = 0, first = 0;
  for (let t = 0; t < trials; t++) {
    await send('Page.navigate', { url: 'https://login.sufe.edu.cn/login/?t=' + t });
    await new Promise(r => setTimeout(r, 3500));
    await send('Runtime.evaluate', { expression: code });
    let title = '';
    for (let i = 0; i < 25; i++) {
      await new Promise(r => setTimeout(r, 1000));
      const r = await send('Runtime.evaluate', { expression: 'document.title', returnByValue: true });
      title = (r.result && r.result.value) || '';
      if (/CLOSEDLOOP/.test(title)) break;
    }
    console.log('[' + t + ']', title);
    if (/CLOSEDLOOP (cnn|seg)/.test(title) && !/FAIL|ERR/.test(title)) {
      pass++;
      if (/probes=1/.test(title)) first++;
    }
  }
  console.log(`pass ${pass}/${trials}, 首探测命中 ${first}/${trials}`);
  ws.close();
  process.exit(pass === trials ? 0 : 2);
}
main();

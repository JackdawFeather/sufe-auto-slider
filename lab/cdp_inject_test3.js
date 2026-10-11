/* 浏览器闭环（v4 流程，与 content.js candidateStream 同构）：
 * 严格分割 → 宽松分割 → CNN → （盲试，测试中略） */
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

    function offs(x,m){ if(x==null||!isFinite(x))return[]; x=Math.round(x); var o=[];
      [[0,''],[1,'+1'],[-1,'-1'],[2,'+2'],[-2,'-2']].forEach(function(p){var v=x+p[0]; if(v>=0)o.push([v,m+p[1]]);}); return o; }
    var t0=Date.now();
    var strict=SufeCNN.segmentStrict(im.data,cv.width,cv.height);
    var segMs=Date.now()-t0;
    var loose=null, cnnMs=0, cx=null, conf=0, probes=0, done=null;

    async function tryList(list){
      for(var k=0;k<list.length;k++){
        probes++;
        var r=await api('/api/v3/sliderCaptcha/check?X='+list[k][0]+'&Y='+d.Y+'&token='+encodeURIComponent(d.token));
        if(String(r.code)==='0'){ done=list[k]; return true; }
      }
      return false;
    }
    if(strict){ if(await tryList(offs(strict.x,'seg'))) { /* done */ } }
    if(!done){
      var tl=Date.now(); loose=SufeCNN.segmentX(im.data,cv.width,cv.height); segMs+=Date.now()-tl;
      if(loose!=null && await tryList(offs(loose,'seg'))) { /* done */ }
    }
    if(!done){
      var tc=Date.now(); var rr=SufeCNN.locate(im.data,cv.width,cv.height); cnnMs=Date.now()-tc;
      cx=Math.round(rr.x); conf=rr.conf;
      await tryList(offs(cx,'cnn'));
    }
    document.title='CLOSEDLOOP '+(done?done[1]+' X='+done[0]:'FAIL')
      +' probes='+probes+' strict='+(strict?strict.x:'null')+' loose='+loose+' cnnX='+cx
      +' W='+cv.width+' segMs='+segMs+' cnnMs='+cnnMs;
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
  let pass = 0, first = 0, segFast = 0, cnnUsed = 0;
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
    if (/CLOSEDLOOP (seg|cnn)/.test(title) && !/FAIL|ERR/.test(title)) {
      pass++;
      if (/probes=1/.test(title)) first++;
      if (/strict=\d/.test(title)) segFast++;
      if (/cnnMs=[1-9]/.test(title)) cnnUsed++;
    }
  }
  console.log(`pass ${pass}/${trials}, 首探测命中 ${first}/${trials}, 走严格分割(2ms) ${segFast}/${trials}, 触发CNN ${cnnUsed}/${trials}`);
  ws.close();
  process.exit(pass === trials ? 0 : 2);
}
main();

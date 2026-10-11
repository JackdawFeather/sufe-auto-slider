/* Browser closed-loop: run the SHIPPED extension JS (weights + CNN + ladder) in the
 * live login page (main world, same-origin fetch) and report pass/fail via title. */
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
    var cnn=SufeCnn.predict(im.data,cv.width,cv.height,d.Y);
    var seg=SufeCnn.segment(im.data,cv.width,cv.height);
    var segX=seg?seg.x0-2:null;
    var ladder=[['cnn',cnn.X]];
    if(segX!=null&&segX!==cnn.X)ladder.push(['seg',segX]);
    [1,-1,2,-2].forEach(function(dx){ladder.push(['cnn'+(dx>0?'+':'')+dx,cnn.X+dx]);});
    var done=null;
    for(var k=0;k<ladder.length;k++){
      var nm=ladder[k][0],x=ladder[k][1];
      var r=await api('/api/v3/sliderCaptcha/check?X='+x+'&Y='+d.Y+'&token='+encodeURIComponent(d.token));
      if(String(r.code)==='0'){done=nm+' X='+x;break;}
    }
    document.title='CLOSEDLOOP '+(done||'FAIL')+' cnnX='+cnn.X+' conf='+cnn.conf.toFixed(2)+' segX='+segX+' W='+cv.width;
  } catch(e){ document.title='CLOSEDLOOP ERR '+e.message; }
})();
`;

async function main() {
  const list = await (await fetch('http://127.0.0.1:9222/json/list')).json();
  const page = list.find(t => t.type === 'page' && /login\.sufe\.edu\.cn/.test(t.url));
  if (!page) { console.error('no login target'); process.exit(1); }
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
  const trials = parseInt(process.argv[2] || '5', 10);
  let pass = 0, cnnFirst = 0;
  for (let t = 0; t < trials; t++) {
    await send('Page.navigate', { url: 'https://login.sufe.edu.cn/login/?t=' + t });
    await new Promise(r => setTimeout(r, 4000));
    await send('Runtime.evaluate', { expression: code });
    let title = '';
    for (let i = 0; i < 20; i++) {
      await new Promise(r => setTimeout(r, 1000));
      const r = await send('Runtime.evaluate', { expression: 'document.title', returnByValue: true });
      title = r.result.value || '';
      if (/CLOSEDLOOP/.test(title)) break;
    }
    console.log('[' + t + ']', title);
    if (/CLOSEDLOOP (cnn|seg|cnn[+-])/.test(title) && !/FAIL|ERR/.test(title)) { pass++; if (/CLOSEDLOOP cnn /.test(title)) cnnFirst++; }
  }
  console.log(`pass ${pass}/${trials}, first-candidate(cnn) ${cnnFirst}/${trials}`);
  ws.close();
  process.exit(pass === trials ? 0 : 2);
}
main();

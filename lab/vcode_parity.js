'use strict';
// JS↔Python 前向一致性校验：读取 train_vcode.py 导出的 parity fixtures
const fs = require('fs');
const path = require('path');
global.window = global;
new Function(fs.readFileSync('sufe-vcode-weights.js', 'utf8'))();
const C = require(path.resolve('sufe-vcode.js'));
C.load();
const DIR = 'lab/output/vcode';
const fx = JSON.parse(fs.readFileSync(path.join(DIR, 'parity.json'), 'utf8'));
let ok = 0;
for (const f of fx) {
  const bin = fs.readFileSync(path.join(DIR, 'parity_' + f.file.replace('.jpg', '') + '.bin'));
  const t0 = Date.now();
  const r = C.recognize(new Uint8Array(bin), f.w, f.h);
  const ms = Date.now() - t0;
  const pass = r.text === f.py;
  ok += pass ? 1 : 0;
  console.log(`${f.file} label=${f.label} py=${f.py} js=${r.text} conf=${r.conf.toFixed(3)} ${pass ? 'OK' : 'MISMATCH'} ${ms}ms`);
}
console.log(`parity ${ok}/${fx.length}`);
process.exit(ok === fx.length ? 0 : 1);

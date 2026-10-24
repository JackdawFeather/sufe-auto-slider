'use strict';
// 推理耗时基准：node lab/vcode_bench.js <weights.js> [iters=6]
const fs = require('fs');
const path = require('path');
const weightsFile = process.argv[2] || 'sufe-vcode-weights.js';
const iters = parseInt(process.argv[3] || '6', 10);
global.window = global;
new Function(fs.readFileSync(weightsFile, 'utf8'))();
const C = require(path.resolve(__dirname, '..', 'sufe-vcode.js'));
C.load();
const DIR = 'lab/output/vcode';
const fx = JSON.parse(fs.readFileSync(path.join(DIR, 'parity.json'), 'utf8'))[0];
const px = new Uint8Array(fs.readFileSync(path.join(DIR, 'parity_' + fx.file.replace('.jpg', '') + '.bin')));
let t0 = Date.now();
C.recognize(px, fx.w, fx.h); // 预热
const warm = Date.now() - t0;
const times = [];
let r = null;
for (let i = 0; i < iters; i++) {
  const t = Date.now();
  r = C.recognize(px, fx.w, fx.h);
  times.push(Date.now() - t);
}
const avg = times.reduce((a, b) => a + b, 0) / times.length;
console.log('%s | 预热 %dms | 平均 %sms | 识别=%s (label %s)', path.basename(weightsFile), warm, avg.toFixed(0), r.text, fx.label);

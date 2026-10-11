/* 对比：经典分割(segment) vs CNN(locate) 在留出集上的精度与耗时。 */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

global.window = {};
vm.runInThisContext(fs.readFileSync(path.resolve(__dirname, '..', 'sufe-cnn-weights.js'), 'utf8'));
const SufeCNN = require(path.resolve(__dirname, '..', 'sufe-cnn.js'));

const OUT = path.join(__dirname, 'output', 'parity3');
const meta = JSON.parse(fs.readFileSync(path.join(OUT, 'meta.json'), 'utf8'));

let segMs = [], cnnMs = [], segErr = [], cnnErr = [], segNull = 0;
for (const m of meta) {
  const px = new Uint8Array(fs.readFileSync(path.join(OUT, `${m.k}.raw`)));
  let t = Date.now();
  const sx = SufeCNN.segmentX(px, m.w, m.h);
  segMs.push(Date.now() - t);
  t = Date.now();
  const r = SufeCNN.locate(px, m.w, m.h);
  cnnMs.push(Date.now() - t);
  if (sx == null) segNull++;
  else segErr.push(sx - m.true);
  cnnErr.push(r.x - m.true);
}
const abs = (a) => a.map(Math.abs);
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
const p = (a) => a.slice().sort((x, y) => x - y)[Math.floor(a.length / 2)];
console.log(`样本 ${meta.length} 张（留出集）`);
console.log(`经典分割: 失败(null) ${segNull}/${meta.length}；误差 MAE ${mean(abs(segErr)).toFixed(3)}px ` +
  `max ${Math.max(...abs(segErr)).toFixed(2)}px；耗时 中位 ${p(segMs)}ms / 均 ${Math.round(mean(segMs))}ms`);
console.log(`CNN: 误差 MAE ${mean(abs(cnnErr)).toFixed(3)}px max ${Math.max(...abs(cnnErr)).toFixed(2)}px；` +
  `耗时 中位 ${p(cnnMs)}ms / 均 ${Math.round(mean(cnnMs))}ms`);
console.log(`速度比: ${(mean(cnnMs) / mean(segMs)).toFixed(0)}x`);

/* parity3：JS(SufeCNN.locate) vs torch 预测逐样本比对。 */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

global.window = {};
vm.runInThisContext(fs.readFileSync(path.resolve(__dirname, '..', 'sufe-cnn-weights.js'), 'utf8'));
const SufeCNN = require(path.resolve(__dirname, '..', 'sufe-cnn.js'));

const OUT = path.join(__dirname, 'output', 'parity3');
const meta = JSON.parse(fs.readFileSync(path.join(OUT, 'meta.json'), 'utf8'));

let worst = 0, badX = 0, errs = [], ms = [];
for (const m of meta) {
  const px = new Uint8Array(fs.readFileSync(path.join(OUT, `${m.k}.raw`)));
  const t0 = Date.now();
  const r = SufeCNN.locate(px, m.w, m.h);
  ms.push(Date.now() - t0);
  const d = Math.abs(r.x - m.torch);
  worst = Math.max(worst, d);
  if (Math.round(r.x) !== Math.round(m.torch)) badX++;
  errs.push(r.x - m.true);
}
const abs = errs.map(Math.abs);
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
console.log(`JS vs torch: worst |d| = ${worst.toExponential(2)}px, 取整不一致 ${badX}/${meta.length}`);
console.log(`JS 端精度(vs 分割真值): MAE ${mean(abs).toFixed(3)}px  max ${Math.max(...abs).toFixed(2)}px  ` +
  `≤2px ${abs.filter((e) => e <= 2).length}/${abs.length}`);
console.log(`纯 JS 推理耗时: 平均 ${Math.round(mean(ms))}ms, 最大 ${Math.max(...ms)}ms`);
process.exit(worst < 0.2 && badX === 0 ? 0 : 1);

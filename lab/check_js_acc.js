/* 用扩展当前权重对留出集图片做 JS 侧精度/耗时复测（真值来自分割标签，不依赖 torch 预测）。 */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

global.window = {};
vm.runInThisContext(fs.readFileSync(path.resolve(__dirname, '..', 'sufe-cnn-weights.js'), 'utf8'));
const SufeCNN = require(path.resolve(__dirname, '..', 'sufe-cnn.js'));

const OUT = path.join(__dirname, 'output', 'parity3');
const meta = JSON.parse(fs.readFileSync(path.join(OUT, 'meta.json'), 'utf8'));

const errs = [], ms = [];
for (const m of meta) {
  const px = new Uint8Array(fs.readFileSync(path.join(OUT, `${m.k}.raw`)));
  const t0 = Date.now();
  const r = SufeCNN.locate(px, m.w, m.h);
  ms.push(Date.now() - t0);
  errs.push(r.x - m.true);
}
const abs = errs.map(Math.abs);
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
const sorted = [...ms].sort((a, b) => a - b);
console.log(`样本 ${meta.length} 张`);
console.log(`JS 精度(vs 分割真值): MAE ${mean(abs).toFixed(3)}px  max ${Math.max(...abs).toFixed(2)}px  ` +
  `≤2px ${abs.filter((e) => e <= 2).length}/${abs.length}`);
console.log(`JS 推理耗时: 平均 ${Math.round(mean(ms))}ms | 中位 ${sorted[sorted.length >> 1]}ms | ` +
  `慢于 1s 的 ${ms.filter((e) => e > 1000).length}/${ms.length} 张`);

/* parity2：用扩展里的 sufe-cnn.js 对同一批 raw 复算，与 numpy 预测比对。 */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

global.window = {};
vm.runInThisContext(fs.readFileSync(path.resolve(__dirname, '..', 'sufe-cnn-weights.js'), 'utf8'));
const SufeCNN = require(path.resolve(__dirname, '..', 'sufe-cnn.js'));

const OUT = path.join(__dirname, 'output', 'parity2');
const meta = JSON.parse(fs.readFileSync(path.join(OUT, 'meta.json'), 'utf8'));

let worst = 0, bad = 0;
for (const m of meta) {
  const rgb = new Uint8Array(fs.readFileSync(path.join(OUT, `${m.i}.raw`)));
  const t0 = Date.now();
  const r = SufeCNN.locate(rgb, m.w, m.h);
  const ms = Date.now() - t0;
  const d = Math.abs(r.x - m.pyx);
  worst = Math.max(worst, d);
  if (Math.round(r.x) !== Math.round(m.pyx)) bad++;
  console.log(`[${m.i}] W=${m.w} js=${r.x.toFixed(3)} py=${m.pyx.toFixed(3)} d=${d.toExponential(1)} ` +
    `conf=${r.conf.toFixed(3)} trueX=${m.trueX} jsErr=${(r.x - m.trueX).toFixed(2)} ${ms}ms`);
}
console.log(`worst |js-py| = ${worst.toExponential(2)}px; 取整后不一致 ${bad}/${meta.length}`);
process.exit(worst < 0.05 ? 0 : 1);

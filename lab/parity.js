/* Node parity harness: JS forward vs Python forward on identical pixels. */
const fs = require('fs');
const vm = require('vm');

const weightsJson = JSON.parse(fs.readFileSync('data/cnn_weights.json', 'utf8'));
global.window = { SUFE_CNN_WEIGHTS: weightsJson };
vm.runInThisContext(fs.readFileSync('../sufe-cnn.js', 'utf8'));

const meta = JSON.parse(fs.readFileSync('data/parity_meta.json', 'utf8'));
let worst = 0, badX = 0;
for (const m of meta) {
  const px = new Uint8ClampedArray(fs.readFileSync(`data/parity_${m.i}.raw`));
  // bufw=像素缓冲宽度（行距），m.w=原图宽度（仅掩码用；>512 时缓冲被裁到 512）
  const r = SufeCnn.predict(px, m.bufw, m.h, m.Y, m.w);
  const d = Math.abs(r.x - m.pyx);
  worst = Math.max(worst, d);
  if (r.X !== m.pyX) badX++;
  const seg = SufeCnn.segment(px, m.bufw, m.h);
  console.log(
    `[${m.i}] js x=${r.x.toFixed(3)} py x=${m.pyx.toFixed(3)} d=${d.toExponential(1)} ` +
    `jsX=${r.X} pyX=${m.pyX} trueX=${m.trueX} conf=${r.conf.toFixed(3)} ` +
    `segX=${seg ? seg.x0 - 2 : 'null'}`);
}
console.log(`worst |js-py| = ${worst.toExponential(2)}; X mismatches = ${badX}/${meta.length}`);
process.exit(worst < 1e-3 && badX === 0 ? 0 : 1);

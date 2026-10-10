/* 对比 JS 与 Python 逐层中间值（sample 0005） */
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const HERE = __dirname;
const ROOT = path.join(HERE, '..');
const DBG = path.join(ROOT, 'output', 'debug');

const WEIGHTS = fs.existsSync(path.join(ROOT, 'sufe-cnn-weights.js'))
  ? path.join(ROOT, 'sufe-cnn-weights.js') : path.join(ROOT, 'output', 'weights.js');

const sandbox = { console, atob, module: { exports: {} } };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(WEIGHTS, 'utf-8'), sandbox);
vm.runInContext(fs.readFileSync(path.join(ROOT, 'sufe-cnn.js'), 'utf-8'), sandbox);

const meta = JSON.parse(fs.readFileSync(path.join(DBG, 'rgb_meta.json'), 'utf-8'));
const rgb = new Uint8Array(fs.readFileSync(path.join(DBG, 'rgb.bin')));
const py = JSON.parse(fs.readFileSync(path.join(DBG, 'py_stages.json'), 'utf-8'));

const js = sandbox.SufeCNN.debugStages(rgb, meta.w, meta.h);

function cmp(name, a, b) {
  const n = Math.min(a.length, b.length);
  let maxdiff = 0, at = -1;
  for (let i = 0; i < n; i++) {
    const d = Math.abs(a[i] - (b[i] || 0));
    if (d > maxdiff) { maxdiff = d; at = i; }
  }
  const badge = maxdiff < 0.05 ? 'OK ' : (maxdiff < 1 ? '~~ ' : 'DIFF');
  const detail = at >= 0 ? ('js=' + a[at].toFixed(4) + ' py=' + (b[at] || 0).toFixed(4)) : '(all equal)';
  console.log(badge + ' ' + name.padEnd(12) + ' maxdiff=' + maxdiff.toFixed(5) +
    ' at=' + at + '  ' + detail);
  return maxdiff;
}

cmp('input', js.input_head30, py.stages[0].head);
for (let i = 0; i < js.stages.length; i++) {
  cmp(js.stages[i].stage, js.stages[i].head, py.stages[i + 1].head);
}
console.log('py pred_in=%s  argmax=%s', py.pred_in, py.argmax_bin);

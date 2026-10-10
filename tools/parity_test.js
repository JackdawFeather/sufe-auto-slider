/*
 * JS/Python 推理一致性测试:
 *   python dump_test.py  (生成 output/parity/manifest.json + raw.bin)
 *   node parity_test.js
 * 对每个样本，用 sufe-cnn.js 的 locate() 与 Python 端预测值比较。
 */
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const HERE = __dirname;
const ROOT = path.join(HERE, '..');
const PARITY = path.join(ROOT, 'output', 'parity');

const manifest = JSON.parse(fs.readFileSync(path.join(PARITY, 'manifest.json'), 'utf-8'));
const raw = fs.readFileSync(path.join(PARITY, 'raw.bin'));

const WEIGHTS_PATH = fs.existsSync(path.join(ROOT, 'sufe-cnn-weights.js'))
  ? path.join(ROOT, 'sufe-cnn-weights.js')
  : path.join(ROOT, 'output', 'weights.js');

const sandbox = { console, atob, module: { exports: {} } };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(WEIGHTS_PATH, 'utf-8'), sandbox);
vm.runInContext(fs.readFileSync(path.join(ROOT, 'sufe-cnn.js'), 'utf-8'), sandbox);
const SufeCNN = sandbox.SufeCNN;

let maxDiff = 0, n = 0, worst = null;
const t0 = Date.now();
for (const it of manifest) {
  const buf = raw.subarray(it.offset, it.offset + it.len);
  const rgb = new Uint8Array(buf);
  const r = SufeCNN.locate(rgb, it.w, it.h);
  const diff = Math.abs(r.x - it.py_pred);
  n++;
  if (diff > maxDiff) { maxDiff = diff; worst = it.dir; }
  if (diff > 0.75) console.log('偏差大:', it.dir, 'js=', r.x.toFixed(2), 'py=', it.py_pred.toFixed(2), 'conf=', r.conf.toFixed(3));
}
const ms = (Date.now() - t0) / n;
console.log('样本数 ' + n + ', JS≈' + ms.toFixed(1) + ' ms/张, 最大偏差 ' + maxDiff.toFixed(3) + ' px (' + worst + ')');
let sum = 0;
for (const it of manifest) {
  const rgb = new Uint8Array(raw.subarray(it.offset, it.offset + it.len));
  sum += Math.abs(SufeCNN.locate(rgb, it.w, it.h).x - it.py_pred);
}
console.log('平均偏差 ' + (sum / n).toFixed(3) + ' px');
console.log(maxDiff <= 0.75 ? 'PASS: JS 与 Python 推理一致' : 'FAIL: 偏差超阈值');

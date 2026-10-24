'use strict';
// OCR CLI：node lab/vcode_ocr_cli.js <rgba.bin> <w> <h>  →  {"text": "...", "conf": 0.x, "ms": 105}
const fs = require('fs');
const path = require('path');
global.window = global;
new Function(fs.readFileSync(path.join(__dirname, '..', 'sufe-vcode-weights.js'), 'utf8'))();
const C = require(path.resolve(__dirname, '..', 'sufe-vcode.js'));
const bin = process.argv[2], w = parseInt(process.argv[3], 10), h = parseInt(process.argv[4], 10);
const px = new Uint8Array(fs.readFileSync(bin));
const t0 = Date.now();
const r = C.recognize(px, w, h);
process.stdout.write(JSON.stringify({ text: r.text, conf: r.conf, ms: Date.now() - t0 }));

/*
 * 上财自动滑块 · 缺口识别 CNN（纯 JS 前向，无依赖）
 * 结构（与 lab/cnn.py 训练时完全一致）：
 *   输入 (2,58,256) = [gray, sat]，取背景图 rows [Y-8, Y+108) 的横带，
 *   宽度统一为 512（>512 左对齐裁剪，<512 reflect 补齐；不缩放以保持缺口物理尺寸），
 *   再 2x2 平均下采样。
 *   → conv3x3 2→10 + relu + maxpool2x2 → conv3x3 10→16 + relu + maxpool(整列取最大)
 *   → conv1x1 16→1 → 128 列 softmax；列 c 对应原图 x = 4c（±2 列质心细化）
 * 权重由 sufe-cnn-weights.js 注入 window.SUFE_CNN_WEIGHTS。
 */
var SufeCnn = (function () {
  'use strict';
  var NCOL = 128, STRIDE = 4, PIECE_DX = 2, BAND_TOP = -8, BAND_H = 116;
  var CANON_W = 512, BW = 256, BH = 58;
  var W1, B1, W2, B2, W3, B3; // Float32Array

  function dec(b64) {
    var bin = atob(b64);
    var n = bin.length;
    var buf = new Uint8Array(n);
    for (var i = 0; i < n; i++) buf[i] = bin.charCodeAt(i);
    return new Float32Array(buf.buffer);
  }

  function load() {
    var w = window.SUFE_CNN_WEIGHTS.w;
    W1 = dec(w['l0.W'].b64); B1 = dec(w['l0.b'].b64);
    W2 = dec(w['l3.W'].b64); B2 = dec(w['l3.b'].b64);
    W3 = dec(w['l6.W'].b64); B3 = dec(w['l6.b'].b64);
  }

  function refl(i, n) { // reflect 索引（不重复边界）
    if (i < n) return i;
    var p = 2 * (n - 1), m = i % p;
    return m < n ? m : p - m;
  }

  // px: RGBA Uint8ClampedArray；w,h 原图尺寸；Y 服务端纵坐标 → (2,58,256)
  function prep(px, w, h, Y) {
    var out = new Float32Array(2 * BH * BW);
    var top = Y + BAND_TOP;
    for (var c = 0; c < 2; c++) {
      for (var br = 0; br < BH; br++) {
        var r0 = top + 2 * br;
        for (var bc = 0; bc < BW; bc++) {
          var c0 = refl(2 * bc, w), c1 = refl(2 * bc + 1, w);
          var a = srcAt(px, w, h, r0, c0, c), b = srcAt(px, w, h, r0, c1, c);
          var d = srcAt(px, w, h, r0 + 1, c0, c), e = srcAt(px, w, h, r0 + 1, c1, c);
          out[(c * BH + br) * BW + bc] = (a + b + d + e) * 0.25;
        }
      }
    }
    return out;
  }

  function srcAt(px, w, h, r, c, ch) {
    if (r < 0) r = 0; else if (r >= h) r = h - 1;
    if (c < 0) c = 0; else if (c >= w) c = w - 1;
    var p = (r * w + c) * 4;
    var R = px[p], G = px[p + 1], B = px[p + 2];
    if (ch === 0) return (R + G + B) / 765.0;
    var mx = R > G ? (R > B ? R : B) : (G > B ? G : B);
    var mn = R < G ? (R < B ? R : B) : (G < B ? G : B);
    return (mx - mn) / 255.0;
  }

  // inp: (cin,hin,win)；3x3 conv(pad=edge)+relu，尺寸不变
  function convRelu(inp, cin, hin, win, Wt, Bt) {
    var cout = (Wt.length / (cin * 9)) | 0;
    var out = new Float32Array(cout * hin * win);
    for (var oc = 0; oc < cout; oc++) {
      for (var y = 0; y < hin; y++) {
        for (var x = 0; x < win; x++) {
          var s = Bt[oc];
          for (var ic = 0; ic < cin; ic++) {
            for (var ky = -1; ky <= 1; ky++) {
              var yy = y + ky; if (yy < 0) yy = 0; else if (yy >= hin) yy = hin - 1;
              for (var kx = -1; kx <= 1; kx++) {
                var xx = x + kx; if (xx < 0) xx = 0; else if (xx >= win) xx = win - 1;
                s += inp[(ic * hin + yy) * win + xx] * Wt[(oc * cin + ic) * 9 + (ky + 1) * 3 + (kx + 1)];
              }
            }
          }
          out[(oc * hin + y) * win + x] = s > 0 ? s : 0;
        }
      }
    }
    return out;
  }

  function maxpool2(inp, c, h, w) {
    var oh = h >> 1, ow = w >> 1;
    var out = new Float32Array(c * oh * ow);
    for (var cc = 0; cc < c; cc++) {
      for (var y = 0; y < oh; y++) {
        for (var x = 0; x < ow; x++) {
          var i0 = (cc * h + 2 * y) * w + 2 * x;
          var a = inp[i0], b = inp[i0 + 1], d = inp[i0 + w], e = inp[i0 + w + 1];
          var m = a > b ? a : b, m2 = d > e ? d : e;
          out[(cc * oh + y) * ow + x] = m > m2 ? m : m2;
        }
      }
    }
    return out;
  }

  function maxpoolRows(inp, c, h, w) {
    var out = new Float32Array(c * w);
    for (var cc = 0; cc < c; cc++) {
      for (var x = 0; x < w; x++) {
        var m = -Infinity;
        for (var y = 0; y < h; y++) { var v = inp[(cc * h + y) * w + x]; if (v > m) m = v; }
        out[cc * w + x] = m;
      }
    }
    return out;
  }

  // px: RGBA Uint8ClampedArray；w=像素缓冲宽度（行距），h 缓冲高；Y 服务端纵坐标；
  // origW=背景图原始宽度（默认取 w），仅用于屏蔽不可能的缺口列。
  function predict(px, w, h, Y, origW) {
    if (!W1) load();
    var x0 = prep(px, w, h, Y);                    // (2,58,256)
    var a1 = convRelu(x0, 2, BH, BW, W1, B1);      // (10,58,256)
    var p1 = maxpool2(a1, 10, BH, BW);             // (10,29,128)
    var a2 = convRelu(p1, 10, 29, NCOL, W2, B2);   // (16,29,128)
    var p2 = maxpoolRows(a2, 16, 29, NCOL);        // (16,1,128)
    var logits = new Float32Array(NCOL);
    // 缺口左缘 nx ∈ [0, origW-96]；列 c 对应 nx≈4c → 屏蔽不可能列（含 reflect 补齐区）
    var maxCol = Math.floor(((origW || w) - 92) / STRIDE);
    for (var x = 0; x < NCOL; x++) {
      var s = B3[0];
      for (var ic = 0; ic < 16; ic++) s += p2[ic * NCOL + x] * W3[ic];
      logits[x] = (x <= maxCol) ? s : -1e9;
    }
    var mx = -Infinity;
    for (x = 0; x < NCOL; x++) if (logits[x] > mx) mx = logits[x];
    var sum = 0, probs = new Float32Array(NCOL);
    for (x = 0; x < NCOL; x++) { probs[x] = Math.exp(logits[x] - mx); sum += probs[x]; }
    var c = 0, bp = 0;
    for (x = 0; x < NCOL; x++) { probs[x] /= sum; if (probs[x] > bp) { bp = probs[x]; c = x; } }
    var lo = Math.max(0, c - 2), hi = Math.min(NCOL, c + 3), ws = 0, acc = 0;
    for (x = lo; x < hi; x++) { ws += probs[x]; acc += x * probs[x]; }
    var xpix = (acc / ws) * STRIDE;
    return { x: xpix, X: Math.round(xpix) - PIECE_DX, col: c, conf: bp };
  }

  // 兜底：经典分割（平坦高亮低饱和连通块 = 缺口）→ {x0,y0,w,h,cnt} 或 null
  function segment(px, w, h) {
    var n = w * h;
    var flat = new Uint8Array(n);
    for (var y = 1; y < h - 1; y++) {
      for (var x = 1; x < w - 1; x++) {
        var i = y * w + x, p = i * 4;
        var R = px[p], G = px[p + 1], B = px[p + 2];
        var mx = R > G ? (R > B ? R : B) : (G > B ? G : B);
        var mn = R < G ? (R < B ? R : B) : (G < B ? G : B);
        if ((R + G + B) / 3 <= 200 || (mx - mn) >= 16) continue;
        var s = 0, s2 = 0;
        for (var dy = -1; dy <= 1; dy++) for (var dx = -1; dx <= 1; dx++) {
          var q = ((y + dy) * w + (x + dx)) * 4;
          var gv = (px[q] + px[q + 1] + px[q + 2]) / 3;
          s += gv; s2 += gv * gv;
        }
        var mean = s / 9;
        if (s2 / 9 - mean * mean < 6) flat[i] = 1;
      }
    }
    var lab = new Int32Array(n), nl = 0, best = null;
    var stack = new Int32Array(n);
    for (var i0 = 0; i0 < n; i0++) {
      if (!flat[i0] || lab[i0]) continue;
      nl++;
      var sp = 0, cnt = 0, x0 = w, x1 = 0, y0 = h, y1 = 0;
      stack[sp++] = i0; lab[i0] = nl;
      while (sp) {
        var cur = stack[--sp];
        cnt++;
        var cy = (cur / w) | 0, cx = cur % w;
        if (cx < x0) x0 = cx; if (cx > x1) x1 = cx;
        if (cy < y0) y0 = cy; if (cy > y1) y1 = cy;
        for (var dy2 = -1; dy2 <= 1; dy2++) for (var dx2 = -1; dx2 <= 1; dx2++) {
          var ny = cy + dy2, nx = cx + dx2;
          if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
          var ni = ny * w + nx;
          if (flat[ni] && !lab[ni]) { lab[ni] = nl; stack[sp++] = ni; }
        }
      }
      var bw = x1 - x0 + 1, bh = y1 - y0 + 1, fill = cnt / (bw * bh);
      if (cnt >= 2000 && bw >= 70 && bw <= 110 && bh >= 70 && bh <= 110 && fill > 0.5 && fill < 0.9) {
        if (!best || cnt > best.cnt) best = { x0: x0, y0: y0, w: bw, h: bh, cnt: cnt };
      }
    }
    return best;
  }

  return { predict: predict, segment: segment, NCOL: NCOL, STRIDE: STRIDE };
})();

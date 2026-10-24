/*
 * 上财图片验证码 OCR（纯 JS 前向，无依赖）
 *
 * 输入：/esc-sso/api/v1/image/getRandcode 返回的 JPEG（80x34 左右）
 * 处理：RGBA → BT.601 灰度（与 OpenCV 取整一致）→ area-resize 到 128x48 → [-1,1]
 * 结构（权重 ops 顺序执行，BN 已折叠进卷积）：
 *   conv3x3+ReLU → pool 2x2 → conv3x3+ReLU → conv3x3+ReLU → pool 2x2
 *   → conv3x3+ReLU → conv3x3+ReLU → pool 2x1 → 高度 max 折叠
 *   → 按 4 个字符位置分箱，每箱列向 max → 线性头 → softmax → 逐位置 argmax
 * 输出：4 位小写字母数字（服务端验证码不区分大小写，实测）
 * 权重：sufe-vcode-weights.js 注入 window.SUFE_VCODE_WEIGHTS（lab/train_vcode.py 训练导出）
 *
 * 用法：SufeVcode.recognize(rgbaU8, w, h) -> { text, conf }
 */
var SufeVcode = (function () {
  'use strict';

  function b64ToF32(b64) {
    var bin = atob(b64);
    var u8 = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return new Float32Array(u8.buffer);
  }

  var META = null, OPS = null, HEAD = null;

  function load() {
    var W = (typeof window !== 'undefined' && window.SUFE_VCODE_WEIGHTS) ||
            (typeof SUFE_VCODE_WEIGHTS !== 'undefined' ? SUFE_VCODE_WEIGHTS : null);
    if (!W) throw new Error('SUFE_VCODE_WEIGHTS 未加载');
    META = W.meta;
    OPS = W.ops.map(function (op) {
      if (op.t !== 'c') return op;
      return { t: 'c', k: op.k, cin: op.cin, cout: op.cout, relu: !!op.relu,
               w: b64ToF32(op.w), b: b64ToF32(op.b) };
    });
    HEAD = { cin: W.head.cin, cout: W.head.cout, w: b64ToF32(W.head.w), b: b64ToF32(W.head.b) };
  }

  // 灰度（OpenCV BT.601 定点）+ 可分离面积缩放 → 归一 [-1,1]
  var resizeCache = {};
  function resizeWeights(sw, sh, dw, dh) {
    var key = sw + 'x' + sh + '>' + dw + 'x' + dh;
    if (resizeCache[key]) return resizeCache[key];
    function axis(srcN, dstN) {
      var W = new Float32Array(dstN * srcN);
      for (var d = 0; d < dstN; d++) {
        var a0 = d * srcN / dstN, a1 = (d + 1) * srcN / dstN;
        var i0 = Math.floor(a0), i1 = Math.min(srcN, Math.ceil(a1));
        for (var i = i0; i < i1; i++) {
          W[d * srcN + i] = (Math.min(i + 1, a1) - Math.max(i, a0)) / (a1 - a0);
        }
      }
      return W;
    }
    var r = { Wx: axis(sw, dw), Wy: axis(sh, dh) };
    resizeCache[key] = r;
    return r;
  }

  function inputTensor(px, w, h) {
    if (!META) load();
    var dw = META.in_w, dh = META.in_h;
    var g = new Float32Array(w * h);
    for (var i = 0, p = 0; i < w * h; i++, p += 4) {
      g[i] = ((px[p] * 4899 + px[p + 1] * 9617 + px[p + 2] * 1868 + 8192) >> 14) / 255;
    }
    var W = resizeWeights(w, h, dw, dh);
    var tmp = new Float32Array(dw * h);
    for (var r = 0; r < h; r++) {
      for (var dx = 0; dx < dw; dx++) {
        var acc = 0, base = dx * w;
        for (var x = 0; x < w; x++) acc += W.Wx[base + x] * g[r * w + x];
        tmp[r * dw + dx] = acc;
      }
    }
    var out = new Float32Array(dw * dh);
    for (var dy = 0; dy < dh; dy++) {
      var wbase = dy * h;
      for (var xx = 0; xx < dw; xx++) {
        var acc2 = 0;
        for (var yy = 0; yy < h; yy++) acc2 += W.Wy[wbase + yy] * tmp[yy * dw + xx];
        out[dy * dw + xx] = (acc2 - 0.5) / 0.5;
      }
    }
    return { data: out, w: dw, h: dh };
  }

  // k x k 卷积（zero pad），平面布局 c*H*W。scatter 累加（ic/kx/ky 外循环）以提升缓存局部性。
  function conv(src, sw, sh, cin, L) {
    var cout = L.cout, k = L.k, half = (k - 1) >> 1;
    var n = sw * sh;
    var dst = new Float32Array(cout * n);
    for (var oc0 = 0; oc0 < cout; oc0++) {
      var b0 = L.b[oc0], off0 = oc0 * n;
      for (var i = 0; i < n; i++) dst[off0 + i] = b0;
    }
    for (var ic = 0; ic < cin; ic++) {
      var sOff = ic * n;
      for (var ky = 0; ky < k; ky++) {
        var dy0 = half - ky; if (dy0 < 0) dy0 = 0;
        var dy1 = sh + half - ky; if (dy1 > sh) dy1 = sh;
        for (var kx = 0; kx < k; kx++) {
          var dx0 = half - kx; if (dx0 < 0) dx0 = 0;
          var dx1 = sw + half - kx; if (dx1 > sw) dx1 = sw;
          if (dy0 >= dy1 || dx0 >= dx1) continue;
          var wOff = ic * k * k + ky * k + kx;
          for (var oc = 0; oc < cout; oc++) {
            var w = L.w[oc * cin * k * k + wOff];
            if (w === 0) continue;
            var dOff = oc * n;
            for (var y = dy0; y < dy1; y++) {
              var sRow = sOff + (y + ky - half) * sw;
              var dRow = dOff + y * sw;
              for (var x = dx0; x < dx1; x++) dst[dRow + x] += w * src[sRow + x + kx - half];
            }
          }
        }
      }
    }
    if (L.relu) {
      for (var q = 0; q < dst.length; q++) if (dst[q] < 0) dst[q] = 0;
    }
    return dst;
  }

  function maxPool(src, sw, sh, ch, ph, pw) {
    var dw = (sw / pw) | 0, dh = (sh / ph) | 0;
    var dst = new Float32Array(ch * dw * dh);
    for (var c = 0; c < ch; c++) {
      var sBase = c * sw * sh, dBase = c * dw * dh;
      for (var y = 0; y < dh; y++) {
        for (var x = 0; x < dw; x++) {
          var m = -1e30;
          for (var yy = 0; yy < ph; yy++) {
            for (var xx = 0; xx < pw; xx++) {
              var v = src[sBase + (y * ph + yy) * sw + x * pw + xx];
              if (v > m) m = v;
            }
          }
          dst[dBase + y * dw + x] = m;
        }
      }
    }
    return dst;
  }

  function collapseH(src, fw, fh, ch) {
    var dst = new Float32Array(ch * fw);
    for (var c = 0; c < ch; c++) {
      var base = c * fw * fh, o = c * fw;
      for (var x = 0; x < fw; x++) {
        var m = -1e30;
        for (var y = 0; y < fh; y++) {
          var v = src[base + y * fw + x];
          if (v > m) m = v;
        }
        dst[o + x] = m;
      }
    }
    return dst;
  }

  function recognize(px, w, h) {
    if (!META || !OPS) load();
    var t = inputTensor(px, w, h);
    var feat = t.data, fw = t.w, fh = t.h, fc = 1;
    for (var i = 0; i < OPS.length; i++) {
      var op = OPS[i];
      if (op.t === 'c') {
        feat = conv(feat, fw, fh, fc, op);
        fc = op.cout;
      } else if (op.t === 'p') {
        var ph = op.k[0], pw = op.k[1];
        feat = maxPool(feat, fw, fh, fc, ph, pw);
        fw = (fw / pw) | 0;
        fh = (fh / ph) | 0;
      } else if (op.t === 'x') {
        feat = collapseH(feat, fw, fh, fc);
        fh = 1;
      }
    }
    // 此时 feat = 96 x fw；按 4 个字符位置分箱，每箱列向 max → 96 维向量 → 线性头 → softmax
    // 概率决策：0/o 形近且在置信区间内 → 按先验（0 明显多于 o）取 0；同时返回联合置信度供上层门控
    var bins = (META.bins || 4), nclass = HEAD.cout, charset = META.charset;
    var bw = (fw / bins) | 0;
    var convW = HEAD.w, convB = HEAD.b;
    var O_CHAR = 24, ZERO_CHAR = 10;   // charset 中 'o' / '0' 的下标（"0123456789abcdefghijklmnopqrstuvwxyz"）
    var OZ_DELTA = 0.12;               // 置信区间：概率差在此以内视为 0/o 歧义
    var text = '', confSum = 0;
    var vec = new Float32Array(HEAD.cin);
    var logits = new Float32Array(nclass);
    var probs = new Float32Array(nclass);
    for (var bi = 0; bi < bins; bi++) {
      var lo = bi * bw, hi = bi < bins - 1 ? (bi + 1) * bw : fw;
      for (var ci = 0; ci < HEAD.cin; ci++) {
        var m = -1e30, base2 = ci * fw;
        for (var xx = lo; xx < hi; xx++) {
          if (feat[base2 + xx] > m) m = feat[base2 + xx];
        }
        vec[ci] = m;
      }
      var bestI = 0, bestV = -1e30;
      for (var oc = 0; oc < nclass; oc++) {
        var s = convB[oc];
        var wOff = oc * HEAD.cin;
        for (var ci2 = 0; ci2 < HEAD.cin; ci2++) s += convW[wOff + ci2] * vec[ci2];
        logits[oc] = s;
        if (s > bestV) { bestV = s; bestI = oc; }
      }
      var sum = 0;
      for (var q = 0; q < nclass; q++) { probs[q] = Math.exp(logits[q] - bestV); sum += probs[q]; }
      var pBest = 0;
      for (var q2 = 0; q2 < nclass; q2++) probs[q2] /= sum;
      pBest = probs[bestI];
      confSum += pBest;
      var ch = charset.charAt(bestI);
      // 0/o 歧义：落在置信区间内 → 按先验取 '0'
      if (charset === '0123456789abcdefghijklmnopqrstuvwxyz' &&
          ((bestI === O_CHAR && probs[ZERO_CHAR] > 0) || (bestI === ZERO_CHAR && probs[O_CHAR] > 0))) {
        var other = bestI === O_CHAR ? ZERO_CHAR : O_CHAR;
        if (pBest - probs[other] <= OZ_DELTA) ch = '0';
      }
      if (ch) text += ch;
    }
    var conf = bins ? confSum / bins : 0;
    return { text: text, conf: conf };
  }

  return { recognize: recognize, load: load };
})();
if (typeof module !== 'undefined') module.exports = SufeVcode;

/*
 * 上财滑块 · 缺口定位 CNN（纯 JS 前向推理，无依赖）
 * 结构: Conv3x3+ReLU ×1 → MaxPool(2,1) ×4 → Conv1x1 → x 向 softmax → soft-argmax
 * 权重来自 tools/train_cnn.py 训练导出（sufe-cnn-weights.js）。
 * 用法: SufeCNN.locate(rgbU8, w, h) -> { x, conf }
 *   rgbU8: Uint8Array/Array, 原始背景图 RGB 字节（w*h*3, 与 canvas.getImageData 一致）
 *   x: 预测的缺口左边缘在原图像素坐标（浮点）; conf: 热图峰值 softmax 概率
 */
var SufeCNN = (function () {
  'use strict';

  function b64ToF32(b64) {
    var bin = atob(b64);
    var u8 = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return new Float32Array(u8.buffer);
  }

  var W = (typeof SUFE_CNN_WEIGHTS !== 'undefined') ? SUFE_CNN_WEIGHTS : null;
  var layers = null;
  if (W) {
    layers = W.layers.map(function (L) {
      return { k: L.k, cin: L.cin, cout: L.cout, w: b64ToF32(L.w), b: b64ToF32(L.b) };
    });
  }

  // 面积平均缩放（与训练端 cv2.INTER_AREA 对齐；确定性实现，不依赖浏览器重采样）
  // 输出 CHW 平面布局（与 PyTorch 输入一致）: out[c*H*W + y*W + x] ∈ [-1,1]
  function resizeArea(src, sw, sh, dw, dh) {
    var plane = dw * dh;
    var out = new Float32Array(plane * 3);
    for (var dy = 0; dy < dh; dy++) {
      var y0 = dy * sh / dh, y1 = (dy + 1) * sh / dh;
      var iy0 = Math.floor(y0), iy1 = Math.min(sh, Math.ceil(y1));
      for (var dx = 0; dx < dw; dx++) {
        var x0 = dx * sw / dw, x1 = (dx + 1) * sw / dw;
        var ix0 = Math.floor(x0), ix1 = Math.min(sw, Math.ceil(x1));
        var r = 0, g = 0, b = 0, wt = 0;
        for (var y = iy0; y < iy1; y++) {
          var wy = Math.min(y + 1, y1) - Math.max(y, y0);
          if (wy <= 0) continue;
          for (var x = ix0; x < ix1; x++) {
            var wx = Math.min(x + 1, x1) - Math.max(x, x0);
            if (wx <= 0) continue;
            var wgt = wx * wy, p = (y * sw + x) * 3;
            r += src[p] * wgt; g += src[p + 1] * wgt; b += src[p + 2] * wgt;
            wt += wgt;
          }
        }
        var o = dy * dw + dx;
        out[o] = r / wt / 255 * 2 - 1;
        out[o + plane] = g / wt / 255 * 2 - 1;
        out[o + 2 * plane] = b / wt / 255 * 2 - 1;
      }
    }
    return out;
  }

  function conv3Relu(x, ci, h, w, L, out) {
    var co = L.cout, kw = L.w, kb = L.b;
    for (var o = 0; o < co; o++) {
      var ob = o * h * w;
      for (var y = 0; y < h; y++) {
        var ylo = y > 0 ? y - 1 : 0, yhi = y < h - 1 ? y + 1 : h - 1;
        for (var xx = 0; xx < w; xx++) {
          var xlo = xx > 0 ? xx - 1 : 0, xhi = xx < w - 1 ? xx + 1 : w - 1;
          var acc = kb[o];
          for (var c = 0; c < ci; c++) {
            var kc = (o * ci + c) * 9, ch = c * h * w;
            for (var dy = -1; dy <= 1; dy++) {
              var yy = y + dy; if (yy < ylo || yy > yhi) continue;
              for (var dx = -1; dx <= 1; dx++) {
                var xa = xx + dx; if (xa < xlo || xa > xhi) continue;
                acc += kw[kc + (dy + 1) * 3 + (dx + 1)] * x[ch + yy * w + xa];
              }
            }
          }
          out[ob + y * w + xx] = acc > 0 ? acc : 0;
        }
      }
    }
  }

  function maxPoolY2(x, c, h, w, out) {
    var oh = h >> 1;
    for (var i = 0; i < c; i++) {
      for (var y = 0; y < oh; y++) {
        for (var xx = 0; xx < w; xx++) {
          var a = x[(i * h + y * 2) * w + xx], b = x[(i * h + y * 2 + 1) * w + xx];
          out[(i * oh + y) * w + xx] = a > b ? a : b;
        }
      }
    }
    return oh;
  }

  function forward(featIn) {
    var cur = featIn, ci = 3, h = W.in_h, w = W.in_w;
    var li = 0;
    for (var l = 0; l < layers.length; l++) {
      var L = layers[l];
      if (L.k === 3) {
        var buf = new Float32Array(L.cout * h * w);
        conv3Relu(cur, ci, h, w, L, buf);
        cur = buf; ci = L.cout;
        var pooled = new Float32Array(ci * (h >> 1) * w);
        h = maxPoolY2(cur, ci, h, w, pooled);
        cur = pooled;
      } else { // 1x1
        var head = new Float32Array(h * w);
        for (var xx = 0; xx < w; xx++) {
          for (var y = 0; y < h; y++) {
            var acc = L.b[0];
            for (var c = 0; c < ci; c++) acc += L.w[c] * cur[(c * h + y) * w + xx];
            head[y * w + xx] = acc;
          }
        }
        // y 向 max
        var logits = new Float32Array(w);
        for (var x2 = 0; x2 < w; x2++) {
          var m = -1e30;
          for (var y2 = 0; y2 < h; y2++) m = head[y2 * w + x2] > m ? head[y2 * w + x2] : m;
          logits[x2] = m;
        }
        // softmax + soft-argmax
        var mx = -1e30;
        for (var i = 0; i < w; i++) mx = logits[i] > mx ? logits[i] : mx;
        var sum = 0, acc2 = 0, best = 0, bestI = 0;
        for (var j = 0; j < w; j++) {
          var p = Math.exp(logits[j] - mx);
          sum += p;
          acc2 += p * (j + 0.5);
          if (p > best) { best = p; bestI = j; }
        }
        return { xIn: acc2 / sum, conf: best / sum, peakBin: bestI };
      }
    }
    throw new Error('权重缺少 1x1 头');
  }

  function toRgb(pix, w, h) {
    // canvas ImageData(RGBA) 与训练端 RGB 均可
    if (pix.length === w * h * 4) {
      var rgb = new Uint8Array(w * h * 3);
      for (var i = 0, j = 0; i < pix.length; i += 4, j += 3) {
        rgb[j] = pix[i]; rgb[j + 1] = pix[i + 1]; rgb[j + 2] = pix[i + 2];
      }
      return rgb;
    }
    return pix;
  }

  function locate(pix, w, h) {
    if (!layers) throw new Error('SUFE_CNN_WEIGHTS 未加载');
    var feat = resizeArea(toRgb(pix, w, h), w, h, W.in_w, W.in_h);
    var r = forward(feat);
    return { x: r.xIn * (w / W.in_w), conf: r.conf, peakBin: r.peakBin };
  }

  // 逐层中间结果（调试/一致性测试用）
  function debugStages(pix, w, h) {
    if (!layers) throw new Error('SUFE_CNN_WEIGHTS 未加载');
    var feat = resizeArea(toRgb(pix, w, h), w, h, W.in_w, W.in_h);
    var stages = [Array.from(feat.slice(0, 30))];
    var cur = feat, ci = 3, hh = W.in_h, ww = W.in_w;
    var meta = [];
    for (var l = 0; l < layers.length; l++) {
      var L = layers[l];
      if (L.k === 3) {
        var buf = new Float32Array(L.cout * hh * ww);
        conv3Relu(cur, ci, hh, ww, L, buf);
        cur = buf; ci = L.cout;
        var pooled = new Float32Array(ci * (hh >> 1) * ww);
        hh = maxPoolY2(cur, ci, hh, ww, pooled);
        cur = pooled;
        meta.push({ stage: 'conv+pool#' + l, cout: ci, h: hh, w: ww, head: Array.from(cur.slice(0, 30)) });
      } else {
        var head = new Float32Array(hh * ww);
        for (var xx = 0; xx < ww; xx++) {
          for (var y = 0; y < hh; y++) {
            var acc = L.b[0];
            for (var c = 0; c < ci; c++) acc += L.w[c] * cur[(c * hh + y) * ww + xx];
            head[y * ww + xx] = acc;
          }
        }
        var logits = new Float32Array(ww);
        for (var x2 = 0; x2 < ww; x2++) {
          var m = -1e30;
          for (var y2 = 0; y2 < hh; y2++) m = head[y2 * ww + x2] > m ? head[y2 * ww + x2] : m;
          logits[x2] = m;
        }
        meta.push({ stage: 'logits', head: Array.from(logits.slice(0, 30)) });
      }
    }
    return { input_head30: stages[0], stages: meta };
  }

  return { locate: locate, debugStages: debugStages };
})();
if (typeof module !== 'undefined') module.exports = SufeCNN;

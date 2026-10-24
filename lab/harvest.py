"""Harvest captcha samples from the live API and label the gap via notch segmentation.

Label convention: stored X (what /check expects) = notch_left - piece_alpha_x0.
Polite rate limiting: random 1.2-2.4s between inits.
"""
import base64
import io
import json
import random
import sys
import time
import urllib.request

import numpy as np
from PIL import Image
from scipy import ndimage

BASE = "https://login.sufe.edu.cn/esc-sso"
OUT = "data/dataset.npz"
CANON_W = 512  # 统一宽度：>则左对齐裁剪，<则 reflect 补齐（不缩放，保持缺口物理尺寸）
BAND_H = 116   # 横带高度（Y-8 .. Y+108）


def canon(src):
    w = src.shape[1]
    if w >= CANON_W:
        return src[:, :CANON_W], w
    return np.pad(src, ((0, 0), (0, CANON_W - w), (0, 0)), mode="reflect"), w


def api(path, body=None, opener=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method="POST" if data else "GET",
        headers={"Accept": "application/json",
                 **({"Content-Type": "application/json"} if data else {})},
    )
    op = opener or urllib.request.urlopen
    with op(req, timeout=30) as r:
        return json.loads(r.read().decode())


def decode(b64):
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def notch_label(src_rgb):
    """Return (x0, y0, w, h, area) of the puzzle-shaped white notch, or None."""
    im = src_rgb.astype(np.int16)
    gray = im.mean(2)
    sat = im.max(2) - im.min(2)
    # local variance 3x3
    p = np.pad(gray, 1, mode="edge")
    acc = np.zeros_like(gray)
    for dy in range(3):
        for dx in range(3):
            acc += p[dy:dy + gray.shape[0], dx:dx + gray.shape[1]]
    mean = acc / 9.0
    p2 = np.pad(gray ** 2, 1, mode="edge")
    acc2 = np.zeros_like(gray)
    for dy in range(3):
        for dx in range(3):
            acc2 += p2[dy:dy + gray.shape[0], dx:dx + gray.shape[1]]
    var = np.maximum(acc2 / 9.0 - mean ** 2, 0)
    mask = (var < 6) & (sat < 16) & (gray > 200)
    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    best = None
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        a = len(ys)
        if a < 2000:
            continue
        w = xs.max() - xs.min() + 1
        h = ys.max() - ys.min() + 1
        fill = a / (w * h)
        # puzzle shape: ~90-100 px box, fill ratio ~0.6-0.8
        if 70 <= w <= 110 and 70 <= h <= 110 and 0.5 < fill < 0.9:
            score = a
            if best is None or score > best[4]:
                best = (int(xs.min()), int(ys.min()), int(w), int(h), int(a))
    return best


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    rows = []
    for k in range(n):
        t0 = time.time()
        try:
            d = api("/api/v3/sliderCaptcha/init", {})["data"]
        except Exception as e:
            print(f"[{k}] init error: {e}")
            time.sleep(3)
            continue
        src = np.asarray(decode(d["sourceImage"]).convert("RGB"))
        piece = decode(d["newImage"])
        pa = np.asarray(piece.getchannel("A"))
        pys, pxs = np.nonzero(pa > 0)
        px0 = int(pxs.min())
        lbl = notch_label(src)
        if lbl is None:
            print(f"[{k}] no notch found (Y={d['Y']})")
        else:
            x0, y0, w, h, a = lbl
            csrc, worig = canon(src)
            Yi = int(d["Y"])
            pad = np.pad(csrc, ((BAND_H, BAND_H), (0, 0), (0, 0)), mode="edge")
            band = pad[Yi - 8 + BAND_H:Yi - 8 + 2 * BAND_H]
            rows.append(dict(src=band, W=worig, Y=8, Yorig=Yi, nx=x0, ny=y0,
                             nw=w, nh=h, px0=px0, X=x0 - px0))
            ok_y = "Ymatch" if abs(y0 - Yi) <= 2 else f"YDIFF y0={y0} vs Y={Yi}"
            print(f"[{k}] W={worig} notch=({x0},{y0},{w}x{h},a={a}) px0={px0} -> X={x0-px0}  {ok_y}  "
                  f"({time.time()-t0:.1f}s)")
        time.sleep(random.uniform(1.2, 2.4))
    if not rows:
        print("no rows harvested")
        return
    np.savez_compressed(
        OUT,
        src=np.stack([r["src"] for r in rows]),
        W=np.array([r["W"] for r in rows]),
        Y=np.array([r["Y"] for r in rows]),
        Yorig=np.array([r["Yorig"] for r in rows]),
        X=np.array([r["X"] for r in rows]),
        nx=np.array([r["nx"] for r in rows]),
        ny=np.array([r["ny"] for r in rows]),
        nw=np.array([r["nw"] for r in rows]),
        px0=np.array([r["px0"] for r in rows]),
    )
    X = np.array([r["X"] for r in rows])
    print(f"\nsaved {len(rows)} samples to {OUT}")
    print("X stats: min=%d max=%d mean=%.1f std=%.1f" % (X.min(), X.max(), X.mean(), X.std()))
    print("X histogram:", np.histogram(X, bins=10, range=(0, 512))[0].tolist())


if __name__ == "__main__":
    main()

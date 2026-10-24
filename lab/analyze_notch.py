"""Analyze the notch (gap) pixel signature in a source image."""
import sys
import numpy as np
from PIL import Image

path = sys.argv[1]
im = np.asarray(Image.open(path).convert("RGB")).astype(np.int16)
H, W, _ = im.shape
print("size", W, H)

r, g, b = im[..., 0], im[..., 1], im[..., 2]
mx = im.max(2)
mn = im.min(2)
sat = mx - mn  # chroma spread; gray fill => low

# local variance via box filter (3x3)
def box(a, k=3):
    p = np.pad(a, k // 2, mode="edge")
    s = p.copy()
    for dy in range(k):
        for dx in range(k):
            pass
    acc = np.zeros_like(a, dtype=np.float64)
    for dy in range(k):
        for dx in range(k):
            acc += p[dy:dy + a.shape[0], dx:dx + a.shape[1]]
    return acc / (k * k)

mean = box(im.mean(2))
sq = box((im.mean(2)) ** 2)
var = np.maximum(sq - mean ** 2, 0)

# candidate notch mask: flat AND desaturated AND brightish
mask = (var < 8) & (sat < 18) & (mean > 150)
ys, xs = np.nonzero(mask)
print("mask px:", mask.sum())
if mask.sum():
    print("bbox x:[%d,%d] y:[%d,%d]" % (xs.min(), xs.max(), ys.min(), ys.max()))
    # largest connected-ish: histogram of x
    import collections
    # sample colors inside bbox center
    cx, cy = int(xs.mean()), int(ys.mean())
    print("center color:", im[cy, cx], "var:", var[cy, cx], "sat:", sat[cy, cx])

# report stats of a few regions for comparison
for name, (x0, x1, y0, y1) in {
    "notch-ish": (110, 190, 140, 220),
    "water": (300, 450, 20, 120),
    "rock": (200, 300, 60, 140),
}.items():
    sub_v = var[y0:y1, x0:x1]
    sub_s = sat[y0:y1, x0:x1]
    sub_m = mean[y0:y1, x0:x1]
    print(f"{name:10s} var mean={sub_v.mean():8.1f} sat mean={sub_s.mean():6.1f} bright={sub_m.mean():6.1f}")

"""Verify the X mapping against the live check oracle on one fresh captcha."""
import json
import sys

import numpy as np

from harvest import api, decode, notch_label

d = api("/api/v3/sliderCaptcha/init", {})["data"]
src = np.asarray(decode(d["sourceImage"]).convert("RGB"))
piece = decode(d["newImage"])
pa = np.asarray(piece.getchannel("A"))
pys, pxs = np.nonzero(pa > 0)
px0, py0 = int(pxs.min()), int(pys.min())
lbl = notch_label(src)
print("piece alpha bbox x0,y0 =", px0, py0, " size=", piece.size)
print("notch =", lbl, " Y =", d["Y"], " token =", d["token"])
if lbl is None:
    sys.exit("no notch")
x0, y0, w, h, a = lbl
cands = {
    "notch-px0": x0 - px0,
    "notch": x0,
    "notch-px0-center": x0 - px0 + w // 2,
}
pick = sys.argv[1] if len(sys.argv) > 1 else "notch-px0"
X = cands[pick]
r = api(f"/api/v3/sliderCaptcha/check?X={X}&Y={d['Y']}&token={d['token']}")
print(f"hypothesis {pick}: X={X} -> {json.dumps(r)[:200]}")

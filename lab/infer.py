"""Inference with exported weights: mirrors train.py prep exactly (no augmentation)."""
import base64
import json

import numpy as np

from cnn import Net, softmax
from train import prep, NCOL, STRIDE


def load_net(path="data/cnn_weights.json"):
    j = json.load(open(path, encoding="utf-8"))
    net = Net()
    for li, L in enumerate(net.layers):
        if hasattr(L, "W"):
            W = np.frombuffer(base64.b64decode(j["w"][f"l{li}.W"]["b64"]), np.float32)
            b = np.frombuffer(base64.b64decode(j["w"][f"l{li}.b"]["b64"]), np.float32)
            L.W = W.reshape(j["w"][f"l{li}.W"]["shape"]).astype(np.float32)
            L.b = b.astype(np.float32)
    return net


def predict(net, src, Y, origW=None):
    """与 sufe-cnn.js 的 predict 完全一致：origW 给定时屏蔽不可能的缺口列。"""
    x0 = prep(src, int(Y))[None]
    logits = net.forward(x0)[0].copy()
    if origW is not None:
        max_col = int((origW - 92) // STRIDE)
        logits[max_col + 1:] = -1e9
    probs = softmax(logits)
    c = int(probs.argmax())
    lo, hi = max(0, c - 2), min(NCOL, c + 3)
    w = probs[lo:hi]
    xc = (np.arange(lo, hi) * w).sum() / w.sum()
    xpix = xc * STRIDE
    return {"x": float(xpix), "X": int(round(xpix)) - 2, "col": c, "conf": float(probs[c])}


if __name__ == "__main__":
    import sys
    net = load_net()
    d = np.load("data/dataset.npz")
    n = len(d["src"])
    idxs = list(range(min(n, 12)))
    out = []
    for i in idxs:
        r = predict(net, d["src"][i], d["Y"][i])
        r["i"] = i
        r["trueX"] = int(d["X"][i])
        out.append(r)
        print(f"[{i}] pred x={r['x']:.2f} X={r['X']} true X={r['trueX']} "
              f"err={r['X']-r['trueX']:+d} conf={r['conf']:.3f}")
    if sys.argv[1:]:
        with open(sys.argv[1], "w") as f:
            json.dump(out, f)

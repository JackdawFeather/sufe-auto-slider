# -*- coding: utf-8 -*-
"""诊断：检查点模型在训练集/验证集上的表现（区分记忆 vs 泛化）。
用法: python lab/debug_vcode.py
"""
import json
import os
import random
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_vcode import CHARSET, Net, decode_greedy, to_input  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, "samples", "vcode")
OUT = os.path.join(HERE, "output", "vcode")

labels = json.load(open(os.path.join(OUT, "labels.json"), encoding="utf-8"))
items = []
for f in sorted(labels):
    t = str(labels[f]["text"]).lower()
    if len(t) == 4 and all(c in CHARSET for c in t):
        items.append((f, t))
random.seed(7)
random.shuffle(items)
nval = max(60, len(items) // 10)
val, train = items[:nval], items[nval:]

net = Net(len(CHARSET) + 1)
ck = torch.load(os.path.join(OUT, "ckpt.pt"), map_location="cpu")
net.load_state_dict(ck["net"])
net.eval()
print("ckpt epoch:", ck["ep"])


def acc(subset, name, n=60):
    ok = cok = tot = 0
    shown = []
    with torch.no_grad():
        for f, t in subset[:n]:
            img = cv2.imread(os.path.join(SAMPLES, f))
            x = torch.from_numpy(to_input(img)[None, None].astype(np.float32))
            pr = decode_greedy(net(x)[:, 0, :].numpy())
            ok += (pr == t)
            cok += sum(1 for a, b in zip(pr, t) if a == b)
            tot += len(t)
            if len(shown) < 8:
                shown.append(t + "→" + pr)
    print("%s(n=%d): 整串 %d = %.0f%% | 字符 %.0f%% | %s" % (
        name, min(n, len(subset)), ok, 100.0 * ok / max(1, min(n, len(subset))),
        100.0 * cok / max(1, tot), " ".join(shown)))


acc(train, "训练集")
acc(val, "验证集")

# 单样本逐列 argmax
f, t = val[0]
img = cv2.imread(os.path.join(SAMPLES, f))
with torch.no_grad():
    x = torch.from_numpy(to_input(img)[None, None].astype(np.float32))
    col = net(x)[:, 0, :]
seq = col.argmax(dim=1).tolist()
print("样本", t, "列数", len(seq), "argmax:", seq)

# -*- coding: utf-8 -*-
"""管线自检：30 张样本过拟合（无增强），应很快 100% 整串正确。
用法: python lab/train_vcode_overfit.py
"""
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_vcode import CHARSET, Net, decode_greedy, encode, to_input  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, "samples", "vcode")
OUT = os.path.join(HERE, "output", "vcode")

labels = json.load(open(os.path.join(OUT, "labels.json"), encoding="utf-8"))
picked = []
for f in sorted(labels):
    t = str(labels[f]["text"]).lower()
    if len(t) == 4 and all(c in CHARSET for c in t):
        picked.append((f, t))
    if len(picked) >= 30:
        break

xs, ys, yl = [], [], []
for f, t in picked:
    img = cv2.imread(os.path.join(SAMPLES, f))
    xs.append(to_input(img)[None, :, :])
    ys += encode(t)
    yl.append(4)
x = torch.from_numpy(np.stack(xs).astype(np.float32))
y = torch.tensor(ys, dtype=torch.long)
yl = torch.tensor(yl, dtype=torch.long)

net = Net(len(CHARSET) + 1)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
ctc = nn.CTCLoss(blank=0, zero_infinity=True)
for ep in range(1, 101):
    net.train()
    logits = net(x)
    loss = ctc(logits, y, torch.full((x.shape[0],), logits.shape[0], dtype=torch.long), yl)
    opt.zero_grad(); loss.backward(); opt.step()
    if ep % 20 == 0 or ep == 1:
        net.eval()
        with torch.no_grad():
            pred = [decode_greedy(net(x)[:, i, :].numpy()) for i in range(len(picked))]
        ok = sum(p == t for p, (_, t) in zip(pred, picked))
        print("ep %d loss %.3f ok %d/%d" % (ep, loss.item(), ok, len(picked)), flush=True)
        if ok == len(picked):
            break

net.eval()
with torch.no_grad():
    logits = net(x)
for i in range(6):
    col = logits[:, i, :]
    seq = col.argmax(dim=1).tolist()
    print("label=%s pred=%r len=%d | argmax前40列=%s" % (
        picked[i][1], decode_greedy(col.numpy()), len(decode_greedy(col.numpy())), seq[:40]), flush=True)


# -*- coding: utf-8 -*-
"""训练图片验证码 OCR（CRNN + CTC，torch CPU），导出纯 JS 权重。

输入: 80x34 JPEG → 灰度 → area-resize 128x48 → [-1,1]
结构: Conv3x3(1→32)+ReLU+MaxPool2x2 → Conv3x3(32→64)+ReLU+MaxPool(2,1)
      → Conv3x3(64→96)+ReLU+MaxPool(2,1) → 高度方向 max 折叠 → Conv1x1(96→37) → CTC
标签: ddddocr 标签折叠小写（服务端验证码不区分大小写，实测佐证）
导出: lab/output/vcode/weights.json + ../sufe-vcode-weights.js

用法: python lab/train_vcode.py [epochs=50]
"""
import base64
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, "samples", "vcode")
OUT = os.path.join(HERE, "output", "vcode")
CHARSET = "0123456789abcdefghijklmnopqrstuvwxyz"  # blank = 0，字符从 1 开始
IN_W, IN_H = 128, 48


_SIZE_CACHE = {}


def resize_matrices(sw, sh):
    """可分离面积平均矩阵（与 sufe-cnn.js resizeArea 同一数学），按源尺寸缓存。"""
    key = (sw, sh)
    if key in _SIZE_CACHE:
        return _SIZE_CACHE[key]
    Wx = np.zeros((IN_W, sw), np.float32)
    for dx in range(IN_W):
        x0, x1 = dx * sw / IN_W, (dx + 1) * sw / IN_W
        ix0, ix1 = int(np.floor(x0)), min(sw, int(np.ceil(x1)))
        for x in range(ix0, ix1):
            Wx[dx, x] = (min(x + 1, x1) - max(x, x0)) / (x1 - x0)
    Wy = np.zeros((IN_H, sh), np.float32)
    for dy in range(IN_H):
        y0, y1 = dy * sh / IN_H, (dy + 1) * sh / IN_H
        iy0, iy1 = int(np.floor(y0)), min(sh, int(np.ceil(y1)))
        for y in range(iy0, iy1):
            Wy[dy, y] = (min(y + 1, y1) - max(y, y0)) / (y1 - y0)
    _SIZE_CACHE[key] = (Wy, Wx)
    return Wy, Wx


def to_input(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    Wy, Wx = resize_matrices(g.shape[1], g.shape[0])
    r = Wy @ g @ Wx.T
    return (r - 0.5) / 0.5


def augment(img):
    h, w = img.shape[:2]
    ang = random.uniform(-2, 2)
    scale = random.uniform(0.96, 1.04)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, scale)
    M[0, 2] += random.uniform(-1.5, 1.5)
    M[1, 2] += random.uniform(-1.5, 1.5)
    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if random.random() < 0.5:
        alpha = random.uniform(0.9, 1.1)
        beta = random.uniform(-12, 12)
        out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return out


class Net(nn.Module):
    """CRNN：3 个卷积块（BN 在导出时折叠进卷积），高度 max 折叠 → 1x1 分类头 → CTC。"""

    def __init__(self, nclass):
        super().__init__()
        self.c1 = nn.Conv2d(1, 24, 3, padding=1)
        self.n1 = nn.BatchNorm2d(24)
        self.c2 = nn.Conv2d(24, 48, 3, padding=1)
        self.n2 = nn.BatchNorm2d(48)
        self.c3 = nn.Conv2d(48, 48, 3, padding=1)
        self.n3 = nn.BatchNorm2d(48)
        self.c4 = nn.Conv2d(48, 96, 3, padding=1)
        self.n4 = nn.BatchNorm2d(96)
        self.c5 = nn.Conv2d(96, 96, 3, padding=1)
        self.n5 = nn.BatchNorm2d(96)
        self.head = nn.Conv2d(96, nclass, 1)

    def forward(self, x):
        x = torch.nn.functional.max_pool2d(torch.relu(self.n1(self.c1(x))), (2, 2))
        x = torch.relu(self.n2(self.c2(x)))
        x = torch.nn.functional.max_pool2d(torch.relu(self.n3(self.c3(x))), (2, 2))
        x = torch.relu(self.n4(self.c4(x)))
        x = torch.nn.functional.max_pool2d(torch.relu(self.n5(self.c5(x))), (2, 1))
        x = x.max(dim=2).values                          # B,96,W
        x = self.head(x.unsqueeze(2)).squeeze(2)         # B,nclass,W
        return torch.log_softmax(x.permute(2, 0, 1), dim=2)  # T,B,C


def encode(text):
    return [CHARSET.index(c) + 1 for c in text]


def decode_greedy(logits):
    """logits: T x C (numpy) → 文本"""
    idx = logits.argmax(axis=1)
    out, prev = [], -1
    for i in idx:
        if i != prev and i != 0:
            out.append(CHARSET[i - 1])
        prev = i
    return "".join(out)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--no-aug", action="store_true", help="关闭数据增强（收敛诊断用）")
    ap.add_argument("--resume", action="store_true", help="从 lab/output/vcode/ckpt.pt 续训")
    args = ap.parse_args()
    epochs = args.epochs
    labels = json.load(open(os.path.join(OUT, "labels.json"), encoding="utf-8"))
    files = sorted(f for f in labels if os.path.exists(os.path.join(SAMPLES, f)))
    data = []
    dropped = 0
    for f in files:
        t = str(labels[f]["text"]).lower()
        if len(t) == 4 and all(c in CHARSET for c in t):
            data.append((f, t))
        else:
            dropped += 1
    print("样本: 可用 %d / %d（丢弃 %d）" % (len(data), len(files), dropped), flush=True)

    random.seed(7)
    random.shuffle(data)
    nval = max(60, len(data) // 10)
    val, train = data[:nval], data[nval:]
    torch.manual_seed(7)
    net = Net(len(CHARSET) + 1)
    print("参数量: %.1fk" % (sum(p.numel() for p in net.parameters()) / 1000), flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(10, epochs // 3), gamma=0.5)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    bs = args.bs
    start_ep = 0
    ck = os.path.join(OUT, "ckpt.pt")
    if args.resume and os.path.exists(ck):
        st = torch.load(ck, map_location="cpu")
        net.load_state_dict(st["net"])
        opt.load_state_dict(st["opt"])
        start_ep = st["ep"]
        print("续训：从 epoch", start_ep + 1, "开始", flush=True)
    t0 = time.time()
    for ep in range(start_ep + 1, epochs + 1):
        net.train()
        random.shuffle(train)
        tot, nb = 0.0, 0
        for i in range(0, len(train), bs):
            batch = train[i:i + bs]
            xs, ys, ylens = [], [], []
            for f, t in batch:
                img = cv2.imread(os.path.join(SAMPLES, f))
                if img is None:
                    continue
                xs.append(to_input(img if args.no_aug else augment(img))[None, :, :])
                ys += encode(t)
                ylens.append(4)
            if not xs:
                continue
            x = torch.from_numpy(np.stack(xs).astype(np.float32))
            y = torch.tensor(ys, dtype=torch.long)
            yl = torch.tensor(ylens, dtype=torch.long)
            logits = net(x)
            loss = ctc(logits, y, torch.full((x.shape[0],), logits.shape[0], dtype=torch.long), yl)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "ep": ep}, ck)
        if ep % 5 == 0 or ep == 1:
            net.eval()
            ok = cok = tot_ch = 0
            samples = []
            with torch.no_grad():
                for idx, (f, t) in enumerate(val):
                    img = cv2.imread(os.path.join(SAMPLES, f))
                    x = torch.from_numpy(to_input(img)[None, None].astype(np.float32))
                    pr = decode_greedy(net(x)[:, 0, :].numpy())
                    ok += (pr == t)
                    cok += sum(1 for a, b in zip(pr, t) if a == b)
                    tot_ch += len(t)
                    if idx < 3:
                        samples.append(t + "→" + pr)
            print("epoch %d/%d loss %.3f | val整串 %d/%d = %.1f%% | 字符 %.1f%% | %s | %.0fs" % (
                ep, epochs, tot / max(1, nb), ok, len(val), 100.0 * ok / len(val),
                100.0 * cok / max(1, tot_ch), " ".join(samples), time.time() - t0), flush=True)

    # 导出权重：BN 折叠进卷积；ops 顺序 = conv/pool/collapse/head
    net.eval()

    def fold(conv, bn):
        w = conv.weight.detach().numpy()
        b = conv.bias.detach().numpy()
        gamma = bn.weight.detach().numpy()
        beta = bn.bias.detach().numpy()
        scale = gamma / np.sqrt(bn.running_var.numpy() + bn.eps)
        w2 = (w * scale[:, None, None, None]).astype(np.float32)
        b2 = ((b - bn.running_mean.numpy()) * scale + beta).astype(np.float32)
        return w2, b2

    def conv_op(conv, bn=None, relu=True):
        w, b = fold(conv, bn) if bn is not None else (conv.weight.detach().numpy().astype(np.float32),
                                                      conv.bias.detach().numpy().astype(np.float32))
        return {"t": "c", "k": int(w.shape[2]), "cin": int(w.shape[1]), "cout": int(w.shape[0]),
                "relu": relu,
                "w": base64.b64encode(w.reshape(w.shape[0], -1).tobytes()).decode(),
                "b": base64.b64encode(b.tobytes()).decode()}

    ops = [
        conv_op(net.c1, net.n1), {"t": "p", "k": [2, 2]},
        conv_op(net.c2, net.n2), conv_op(net.c3, net.n3), {"t": "p", "k": [2, 2]},
        conv_op(net.c4, net.n4), conv_op(net.c5, net.n5), {"t": "p", "k": [2, 1]},
        {"t": "x"},                       # 高度方向 max 折叠
        conv_op(net.head, None, relu=False),
    ]
    blob = {"meta": {"in_w": IN_W, "in_h": IN_H, "charset": CHARSET, "blank": 0}, "ops": ops}
    json.dump(blob, open(os.path.join(OUT, "weights.json"), "w"), separators=(",", ":"))
    js = os.path.join(HERE, "..", "sufe-vcode-weights.js")
    with open(js, "w", encoding="utf-8") as f:
        f.write("/* 训练产物：图片验证码 OCR 权重（lab/train_vcode.py 导出，勿手改） */\n")
        f.write("window.SUFE_VCODE_WEIGHTS = " + json.dumps(blob, separators=(",", ":")) + ";\n")

    # parity fixtures：留出集前 8 张的原始 RGBA + python 预测
    fx = []
    for f, t in val[:8]:
        img = cv2.imread(os.path.join(SAMPLES, f))
        h, w = img.shape[:2]
        rgba = np.dstack([cv2.cvtColor(img, cv2.COLOR_BGR2RGB), np.full((h, w), 255, np.uint8)]).tobytes()
        with open(os.path.join(OUT, "parity_%s.bin" % f.replace(".jpg", "")), "wb") as fp:
            fp.write(rgba)
        with torch.no_grad():
            x = torch.from_numpy(to_input(img)[None, None].astype(np.float32))
            py = decode_greedy(net(x)[:, 0, :].detach().numpy())
        fx.append({"file": f, "w": w, "h": h, "label": t, "py": py})
    json.dump(fx, open(os.path.join(OUT, "parity.json"), "w"), ensure_ascii=False, indent=1)
    print("已导出 sufe-vcode-weights.js（%.0f KB）" % (os.path.getsize(js) / 1024), flush=True)


if __name__ == "__main__":
    main()

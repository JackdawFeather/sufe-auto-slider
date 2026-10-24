# -*- coding: utf-8 -*-
"""对照实验：4 位置分类头（每位置 CE），检验字形特征能否学到。

用法: python lab/train_vcode4.py [--epochs 15] [--lr 2e-3] [--no-aug]
"""
import argparse
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_vcode import CHARSET, IN_H, IN_W, augment, to_input  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, "samples", "vcode")
OUT = os.path.join(HERE, "output", "vcode")


class Net4(nn.Module):
    def __init__(self, nclass, width=24):
        super().__init__()
        w = width
        self.c1 = nn.Conv2d(1, w, 3, padding=1)
        self.n1 = nn.BatchNorm2d(w)
        self.c2 = nn.Conv2d(w, 2 * w, 3, padding=1)
        self.n2 = nn.BatchNorm2d(2 * w)
        self.c3 = nn.Conv2d(2 * w, 2 * w, 3, padding=1)
        self.n3 = nn.BatchNorm2d(2 * w)
        self.c4 = nn.Conv2d(2 * w, 4 * w, 3, padding=1)
        self.n4 = nn.BatchNorm2d(4 * w)
        self.c5 = nn.Conv2d(4 * w, 4 * w, 3, padding=1)
        self.n5 = nn.BatchNorm2d(4 * w)
        self.head = nn.Linear(4 * w, nclass)

    def forward(self, x):
        x = torch.nn.functional.max_pool2d(torch.relu(self.n1(self.c1(x))), (2, 2))
        x = torch.relu(self.n2(self.c2(x)))
        x = torch.nn.functional.max_pool2d(torch.relu(self.n3(self.c3(x))), (2, 2))
        x = torch.relu(self.n4(self.c4(x)))
        x = torch.nn.functional.max_pool2d(torch.relu(self.n5(self.c5(x))), (2, 1))
        x = x.max(dim=2).values                     # B,96,32
        W = x.shape[2]
        bw = W // 4
        outs = []
        for i in range(4):
            lo, hi = i * bw, (i + 1) * bw if i < 3 else W
            v = x[:, :, lo:hi].max(dim=2).values    # B,96
            outs.append(self.head(v))               # B,36
        return torch.stack(outs, dim=1)             # B,4,36


def decode(logits):
    idx = logits.argmax(dim=2)[0].tolist()
    return "".join(CHARSET[i] for i in idx), float(torch.softmax(logits, dim=2).max(dim=2).values.mean())


def export(net, val):
    """导出 slots 格式权重（BN 折叠）+ JS 全局文件 + parity fixtures。"""
    net.eval()

    def fold(conv, bn):
        w = conv.weight.detach().numpy()
        b = conv.bias.detach().numpy()
        scale = bn.weight.detach().numpy() / np.sqrt(bn.running_var.numpy() + bn.eps)
        w2 = (w * scale[:, None, None, None]).astype(np.float32)
        b2 = ((b - bn.running_mean.numpy()) * scale + bn.bias.detach().numpy()).astype(np.float32)
        return w2, b2

    def conv_op(conv, bn):
        w, b = fold(conv, bn)
        return {"t": "c", "k": int(w.shape[2]), "cin": int(w.shape[1]), "cout": int(w.shape[0]),
                "relu": True,
                "w": base64.b64encode(w.reshape(w.shape[0], -1).tobytes()).decode(),
                "b": base64.b64encode(b.tobytes()).decode()}

    ops = [conv_op(net.c1, net.n1), {"t": "p", "k": [2, 2]},
           conv_op(net.c2, net.n2), conv_op(net.c3, net.n3), {"t": "p", "k": [2, 2]},
           conv_op(net.c4, net.n4), conv_op(net.c5, net.n5), {"t": "p", "k": [2, 1]},
           {"t": "x"}]
    hw = net.head.weight.detach().numpy().astype(np.float32)
    hb = net.head.bias.detach().numpy().astype(np.float32)
    head = {"cin": int(hw.shape[1]), "cout": int(hw.shape[0]),
            "w": base64.b64encode(hw.tobytes()).decode(),
            "b": base64.b64encode(hb.tobytes()).decode()}
    blob = {"meta": {"in_w": IN_W, "in_h": IN_H, "charset": CHARSET, "mode": "slots", "bins": 4},
            "ops": ops, "head": head}
    json.dump(blob, open(os.path.join(OUT, "weights4.json"), "w"), separators=(",", ":"))
    js = os.path.join(HERE, "..", "sufe-vcode-weights.js")
    with open(js, "w", encoding="utf-8") as f:
        f.write("/* 训练产物：图片验证码 OCR 权重（lab/train_vcode4.py 导出，勿手改） */\n")
        f.write("window.SUFE_VCODE_WEIGHTS = " + json.dumps(blob, separators=(",", ":")) + ";\n")

    fx = []
    with torch.no_grad():
        for f, t in val[:8]:
            img = cv2.imread(os.path.join(SAMPLES, f))
            h, w = img.shape[:2]
            rgba = np.dstack([cv2.cvtColor(img, cv2.COLOR_BGR2RGB), np.full((h, w), 255, np.uint8)]).tobytes()
            with open(os.path.join(OUT, "parity_%s.bin" % f.replace(".jpg", "")), "wb") as fp:
                fp.write(rgba)
            x = torch.from_numpy(to_input(img)[None, None].astype(np.float32))
            pr, _ = decode(net(x))
            fx.append({"file": f, "w": w, "h": h, "label": t, "py": pr})
    json.dump(fx, open(os.path.join(OUT, "parity.json"), "w"), ensure_ascii=False, indent=1)
    print("已导出 sufe-vcode-weights.js（%.0f KB，slots 格式）" % (os.path.getsize(js) / 1024), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--width", type=int, default=24, help="主干基础通道（24=标准，16≈提速 2 倍）")
    args = ap.parse_args()

    labels = json.load(open(os.path.join(OUT, "labels.json"), encoding="utf-8"))
    data = []
    for f in sorted(labels):
        t = str(labels[f]["text"]).lower()
        if len(t) == 4 and all(c in CHARSET for c in t):
            data.append((f, t))
    random.seed(7)
    random.shuffle(data)
    # 黄金人工标注集不进训练（保证评估诚实）
    golden_path = os.path.join(OUT, "golden_files.txt")
    if os.path.exists(golden_path):
        gold = set(open(golden_path).read().split())
        before = len(data)
        data = [d for d in data if d[0] not in gold]
        print("排除黄金集 %d 张（剩 %d）" % (before - len(data), len(data)), flush=True)
    nval = max(60, len(data) // 10)
    val, train = data[:nval], data[nval:]
    print("训练 %d / 验证 %d" % (len(train), len(val)), flush=True)

    net = Net4(len(CHARSET), args.width)
    print("参数量 %.1fk" % (sum(p.numel() for p in net.parameters()) / 1000), flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(8, args.epochs // 3), gamma=0.5)
    ce = nn.CrossEntropyLoss()
    bs = 64
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        net.train()
        random.shuffle(train)
        tot = nb = 0
        for i in range(0, len(train), bs):
            batch = train[i:i + bs]
            xs, ys = [], []
            for f, t in batch:
                img = cv2.imread(os.path.join(SAMPLES, f))
                xs.append(to_input(img if args.no_aug else augment(img))[None, :, :])
                ys.append([CHARSET.index(c) for c in t])
            x = torch.from_numpy(np.stack(xs).astype(np.float32))
            y = torch.tensor(ys, dtype=torch.long)
            logits = net(x)                          # B,4,36
            loss = sum(ce(logits[:, i, :], y[:, i]) for i in range(4)) / 4
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        torch.save({"net": net.state_dict(), "ep": ep}, os.path.join(OUT, "ckpt4.pt"))
        if ep % 3 == 0 or ep == 1:
            net.eval()
            with torch.no_grad():
                def ev(sub):
                    ok = cok = 0
                    for f, t in sub[:100]:
                        img = cv2.imread(os.path.join(SAMPLES, f))
                        x = torch.from_numpy(to_input(img)[None, None].astype(np.float32))
                        pr, _ = decode(net(x))
                        ok += (pr == t)
                        cok += sum(1 for a, b in zip(pr, t) if a == b)
                    n = min(100, len(sub))
                    return ok, n, cok
                vok, vn, vc = ev(val)
                tok, tn, tc = ev(train)
            print("ep %d loss %.2f | val整串 %d/%d=%.0f%% 字符%.0f%% | train整串 %d/%d=%.0f%% 字符%.0f%% | %.0fs" % (
                ep, tot / max(1, nb), vok, vn, 100.0 * vok / vn, 100.0 * vc / (4 * vn),
                tok, tn, 100.0 * tok / tn, 100.0 * tc / (4 * tn), time.time() - t0), flush=True)

    export(net, val)
    # 保存带尺寸/宽度标识的检查点，并做黄金集评估（黄金集已从训练中排除）
    import shutil
    shutil.copy(os.path.join(OUT, "ckpt4.pt"),
                os.path.join(OUT, "ckpt4_w%d_%dx%d.pt" % (args.width, IN_W, IN_H)))
    gp = os.path.join(OUT, "golden.json")
    if os.path.exists(gp):
        g = json.load(open(gp, encoding="utf-8"))
        net.eval()
        ok = 0
        with torch.no_grad():
            for gf, gt in g.items():
                im = cv2.imread(os.path.join(SAMPLES, gf))
                x = torch.from_numpy(to_input(im)[None, None].astype(np.float32))
                pr, _ = decode(net(x))
                ok += (pr == gt)
        print("黄金集: %d/%d = %.0f%%（in %dx%d, width=%d）" % (
            ok, len(g), 100.0 * ok / len(g), IN_W, IN_H, args.width), flush=True)


if __name__ == "__main__":
    main()

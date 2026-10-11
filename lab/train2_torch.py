"""整图缺口模型 · torch(CPU) 训练版（远端大机器用）。

与 train2.py 同架构同标签（窗口中心 = 缺口左缘-3，输入 256x128 整图 resize），
但 conv 走 oneDNN、增强在张量上按批做（每轮重新生成），速度比 numpy 手写 im2col 快一个量级。
注意：torch Conv2d padding=1 是**零填充**，扩展端 sufe-cnn.js 已相应改为零填充（edge→zero）。

用法: python train2_torch.py [epochs]
环境变量: SUFE_CHANNELS(默认 24,32,48,64) SUFE_BS(128) SUFE_EPOCHS ...
"""
import base64
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from resize import area_resize
from train2 import LABEL_SHIFT, load_samples, notch_mask_in

HERE = os.path.dirname(os.path.abspath(__file__))
IN_W, IN_H = 256, 128
NCOL = IN_W
CACHE = os.path.join(HERE, "output", "prep_cache.npz")


def load_cached():
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        return d["X"], d["C"], list(d["aux"]), d["mask"]
    X, C, aux = load_samples()
    mask = np.stack([notch_mask_in(m) for m in aux])
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, X=X, C=C, aux=np.array(aux, dtype=object), mask=mask)
    return X, C, aux, mask


class GapNet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        layers, ci = [], 3
        for co in channels:
            layers += [nn.Conv2d(ci, co, 3, padding=1, bias=True),
                       nn.ReLU(inplace=True),
                       nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))]
            ci = co
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(ci, 1, 1, bias=True)

    def forward(self, x):
        f = self.body(x)                 # (B,C,8,W)
        f = f.amax(dim=2)                # 高度方向全池化 → (B,C,W)
        logits = self.head(f.unsqueeze(2)).squeeze(2).squeeze(1)  # (B,W)
        return logits


def soft_target(cols, ncol=NCOL, sigma=1.2):
    idx = torch.arange(ncol, dtype=torch.float32, device=cols.device)
    t = torch.exp(-0.5 * ((idx[None, :] - cols[:, None]) / sigma) ** 2)
    return t / t.sum(1, keepdim=True)


def augment_torch(x, c, mask, rng, flip_fix=None):
    """x (B,3,H,W) float [-1,1]，c (B,) 列坐标（缺口**左缘**-3 的输入列坐标），mask (B,H,W) bool。
    flip_fix (B,)=(nw+2*LABEL_SHIFT)*scale：翻转后左缘 = IN_W - c - fix。"""
    B, _, H, W = x.shape
    dev = x.device
    # 水平翻转
    flip = torch.rand(B, device=dev) < 0.5
    if flip.any() and flip_fix is not None:
        x[flip] = x[flip].flip(3)
        c[flip] = W - c[flip] - flip_fix[flip]
        mask[flip] = mask[flip].flip(2)
    # 水平平移 ±8
    k = torch.randint(-8, 9, (B,), device=dev)
    for b in range(B):
        kk = int(k[b])
        if kk:
            x[b] = torch.roll(x[b], kk, dims=2)
            if kk > 0:
                x[b, :, :, :kk] = x[b, :, :, kk:kk + 1]
            else:
                x[b, :, :, kk:] = x[b, :, :, kk - 1:kk]
            mask[b] = torch.roll(mask[b], kk, dims=1)
    c += k.float()
    # 光度
    a = (torch.rand(B, 1, 1, 1, device=dev) * 0.6 + 0.7)
    bb = (torch.rand(B, 1, 1, 1, device=dev) * 0.5 - 0.25)
    x = torch.clamp(x * a + bb, -1, 1)
    g = torch.rand(B, 1, 1, 1, device=dev) * 0.4 + 0.6
    x = torch.sign(x) * torch.abs(x) ** g
    # 噪声
    sig = torch.rand(B, 1, 1, 1, device=dev) * 0.05 + 0.01
    x = torch.clamp(x + torch.randn_like(x) * sig, -1, 1)
    # 缺口改色（去纯白依赖）
    recolor = (torch.rand(B, 1, 1, 1, device=dev) < 0.5)
    lvl = torch.rand(B, 1, 1, 1, device=dev) * 1.0 - 0.1
    x = torch.where(recolor & mask.unsqueeze(1), lvl.expand_as(x), x)
    # 亮块干扰（雪/云相邻的困难样本）
    dist = (torch.rand(B, device=dev) < 0.3)
    for b in torch.nonzero(dist).flatten().tolist():
        for _ in range(3):
            h = int(torch.randint(6, 30, (1,))); w = int(torch.randint(10, 40, (1,)))
            y0 = int(torch.randint(0, H - h, (1,))); x0 = int(torch.randint(0, W - w, (1,)))
            if mask[b, y0:y0 + h, x0:x0 + w].any():
                continue
            x[b, :, y0:y0 + h, x0:x0 + w] = float(torch.rand(1)) * 0.4 + 0.6
            break
    return x, c.clamp(0, W - 1e-3), mask


@torch.no_grad()
def predict_cols(model, x):
    """x (N,3,H,W) → (cols (N,), conf (N,))，soft-argmax ±2 细化。"""
    logits = model(x)
    p = F.softmax(logits, dim=1)
    best, bi = p.max(dim=1)
    cols = torch.zeros_like(best)
    B, W = p.shape
    for b in range(B):
        lo = max(0, int(bi[b]) - 2); hi = min(W, int(bi[b]) + 3)
        w = p[b, lo:hi]
        cols[b] = (torch.arange(lo, hi, dtype=torch.float32, device=p.device) * w).sum() / w.sum()
    return cols, best


def main():
    epochs = int(sys.argv[1]) if sys.argv[1:] else int(os.environ.get("SUFE_EPOCHS", "60"))
    channels = [int(v) for v in os.environ.get("SUFE_CHANNELS", "24,32,48,64").split(",")]
    bs = int(os.environ.get("SUFE_BS", "128"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    if dev.type == "cuda":
        print(f"设备: {dev} ({torch.cuda.get_device_name(0)})", flush=True)
        torch.backends.cudnn.benchmark = True
    else:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "48")))
        print("设备: cpu", flush=True)

    t0 = time.time()
    X, C, aux, mask = load_cached()
    Ws = np.array([m["W"] for m in aux], np.float32)
    n = len(X)
    print(f"样本 {n}，缓存载入 {time.time()-t0:.1f}s", flush=True)

    Xt = torch.from_numpy(X)
    Ct = torch.from_numpy(C.astype(np.float32))
    Mt = torch.from_numpy(mask)
    # 翻转标签修正量：(nw + 2*LABEL_SHIFT) * (IN_W/W)
    FIXt = torch.from_numpy(((np.array([m["nw"] for m in aux], np.float32) + 2 * LABEL_SHIFT)
                             * IN_W / Ws).astype(np.float32))
    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    nval = max(12, n // 5)
    val_i, tr_i = order[:nval], order[nval:]

    model = GapNet(channels).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"模型通道 {channels}，参数 {nparam}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=1.2e-3)

    def to_dev(idx):
        return Xt[idx].to(dev), Ct[idx].to(dev), Mt[idx].to(dev), FIXt[idx].to(dev)

    val_x, _, _, _ = to_dev(val_i)
    val_true_np = (np.array([aux[i]["nx"] for i in val_i], np.float32) - LABEL_SHIFT)
    val_true = torch.from_numpy(val_true_np * IN_W / Ws[val_i]).to(dev)

    for ep in range(epochs):
        ep_t0 = time.time()
        model.train()
        perm = np.random.permutation(len(tr_i))
        loss_sum, nb = 0.0, 0
        for s in range(0, len(perm), bs):
            idx = tr_i[perm[s:s + bs]]
            xb, cb, mb, fb = to_dev(idx)
            xb, cb, _ = augment_torch(xb, cb, mb, rng, fb)
            logits = model(xb)
            tgt = soft_target(cb)
            loss = -(tgt * F.log_softmax(logits, dim=1)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item(); nb += 1
        if (ep + 1) % max(1, epochs // 20) == 0 or ep == 0:
            model.eval()
            with torch.no_grad():
                cols, conf = predict_cols(model, val_x)
            pred_px = (cols * torch.from_numpy(Ws[val_i]).to(dev) / IN_W).cpu().numpy()
            err = np.abs(pred_px - val_true_np)
            print(f"ep{ep+1:4d} loss={loss_sum/max(nb,1):.4f} val MAE={err.mean():.2f}px "
                  f"≤2px {(err<=2).mean():.2f} ≤4px {(err<=4).mean():.2f} max={err.max():.2f} "
                  f"[{time.time()-ep_t0:.2f}s/ep]", flush=True)

    model.eval()
    with torch.no_grad():
        cols, conf = predict_cols(model, val_x)
    pred_px = (cols * torch.from_numpy(Ws[val_i]).to(dev) / IN_W).cpu().numpy()
    true_px = val_true_np
    err = np.abs(pred_px - true_px)
    print(f"\n=== 留出集（{len(val_i)} 张）===\nMAE={err.mean():.2f}px ≤2px {(err<=2).mean():.2f} "
          f"≤4px {(err<=4).mean():.2f} max={err.max():.2f}", flush=True)

    # 导出（与 train2.py 相同格式，供 sufe-cnn.js 使用）
    layers = []
    convs = [m for m in model.body if isinstance(m, nn.Conv2d)] + [model.head]
    for c in convs:
        w = c.weight.detach().cpu().numpy().astype(np.float32)
        b = c.bias.detach().cpu().numpy().astype(np.float32)
        k = int(w.shape[2])
        layers.append({"k": k, "cin": int(w.shape[1]), "cout": int(w.shape[0]),
                       "w": base64.b64encode(w.reshape(w.shape[0], -1).tobytes()).decode(),
                       "b": base64.b64encode(b.tobytes()).decode()})
    blob = {"meta": {"in_w": IN_W, "in_h": IN_H, "label_shift": LABEL_SHIFT, "ncol": NCOL,
                     "pad": "zero"}, "layers": layers}
    os.makedirs(os.path.join(HERE, "output"), exist_ok=True)
    json.dump(blob, open(os.path.join(HERE, "output", "weights2.json"), "w"))
    js_path = os.path.join(HERE, "..", "sufe-cnn-weights.js")
    with open(js_path, "w", encoding="utf-8") as f:
        f.write("/* 训练产物：整图缺口 CNN 权重（lab/train2_torch.py 导出，勿手改） */\n")
        f.write("window.SUFE_CNN_WEIGHTS = " + json.dumps(blob, separators=(",", ":")) + ";\n")
    # 留出集逐样本预测（供 JS parity 复核）
    json.dump({"idx": [int(i) for i in val_i], "pred": pred_px.tolist(), "true": true_px.tolist()},
              open(os.path.join(HERE, "output", "val_pred.json"), "w"))
    print(f"导出 output/weights2.json 与 ../sufe-cnn-weights.js（总用时 {time.time()-t0:.0f}s）", flush=True)


if __name__ == "__main__":
    main()

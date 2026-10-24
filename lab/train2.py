"""整图输入缺口模型（参考 v2.1.0 思路，容量更大）：训练 + 评测 + 导出。

输入 (3,128,256) = 原图 area-resize 到 256x128 后归一 [-1,1]（CHW）。
结构: 4×[Conv3x3+ReLU → MaxPoolY2] → MaxPoolRows → Conv1x1 → 256 列 logits → softmax → soft-argmax。
标签: 服务端可通过窗口 = 5 个连续整数、中心 = 缺口左缘分割值 - 3（线上扫测确认），
      故直接回归「窗口中心」，预测值四舍五入即 X，两侧各有 ±2px 余量。
"""
import base64
import json
import os
import sys
import time

import numpy as np
from PIL import Image

from cnn import Conv, Conv1x1Out, MaxPoolRows, MaxPoolY2, Net, ReLU, softmax, soft_target
from resize import area_resize

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.join(HERE, "samples", "full")
IN_W, IN_H = 256, 128
LABEL_SHIFT = 3          # X_center = notch_left - LABEL_SHIFT
NCOL = IN_W


def load_samples():
    # 优先用从 PNG 重建的自洽标签（regen_meta.py 产物）
    p_fixed = os.path.join(FULL, "meta_fixed.json")
    p_raw = os.path.join(FULL, "meta.json")
    metas = json.load(open(p_fixed if os.path.exists(p_fixed) else p_raw, encoding="utf-8"))
    X, Cin, aux = [], [], []
    for m in metas:
        p = os.path.join(FULL, m["file"])
        if not os.path.exists(p):
            continue
        rgb = np.asarray(Image.open(p).convert("RGB"))
        r = area_resize(rgb, IN_W, IN_H)
        X.append(np.transpose(r / 255.0 * 2 - 1, (2, 0, 1)).astype(np.float32))
        # 目标：窗口中心在“输入列”坐标系里的位置
        Cin.append((m["nx"] - LABEL_SHIFT) * IN_W / m["W"])
        aux.append(m)
    return np.stack(X), np.array(Cin, np.float32), aux


class Net2(Net):
    """通道数可用环境变量 SUFE_CHANNELS="24,32,48,64" 覆盖（远端大机器放大容量用）。"""

    def __init__(self, channels=None):
        if channels is None:
            channels = [int(x) for x in os.environ.get("SUFE_CHANNELS", "16,24,32,40").split(",")]
        ci = 3
        layers = []
        for co in channels:
            layers += [Conv(ci, co, 3), ReLU(), MaxPoolY2()]
            ci = co
        layers += [MaxPoolRows(), Conv1x1Out(ci)]
        self.layers = layers


def notch_mask_in(m):
    """把原图上的缺口 bbox 映射到输入分辨率，用于“改色/干扰块”增强（只用 meta，无需原图）。"""
    mask = np.zeros((IN_H, IN_W), bool)
    x0 = int(np.floor(m["nx"] * IN_W / m["W"]))
    x1 = int(np.ceil((m["nx"] + m["nw"]) * IN_W / m["W"]))
    y0 = int(np.floor(m["ny"] * IN_H / m["H"]))
    y1 = int(np.ceil((m["ny"] + m["nh"]) * IN_H / m["H"]))
    mask[max(0, y0):y1, max(0, x0):x1] = True
    return mask


def augment(x, c, rng, masks=None, mi=None, flip_fix=0.0):
    """在输入分辨率上做增强；返回 (x, c)。
    c 是缺口**左缘**（-3）的输入列坐标；水平翻转后左缘 = IN_W - c - flip_fix，
    flip_fix = (nw + 2*LABEL_SHIFT) * IN_W / W（忽略 nw 会把标签偏掉半个缺口宽）。"""
    x = x.copy()
    c = float(c)
    # 水平翻转
    if rng.random() < 0.5:
        x = x[:, :, ::-1].copy()
        c = IN_W - c - flip_fix
    # 水平平移（±8 输入像素）
    k = int(rng.integers(-8, 9))
    if k:
        x = np.roll(x, k, axis=2)
        if k > 0:
            x[:, :, :k] = x[:, :, k:k + 1]
        else:
            x[:, :, k:] = x[:, :, k - 1:k]
        c += k
    # 光度
    a = rng.uniform(0.7, 1.3)
    b = rng.uniform(-0.25, 0.25)
    x = np.clip(x * a + b, -1, 1)
    if rng.random() < 0.5:
        g = rng.uniform(0.6, 1.0)
        x = np.sign(x) * np.abs(x) ** g
    if rng.random() < 0.4:
        x = np.clip(x + rng.normal(0, rng.uniform(0.01, 0.06), x.shape), -1, 1)
    # 缺口改色（去“纯白依赖”）
    if masks is not None and mi is not None and rng.random() < 0.5:
        lvl = rng.uniform(-0.1, 0.9)
        x[:, masks[mi]] = lvl
    # 亮块干扰（模拟雪/云与缺口相邻的困难样本）
    if rng.random() < 0.3:
        h = int(rng.integers(6, 30)); w = int(rng.integers(10, 40))
        y0 = int(rng.integers(0, IN_H - h)); x0 = int(rng.integers(0, IN_W - w))
        if not (masks is not None and mi is not None and masks[mi][y0:y0 + h, x0:x0 + w].any()):
            x[:, y0:y0 + h, x0:x0 + w] = rng.uniform(0.6, 1.0)
    return x, min(NCOL - 1e-3, max(0.0, c))


def predict_orig_cols(net, x, W):
    """x: (3,128,256) → 预测的窗口中心（原图像素）"""
    logits = net.forward(x[None])[0]
    p = softmax(logits)
    c = int(p.argmax())
    lo, hi = max(0, c - 2), min(NCOL, c + 3)
    w = p[lo:hi]
    xin = (np.arange(lo, hi) * w).sum() / w.sum()
    return xin * W / IN_W, float(p[c])


def main():
    epochs = int(sys.argv[1]) if sys.argv[1:] else 40
    AUG = int(os.environ.get("SUFE_AUG", "4"))
    bs = int(os.environ.get("SUFE_BS", "24"))
    t0 = time.time()
    X, C, aux = load_samples()
    n = len(X)
    print(f"样本 {n} 张，载入+缩放 {time.time()-t0:.1f}s  X={X.shape}")

    masks = [notch_mask_in(m) for m in aux]

    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    nval = max(12, n // 5)
    val_i, tr_i = order[:nval], order[nval:]

    # 预生成增强池（每样本 AUG 份），int8 存量化值省内存
    pool_x, pool_c = [], []
    for i in tr_i:
        for _ in range(AUG):
            fix = (aux[i]["nw"] + 2 * LABEL_SHIFT) * IN_W / aux[i]["W"]
            xx, cc = augment(X[i], C[i], rng, masks=masks, mi=i, flip_fix=fix)
            pool_x.append((xx * 127).astype(np.int8))
            pool_c.append(cc)
    pool_x = np.stack(pool_x)
    pool_c = np.array(pool_c, np.float32)
    print(f"增强池 {pool_x.shape}，内存 {pool_x.nbytes/1e6:.0f}MB，用时 {time.time()-t0:.1f}s")

    net = Net2()
    t = 0
    nprint = max(1, int(os.environ.get("SUFE_LOG_EVERY", "5")))
    for ep in range(epochs):
        perm = rng.permutation(len(pool_x))
        loss_sum = 0.0
        ep_t0 = time.time()
        for s in range(0, len(pool_x), bs):
            st0 = time.time()
            idx = perm[s:s + bs]
            xb = (pool_x[idx].astype(np.float32) / 127.0)
            cb = pool_c[idx]
            logits = net.forward(xb)
            probs = softmax(logits)
            tgt = soft_target(np.round(cb).astype(int), NCOL, sigma=1.2)
            loss = -(tgt * np.log(probs + 1e-9)).sum(1).mean()
            loss_sum += loss * len(idx)
            net.backward((probs - tgt) / len(idx))
            net.step(1.2e-3 if ep < epochs * 0.75 else 3e-4, t)
            t += 1
            if nprint and t % nprint == 0:
                print(f"  ep{ep+1} step{t} bs={len(idx)} {time.time()-st0:.2f}s "
                      f"loss={float(loss):.3f} [{time.time()-ep_t0:.0f}s/epoch]", flush=True)
        if (ep + 1) % 5 == 0 or ep == 0:
            errs = []
            for i in val_i:
                pc, _ = predict_orig_cols(net, X[i], aux[i]["W"])
                errs.append(pc - (aux[i]["nx"] - LABEL_SHIFT))
            e = np.abs(np.array(errs))
            print(f"ep{ep+1:3d} loss={loss_sum/len(pool_x):.4f} val MAE={e.mean():.2f}px "
                  f"≤2px {np.mean(e<=2):.2f} ≤4px {np.mean(e<=4):.2f} max={e.max():.2f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    # 最终评测（含旧模型对比材料）
    print("\n=== 留出集最终指标 ===")
    errs = []
    for i in val_i:
        pc, conf = predict_orig_cols(net, X[i], aux[i]["W"])
        errs.append(pc - (aux[i]["nx"] - LABEL_SHIFT))
    e = np.abs(np.array(errs))
    print(f"新模型(整图256x128): MAE={e.mean():.2f}px  ≤2px {np.mean(e<=2):.2f}  "
          f"≤4px {np.mean(e<=4):.2f}  max={e.max():.2f}")

    os.makedirs(os.path.join(HERE, "output"), exist_ok=True)
    np.savez_compressed(os.path.join(HERE, "output", "holdout_idx.npz"), val=np.array(val_i))
    json.dump([int(i) for i in val_i], open(os.path.join(HERE, "output", "holdout.json"), "w"))

    # 导出权重（json + JS）
    ws = {}
    for li, L in enumerate(net.layers):
        if hasattr(L, "W"):
            ws[f"l{li}.W"] = L.W.astype(np.float32)
            ws[f"l{li}.b"] = L.b.astype(np.float32)
    blob = {"meta": {"in_w": IN_W, "in_h": IN_H, "label_shift": LABEL_SHIFT, "ncol": NCOL},
            "layers": []}
    for li, L in enumerate(net.layers):
        if hasattr(L, "W"):
            blob["layers"].append({
                "k": L.k, "cin": L.W.shape[1] // (L.k * L.k), "cout": L.W.shape[0],
                "w": base64.b64encode(L.W.astype(np.float32).tobytes()).decode(),
                "b": base64.b64encode(L.b.astype(np.float32).tobytes()).decode()})
    json.dump(blob, open(os.path.join(HERE, "output", "weights2.json"), "w"))
    js_path = os.path.join(HERE, "..", "sufe-cnn-weights.js")
    with open(js_path, "w", encoding="utf-8") as f:
        f.write("/* 训练产物：整图缺口 CNN 权重（lab/train2.py 导出，勿手改） */\n")
        f.write("window.SUFE_CNN_WEIGHTS = " + json.dumps(blob, separators=(",", ":")) + ";\n")
    print(f"导出 output/weights2.json 与 {os.path.relpath(js_path, HERE)}")


if __name__ == "__main__":
    main()

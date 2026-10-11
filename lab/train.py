"""Prepare band tensors from harvested samples, train the gap CNN, evaluate, export."""
import json
import sys

import numpy as np

from cnn import Net, softmax, soft_target

W_SRC = 512         # 统一宽度（harvest 已裁剪/reflect 补齐）
NCOL = 128          # width after /4 stride
STRIDE = 4
BAND_TOP, BAND_H = -8, 116   # rows relative to Y


def prep(src, Y, rng=None, aug=False, notch_mask=None):
    """src: (H,W,3) uint8 -> (2,58,250) float32 band tensor."""
    im = src.astype(np.float32)
    if aug and rng is not None:
        # photometric jitter
        a = rng.uniform(0.75, 1.25)
        b = rng.uniform(-25, 25)
        im = np.clip(im * a + b, 0, 255)
        if rng.random() < 0.5:
            g = rng.uniform(0.55, 1.0)
            im = 255.0 * (im / 255.0) ** g
        if rng.random() < 0.5 and notch_mask is not None:
            # recolor the notch fill so the net cannot key on pure white alone
            lvl = rng.uniform(120, 255)
            im[notch_mask] = lvl
        if rng.random() < 0.4:
            im = im + rng.normal(0, rng.uniform(2, 10), im.shape)
            im = np.clip(im, 0, 255)
    gray = im.mean(2) / 255.0
    sat = (im.max(2) - im.min(2)) / 255.0
    chans = np.stack([gray, sat])
    top = Y + BAND_TOP
    if rng is not None and aug:
        top += int(rng.integers(-4, 5))
    rows = []
    for c in range(2):
        r = chans[c]
        pad = np.pad(r, ((BAND_H, BAND_H), (0, 0)), mode="edge")
        rows.append(pad[top + BAND_H:top + 2 * BAND_H])
    band = np.stack(rows)  # (2,116,W)
    b = band[:, :, :W_SRC].reshape(2, 58, 2, W_SRC // 2, 2).mean((2, 4))
    return b.astype(np.float32)


def notch_mask_of(src, nx, ny, nw, nh):
    m = np.zeros(src.shape[:2], bool)
    m[ny:ny + nh, nx:nx + nw] = True
    return m


def main():
    from harvest import notch_label
    d = np.load("data/dataset.npz")
    src, Y, X = d["src"], d["Y"], d["X"]
    nx = X + 2
    n = len(src)
    boxes = []
    for i in range(n):
        b = notch_label(src[i])
        if b is None:
            b = (int(nx[i]), int(Y[i]), 96, 94, 0)
        boxes.append(b)
    NW = np.array([b[2] for b in boxes])
    masks = [notch_mask_of(src[i], boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])
             for i in range(n)]
    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    nval = max(8, n // 5)
    val_i, tr_i = order[:nval], order[nval:]
    print(f"samples={n} train={len(tr_i)} val={len(val_i)}")

    AUG = 8
    trX, trC = [], []
    for i in tr_i:
        for _ in range(AUG):
            p = prep(src[i], int(Y[i]), rng, aug=True, notch_mask=masks[i])
            flip = rng.random() < 0.5
            if flip:
                p = p[:, :, ::-1].copy()
                nxf = W_SRC - NW[i] - nx[i]
            else:
                nxf = nx[i]
            trX.append(p)
            trC.append(min(NCOL - 1, max(0, int(round(nxf / STRIDE)))))
    trX = np.stack(trX)
    trC = np.array(trC)
    valX = np.stack([prep(src[i], int(Y[i])) for i in val_i])
    valC = np.array([min(NCOL - 1, max(0, int(round(nx[i] / STRIDE)))) for i in val_i])
    print("train pool", trX.shape, "val", valX.shape)

    net = Net()
    bs = 64
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    t = 0
    for ep in range(epochs):
        perm = rng.permutation(len(trX))
        loss_sum = 0.0
        for s in range(0, len(trX), bs):
            idx = perm[s:s + bs]
            xb = trX[idx]
            cb = trC[idx]
            logits = net.forward(xb)
            probs = softmax(logits)
            tgt = soft_target(cb, NCOL, sigma=1.0)
            loss = -(tgt * np.log(probs + 1e-9)).sum(1).mean()
            loss_sum += loss * len(idx)
            g = (probs - tgt) / len(idx)
            net.backward(g)
            lr = 1e-3 if ep < epochs * 0.7 else 3e-4
            net.step(lr, t)
            t += 1
        if (ep + 1) % 5 == 0 or ep == 0:
            vp = softmax(net.forward(valX))
            vc = vp.argmax(1)
            err_col = vc - valC
            err_px = err_col * STRIDE
            acc4 = (np.abs(err_px) <= 4).mean()
            acc2 = (np.abs(err_px) <= 2).mean()
            print(f"ep{ep+1:3d} loss={loss_sum/len(trX):.4f} "
                  f"val|err|<=2px:{acc2:.3f} <=4px:{acc4:.3f} "
                  f"mae_px={np.abs(err_px).mean():.2f} max={np.abs(err_px).max()}")

    # refine helper: centroid within +-2 cols
    def predict(x):
        p = softmax(net.forward(x[None]))[0]
        c = int(p.argmax())
        lo, hi = max(0, c - 2), min(NCOL, c + 3)
        w = p[lo:hi]
        xc = (np.arange(lo, hi) * w).sum() / w.sum()
        return xc * STRIDE

    errs = []
    for i in val_i:
        xhat = predict(prep(src[i], int(Y[i])))
        errs.append(xhat - nx[i])
    errs = np.array(errs)
    print("refined val err px: mae=%.2f  |e|<=2:%.3f  |e|<=4:%.3f  |e|<=5:%.3f  max=%.1f"
          % (np.abs(errs).mean(), (np.abs(errs) <= 2).mean(), (np.abs(errs) <= 4).mean(),
             (np.abs(errs) <= 5).mean(), np.abs(errs).max()))

    # export weights
    ws = {}
    for li, L in enumerate(net.layers):
        if hasattr(L, "W"):
            ws[f"l{li}.W"] = L.W.astype(np.float32)
            ws[f"l{li}.b"] = L.b.astype(np.float32)
    np.savez_compressed("data/cnn_weights.npz", **ws)
    blob = {}
    for k, v in ws.items():
        blob[k] = {"shape": list(v.shape),
                   "b64": __import__("base64").b64encode(v.tobytes()).decode()}
    with open("data/cnn_weights.json", "w") as f:
        json.dump({"meta": {"ncol": NCOL, "stride": STRIDE, "piece_dx": 2,
                            "band_top": BAND_TOP, "band_h": BAND_H},
                   "w": blob}, f)
    print("exported data/cnn_weights.npz + .json")


if __name__ == "__main__":
    main()

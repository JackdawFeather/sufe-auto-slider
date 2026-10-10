# -*- coding: utf-8 -*-
"""训练缺口定位 CNN（卷积+池化 + x 向热图 + soft-argmax 回归缺口左边缘）。

输入: samples/*/bg.png（标签由 detect_gap 教师检测器生成 gap_left）
输出: output/cnn.pt（torch 权重）、output/weights.js（浏览器扩展用）、output/eval.json（留出集指标与逐样本误差）
"""
import base64, json, os, random, sys, time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from detect_gap import white_mask, piece_mask, find_by_components, find_by_template, refine

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, '..', 'samples')
OUT = os.path.join(HERE, '..', 'output')
IN_W, IN_H = 248, 128
# 原图尺寸不定（460~594 宽），换算: 原图像素 = 输入像素 * (图宽 / IN_W)

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# ---------------- 数据 ----------------
def collect():
    items = []
    for d in sorted(os.listdir(SAMPLES)):
        p = os.path.join(SAMPLES, d)
        if not os.path.isdir(p):
            continue
        try:
            bg = cv2.imread(os.path.join(p, 'bg.png'), cv2.IMREAD_UNCHANGED)
            pc = cv2.imread(os.path.join(p, 'piece.png'), cv2.IMREAD_UNCHANGED)
            if bg is None or pc is None:
                continue
            meta = json.load(open(os.path.join(p, 'meta.json'), encoding='utf-8'))
            Y = int(meta['Y'])
            m = white_mask(bg)
            pm = piece_mask(pc)
            ox, oy = np.nonzero(pm)[1].min(), np.nonzero(pm)[0].min()
            comp = find_by_components(m, Y)
            if comp:
                _, gx, gy, _, _, _, _ = comp
                _, gx, gy = refine(m, pm, gx - ox, gy - oy)
            else:
                hit = find_by_template(m, pc)
                if hit is None:
                    continue
                gx, gy = hit[0] + ox, hit[1] + oy
            items.append((d, bg, float(gx)))
        except Exception as e:
            print('skip', d, e)
    return items

def to_input(bg):
    rgb = cv2.cvtColor(bg[:, :, :3], cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (IN_W, IN_H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return img

def augment(img, label_in):
    """轻量增强: 亮度/对比度/噪声/模糊/小位移。label 为输入像素坐标。"""
    if random.random() < 0.8:
        a = random.uniform(0.85, 1.15)
        b = random.uniform(-0.06, 0.06)
        img = np.clip(img * a + b, 0, 1)
    if random.random() < 0.3:
        s = random.uniform(0.3, 0.9)
        img = cv2.GaussianBlur(img, (0, 0), s)
    if random.random() < 0.4:
        img = np.clip(img + np.random.normal(0, 0.015, img.shape).astype(np.float32), 0, 1)
    if random.random() < 0.5:
        tx = random.uniform(-8, 8)
        M = np.float32([[1, 0, tx], [0, 1, 0]])
        img = cv2.warpAffine(img, M, (IN_W, IN_H), borderMode=cv2.BORDER_REPLICATE)
        label_in = label_in + tx
    return img, label_in

# ---------------- 模型 ----------------
class GapNet(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(ci, co):
            return [nn.Conv2d(ci, co, 3, padding=1), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(
            *blk(3, 12), nn.MaxPool2d((2, 1)),      # 64x248
            *blk(12, 16), nn.MaxPool2d((2, 1)),     # 32x248
            *blk(16, 24), nn.MaxPool2d((2, 1)),     # 16x248
            *blk(24, 32), nn.MaxPool2d((2, 1)),     # 8x248
        )
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        f = self.features(x)
        h = self.head(f)                     # B,1,8,248
        logits = h.max(dim=2).values.squeeze(1)   # B,248
        return logits

def soft_argmax(logits):
    p = F.softmax(logits, dim=1)
    centers = torch.arange(IN_W, dtype=logits.dtype, device=logits.device) + 0.5
    return (p * centers).sum(dim=1)

def gauss_target(label_in, sigma=2.0):
    centers = torch.arange(IN_W, dtype=torch.float32)
    t = torch.exp(-0.5 * ((centers[None, :] - label_in[:, None]) / sigma) ** 2)
    return t / t.sum(dim=1, keepdim=True)

def main():
    os.makedirs(OUT, exist_ok=True)
    items = collect()
    print('样本数:', len(items))
    if len(items) < 30:
        print('样本太少，退出'); return
    base = [it for it in items if it[0] != '0000' or True]
    random.shuffle(base)
    n_val = max(15, int(len(base) * 0.12))
    val, train = base[:n_val], base[n_val:]
    print('train %d / val %d' % (len(train), len(val)))

    def make_batch(rows, aug):
        imgs, labels, widths = [], [], []
        for _, bg, gx in rows:
            sw = bg.shape[1]
            img = to_input(bg)
            lab = gx * IN_W / sw          # 原图像素 -> 输入像素
            if aug:
                img, lab = augment(img, lab)
            imgs.append(img.transpose(2, 0, 1))
            labels.append(lab)
            widths.append(sw)
        X = torch.from_numpy(np.stack(imgs).astype(np.float32))
        y = torch.tensor(labels, dtype=torch.float32)
        return X * 2 - 1, y, np.array(widths, dtype=np.float32)

    EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    SIGMA = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    model = GapNet()
    opt = torch.optim.Adam(model.parameters(), lr=1.5e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    for ep in range(1, EPOCHS + 1):
        model.train()
        idx = list(range(len(train)))
        random.shuffle(idx)
        tot, nb = 0.0, 0
        for i in range(0, len(idx), BATCH):
            rows = [train[j] for j in idx[i:i + BATCH]]
            X, y, _ = make_batch(rows, aug=True)
            logits = model(X)
            logp = F.log_softmax(logits, dim=1)
            loss_ce = -(gauss_target(y, SIGMA) * logp).sum(dim=1).mean()
            loss_l1 = (soft_argmax(logits) - y).abs().mean()
            loss = loss_ce + 0.2 * loss_l1
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        sched.step()
        if ep % 10 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                Xv, yv, wv = make_batch(val, aug=False)
                pred = soft_argmax(model(Xv))
                err = (pred - yv).abs() * torch.from_numpy(wv) / IN_W
                print('ep%3d loss=%.4f val_MAE=%.2fpx max=%.2fpx  <=2px:%.1f%% <=4px:%.1f%%'
                      % (ep, tot / nb, err.mean(), err.max(),
                         (err <= 2).float().mean() * 100, (err <= 4).float().mean() * 100))

    # 最终评估
    model.eval()
    with torch.no_grad():
        Xv, yv, wv = make_batch(val, aug=False)
        pred = soft_argmax(model(Xv))
        err = (pred - yv).abs() * torch.from_numpy(wv) / IN_W
        per = [{'dir': val[i][0], 'gt': round(val[i][2], 1),
                'pred': round(float(pred[i]) * float(wv[i]) / IN_W, 1), 'err': round(float(err[i]), 2)}
               for i in range(len(val))]
    metrics = {
        'n_train': len(train), 'n_val': len(val), 'epochs': EPOCHS,
        'mae_px': round(float(err.mean()), 3), 'max_px': round(float(err.max()), 2),
        'pct_le2': round(float((err <= 2).float().mean() * 100), 1),
        'pct_le4': round(float((err <= 4).float().mean() * 100), 1),
        'per_sample': per,
    }
    print(json.dumps({k: v for k, v in metrics.items() if k != 'per_sample'}, ensure_ascii=False, indent=1))
    torch.save(model.state_dict(), os.path.join(OUT, 'cnn.pt'))
    json.dump(metrics, open(os.path.join(OUT, 'eval.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    # 导出 JS 权重
    export_js(model)

def export_js(model):
    def b64(t):
        return base64.b64encode(t.detach().cpu().numpy().astype('<f4').tobytes()).decode()
    layers = []
    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            layers.append({'k': mod.kernel_size[0], 'cin': mod.in_channels,
                           'cout': mod.out_channels, 'w': b64(mod.weight), 'b': b64(mod.bias)})
    js = '// 自动生成: tools/train_cnn.py 导出, 勿手改\nvar SUFE_CNN_WEIGHTS = ' + json.dumps(
        {'in_w': IN_W, 'in_h': IN_H, 'layers': layers}, ensure_ascii=False) + ';\n'
    with open(os.path.join(OUT, 'weights.js'), 'w', encoding='utf-8') as f:
        f.write(js)
    print('weights.js 已导出 (%d 层)' % len(layers))

if __name__ == '__main__':
    main()

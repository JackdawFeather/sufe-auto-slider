# -*- coding: utf-8 -*-
"""缺口检测（教师标签器）：
1) 连通域找纯白缺口（面积/宽高近似拼图块、顶边 y 与接口 Y 吻合）；
2) 若缺口与照片中其他白色区域连通，退化为形状模板匹配（拼图块 alpha 掩码 vs 白色掩码）；
3) 输出服务端期望的 X = 拼图块图像左上角应落点 = 缺口左边缘 - 拼图块内部左边距。

labels.json: {样本目录: {gap_left, piece_off_x, X, ...}}
"""
import json, os
import numpy as np
import cv2

SAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'samples')

def load(d):
    bg = cv2.imread(os.path.join(d, 'bg.png'), cv2.IMREAD_UNCHANGED)
    pc = cv2.imread(os.path.join(d, 'piece.png'), cv2.IMREAD_UNCHANGED)
    meta = json.load(open(os.path.join(d, 'meta.json'), encoding='utf-8'))
    return bg, pc, meta

def white_mask(bg):
    b, g, r = bg[:, :, 0].astype(np.int16), bg[:, :, 1].astype(np.int16), bg[:, :, 2].astype(np.int16)
    m = ((b > 244) & (g > 244) & (r > 244)).astype(np.uint8)
    # 去掉小噪点
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    for i in range(1, n):
        if stats[i][4] < 150:
            m[lab == i] = 0
    return m

def piece_mask(pc):
    return (pc[:, :, 3] > 128).astype(np.uint8)

def find_by_components(m, Y):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    best = None
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (3500 < area < 9500 or (area < 9500 and abs(int(y) - int(Y)) <= 6 and w >= 88)):
            pass
        if not (abs(int(y) - int(Y)) <= 6):
            continue
        if not (85 <= w <= 125 and 80 <= h <= 115):
            continue
        score = abs(int(y) - int(Y)) * 10 + abs(w - 100) + abs(h - 98)
        if best is None or score < best[0]:
            best = (score, int(x), int(y), int(w), int(h), int(area), 'component')
    return best

def find_by_template(m, pc):
    pm = piece_mask(pc)
    res = cv2.matchTemplate(m.astype(np.float32), pm.astype(np.float32), cv2.TM_CCORR)
    max_dir = None
    for _ in range(8):
        uniq, vals = np.unique(res, return_counts=True)
        _, _, minloc, maxloc = cv2.minMaxLoc(res)
        x, y = maxloc
        if m[y:y + 100, x:x + 100].sum() > 0:
            return (int(x), int(y))
        res[y, x] = -1  # 命中全零区域，屏蔽后找下一峰
    return None

def refine(m, pm, gx, gy):
    """在初值附近 ±6px 精调（最大交集）"""
    ph, pw = pm.shape
    best = None
    for dy in range(gy - 6, gy + 7):
        for dx in range(gx - 6, gx + 7):
            if dy < 0 or dx < 0 or dy + ph > m.shape[0] or dx + pw > m.shape[1]:
                continue
            inter = int(np.logical_and(m[dy:dy + ph, dx:dx + pw], pm).sum())
            if best is None or inter > best[0]:
                best = (inter, dx, dy)
    return best

def main():
    labels = {}
    dirs = sorted([d for d in os.listdir(SAMPLES) if os.path.isdir(os.path.join(SAMPLES, d))])
    ok = 0
    for d in dirs:
        p = os.path.join(SAMPLES, d)
        try:
            bg, pc, meta = load(p)
            m = white_mask(bg)
            pm = piece_mask(pc)
            oy, ox = np.nonzero(pm)[0].min(), np.nonzero(pm)[1].min()
            Y = int(meta['Y'])
            comp = find_by_components(m, Y)
            src = 'component'
            if comp:
                _, gx, gy, gw, gh, ga, src = comp
                inter, gx, gy = refine(m, pm, gx - ox, gy - oy)
            else:
                hit = find_by_template(m, pc)
                if hit is None:
                    print('%s: 检测失败（无组件、模板无峰）' % d)
                    continue
                gx, gy = hit
                src = 'template'
                inter = int(np.logical_and(m[gy:gy + 100, gx:gx + 100], pm).sum())
                # 转为缺口左边缘坐标
                gx, gy = gx + ox, gy + oy
            gap_area = int(m[gy:gy + 100, gx:gx + 100].sum())
            iou = inter / max(1, (pm.sum() + gap_area - inter))
            labels[d] = {
                'token': meta['token'], 'Y': Y,
                'img_w': int(bg.shape[1]), 'img_h': int(bg.shape[0]),
                'piece_off_x': int(ox), 'piece_off_y': int(oy),
                'gap_left': int(gx), 'gap_top': int(gy),
                'X': int(gx - ox), 'Y_calc': int(gy - oy),
                'align_iou': round(float(iou), 3), 'method': src,
            }
            print('%s: Y=%3d gap_left=%3d piece_off=(%d,%d) -> X=%3d iou=%.3f [%s]'
                  % (d, Y, gx, ox, oy, gx - ox, iou, src))
            ok += 1
        except Exception as e:
            print('%s: 异常 %s' % (d, e))
    out = os.path.join(SAMPLES, 'labels.json')
    json.dump(labels, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, sort_keys=True)
    print('detected %d / %d' % (ok, len(dirs)))

if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""导出 JS/Python 一致性测试数据：原图 RGB 字节 + Python 端预测值。
取 val 集样本（含误差最大者），供 node parity_test.js 对比。"""
import json, os
import numpy as np
import cv2
import torch
from train_cnn import GapNet, to_input, soft_argmax, collect, IN_W, IN_H

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'output')
PARITY = os.path.join(OUT, 'parity')

def main():
    os.makedirs(PARITY, exist_ok=True)
    model = GapNet()
    model.load_state_dict(torch.load(os.path.join(OUT, 'cnn.pt')))
    model.eval()

    items = collect()
    items.sort(key=lambda t: t[0])
    # 取 60 张（含首尾与中段），保障覆盖不同缺口位置
    if len(items) > 60:
        idx = np.linspace(0, len(items) - 1, 60).astype(int)
        items = [items[i] for i in idx]

    manifest, blob = [], bytearray()
    with torch.no_grad():
        for d, bg, gx in items:
            inp = to_input(bg)
            X = torch.from_numpy(inp.transpose(2, 0, 1)[None].astype(np.float32)) * 2 - 1
            pred = float(soft_argmax(model(X))[0]) * bg.shape[1] / IN_W
            rgb = cv2.cvtColor(bg[:, :, :3], cv2.COLOR_BGR2RGB).tobytes()
            manifest.append({'dir': d, 'gt': round(float(gx), 1), 'py_pred': round(pred, 3),
                             'w': int(bg.shape[1]), 'h': int(bg.shape[0]),
                             'offset': len(blob), 'len': len(rgb)})
            blob += rgb
    json.dump(manifest, open(os.path.join(PARITY, 'manifest.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    open(os.path.join(PARITY, 'raw.bin'), 'wb').write(bytes(blob))
    print('导出 %d 张 -> %s' % (len(manifest), PARITY))

if __name__ == '__main__':
    main()

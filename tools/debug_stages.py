# -*- coding: utf-8 -*-
"""导出 Python 端逐层中间值（样例 0005），供 tools/debug_stages_js.js 对比。"""
import json, os
import numpy as np
import cv2
import torch
import torch.nn as nn
from train_cnn import GapNet, to_input, soft_argmax, IN_W, IN_H

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'output', 'debug')
SAMPLE = os.path.join(HERE, '..', 'samples', '0005')

def main():
    os.makedirs(OUT, exist_ok=True)
    bg = cv2.imread(os.path.join(SAMPLE, 'bg.png'), cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(bg[:, :, :3], cv2.COLOR_BGR2RGB)
    open(os.path.join(OUT, 'rgb.bin'), 'wb').write(rgb.tobytes())
    json.dump({'w': int(bg.shape[1]), 'h': int(bg.shape[0])}, open(os.path.join(OUT, 'rgb_meta.json'), 'w'))

    inp = to_input(bg)
    X = torch.from_numpy(inp.transpose(2, 0, 1)[None].astype(np.float32)) * 2 - 1
    model = GapNet()
    model.load_state_dict(torch.load(os.path.join(HERE, '..', 'output', 'cnn.pt')))
    model.eval()

    stages = []
    x = X
    li = 0
    with torch.no_grad():
        for mod in model.features:
            x = mod(x)
            if isinstance(mod, nn.Conv2d):
                pass
            else:
                # 下一个 conv 前先记录：判断是否刚过 pool
                continue
        # 手动逐层跑
        x = X
        stages.append({'stage': 'input', 'head': [round(float(v), 4) for v in x.flatten()[:30].tolist()]})
        idx = 0
        convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        conv_i = 0
        cur = X
        for mod in model.features:
            cur = mod(cur)
            if isinstance(mod, nn.Conv2d):
                conv_i += 1
            if isinstance(mod, nn.MaxPool2d):
                stages.append({'stage': 'conv+pool#%d' % conv_i, 'cout': cur.shape[1],
                               'h': cur.shape[2], 'w': cur.shape[3],
                               'head': [round(float(v), 4) for v in cur.flatten()[:30].tolist()]})
        head = model.head(cur)
        logits = head.max(dim=2).values.squeeze(1)[0]
        stages.append({'stage': 'logits', 'head': [round(float(v), 4) for v in logits[:30].tolist()]})
        pred = float(soft_argmax(model(X))[0])
    out = {'pred_in': pred, 'pred_px': pred * bg.shape[1] / IN_W, 'stages': stages,
           'argmax_bin': int(torch.argmax(logits))}
    json.dump(out, open(os.path.join(OUT, 'py_stages.json'), 'w'), indent=1)
    print('pred_in=%.2f px=%.2f argmax=%d' % (pred, out['pred_px'], out['argmax_bin']))

if __name__ == '__main__':
    main()

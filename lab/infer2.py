"""整图模型推理（numpy）：从 output/weights2.json 重建 Net2 并预测。"""
import base64
import json
import os

import numpy as np

from resize import area_resize
from train2 import IN_H, IN_W, Net2, predict_orig_cols

HERE = os.path.dirname(os.path.abspath(__file__))


def load_net2(path=None):
    path = path or os.path.join(HERE, "output", "weights2.json")
    j = json.load(open(path, encoding="utf-8"))
    net = Net2()
    bi = 0
    for L in net.layers:
        if hasattr(L, "W"):
            d = j["layers"][bi]
            bi += 1
            L.W = np.frombuffer(base64.b64decode(d["w"]), np.float32).reshape(L.W.shape).astype(np.float32)
            L.b = np.frombuffer(base64.b64decode(d["b"]), np.float32).astype(np.float32)
    return net


def predict_rgb(net, rgb):
    """rgb: (h,w,3) uint8 → (预测窗口中心原图 px, conf)"""
    r = area_resize(rgb, IN_W, IN_H)
    x = np.transpose(r / 255.0 * 2 - 1, (2, 0, 1)).astype(np.float32)
    return predict_orig_cols(net, x, rgb.shape[1])

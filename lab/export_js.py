"""Export trained weights as a JS global for the extension, plus parity fixtures."""
import json

import numpy as np

j = json.load(open("data/cnn_weights.json", encoding="utf-8"))
with open("../sufe-cnn-weights.js", "w", encoding="utf-8") as f:
    f.write("/* 训练产物：缺口 CNN 权重（lab/train.py 导出，勿手改） */\n")
    f.write("window.SUFE_CNN_WEIGHTS = " + json.dumps(j, separators=(",", ":")) + ";\n")
print("wrote ../sufe-cnn-weights.js")

d = np.load("data/dataset.npz")
K = 10
meta = []
for i in range(K):
    # 与浏览器一致：喂原始宽度（>512 只取前 512 列，JS 也只读这么多），掩码按原始 W 计算
    W0 = int(d["W"][i])
    src = d["src"][i][:, :min(W0, 512)]
    h, w, _ = src.shape
    rgba = np.dstack([src, np.full((h, w), 255, np.uint8)]).tobytes()
    with open(f"data/parity_{i}.raw", "wb") as f:
        f.write(rgba)
    meta.append({"i": int(i), "w": W0, "bufw": int(w), "h": int(h),
                 "Y": int(d["Y"][i]), "trueX": int(d["X"][i])})
from infer import load_net, predict  # noqa: E402
net = load_net()
for m in meta:
    r = predict(net, d["src"][m["i"]], d["Y"][m["i"]], origW=m["w"])
    m["pyx"] = r["x"]
    m["pyX"] = r["X"]
    m["conf"] = r["conf"]
with open("data/parity_meta.json", "w") as f:
    json.dump(meta, f, indent=1)
print("wrote parity fixtures for", K, "samples")

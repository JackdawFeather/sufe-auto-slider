"""parity2：导出样本 raw + numpy 预测 → 交给 node 用扩展里的 sufe-cnn.js 复算并比对。"""
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

from infer2 import load_net2, predict_rgb
from train2 import load_samples

HERE = os.path.dirname(os.path.abspath(__file__))
K = int(sys.argv[1]) if sys.argv[1:] else 12
OUT = os.path.join(HERE, "output", "parity2")
os.makedirs(OUT, exist_ok=True)


def main():
    X, C, aux = load_samples()
    net = load_net2()
    idxs = list(range(min(K, len(X))))
    meta = []
    for i in idxs:
        m = aux[i]
        rgb = np.asarray(Image.open(os.path.join(HERE, "samples", "full", m["file"])).convert("RGB"))
        open(os.path.join(OUT, f"{i}.raw"), "wb").write(rgb.tobytes())
        px, conf = predict_rgb(net, rgb)
        meta.append({"i": i, "w": rgb.shape[1], "h": rgb.shape[0],
                     "pyx": float(px), "trueX": m["X"]})
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"))
    print(f"phaseA: {len(meta)} 张 raw + numpy 预测已导出")

    r = subprocess.run(["node", os.path.join(HERE, "parity2.js")],
                       capture_output=True, text=True, cwd=HERE)
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr.strip()[-800:])
        sys.exit(r.returncode)


if __name__ == "__main__":
    main()

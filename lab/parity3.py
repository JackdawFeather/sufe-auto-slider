"""parity3：扩展 sufe-cnn.js（纯 JS）与 torch 训练产物在同一批留出集图片上的一致性。

前提：已从远端取回 output/weights2.json 与 output/val_pred.json（torch 的逐样本预测），
并已用 export 更新 ../sufe-cnn-weights.js。
"""
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.join(HERE, "samples", "full")
OUT = os.path.join(HERE, "output", "parity3")


def main():
    vp = json.load(open(os.path.join(HERE, "output", "val_pred.json"), encoding="utf-8"))
    metas = json.load(open(os.path.join(FULL, "meta_fixed.json"), encoding="utf-8"))
    # val_pred 的 idx 是 load_samples 返回数组的下标（= meta_fixed 的位置），不是文件名号
    os.makedirs(OUT, exist_ok=True)
    meta = []
    for k, i in enumerate(vp["idx"]):
        m = metas[i]
        rgb = np.asarray(Image.open(os.path.join(FULL, m["file"])).convert("RGB"))
        open(os.path.join(OUT, f"{k}.raw"), "wb").write(rgb.tobytes())
        meta.append({"k": k, "i": i, "file": m["file"], "w": rgb.shape[1], "h": rgb.shape[0],
                     "torch": vp["pred"][k], "true": vp["true"][k]})
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"))
    print(f"导出 {len(meta)} 张留出集图片 → node 复算")
    rc = subprocess.call(["node", os.path.join(HERE, "parity3.js")], cwd=HERE)
    sys.exit(rc)


if __name__ == "__main__":
    main()

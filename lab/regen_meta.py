"""从 PNG 重建标签（修 harvest2 并发/续跑造成的 meta 与文件脱钩）。

只依赖图像本身：notch_label 给出缺口 bbox → 标签 X_center = nx - LABEL_SHIFT。
输出 samples/full/meta_fixed.json，并报告分割失败/形状异常的样本。
"""
import json
import os
import sys

import numpy as np
from PIL import Image

from harvest import notch_label
from train2 import LABEL_SHIFT

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.join(HERE, "samples", "full")


def main():
    files = sorted([f for f in os.listdir(FULL) if f.endswith(".png")],
                   key=lambda s: int(s.split(".")[0]))
    out, bad = [], []
    for f in files:
        p = os.path.join(FULL, f)
        rgb = np.asarray(Image.open(p).convert("RGB"))
        lbl = notch_label(rgb)
        if lbl is None:
            bad.append((f, "no-notch"))
            continue
        nx, ny, nw, nh, area = lbl
        if not (70 <= nw <= 110 and 70 <= nh <= 110 and 2000 <= area <= 9000):
            bad.append((f, f"odd-shape {nw}x{nh} area={area}"))
            continue
        out.append({"i": int(f.split(".")[0]), "file": f, "W": rgb.shape[1], "H": rgb.shape[0],
                    "nx": int(nx), "ny": int(ny), "nw": int(nw), "nh": int(nh),
                    "X": int(nx - LABEL_SHIFT), "area": int(area)})
    json.dump(out, open(os.path.join(FULL, "meta_fixed.json"), "w"), ensure_ascii=False, indent=1)
    print(f"重建完成：可用 {len(out)} 张，异常 {len(bad)} 张 -> samples/full/meta_fixed.json")
    for b in bad[:10]:
        print("   异常:", b)
    ws = np.array([m["W"] for m in out])
    xs = np.array([m["X"] for m in out])
    print(f"W 分布 {sorted(set(ws.tolist()))}")
    print(f"X 范围 {xs.min()}..{xs.max()}，nx+96<=W 的比例 "
          f"{np.mean([m['nx'] + 96 <= m['W'] for m in out]):.3f}")


if __name__ == "__main__":
    main()

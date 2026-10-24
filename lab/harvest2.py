"""采集整图样本（整图输入模型用）：原图 PNG + meta，标签仍由白色缺口分割给出。

存到 lab/samples/full/<序号>.png，meta 追加到 lab/samples/full/meta.json。
"""
import base64
import io
import json
import os
import random
import sys
import time

import numpy as np
from PIL import Image

from harvest import api, notch_label

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "samples", "full")


def main():
    n = int(sys.argv[1]) if sys.argv[1:] else 220
    os.makedirs(OUT, exist_ok=True)
    metas = json.load(open(os.path.join(OUT, "meta.json"), encoding="utf-8")) if \
        os.path.exists(os.path.join(OUT, "meta.json")) else []
    start = len(metas)
    ok = 0
    for k in range(start, start + n):
        try:
            d = api("/api/v3/sliderCaptcha/init", {})["data"]
        except Exception as e:
            print(f"[{k}] init error: {e}")
            time.sleep(3)
            continue
        im = Image.open(io.BytesIO(base64.b64decode(d["sourceImage"]))).convert("RGB")
        rgb = np.asarray(im)
        lbl = notch_label(rgb)
        if lbl is None:
            print(f"[{k}] 缺口分割失败，跳过")
            time.sleep(random.uniform(1.2, 2.2))
            continue
        nx, ny, nw, nh, area = lbl
        im.save(os.path.join(OUT, f"{k}.png"))
        metas.append({"i": k, "file": f"{k}.png", "W": im.size[0], "H": im.size[1],
                      "Y": int(d["Y"]), "nx": int(nx), "ny": int(ny), "nw": int(nw), "nh": int(nh),
                      "X": int(nx - 2), "area": int(area)})
        ok += 1
        if ok % 10 == 0:
            json.dump(metas, open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8"))
        print(f"[{k}] W={im.size[0]} notch=({nx},{ny},{nw}x{nh}) X={nx-2} Y={d['Y']}", flush=True)
        time.sleep(random.uniform(1.4, 2.4))
    json.dump(metas, open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8"))
    print(f"\n完成：新增 {ok} 张，累计 {len(metas)} 张 -> {OUT}")


if __name__ == "__main__":
    main()

"""实测 check 接口的通过窗口：以「缺口左缘分割值」为基准，扫 ±6。

输出每个验证码可通过的 X 集合，并换算成「相对 (nx-3) 的偏移」，用于确认
标签中心 = nx-3（窗口宽 5、半径 ±2）这一约定是否稳定。
"""
import base64
import io
import random
import sys
import time

import numpy as np
from PIL import Image

from harvest import api, notch_label

N = int(sys.argv[1]) if sys.argv[1:] else 4


def probe(X, Y, token):
    r = api(f"/api/v3/sliderCaptcha/check?X={X}&Y={Y}&token={token}")
    return str(r.get("code")) == "0", r.get("code"), r.get("msg")


rows = []
for k in range(N):
    d = api("/api/v3/sliderCaptcha/init", {})["data"]
    rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(d["sourceImage"]))).convert("RGB"))
    lbl = notch_label(rgb)
    if not lbl:
        print(f"[{k}] 分割失败，跳过")
        continue
    nx = lbl[0]
    passed, codes = [], set()
    for delta in range(-6, 7):
        ok, code, msg = probe(nx + delta, d["Y"], d["token"])
        if ok:
            passed.append(delta)
        else:
            codes.add(f"{code}:{msg}")
        time.sleep(0.12)
    if not passed:
        # 窄窗全灭：粗扫全图找真实窗口（同时暴露限流/异常响应）
        coarse = []
        for X in range(10, rgb.shape[1] - 90, 6):
            ok, code, msg = probe(X, d["Y"], d["token"])
            if ok:
                coarse.append(X)
            else:
                codes.add(f"{code}:{msg}")
            time.sleep(0.12)
        print(f"[{k}] W={rgb.shape[1]} nx={nx} ±6 全灭；粗扫命中 {coarse[:6]}"
              f"{'...' if len(coarse) > 6 else ''}；非通过响应样本 {list(codes)[:3]}", flush=True)
        rows.append((nx, None, coarse))
        continue
    rel = [p - (-3) for p in passed]
    rows.append((nx, passed, rel))
    print(f"[{k}] W={rgb.shape[1]} nx={nx} 通过偏移(相对 nx)={passed} → 相对(nx-3)={rel}", flush=True)
    time.sleep(random.uniform(0.6, 1.2))

print("\n=== 汇总 ===")
ok_rows = [r for r in rows if r[1]]
for nx, passed, rel in ok_rows:
    print(f"nx={nx}: 窗口=[{min(passed)+nx},{max(passed)+nx}] 宽度={len(passed)} 相对中心(nx-3)={rel}")
for nx, passed, coarse in [r for r in rows if not r[1]]:
    print(f"nx={nx}: ±6 全灭，粗扫命中点 {coarse[:8]}（共 {len(coarse)}）")
if ok_rows:
    widths = [len(r[1]) for r in ok_rows]
    centers = [(min(r[1]) + max(r[1])) / 2 for r in ok_rows]
    print(f"窗口宽度: {widths}  窗口中心(相对 nx): {centers}  → 应为 -3")


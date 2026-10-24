# -*- coding: utf-8 -*-
"""采集图片验证码（randcode）样本：lab/samples/vcode/<id>.jpg

用法: python lab/collect_vcode.py [数量=2000] [间隔秒=1.0]
低频顺序采集，断点续采（按已存在文件数续号）。
"""
import os
import random
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "samples", "vcode")
BASE = "https://login.sufe.edu.cn/esc-sso"


def main():
    n = int(sys.argv[1]) if sys.argv[1:] else 2000
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    os.makedirs(OUT, exist_ok=True)
    S = requests.Session()
    S.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36",
        "Accept-Language": "zh-CN,zh;q=0.9",
    })
    S.get("https://login.sufe.edu.cn/login/", timeout=20)
    start = len([f for f in os.listdir(OUT) if f.endswith(".jpg")])
    i, ok = start, 0
    end = start + n
    print("collect start:", start, "->", end, "interval", interval, flush=True)
    while i < end:
        try:
            r = S.get(BASE + "/api/v1/image/getRandcode?" + str(random.random()), timeout=15)
            ct = r.headers.get("content-type", "")
            if r.status_code == 200 and ct.startswith("image") and len(r.content) > 500:
                with open(os.path.join(OUT, "%05d.jpg" % i), "wb") as f:
                    f.write(r.content)
                i += 1
                ok += 1
                if ok % 100 == 0:
                    print("collected", ok, "/", n, flush=True)
            else:
                print("bad resp:", r.status_code, ct, len(r.content), flush=True)
                time.sleep(3)
        except Exception as e:
            print("err:", e, flush=True)
            time.sleep(3)
        time.sleep(interval)
    print("done:", ok, "new files, total", len(os.listdir(OUT)), flush=True)


if __name__ == "__main__":
    main()

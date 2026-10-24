# -*- coding: utf-8 -*-
"""端到端验证：真实拉取图片验证码 → JS OCR（node CLI）→ doLogin 提交（假学号）。

期望：返回 SSO10001（账号不存在）= 验证码被接受。统计命中率。
用法: python lab/vcode_api_test.py [N=8]
"""
import base64
import json
import random
import subprocess
import sys
import time

import cv2
import numpy as np
import requests
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding

HERE = __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0]
BASE = "https://login.sufe.edu.cn/esc-sso"
N = int(sys.argv[1]) if sys.argv[1:] else 12

S = requests.Session()
S.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36",
    "Origin": "https://login.sufe.edu.cn",
    "Referer": "https://login.sufe.edu.cn/login/",
    "Accept-Language": "zh-CN,zh;q=0.9",
})
S.get("https://login.sufe.edu.cn/login/", timeout=20)


def do_login(code):
    qa = S.get(BASE + "/api/v3/auth/queryAllValid", timeout=20).json()
    p = qa["data"]["param"]
    t = qa["data"]["config"].get("loginTicket")
    key = serialization.load_der_public_key(base64.b64decode(p["publicKey"]))
    enc = base64.b64encode(key.encrypt(b"probe-pass-123", padding.PKCS1v15())).decode()
    body = {"authType": "webLocalAuth",
            "dataField": {"username": "9999999999", "password": enc, "vcode": code,
                          "publicKeyId": p["publicKeyId"]},
            "loginTicket": t, "redirectUri": ""}
    return S.post(BASE + "/api/v3/auth/doLogin", json=body, timeout=20).json()


def main():
    hit = 0
    t_net = []
    t_ocr = []
    e2e = []             # 每次成功前累计耗时
    since = time.time()
    for i in range(N):
        t0 = time.time()
        img = S.get(BASE + "/api/v1/image/getRandcode?" + str(random.random()), timeout=20).content
        arr = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        h, w = arr.shape[:2]
        rgba = np.dstack([cv2.cvtColor(arr, cv2.COLOR_BGR2RGB), np.full((h, w), 255, np.uint8)]).tobytes()
        bin_path = "lab/output/vcode/api_test.bin"
        with open(bin_path, "wb") as f:
            f.write(rgba)
        t1 = time.time()
        out = subprocess.run(["node", "lab/vcode_ocr_cli.js", bin_path, str(w), str(h)],
                             capture_output=True, text=True).stdout
        t2 = time.time()
        try:
            rec = json.loads(out)
        except Exception:
            rec = {"text": "", "conf": 0, "ms": 0}
        t_ocr.append(rec.get("ms", 0) / 1000.0)
        r = do_login(rec["text"]) if rec["text"] else {"code": "EMPTY"}
        t3 = time.time()
        ok = str(r.get("code")) == "SSO10001"
        hit += ok
        t_net.append((t1 - t0) + (t3 - t2))
        print("%2d) OCR=%-6s conf=%.2f -> %s %s | OCR %.0fms 网络 %.2fs" % (
            i + 1, rec["text"], rec["conf"], r.get("code"), "HIT" if ok else "",
            rec.get("ms", 0), (t1 - t0) + (t3 - t2)), flush=True)
        if ok:
            e2e.append(time.time() - since)
            since = time.time()
        time.sleep(1.0)
    p = hit / max(1, N)
    print("提交 %d 次，命中 %d（%.0f%%）| OCR均值 %.0fms | 网络均值 %.2fs" % (
        N, hit, 100.0 * p, 1000 * sum(t_ocr) / max(1, len(t_ocr)),
        sum(t_net) / max(1, len(t_net))))
    if e2e:
        print("实测「每次通过耗时」均值 %.2fs（含 node 启动开销，浏览器内应更低）" % (sum(e2e) / len(e2e)))


if __name__ == "__main__":
    main()

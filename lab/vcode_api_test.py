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
N = int(sys.argv[1]) if sys.argv[1:] else 8

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
    for i in range(N):
        img = S.get(BASE + "/api/v1/image/getRandcode?" + str(random.random()), timeout=20).content
        arr = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        h, w = arr.shape[:2]
        rgba = np.dstack([cv2.cvtColor(arr, cv2.COLOR_BGR2RGB), np.full((h, w), 255, np.uint8)]).tobytes()
        bin_path = "lab/output/vcode/api_test.bin"
        with open(bin_path, "wb") as f:
            f.write(rgba)
        out = subprocess.run(["node", "lab/vcode_ocr_cli.js", bin_path, str(w), str(h)],
                             capture_output=True, text=True).stdout
        try:
            rec = json.loads(out)
        except Exception:
            rec = {"text": "", "conf": 0}
        code = rec["text"]
        r = do_login(code) if code else {"code": "EMPTY"}
        ok = str(r.get("code")) == "SSO10001"
        hit += ok
        print("%2d) OCR=%-6s conf=%.2f -> %s %s" % (i + 1, code, rec.get("conf", 0),
                                                    r.get("code"), "HIT" if ok else ""), flush=True)
        time.sleep(1.2)
    print("命中: %d/%d = %.0f%%" % (hit, N, 100.0 * hit / max(1, N)))


if __name__ == "__main__":
    main()

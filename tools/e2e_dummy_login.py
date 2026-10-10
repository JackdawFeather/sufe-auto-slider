# -*- coding: utf-8 -*-
"""端到端闭环验证（单次，不涉及任何真实账号）：
滑块 init → CNN 定位 → 网格探测通过拿 vcode → queryAllValid 取公钥 →
用不存在的学号 + RSA(PKCS#1 v1.5) 加密的假密码调 doLogin。
预期返回「学号或密码错误（SSO10002）」——若如此，证明 vcode 与整条链路被服务端接受，
唯一失败原因是凭据本身（这正是我们要证明的闭环）。
"""
import base64, json, os, sys, urllib.request
import numpy as np
import cv2
import torch
from cryptography.hazmat.primitives.serialization import load_der_public_key
from cryptography.hazmat.primitives.asymmetric import padding
from probe_check import api, BASE, HDR
from train_cnn import GapNet, to_input, soft_argmax, IN_W, IN_H

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = [0, 2, -2, 4, -4, 6, -6, 1, -1, 3, -3, 5, -5, 7, -7]

def post(path, payload):
    req = urllib.request.Request(BASE + path, method='POST', headers=HDR,
                                 data=json.dumps(payload).encode('utf-8'))
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode('utf-8'))

def solve_slider(model):
    d = api('/api/v3/sliderCaptcha/init', 'POST')['data']
    Y, token = int(d['Y']), d['token']
    bg = cv2.imdecode(np.frombuffer(base64.b64decode(d['sourceImage']), np.uint8), cv2.IMREAD_UNCHANGED)
    with torch.no_grad():
        inp = to_input(bg)
        X = torch.from_numpy(inp.transpose(2, 0, 1)[None].astype(np.float32)) * 2 - 1
        pred = float(soft_argmax(model(X))[0]) * bg.shape[1] / IN_W
    base = round(pred)
    for tries, off in enumerate(GRID, 1):
        x = base + off
        if x < 0:
            continue
        r = api('/api/v3/sliderCaptcha/check?X=%d&Y=%d&token=%s' % (x, Y, token))
        if str(r.get('code')) == '0':
            return r['data'], pred, tries
    raise RuntimeError('滑块全部探测点未通过')

def main():
    model = GapNet()
    model.load_state_dict(torch.load(os.path.join(HERE, '..', 'output', 'cnn.pt')))
    model.eval()

    vcode, pred, tries = solve_slider(model)
    print('滑块通过: CNN预测=%.1f, 第%d次探测命中, vcode=%s…' % (pred, tries, str(vcode)[:12]))

    cfg = api('/api/v3/auth/queryAllValid')['data']
    param, ticket = cfg['param'], cfg['config']['loginTicket']
    pk = load_der_public_key(base64.b64decode(param['publicKey']))
    enc = base64.b64encode(pk.encrypt(b'DummyPassword-For-E2E-Only', padding.PKCS1v15())).decode()

    res = post('/api/v3/auth/doLogin', {
        'authType': 'webLocalAuth',
        'dataField': {'username': '98765432100', 'password': enc, 'vcode': vcode,
                      'publicKeyId': param['publicKeyId']},
        'loginTicket': ticket, 'redirectUri': '',
    })
    print('doLogin 返回: code=%s msg=%s' % (res.get('code'), res.get('msg')))
    # SSO10001 账号不存在 / SSO10002 学号或密码错误 —— 都说明 vcode 已被接受，仅凭据被拒
    if str(res.get('code')) in ('SSO10001', 'SSO10002'):
        print('== 闭环成功 ==: vcode 被接受，失败原因仅为假账号凭据（预期行为）')
        return 0
    print('== 注意 ==: 返回非凭据类错误，需人工确认含义')
    return 1

if __name__ == '__main__':
    sys.exit(main())

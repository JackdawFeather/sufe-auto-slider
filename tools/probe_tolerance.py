# -*- coding: utf-8 -*-
"""探测 check 接口对 X 的容差：每个偏移用一个新 token 提交 base_X+Δ，记录通过与否。"""
import base64, sys, time
import numpy as np
import cv2
from probe_check import api
from detect_gap import white_mask, piece_mask, find_by_components, refine

DELTAS = [int(x) for x in (sys.argv[1].split(',') if len(sys.argv) > 1 else
                           ['-20', '-10', '-5', '-3', '-2', '-1', '1', '2', '3', '5', '10', '20'])]

for delta in DELTAS:
    try:
        d = api('/api/v3/sliderCaptcha/init', 'POST')
        data = d['data']
        Y = int(data['Y']); token = data['token']
        bg = cv2.imdecode(np.frombuffer(base64.b64decode(data['sourceImage']), np.uint8), cv2.IMREAD_UNCHANGED)
        pc = cv2.imdecode(np.frombuffer(base64.b64decode(data['newImage']), np.uint8), cv2.IMREAD_UNCHANGED)
        m = white_mask(bg); pm = piece_mask(pc)
        ox, oy = np.nonzero(pm)[1].min(), np.nonzero(pm)[0].min()
        comp = find_by_components(m, Y)
        if not comp:
            print('delta=%+d: 检测失败，跳过' % delta); time.sleep(1.2); continue
        _, gx, gy, _, _, _, _ = comp
        _, gx, gy = refine(m, pm, gx - ox, gy - oy)
        X = (gx - ox) + delta
        r = api('/api/v3/sliderCaptcha/check?X=%d&Y=%d&token=%s' % (X, Y, token))
        print('delta=%+3d  X=%3d  code=%s  %s' % (delta, X, r.get('code'), 'PASS' if str(r.get('code')) == '0' else 'fail (' + str(r.get('msg')) + ')'))
    except Exception as e:
        print('delta=%+d 异常: %s' % (delta, e))
    time.sleep(1.2)

# -*- coding: utf-8 -*-
"""边界确认 + 失败是否作废 token 的判定。
用例:
  tok1: X=0(必失败) 再 X=gap_left  -> 若第二次通过 => 失败不作废 token
  tok2: X=0(必失败) 再 X=gap_left  -> 复现
  tok3/4: X=gap_left+4 -> 确认上界
  tok5: X=gap_left-1/2/3 -> 确认下界
"""
import base64, time
import numpy as np
import cv2
from probe_check import api
from detect_gap import white_mask, piece_mask, find_by_components, refine

def fresh():
    d = api('/api/v3/sliderCaptcha/init', 'POST')
    data = d['data']
    Y = int(data['Y']); token = data['token']
    bg = cv2.imdecode(np.frombuffer(base64.b64decode(data['sourceImage']), np.uint8), cv2.IMREAD_UNCHANGED)
    pc = cv2.imdecode(np.frombuffer(base64.b64decode(data['newImage']), np.uint8), cv2.IMREAD_UNCHANGED)
    m = white_mask(bg); pm = piece_mask(pc)
    ox, oy = np.nonzero(pm)[1].min(), np.nonzero(pm)[0].min()
    comp = find_by_components(m, Y)
    _, gx, gy, _, _, _, _ = comp
    _, gx, gy = refine(m, pm, gx - ox, gy - oy)
    return token, Y, gx  # gx = gap_left

def chk(X, Y, token):
    r = api('/api/v3/sliderCaptcha/check?X=%d&Y=%d&token=%s' % (X, Y, token))
    return str(r.get('code')) == '0', r.get('code')

# 1) 失败是否作废 token
for i in (1, 2):
    tok, Y, gl = fresh()
    ok1, c1 = chk(0, Y, tok)
    time.sleep(0.8)
    ok2, c2 = chk(gl, Y, tok)
    print('token%d: X=0 -> %s ; 再 X=gap_left=%d -> %s  => %s' % (
        i, c1, gl, c2, '失败不作废' if ok2 else '失败后无法通过(可能作废)'))
    time.sleep(1.2)

# 2) 上界 gap_left+4
for i in (1, 2):
    tok, Y, gl = fresh()
    ok, c = chk(gl + 4, Y, tok)
    print('上界 gap_left+4  X=%d -> %s %s' % (gl + 4, c, 'PASS' if ok else 'fail'))
    time.sleep(1.2)

# 3) 下界 gap_left-1
tok, Y, gl = fresh()
ok, c = chk(gl - 1, Y, tok)
print('下界 gap_left-1  X=%d -> %s %s' % (gl - 1, c, 'PASS' if ok else 'fail'))

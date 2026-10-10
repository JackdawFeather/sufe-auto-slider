# -*- coding: utf-8 -*-
"""实测探测：init 取验证码 -> 图像检测缺口 -> check 提交 X=检测值，验证坐标映射与容差。

用法: python probe_check.py [次数] [偏移列表，逗号分隔，默认 "0"]
  例: python probe_check.py 3 0        # 每轮用检测值直接提交
      python probe_check.py 3 0,-2,2   # 每个 token 依次试 检测值+偏移
"""
import base64, io, json, os, sys, time, urllib.request
import numpy as np
import cv2
from detect_gap import white_mask, piece_mask, find_by_components, find_by_template, refine

BASE = 'https://login.sufe.edu.cn/esc-sso'
HDR = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0 Safari/537.36',
    'Accept': 'application/json', 'Content-Type': 'application/json',
    'Origin': 'https://login.sufe.edu.cn', 'Referer': 'https://login.sufe.edu.cn/login/',
}

def api(path, method='GET'):
    req = urllib.request.Request(BASE + path, method=method, headers=HDR)
    if method == 'POST':
        req.data = b'{}'
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode('utf-8'))

def detect_X_and_Y(bg, pc):
    m = white_mask(bg)
    pm = piece_mask(pc)
    ox = np.nonzero(pm)[1].min()
    oy = np.nonzero(pm)[0].min()
    Y = None
    return m, pm, ox, oy

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    offsets = [int(x) for x in (sys.argv[2].split(',') if len(sys.argv) > 2 else ['0'])]
    for rnd in range(1, n + 1):
        try:
            d = api('/api/v3/sliderCaptcha/init', 'POST')
            data = d['data']
            Y = int(data['Y']); token = data['token']
            bg = cv2.imdecode(np.frombuffer(base64.b64decode(data['sourceImage']), np.uint8), cv2.IMREAD_UNCHANGED)
            pc = cv2.imdecode(np.frombuffer(base64.b64decode(data['newImage']), np.uint8), cv2.IMREAD_UNCHANGED)
            m = white_mask(bg); pm = piece_mask(pc)
            ox, oy = np.nonzero(pm)[1].min(), np.nonzero(pm)[0].min()
            comp = find_by_components(m, Y)
            if comp:
                _, gx, gy, gw, gh, ga, _ = comp
                inter, gx, gy = refine(m, pm, gx - ox, gy - oy)
                gap_left = gx
                method = 'component'
            else:
                hit = find_by_template(m, pc)
                if hit is None:
                    print('#%d 检测失败' % rnd); time.sleep(1.5); continue
                gap_left = hit[0] + ox
                method = 'template'
            base_X = gap_left - ox
            line = '#%d Y=%d gap_left=%d piece_off=%d %s' % (rnd, Y, gap_left, ox, method)
            for off in offsets:
                X = base_X + off
                r = api('/api/v3/sliderCaptcha/check?X=%d&Y=%d&token=%s' % (X, Y, token))
                ok = str(r.get('code')) == '0'
                line += ' | X=%d(code=%s%s)%s' % (X, r.get('code'), (',msg=' + str(r.get('msg'))) if not ok else '', ' PASS' if ok else '')
                if ok:
                    break
            print(line)
        except Exception as e:
            print('#%d 异常: %s' % (rnd, e))
        if rnd < n:
            time.sleep(1.5)

if __name__ == '__main__':
    main()

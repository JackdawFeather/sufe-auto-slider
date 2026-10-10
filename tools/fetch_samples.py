# -*- coding: utf-8 -*-
"""从 login.sufe.edu.cn 采集滑块验证码样本（背景图/拼图块/Y/token），并测量白色缺口位置。

用法: python fetch_samples.py [数量] [间隔秒]
样本保存到 ../samples/<序号>/  (bg.png, piece.png, meta.json)
"""
import base64, io, json, os, random, sys, time, urllib.request
import numpy as np
from PIL import Image

BASE = 'https://login.sufe.edu.cn/esc-sso'
HDR = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0 Safari/537.36',
    'Accept': 'application/json',
    'Content-Type': 'application/json',
    'Origin': 'https://login.sufe.edu.cn',
    'Referer': 'https://login.sufe.edu.cn/login/',
}

def api(path, method='GET'):
    req = urllib.request.Request(BASE + path, method=method, headers=HDR)
    if method == 'POST':
        req.data = b'{}'
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode('utf-8'))

def gap_bbox(bg):
    """白色缺口 bbox（缺口被渲染成接近纯白）"""
    rgb = bg[:, :, :3].astype(np.int16)
    white = (rgb[:, :, 0] > 244) & (rgb[:, :, 1] > 244) & (rgb[:, :, 2] > 244)
    ys, xs = np.nonzero(white)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())

def fetch_one(idx, outdir):
    d = api('/api/v3/sliderCaptcha/init', 'POST')
    if str(d.get('code')) != '0':
        print('  init 失败:', d.get('code'), d.get('msg'))
        return None
    data = d['data']
    bg = np.array(Image.open(io.BytesIO(base64.b64decode(data['sourceImage']))))
    piece = Image.open(io.BytesIO(base64.b64decode(data['newImage'])))
    folder = os.path.join(outdir, '%04d' % idx)
    os.makedirs(folder, exist_ok=True)
    Image.fromarray(bg).save(os.path.join(folder, 'bg.png'))
    piece.save(os.path.join(folder, 'piece.png'))
    bb = gap_bbox(bg)
    meta = {
        'token': data['token'], 'Y': data['Y'],
        'bg_size': [bg.shape[1], bg.shape[0]],
        'gap_bbox_white': bb,
    }
    with open(os.path.join(folder, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return meta

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'samples')
    os.makedirs(outdir, exist_ok=True)
    print('start_idx =', next_idx(outdir))
    for i in range(n):
        idx = next_idx(outdir)
        try:
            meta = fetch_one(idx, outdir)
            if meta:
                bb = meta['gap_bbox_white']
                print('#%04d Y=%d gap_x=[%s,%s] gap_y=[%s,%s]' % (
                    idx, meta['Y'], bb[0] if bb else '-', bb[1] if bb else '-',
                    bb[2] if bb else '-', bb[3] if bb else '-'))
        except Exception as e:
            print('#%04d 异常: %s' % (idx, e))
        if i < n - 1:
            time.sleep(random.uniform(interval * 0.8, interval * 1.2))

def next_idx(outdir):
    if not os.path.isdir(outdir):
        return 0
    subs = [d for d in os.listdir(outdir) if d.isdigit()]
    return max([int(d) for d in subs] + [-1]) + 1

if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""实测闭环评估（对真实 login.sufe.edu.cn 接口）：
对 N 个新 token: init → CNN 预测缺口 → 以其为中心的网格探测 → 记录命中所需次数。
同时用教师检测器测 CNN 误差，并对比旧「盲试 100/98/102」策略的假设通过率。
用法: python live_eval.py [N] [间隔秒]
输出: ../output/live_eval.json
"""
import base64, json, os, sys, time
import numpy as np
import cv2
import torch
from probe_check import api
from detect_gap import white_mask, piece_mask, find_by_components, find_by_template, refine
from train_cnn import GapNet, to_input, soft_argmax, IN_W, IN_H

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'output')
GRID = [0, 2, -2, 4, -4, 6, -6, 1, -1, 3, -3, 5, -5, 7, -7]
BLIND = [100, 98, 102]

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
    model = GapNet()
    model.load_state_dict(torch.load(os.path.join(OUT, 'cnn.pt')))
    model.eval()

    recs = []
    for i in range(1, n + 1):
        try:
            d = api('/api/v3/sliderCaptcha/init', 'POST')
            data = d['data']
            Y = int(data['Y']); token = data['token']
            bg = cv2.imdecode(np.frombuffer(base64.b64decode(data['sourceImage']), np.uint8), cv2.IMREAD_UNCHANGED)
            pc = cv2.imdecode(np.frombuffer(base64.b64decode(data['newImage']), np.uint8), cv2.IMREAD_UNCHANGED)
            W = bg.shape[1]
            with torch.no_grad():
                inp = to_input(bg)
                Xt = torch.from_numpy(inp.transpose(2, 0, 1)[None].astype(np.float32)) * 2 - 1
                pred = float(soft_argmax(model(Xt))[0]) * W / IN_W
            # 教师检测（仅评估用）
            m = white_mask(bg); pm = piece_mask(pc)
            ox, oy = np.nonzero(pm)[1].min(), np.nonzero(pm)[0].min()
            comp = find_by_components(m, Y)
            gl = None
            if comp:
                _, gx, gy, _, _, _, _ = comp
                _, gx, gy = refine(m, pm, gx - ox, gy - oy)
                gl = int(gx)
            else:
                hit = find_by_template(m, pc)
                if hit: gl = int(hit[0] + ox)
            base = int(round(pred))
            hit_try, hit_x = None, None
            for tries, off in enumerate(GRID, 1):
                X = base + off
                if X < 0:
                    continue
                r = api('/api/v3/sliderCaptcha/check?X=%d&Y=%d&token=%s' % (X, Y, token))
                if str(r.get('code')) == '0':
                    hit_try, hit_x = tries, X
                    break
            blind_ok = (gl is not None) and any(abs(gl - b) <= 2 for b in BLIND)
            rec = {'i': i, 'W': int(W), 'pred': round(pred, 2), 'base': base,
                   'teacher_gl': gl, 'err': (None if gl is None else round(pred - gl, 2)),
                   'tries': hit_try, 'pass_x': hit_x, 'blind_would_pass': blind_ok}
            recs.append(rec)
            print('#%02d W=%d pred=%.1f teacher_gl=%s err=%s -> %s (tries=%s)  盲试策略:%s' % (
                i, W, pred, gl, rec['err'],
                ('PASS X=%d' % hit_x) if hit_try else 'ALL-FAIL', hit_try,
                '会通过' if blind_ok else '会失败'))
        except Exception as e:
            print('#%02d 异常: %s' % (i, e))
        if i < n:
            time.sleep(interval)

    ok = [r for r in recs if r['tries']]
    errs = [abs(r['err']) for r in recs if r['err'] is not None]
    blind = [r for r in recs if r['blind_would_pass']]
    summary = {
        'n': len(recs), 'pass': len(ok),
        'pass_rate': round(100.0 * len(ok) / max(1, len(recs)), 1),
        'first_try': sum(1 for r in ok if r['tries'] == 1),
        'avg_tries': round(sum(r['tries'] for r in ok) / max(1, len(ok)), 2),
        'max_tries': max([r['tries'] for r in ok], default=None),
        'pred_mae_px': round(sum(errs) / len(errs), 2) if errs else None,
        'pred_max_px': round(max(errs), 2) if errs else None,
        'blind_would_pass': len(blind),
        'details': recs,
    }
    json.dump(summary, open(os.path.join(OUT, 'live_eval.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n== 汇总 ==')
    print('通过率 %d/%d = %.1f%%，首次命中 %d，平均探测 %.2f 次' % (
        summary['pass'], summary['n'], summary['pass_rate'], summary['first_try'], summary['avg_tries']))
    print('CNN 预测误差: MAE %s px, max %s px' % (summary['pred_mae_px'], summary['pred_max_px']))
    print('旧盲试策略在同样本上会通过: %d/%d' % (summary['blind_would_pass'], summary['n']))

if __name__ == '__main__':
    main()

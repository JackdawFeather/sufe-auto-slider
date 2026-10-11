"""在线闭环评测（新整图模型，走 JS 出货路径）。

1) 抓 N 张新验证码 → 存 raw RGB + token/Y
2) node 用扩展里的 sufe-cnn.js 逐张预测
3) 按扩展的候选梯队在同一 token 内实探：[x, x±1, x±2, seg, seg±1]
4) 统计首探测命中率、探测次数、CNN 误差（对分割真值）
"""
import base64
import io
import json
import os
import random
import statistics
import subprocess
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harvest import api, notch_label  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1]) if sys.argv[1:] else 12
REUSE = "reuse" in sys.argv
OUT = os.path.join(HERE, "output", "live2")
os.makedirs(OUT, exist_ok=True)

meta = []
if REUSE and os.path.exists(os.path.join(OUT, "meta.json")):
    meta = json.load(open(os.path.join(OUT, "meta.json"), encoding="utf-8"))
    print(f"复用已抓样本 {len(meta)} 张（token 可能已过期）")
else:
    for k in range(N):
        d = api("/api/v3/sliderCaptcha/init", {})["data"]
        im = Image.open(io.BytesIO(base64.b64decode(d["sourceImage"]))).convert("RGB")
        rgb = np.asarray(im)
        open(os.path.join(OUT, f"{k}.raw"), "wb").write(rgb.tobytes())
        lbl = notch_label(rgb)
        meta.append({"i": k, "w": im.size[0], "h": im.size[1], "Y": d["Y"], "token": d["token"],
                     "segX": (lbl[0] - 3) if lbl else None})
        time.sleep(random.uniform(0.5, 1.0))
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"))
    print(f"抓取 {len(meta)} 张新验证码")

EXT = os.path.dirname(HERE)
js = """
const fs=require('fs'),path=require('path'),vm=require('vm');
global.window={};
vm.runInThisContext(fs.readFileSync(%s,'utf8'));
const S=require(%s);
const OUT=%s;
const meta=JSON.parse(fs.readFileSync(path.join(OUT,'meta.json'),'utf8'));
const out=[];
for(const m of meta){
  const rgb=new Uint8Array(fs.readFileSync(path.join(OUT,m.i+'.raw')));
  const t0=Date.now(); const r=S.locate(rgb,m.w,m.h);
  out.push({i:m.i,x:r.x,conf:r.conf,ms:Date.now()-t0});
}
fs.writeFileSync(path.join(OUT,'js.json'),JSON.stringify(out));
console.log('node: '+out.length+' 张预测完成，平均 '+Math.round(out.reduce((a,b)=>a+b.ms,0)/out.length)+'ms/张');
""" % (json.dumps(os.path.join(EXT, "sufe-cnn-weights.js").replace("\\", "/")),
       json.dumps(os.path.join(EXT, "sufe-cnn.js").replace("\\", "/")),
       json.dumps(OUT.replace("\\", "/")))
open(os.path.join(OUT, "run.js"), "w").write(js)
r = subprocess.run(["node", os.path.join(OUT, "run.js")], capture_output=True, text=True,
                   encoding="utf-8", errors="replace", cwd=HERE)
print((r.stdout or "").strip() or (r.stderr or "").strip()[-500:])
js_res = {c["i"]: c for c in json.load(open(os.path.join(OUT, "js.json")))}

first = passed = 0
errs, probes = [], []
print(f"{'i':>2} {'W':>4} {'jsX':>6} {'segX':>6} {'err':>6} {'probe':>5} {'hit':>6}")
for m in meta:
    c = js_res[m["i"]]
    cx = round(c["x"])
    ladder = [("cnn", cx), ("cnn+1", cx + 1), ("cnn-1", cx - 1), ("cnn+2", cx + 2), ("cnn-2", cx - 2)]
    if m["segX"] is not None:
        ladder += [("seg", m["segX"]), ("seg+1", m["segX"] + 1), ("seg-1", m["segX"] - 1)]
    hit, used = None, 0
    for name, x in ladder:
        if x < 0:
            continue
        used += 1
        resp = api(f"/api/v3/sliderCaptcha/check?X={x}&Y={m['Y']}&token={m['token']}")
        if str(resp.get("code")) == "0":
            hit = (name, x)
            break
    probes.append(used)
    if hit:
        passed += 1
        if used == 1:
            first += 1
    if m["segX"] is not None:
        errs.append(c["x"] - m["segX"])
    print(f"{m['i']:>2} {m['w']:>4} {cx:>6} {str(m['segX']):>6} "
          f"{(c['x']-m['segX'] if m['segX'] is not None else float('nan')):>6.2f} "
          f"{used:>5} {str(hit):>6}")
    time.sleep(0.3)

print(f"\nn={len(meta)} 首 token 通过 {passed}/{len(meta)} 首探测命中 {first}/{len(meta)} "
      f"平均探测 {statistics.mean(probes):.2f}")
if errs:
    ae = [abs(e) for e in errs]
    print(f"CNN 误差(对分割真值): MAE {statistics.mean(ae):.2f}px 最大 {max(ae):.2f}px "
          f"≤2px {sum(1 for e in ae if e <= 2)}/{len(ae)}")

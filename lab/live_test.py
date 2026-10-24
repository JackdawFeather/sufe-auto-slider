"""Live closed-loop test: fresh captcha -> CNN gap X -> /check, N trials."""
import collections
import random
import sys
import time

import numpy as np

from harvest import api, decode, notch_label, canon
from infer import load_net, predict

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
net = load_net()
stat = collections.Counter()
errs = []
for k in range(N):
    d = api("/api/v3/sliderCaptcha/init", {})["data"]
    src0 = np.asarray(decode(d["sourceImage"]).convert("RGB"))
    src, w0 = canon(src0)
    truth = notch_label(src0)
    r = predict(net, src, d["Y"], origW=w0)
    seg = truth[0] - 2 if truth else None
    ladder = [("cnn", r["X"])]
    if seg is not None and seg != r["X"]:
        ladder.append(("seg", seg))
    for dx, nm in ((1, "cnn+1"), (-1, "cnn-1"), (2, "cnn+2"), (-2, "cnn-2")):
        ladder.append((nm, r["X"] + dx))
    done = None
    for nm, x in ladder:
        c = api(f"/api/v3/sliderCaptcha/check?X={x}&Y={d['Y']}&token={d['token']}")
        if str(c.get("code")) == "0":
            done = nm
            break
    stat[done or "FAIL"] += 1
    if truth:
        errs.append(r["X"] - (truth[0] - 2))
    print(f"[{k}] W={w0} cnnX={r['X']} conf={r['conf']:.3f} segX={seg} "
          f"trueX={truth[0]-2 if truth else '?'} err={errs[-1] if truth else '?':+} -> {done}")
    time.sleep(random.uniform(0.8, 1.6))
e = np.array(errs)
print("\nmethod counts:", dict(stat))
print("cnn err px: mae=%.2f |e|<=2:%.2f |e|<=5:%.2f max=%d"
      % (np.abs(e).mean(), (np.abs(e) <= 2).mean(), (np.abs(e) <= 5).mean(), np.abs(e).max()))
print("first-try (cnn) pass rate: %.2f" % (stat["cnn"] / N))
print("overall pass rate: %.2f" % ((N - stat["FAIL"]) / N))

"""Inspect the check API response on a deliberately WRONG X (token rotation?)."""
import json

from harvest import api, decode, notch_label
import numpy as np

d = api("/api/v3/sliderCaptcha/init", {})["data"]
src = np.asarray(decode(d["sourceImage"]).convert("RGB"))
lbl = notch_label(src)
print("notch", lbl, "Y", d["Y"])
good = lbl[0] - 2
wrong = (good + 120) % 380 + 20
r1 = api(f"/api/v3/sliderCaptcha/check?X={wrong}&Y={d['Y']}&token={d['token']}")
print("wrong X=%d ->" % wrong, json.dumps(r1, ensure_ascii=False)[:300])
tok2 = r1.get("data")
if isinstance(tok2, str) and len(tok2) == 32:
    r2 = api(f"/api/v3/sliderCaptcha/check?X={good}&Y={d['Y']}&token={tok2}")
    print("rotated token, good X=%d ->" % good, json.dumps(r2, ensure_ascii=False)[:300])
else:
    r2 = api(f"/api/v3/sliderCaptcha/check?X={good}&Y={d['Y']}&token={d['token']}")
    print("same token, good X=%d ->" % good, json.dumps(r2, ensure_ascii=False)[:300])

"""Probe sourceImage size distribution over N inits (fast, no long sleeps)."""
import collections
import time

from harvest import api, decode

cnt = collections.Counter()
ys = []
for k in range(int(__import__("sys").argv[1]) if __import__("sys").argv[1:] else 25):
    d = api("/api/v3/sliderCaptcha/init", {})["data"]
    im = decode(d["sourceImage"])
    pc = decode(d["newImage"])
    cnt[(im.size, pc.size)] += 1
    ys.append(d["Y"])
    time.sleep(0.4)
for k, v in cnt.items():
    print(k, v)
print("Y range", min(ys), max(ys))

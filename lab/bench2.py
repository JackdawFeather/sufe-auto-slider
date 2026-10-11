"""训练速度/内存基准：用已采集样本测 每个 batch 的耗时。"""
import sys
import time

import numpy as np

from cnn import softmax, soft_target
from train2 import IN_H, IN_W, NCOL, Net2, load_samples

n_use = int(sys.argv[1]) if sys.argv[1:] else 48
bs = int(sys.argv[2]) if sys.argv[2:] else 12

X, C, aux = load_samples()
print("loaded", len(X), "samples", X.shape)
X = X[:n_use]
C = C[:n_use]

net = Net2()
rng = np.random.default_rng(0)
t = 0
for step in range(6):
    xb = X[rng.integers(0, len(X), bs)]
    cb = C[rng.integers(0, len(C), bs)]
    t0 = time.time()
    logits = net.forward(xb)
    probs = softmax(logits)
    tgt = soft_target(np.round(cb).astype(int), NCOL, sigma=1.2)
    loss = -(tgt * np.log(probs + 1e-9)).sum(1).mean()
    net.backward((probs - tgt) / bs)
    net.step(1e-3, t)
    t += 1
    dt = time.time() - t0
    import os
    try:
        import ctypes
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("pb", ctypes.c_ulong)]
        pmc = PMC()
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), ctypes.sizeof(pmc))
        mem = f"commit {pmc.pb/1e6:.0f}MB"
    except Exception:
        mem = ""
    print(f"step{step} bs={bs} {dt:.2f}s loss={float(loss):.3f} {mem}", flush=True)

print(f"\n估算: 320 训练样本池 → 每 epoch {len(X)//bs if len(X)<320 else 320//bs} 步")

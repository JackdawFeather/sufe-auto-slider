"""Timing probe: how slow is one training batch?"""
import time

import numpy as np

from cnn import Net, softmax, soft_target

rng = np.random.default_rng(0)
net = Net()
x = rng.uniform(-0.5, 0.5, (64, 2, 58, 256)).astype(np.float32)
c = rng.integers(0, 128, 64)

t0 = time.time()
lg = net.forward(x)
t1 = time.time()
p = softmax(lg)
t = soft_target(c, 128, 1.0)
g = (p - t) / 64
net.backward(g)
t2 = time.time()
net.step(1e-3, 0)
t3 = time.time()
print(f"fwd={t1-t0:.3f}s bwd={t2-t1:.3f}s step={t3-t2:.3f}s total={t3-t0:.3f}s")
print("per-epoch estimate (20 batches):", 20 * (t3 - t0), "s")

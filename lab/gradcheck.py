"""Numeric gradient check for cnn.py backprop."""
import numpy as np

from cnn import Net, softmax, soft_target

rng = np.random.default_rng(1)
net = Net()
for L in net.params():
    L.W = L.W.astype(np.float64)
    L.b = L.b.astype(np.float64)
x = rng.normal(0, 0.3, (2, 2, 8, 10)).astype(np.float64)
cols = np.array([3, 7])


def loss():
    lg = net.forward(x)
    p = softmax(lg)
    t = soft_target(cols, lg.shape[1], 1.0)
    return -(t * np.log(p + 1e-9)).sum(1).mean()


lg = net.forward(x)
p = softmax(lg)
t = soft_target(cols, lg.shape[1], 1.0)
net.backward((p - t) / len(x))

worst = 0
for L in net.params():
    for name in ("W", "b"):
        P = getattr(L, name)
        G = getattr(L, "g" + name)
        flatP = P.reshape(-1)
        flatG = G.reshape(-1)
        for _ in range(6):
            i = rng.integers(flatP.size)
            h = 1e-6
            old = flatP[i]
            flatP[i] = old + h
            lp = loss()
            flatP[i] = old - h
            lm = loss()
            flatP[i] = old
            num = (lp - lm) / (2 * h)
            ana = flatG[i]
            rel = abs(num - ana) / max(1e-6, abs(num) + abs(ana))
            worst = max(worst, rel)
            print(f"{type(L).__name__}.{name}[{i}] num={num:.6f} ana={ana:.6f} rel={rel:.2e}")
print("WORST rel err:", worst)
assert worst < 5e-3, "gradient check failed (kink-tolerant)"
print("gradcheck OK")

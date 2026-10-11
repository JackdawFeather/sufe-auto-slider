"""Tiny fully-convolutional CNN for slider-gap column detection (numpy, no deps).

Input : (2, 58, 250)  = [gray, saturation] of the horizontal band around Y,
        downsampled x0.5 from the 497-wide source image.
Output: softmax over 125 columns; column c <-> source-image x = 4*c.

Arch: conv(2->10,k3,p1) relu maxpool2x2 -> conv(10->16,k3,p1) relu maxpool-rows
      -> conv1x1(16->1)  =>  (125,) logits
"""
import numpy as np

RNG = np.random.default_rng(7)


# ───────────────────────── primitives ─────────────────────────
def im2col(x, kh, kw, sh, sw, ph, pw):
    n, c, h, w = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode="edge")
    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1
    cols = np.empty((n, oh, ow, c, kh, kw), dtype=x.dtype)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, :, :, i, j] = xp[:, :, i:i + sh * oh:sh, j:j + sw * ow:sw].transpose(0, 2, 3, 1)
    return cols.reshape(n, oh * ow, c * kh * kw), oh, ow


def col2im_add(grad_cols, shape, kh, kw, sh, sw, ph, pw, oh, ow):
    n, c, h, w = shape
    gc = grad_cols.reshape(n, oh, ow, c, kh, kw)
    gp = np.zeros((n, c, h + 2 * ph, w + 2 * pw), dtype=grad_cols.dtype)
    v = gc.transpose(0, 3, 1, 2, 4, 5)  # (n, c, oh, ow, kh, kw)
    for i in range(kh):
        for j in range(kw):
            gp[:, :, i:i + sh * oh:sh, j:j + sw * ow:sw] += v[:, :, :, :, i, j]
    if ph or pw:
        # edge 填充的梯度折回边界像素（与 forward 的 np.pad(mode='edge') 对应）
        gx = gp[:, :, ph:ph + h, pw:pw + w].copy()
        gx[:, :, 0, :] += gp[:, :, :ph, pw:pw + w].sum(2)
        gx[:, :, h - 1, :] += gp[:, :, ph + h:, pw:pw + w].sum(2)
        gx[:, :, :, 0] += gp[:, :, ph:ph + h, :pw].sum(3)
        gx[:, :, :, w - 1] += gp[:, :, ph:ph + h, pw + w:].sum(3)
        gx[:, :, 0, 0] += gp[:, :, :ph, :pw].sum((2, 3))
        gx[:, :, 0, w - 1] += gp[:, :, :ph, pw + w:].sum((2, 3))
        gx[:, :, h - 1, 0] += gp[:, :, ph + h:, :pw].sum((2, 3))
        gx[:, :, h - 1, w - 1] += gp[:, :, ph + h:, pw + w:].sum((2, 3))
        return gx
    return gp


class Conv:
    def __init__(self, cin, cout, k, stride=1, pad=None):
        self.k, self.s = k, stride
        self.p = (k - 1) // 2 if pad is None else pad
        lim = np.sqrt(2.0 / (cin * k * k))
        self.W = RNG.uniform(-lim, lim, (cout, cin * k * k)).astype(np.float32)
        self.b = np.zeros(cout, np.float32)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        self.cols, self.oh, self.ow = im2col(x, self.k, self.k, self.s, self.s, self.p, self.p)
        out = self.cols @ self.W.T + self.b
        n = x.shape[0]
        return out.reshape(n, self.oh, self.ow, -1).transpose(0, 3, 1, 2)

    def backward(self, g):
        n = self.x.shape[0]
        g = g.transpose(0, 2, 3, 1).reshape(n, self.oh * self.ow, -1)
        gr = g.reshape(-1, self.W.shape[0])          # (M, cout)
        cr = self.cols.reshape(-1, self.W.shape[1])  # (M, cin*k*k)
        self.gW = gr.T @ cr                          # (cout, cin*k*k)
        self.gb = gr.sum(0)
        dcols = g @ self.W                           # (n, oh*ow, cin*k*k)
        return col2im_add(dcols, self.x.shape, self.k, self.k, self.s, self.s,
                          self.p, self.p, self.oh, self.ow)

    def step(self, lr, t):
        for P, G, m, v in ((self.W, self.gW, self.mW, self.vW),
                           (self.b, self.gb, self.mb, self.vb)):
            m *= 0.9
            m += 0.1 * G
            v *= 0.999
            v += 0.001 * G * G
            P -= lr * m / (np.sqrt(v) + 1e-8)


class MaxPool2:
    def forward(self, x):
        n, c, h, w = x.shape
        self.x = x
        xr = x.reshape(n, c, h // 2, 2, w // 2, 2).transpose(0, 1, 2, 4, 3, 5)
        xr = xr.reshape(n, c, h // 2, w // 2, 4)
        self.idx = xr.argmax(-1)
        return np.take_along_axis(xr, self.idx[..., None], -1)[..., 0]

    def backward(self, g):
        n, c, h, w = self.x.shape
        out = np.zeros((n, c, h // 2, w // 2, 4), dtype=g.dtype)
        np.put_along_axis(out, self.idx[..., None], g[..., None], -1)
        return out.reshape(n, c, h // 2, w // 2, 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, h, w)


class MaxPoolRows:
    """max over the height axis: (n,c,h,w) -> (n,c,1,w)"""

    def forward(self, x):
        self.idx = x.argmax(2, keepdims=True)
        self.shape = x.shape
        return np.take_along_axis(x, self.idx, 2)

    def backward(self, g):
        out = np.zeros(self.shape, dtype=g.dtype)
        np.put_along_axis(out, self.idx, g, 2)
        return out


class ReLU:
    def forward(self, x):
        self.m = x > 0
        return x * self.m

    def backward(self, g):
        return g * self.m


class Conv1x1Out(Conv):
    """1x1 conv on (n,c,1,w) producing (n,1,1,w) logits == per-column scores."""

    def __init__(self, cin):
        super().__init__(cin, 1, 1, 1, 0)


class Net:
    def __init__(self):
        self.layers = [Conv(2, 10, 3), ReLU(), MaxPool2(),
                       Conv(10, 16, 3), ReLU(), MaxPoolRows(),
                       Conv1x1Out(16)]

    def forward(self, x):
        for L in self.layers:
            x = L.forward(x)
        return x[:, 0, 0, :]  # (n, W')

    def backward(self, g):
        g = g[:, None, None, :]
        for L in reversed(self.layers):
            g = L.backward(g)
        return g

    def params(self):
        return [L for L in self.layers if isinstance(L, Conv)]

    def step(self, lr, t):
        for L in self.params():
            L.step(lr, t)


def softmax(z, axis=-1):
    z = z - z.max(axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis, keepdims=True)


def soft_target(cols, ncol, sigma=1.0):
    t = np.zeros((len(cols), ncol), np.float32)
    idx = np.arange(ncol)
    for i, c in enumerate(cols):
        t[i] = np.exp(-0.5 * ((idx - c) / sigma) ** 2)
    t /= t.sum(1, keepdims=True)
    return t

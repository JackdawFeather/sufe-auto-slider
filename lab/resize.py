"""面积平均缩放（area resize），与 JS 端 sufe-cnn.js 的 resizeArea 数学等价。

分离式实现：先沿 y 再把沿 x，权重=区间重叠长度（归一化），等价于二维面积积分。
"""
import numpy as np


def axis_weights(src_n, dst_n):
    """返回 (dst_n, src_n) 权重矩阵：dst i ← src j 的重量 = 重叠长度 / dst 区间长度。"""
    sr = src_n / dst_n
    W = np.zeros((dst_n, src_n), np.float64)
    for i in range(dst_n):
        a, b = i * sr, (i + 1) * sr
        j0, j1 = int(np.floor(a)), int(np.ceil(b))
        for j in range(j0, min(j1, src_n)):
            ov = min(j + 1, b) - max(j, a)
            if ov > 0:
                W[i, j] = ov / (b - a)
    return W


def area_resize(img, dw, dh):
    """img: (h,w,3) float 或 uint8 → (dh,dw,3) float32（保留取值尺度）。"""
    h, w = img.shape[:2]
    Wy = axis_weights(h, dh)   # (dh, h)
    Wx = axis_weights(w, dw)   # (dw, w)
    out = np.einsum('ij,jkl->ikl', Wy, img.astype(np.float64))   # (dh, w, 3)
    out = np.einsum('ij,kjl->kil', Wx, out)                      # (dh, dw, 3)
    return out.astype(np.float32)


def prep_full(rgb, in_w=256, in_h=128):
    """原图 RGB uint8 (h,w,3) → CHW float32 [-1,1] (3,in_h,in_w)。"""
    r = area_resize(rgb, in_w, in_h)
    x = np.transpose(r / 255.0 * 2 - 1, (2, 0, 1)).astype(np.float32)
    return x

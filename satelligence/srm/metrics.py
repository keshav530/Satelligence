"""Quality + spectral-consistency metrics. LPIPS needs torch and a pretrained net: TODO."""
import numpy as np


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def _box(x, k):
    c = np.cumsum(np.cumsum(np.pad(x, ((1, 0), (1, 0))), 0), 1)
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def ssim(a, b, k=7):
    """Mean SSIM of two 2-D arrays in [0,1], uniform window."""
    c1, c2 = .01 ** 2, .03 ** 2
    ma, mb = _box(a, k), _box(b, k)
    va, vb = _box(a * a, k) - ma ** 2, _box(b * b, k) - mb ** 2
    cov = _box(a * b, k) - ma * mb
    return float(np.mean(((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))))


def luma(img):
    return .299 * img[0] + .587 * img[1] + .114 * img[2]


def ndvi(img):
    return (img[3] - img[0]) / (img[3] + img[0] + 1e-6)


def ndvi_mae(sr, hr):
    """Spectral check: how far super-resolved NDVI drifts from reference NDVI."""
    return float(np.mean(np.abs(ndvi(sr) - ndvi(hr))))


def report(sr, hr):
    return {"psnr": psnr(sr[:3], hr[:3]), "ssim": ssim(luma(sr), luma(hr)), "ndvi_mae": ndvi_mae(sr, hr)}

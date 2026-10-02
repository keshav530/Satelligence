import numpy as np
from srm.data import synthetic_scene, degrade, tile, normalize
from srm.baseline import bicubic
from srm.metrics import psnr, ssim, ndvi, report

def test_shapes():
    hr = synthetic_scene(128, 1); lr = degrade(hr, 4)
    assert hr.shape == (4, 128, 128) and lr.shape == (4, 32, 32)
    assert bicubic(lr, 4).shape == hr.shape

def test_tiling_count():
    assert len(list(tile(synthetic_scene(128), 64, 32))) == 9

def test_metrics_identity():
    hr = synthetic_scene(64, 2)
    assert psnr(hr, hr) == float("inf") and abs(ssim(hr[0], hr[0]) - 1) < 1e-6

def test_bicubic_beats_nearest():
    hr = synthetic_scene(128, 3); lr = degrade(hr, 4)
    nn = np.kron(lr, np.ones((1, 4, 4), np.float32))
    assert report(bicubic(lr), hr)["psnr"] > report(nn, hr)["psnr"]

def test_ndvi_range():
    v = ndvi(synthetic_scene(64, 4)); assert v.min() >= -1 and v.max() <= 1

def test_normalize_bounds():
    n = normalize(synthetic_scene(64, 5)); assert n.min() >= 0 and n.max() <= 1

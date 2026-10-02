"""Bicubic baseline: the number the learned model has to beat."""
import numpy as np
from PIL import Image


def bicubic(lr, scale=4):
    """(C,h,w) float -> (C,h*scale,w*scale) float, band by band (PIL mode 'F')."""
    bands = []
    for band in lr:
        im = Image.fromarray(band.astype(np.float32), mode="F")
        bands.append(np.asarray(im.resize((band.shape[1] * scale, band.shape[0] * scale), Image.BICUBIC)))
    return np.clip(np.stack(bands), 0, 1).astype(np.float32)

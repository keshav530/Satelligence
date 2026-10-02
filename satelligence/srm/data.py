"""Data utilities: synthetic scenes (stand-in until Sentinel-2/PlanetScope pairs are curated),
degradation to 10 m, tiling and band normalisation. Bands are (R, G, B, NIR) in [0, 1]."""
import numpy as np

# class means: R, G, B, NIR
CLS = {
    "crop1": (.22, .42, .18, .62), "crop2": (.30, .46, .20, .55), "bare": (.52, .44, .32, .40),
    "water": (.08, .18, .30, .04), "road": (.42, .42, .43, .38), "roof": (.58, .35, .30, .36),
}


def synthetic_scene(size=128, seed=0):
    """Fake 'sub-4 m truth': parcels, a river, roads and small buildings. Returns (4, H, W) float32."""
    rng = np.random.default_rng(seed)
    lab = np.empty((size, size), dtype=object)
    x = 0
    while x < size:
        w = int(rng.integers(20, 40)); y = 0
        while y < size:
            h = int(rng.integers(20, 40))
            lab[y:y + h, x:x + w] = rng.choice(["crop1", "crop2", "bare"])
            y += h
        x += w
    xs = np.arange(size)
    cy = (size * .62 + np.sin(xs * .07 + seed) * 10).astype(int)
    for xi in xs:
        lab[max(cy[xi] - 4, 0):cy[xi] + 4, xi] = "water"
    ry, rx = int(rng.integers(25, 40)), int(rng.integers(30, 90))
    lab[ry:ry + 2, :] = "road"; lab[:, rx:rx + 2] = "road"
    for _ in range(40):
        by, bx = rng.integers(2, size - 6, 2)
        if lab[by, bx] != "water":
            lab[by:by + 3, bx:bx + 4] = "roof"
    img = np.zeros((4, size, size), np.float32)
    for name, mean in CLS.items():
        m = lab == name
        for b in range(4):
            img[b][m] = mean[b]
    img += rng.normal(0, .03, img.shape).astype(np.float32)
    return np.clip(img, 0, 1)


def degrade(hr, scale=4):
    """Box-average downsample, a crude model of a 10 m sensor footprint. (C,H,W) -> (C,H/s,W/s)."""
    c, h, w = hr.shape
    return hr.reshape(c, h // scale, scale, w // scale, scale).mean(axis=(2, 4))


def tile(img, size=64, stride=32):
    """Yield (y, x, tile) windows over a (C,H,W) array."""
    _, h, w = img.shape
    for y in range(0, h - size + 1, stride):
        for x in range(0, w - size + 1, stride):
            yield y, x, img[:, y:y + size, x:x + size]


def normalize(img, lo=2, hi=98):
    """Per-band percentile stretch to [0, 1]."""
    out = np.empty_like(img, dtype=np.float32)
    for b in range(img.shape[0]):
        a, z = np.percentile(img[b], [lo, hi])
        out[b] = np.clip((img[b] - a) / max(z - a, 1e-6), 0, 1)
    return out

"""End-to-end baseline run: synthetic truth -> 10 m -> bicubic -> metrics + side-by-side PNG."""
import sys, pathlib
import numpy as np
from PIL import Image
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from srm.data import synthetic_scene, degrade
from srm.baseline import bicubic
from srm.metrics import report

def rgb(a):
    return (np.clip(a[:3].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    hr = synthetic_scene(128, seed)
    lr = degrade(hr, 4)
    sr = bicubic(lr, 4)
    m = report(sr, hr)
    print(f"seed={seed}  PSNR={m['psnr']:.2f} dB  SSIM={m['ssim']:.3f}  NDVI-MAE={m['ndvi_mae']:.3f}")
    nn = np.kron(lr, np.ones((1, 4, 4), np.float32))
    out = pathlib.Path(__file__).resolve().parents[1] / "outputs" / f"demo_{seed}.png"
    Image.fromarray(np.hstack([rgb(nn), rgb(sr), rgb(hr)])).save(out)
    print("wrote", out, "(left: 10 m input | middle: bicubic | right: reference)")

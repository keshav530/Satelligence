# Satelligence: Deep Learning Super-Resolution Mapping (SIH 26142)

x4 super-resolution of 10 m Sentinel-2 style imagery to sub-4 m, with a per-pixel confidence map.
**Status: ~10% (v0.1).** Baseline pipeline + metrics + model scaffold. The learned model is not trained.

## Run it
```bash
pip install numpy Pillow pytest
python scripts/demo.py 0        # synthetic scene -> 10 m -> bicubic -> metrics + outputs/demo_0.png
pytest -q                       # 6 tests
open web/index.html             # offline dev dashboard (same pipeline in JS)
# with torch + fastapi installed:
uvicorn api.main:app --reload   # GET /health, POST /upscale (bicubic for now)
```

## Layout
| Path | What |
|---|---|
| `srm/data.py` | synthetic scenes, 10 m degradation, tiling, percentile normalisation |
| `srm/baseline.py` | bicubic x4 baseline |
| `srm/metrics.py` | PSNR, SSIM, NDVI and NDVI-MAE spectral check |
| `srm/models.py` | RRDB generator + MC-dropout helper (untrained) |
| `api/main.py` | FastAPI stub |
| `web/index.html` | dev dashboard: layer compare slider, live metrics, build status |

## Baseline numbers (synthetic, seed 0)
PSNR 26.6 dB, SSIM 0.696, NDVI-MAE 0.092. The trained model has to beat these.

## Roadmap
- [x] Tiling, normalisation, degradation model
- [x] Bicubic baseline, PSNR/SSIM/NDVI checks, tests
- [~] RRDB generator (defined), FastAPI (baseline only)
- [ ] Curate Sentinel-2 / PlanetScope pairs, sub-pixel co-registration
- [ ] Swin block, discriminator, GAN training, LPIPS
- [ ] MC-dropout confidence calibration
- [ ] Leaflet dashboard, Docker, deployment

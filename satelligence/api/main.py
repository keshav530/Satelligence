"""FastAPI stub. /upscale serves the bicubic baseline until the trained model exists.
Run: uvicorn api.main:app --reload"""
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from PIL import Image
from srm.baseline import bicubic

app = FastAPI(title="Satelligence SRM", version="0.1.0-dev")


@app.get("/health")
def health():
    return {"status": "ok", "model": "bicubic-baseline", "trained": False}


@app.post("/upscale")
async def upscale(file: UploadFile = File(...), scale: int = 4):
    # TODO: swap in Generator, return confidence map alongside the image
    im = Image.open(io.BytesIO(await file.read())).convert("RGB")
    lr = np.asarray(im, np.float32).transpose(2, 0, 1) / 255
    sr = (bicubic(lr, scale).transpose(1, 2, 0) * 255).astype(np.uint8)
    buf = io.BytesIO(); Image.fromarray(sr).save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png")

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from . import detector

app = FastAPI(title="CricVision API")

MAX_UPLOAD = 8 * 1024 * 1024  # 8 MB


@app.get("/api/health")
def health():
    return {"status": "ok", "weights": detector.WEIGHTS}


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "image larger than 8 MB")
    if not data:
        raise HTTPException(400, "empty upload")
    try:
        return detector.detect(data)
    except Exception as exc:  # bad image bytes, unsupported format
        raise HTTPException(400, f"could not process image: {exc}") from exc


# Serve the frontend from the same origin (mounted after the API routes)
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

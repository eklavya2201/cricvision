from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import detector, video

app = FastAPI(title="CricVision API")

MAX_UPLOAD = 8 * 1024 * 1024  # 8 MB
MAX_VIDEO_UPLOAD = 50 * 1024 * 1024  # 50 MB


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


@app.post("/api/video")
async def video_upload(file: UploadFile = File(...), pitch_len_px: float | None = Form(None)):
    data = await file.read()
    if len(data) > MAX_VIDEO_UPLOAD:
        raise HTTPException(413, "clip larger than 50 MB")
    if not data:
        raise HTTPException(400, "empty upload")
    try:
        job_id = video.start_job(data, pitch_len_px)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"job_id": job_id}


@app.get("/api/video/{job_id}")
def video_status(job_id: str):
    job = video.get_job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return {"status": job["status"], "progress": job["progress"],
            "result": job["result"], "error": job["error"]}


@app.get("/api/video/{job_id}/annotated.mp4")
def video_file(job_id: str):
    job = video.get_job(job_id)
    if job is None or job["status"] != "done":
        raise HTTPException(404, "no annotated clip for this job")
    return FileResponse(job["dir"] / "annotated.mp4", media_type="video/mp4")


# Serve the frontend from the same origin (mounted after the API routes)
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

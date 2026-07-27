import os
import tempfile
import time

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _clip_bytes(frames=16, size=(320, 240), fps=10) -> bytes:
    """Tiny synthetic clip: white ball moving across a green pitch."""
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(frames):
        frame = np.full((size[1], size[0], 3), (60, 130, 60), np.uint8)
        cv2.circle(frame, (20 + i * 15, size[1] // 2), 10, (255, 255, 255), -1)
        w.write(frame)
    w.release()
    with open(path, "rb") as f:
        data = f.read()
    os.unlink(path)
    return data


def _wait_done(job_id, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/video/{job_id}").json()
        if r["status"] != "processing":
            return r
        time.sleep(0.5)
    raise TimeoutError("video job did not finish in time")


def test_video_pipeline_end_to_end():
    clip = _clip_bytes()
    r = client.post("/api/video", files={"file": ("clip.mp4", clip, "video/mp4")},
                    data={"pitch_len_px": "200"})
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    status = _wait_done(job_id)
    assert status["status"] == "done", status["error"]
    result = status["result"]
    assert result["frames_processed"] == 16
    assert result["fps"] == 10
    assert result["unique_players"] >= 0
    assert set(result["ball"]) == {"sightings", "speed_px_s_max", "speed_px_s_avg",
                                   "speed_kmh_max", "speed_kmh_avg"}
    assert result["heatmap_jpeg_b64"]

    mp4 = client.get(f"/api/video/{job_id}/annotated.mp4")
    assert mp4.status_code == 200
    assert mp4.headers["content-type"] == "video/mp4"
    assert len(mp4.content) > 1000


def test_garbage_video_rejected():
    r = client.post("/api/video", files={"file": ("x.mp4", b"not a video", "video/mp4")})
    assert r.status_code == 400


def test_empty_video_rejected():
    r = client.post("/api/video", files={"file": ("x.mp4", b"", "video/mp4")})
    assert r.status_code == 400


def test_oversize_video_rejected():
    big = b"0" * (50 * 1024 * 1024 + 1)
    r = client.post("/api/video", files={"file": ("big.mp4", big, "video/mp4")})
    assert r.status_code == 413


def test_unknown_job_404():
    assert client.get("/api/video/nope").status_code == 404
    assert client.get("/api/video/nope/annotated.mp4").status_code == 404

"""Phase 2: video processing.

Per-frame YOLO detection with ByteTrack IDs, stitched into an annotated H.264
clip (imageio-ffmpeg — cv2 on Windows can't encode browser-playable H.264),
plus ball speed from frame deltas and a player-position heatmap.

Jobs run in a background thread; an in-memory registry holds status/progress.
"""
import base64
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from . import detector

MAX_FRAMES = 300          # cap CPU work per clip
MAX_SIDE = 960            # downscale long side before inference
PITCH_LENGTH_M = 20.12    # stumps to stumps (22 yards) — for px→km/h calibration
BALL_MAX_GAP = 5          # frames a ball may go undetected and still count as one flight
MAX_JOBS = 6              # evict oldest finished job (and its temp dir) beyond this

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def get_job(job_id: str) -> dict | None:
    with _lock:
        return _jobs.get(job_id)


def start_job(video_bytes: bytes, pitch_len_px: float | None) -> str:
    """Raises ValueError if the bytes are not a decodable video."""
    job_id = uuid.uuid4().hex[:12]
    workdir = Path(tempfile.mkdtemp(prefix="cricvision_"))
    src = workdir / "input.mp4"
    src.write_bytes(video_bytes)
    cap = cv2.VideoCapture(str(src))
    ok, _ = cap.read()
    cap.release()
    if not ok:
        shutil.rmtree(workdir, ignore_errors=True)
        raise ValueError("could not decode video — upload an MP4/WebM clip")
    job = {"status": "processing", "progress": 0.0, "result": None,
           "error": None, "dir": workdir, "created": time.time()}
    with _lock:
        _jobs[job_id] = job
        _evict_locked()
    threading.Thread(target=_run, args=(job, src, pitch_len_px), daemon=True).start()
    return job_id


def _evict_locked():
    finished = [k for k, j in _jobs.items() if j["status"] != "processing"]
    while len(_jobs) > MAX_JOBS and finished:
        oldest = min(finished, key=lambda k: _jobs[k]["created"])
        finished.remove(oldest)
        shutil.rmtree(_jobs.pop(oldest)["dir"], ignore_errors=True)


def _run(job: dict, src: Path, pitch_len_px: float | None):
    try:
        job["result"] = _process(job, src, pitch_len_px)
        job["status"] = "done"
        job["progress"] = 1.0
    except Exception as exc:
        job["status"] = "error"
        job["error"] = str(exc)


def _process(job: dict, src: Path, pitch_len_px: float | None) -> dict:
    start = time.perf_counter()
    from ultralytics import YOLO
    model = YOLO(detector.WEIGHTS)  # fresh model per job: ByteTrack state must not leak across clips

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1:  # 0 or NaN on broken headers
        fps = 25.0
    total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or MAX_FRAMES, MAX_FRAMES)

    out_path = job["dir"] / "annotated.mp4"
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264",
                                quality=7, macro_block_size=1)

    ball_pts: list[tuple[int, float, float]] = []  # (frame_idx, cx, cy)
    player_ids: set[int] = set()
    heat = None
    first_frame = None

    n = 0
    while n < MAX_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if max(h, w) > MAX_SIDE:
            s = MAX_SIDE / max(h, w)
            frame = cv2.resize(frame, (int(w * s) // 2 * 2, int(h * s) // 2 * 2))
        if first_frame is None:
            first_frame = frame.copy()
            heat = np.zeros(frame.shape[:2], np.float32)

        r = model.track(frame, persist=True, conf=0.25, classes=detector.CLASS_FILTER,
                        tracker="bytetrack.yaml", verbose=False)[0]
        for box in r.boxes:
            label = detector.label_for(int(box.cls[0]), r.names)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if label in detector.BALL_LABELS:
                ball_pts.append((n, (x1 + x2) / 2, (y1 + y2) / 2))
            elif label not in detector.NON_PLAYER_LABELS:
                if box.id is not None:
                    player_ids.add(int(box.id[0]))
                # feet position (bottom-center) → heatmap
                cy = min(int(y2), heat.shape[0] - 1)
                cx = min(int((x1 + x2) / 2), heat.shape[1] - 1)
                heat[cy, cx] += 1.0

        writer.append_data(r.plot()[..., ::-1])  # BGR → RGB
        n += 1
        job["progress"] = round(n / total, 3)

    cap.release()
    writer.close()
    if n == 0:
        raise ValueError("could not decode any frames from the clip")

    return {
        "frames_processed": n,
        "fps": round(fps, 2),
        "duration_s": round(n / fps, 2),
        "unique_players": len(player_ids),
        "ball": _ball_stats(ball_pts, fps, pitch_len_px),
        "heatmap_jpeg_b64": _heatmap_b64(first_frame, heat),
        "latency_ms": int((time.perf_counter() - start) * 1000),
    }


def _ball_stats(pts, fps, pitch_len_px):
    """Speed from displacement between consecutive ball sightings (frame delta)."""
    speeds = []
    for (f0, x0, y0), (f1, x1, y1) in zip(pts, pts[1:]):
        gap = f1 - f0
        if 0 < gap <= BALL_MAX_GAP:
            dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            speeds.append(dist * fps / gap)
    # median-of-3 smoothing knocks out single-frame jitter spikes
    smoothed = [sorted(speeds[max(0, i - 1):i + 2])[len(speeds[max(0, i - 1):i + 2]) // 2]
                for i in range(len(speeds))]
    stats = {
        "sightings": len(pts),
        "speed_px_s_max": round(max(smoothed), 1) if smoothed else None,
        "speed_px_s_avg": round(sum(smoothed) / len(smoothed), 1) if smoothed else None,
        "speed_kmh_max": None,
        "speed_kmh_avg": None,
    }
    if smoothed and pitch_len_px and pitch_len_px > 0:
        to_kmh = PITCH_LENGTH_M / pitch_len_px * 3.6
        stats["speed_kmh_max"] = round(stats["speed_px_s_max"] * to_kmh, 1)
        stats["speed_kmh_avg"] = round(stats["speed_px_s_avg"] * to_kmh, 1)
    return stats


def _heatmap_b64(first_frame, heat) -> str:
    """Player-position density blended over the first frame (fixed-camera assumption)."""
    sigma = max(9, (heat.shape[1] // 40) | 1)  # odd kernel scaled to frame width
    blurred = cv2.GaussianBlur(heat, (0, 0), sigmaX=sigma / 3)
    if blurred.max() > 0:
        blurred = blurred / blurred.max()
    colored = cv2.applyColorMap((blurred * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    # only tint where there is signal, keep the rest of the frame clean
    alpha = (blurred[..., None] * 0.6).clip(0, 0.6)
    overlay = (first_frame * (1 - alpha) + colored * alpha).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode()

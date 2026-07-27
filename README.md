# CricVision 🏏

[![CI](https://github.com/eklavya2201/cricvision/actions/workflows/ci.yml/badge.svg)](https://github.com/eklavya2201/cricvision/actions/workflows/ci.yml)

**Real computer vision on cricket footage.** Upload a match photo — or a short clip — and CricVision runs YOLO object detection locally: every player and the ball boxed with confidences, ByteTrack IDs stitched across frames into an annotated video, ball speed estimated from frame deltas, and a player-position heatmap. No API keys, no cloud inference.

> Unlike my LLM projects ([InterviewOps](https://github.com/eklavya2201/InterviewOPS), [TokenMeter](https://github.com/eklavya2201/Token-Meter), [agenttrace](https://github.com/eklavya2201/agenttrace)) which orchestrate hosted models, CricVision is the *classical ML* track: model weights on disk, tensor inference on your CPU, and a roadmap that ends in fine-tuning on my own labeled dataset.

## How it works

```
photo upload ──► YOLOv8n (COCO-pretrained, runs locally)
                   │  classes filtered: person → "player", sports ball → "ball"
                   ▼
        boxes + confidences + annotated frame ──► scoreboard UI

video upload ──► background job: per-frame YOLOv8n + ByteTrack (persistent IDs)
                   │  ball centres per frame ──► speed from frame deltas
                   │  player foot positions ──► Gaussian-blurred density heatmap
                   ▼
        annotated H.264 clip + speed stats + heatmap ──► video UI (progress-polled)
```

- **Model**: `yolov8n` (nano) — 6 MB of weights, fast enough for CPU inference (~1s per image)
- **Classes**: COCO `person` and `sports ball` map cleanly onto cricket frames out of the box
- Uploads are validated (type, size ≤ 8 MB images / 50 MB clips); images are never stored, video jobs live in a temp dir that's evicted automatically

## Run locally

```bash
cd backend
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt                   # pulls CPU torch + ultralytics (~1 GB)
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000**, drop in any cricket photo, hit **Detect**. First run downloads the 6 MB weights automatically.

## API

| Endpoint | What it does |
|---|---|
| `POST /api/detect` | multipart image → `{detections[], players, balls, latency_ms, annotated_jpeg_b64}` |
| `POST /api/video` | multipart clip (≤ 50 MB, first 300 frames) + optional `pitch_len_px` → `{job_id}`; processing runs in the background |
| `GET /api/video/{job_id}` | `{status, progress, result, error}` — result has `frames_processed`, `unique_players` (ByteTrack IDs), `ball` speed stats, `heatmap_jpeg_b64` |
| `GET /api/video/{job_id}/annotated.mp4` | the stitched, annotated H.264 clip |
| `GET /api/health` | `{status, weights}` |

**Ball speed honesty note**: without calibration the speed is reported in px/s. Pass `pitch_len_px` (pixel distance stumps→stumps, 20.12 m) and it converts to km/h — a rough single-plane estimate, not Hawk-Eye.

## Tests

10 API tests (valid shape on real inference, video pipeline end-to-end on a synthetic clip, empty/garbage/oversize rejection) run on every push — CI performs actual YOLO inference and ByteTrack tracking.

```bash
pip install -r backend/requirements-dev.txt
cd backend && python -m pytest tests -q
```

## Roadmap: pretrained → fine-tuned

### Phase 1 — Working detector ✅ (shipped)
- [x] YOLOv8n inference API + annotated output
- [x] Scoreboard UI with drag-and-drop upload
- [x] Test suite + CI (real inference in CI)

### Phase 2 — Video & tracking ✅ (shipped)
- [x] Short-clip upload: per-frame detection stitched into an annotated video
- [x] Ball trajectory tracking across frames (ByteTrack), speed estimate from frame delta
- [x] Player heatmap over a fixed-camera clip

**How Phase 2 works** (`backend/app/video.py`):

- **Async jobs, not blocking requests** — a clip kicks off a background thread and returns a `job_id`; the UI polls progress (0→100%) every second. Jobs live in an in-memory registry; each gets its own temp dir, evicted oldest-first beyond 6 jobs.
- **Per-frame tracking** — first 300 frames, downscaled to ≤960px for CPU latency, run through `model.track()` with the ByteTrack tracker. A **fresh YOLO model per job** keeps tracker state from leaking between clips. `unique_players` = distinct ByteTrack IDs seen.
- **Ball speed** — consecutive ball sightings (up to a 5-frame detection gap, because a small fast ball drops out of frames) give displacement ÷ time = px/s, median-of-3 smoothed to kill single-frame jitter spikes. Optional `pitch_len_px` calibration converts to km/h.
- **Heatmap** — each player's foot position (bottom-centre of their box) accumulates into a density grid, Gaussian-blurred with a kernel scaled to frame width, rendered as a TURBO colormap alpha-blended over the first frame (fixed-camera assumption).
- **H.264 stitching via `imageio-ffmpeg`** — OpenCV on Windows ships without OpenH264, so `cv2.VideoWriter` mp4s won't play in browsers; imageio's bundled ffmpeg encodes real `libx264`.
- **Tested end-to-end** — CI runs the whole pipeline (real inference + tracking + stitching) on a synthetic moving-ball clip, plus rejection tests for garbage/empty/oversize uploads.

### Phase 3 — Fine-tuning on my own dataset (the real ML)

The full training loop, documented as it happens. Each step has a concrete finish line.

**3.1 — Data collection**
- [ ] Extract frames from match footage with a sampling script (1 frame/2s, dedup near-identical frames by perceptual hash)
- [ ] Target: **1,500+ frames** across formats (Tests/red ball, ODI/white ball), lighting (day/night), and camera angles (broadcast main, side-on)
  - *Done when: `data/raw/` has ≥1,500 diverse frames with a provenance log (source clip, timestamp)*

**3.2 — Labeling**
- [ ] Label in **Label Studio** with a 7-class schema: `batsman / bowler / wicketkeeper / umpire / fielder / ball / stumps`
- [ ] Write a labeling guide first (what counts as "batsman" mid-runout? partial occlusion rules? min box size for the ball) — consistency beats volume
- [ ] Double-label a 10% sample to measure my own annotation agreement; re-label classes below 90% agreement
  - *Done when: every frame labeled, exported to YOLO format, agreement measured and reported*

**3.3 — Dataset engineering**
- [ ] Split **70/20/10 train/val/test by match, not by frame** — frames from the same match must never cross splits (leakage would inflate mAP)
- [ ] Address ball scarcity (tiny object, few pixels): mosaic + copy-paste augmentation, higher input resolution (`imgsz=1280`) for the ball's sake
  - *Done when: `dataset.yaml` checked in, class-count table per split published in the README*

**3.4 — Training**
- [ ] Baseline first: evaluate **pretrained COCO YOLOv8n** on my test set (person→player-classes collapsed, sports-ball→ball) — this is the number to beat
- [ ] Fine-tune YOLOv8n, then YOLOv8s, from pretrained weights: ~100 epochs, early stopping on val mAP, default hyperparameters before any tuning
- [ ] Train on free GPU (Kaggle/Colab); keep the exact notebook + seed in `training/` so results are reproducible
  - *Done when: training curves (box/cls loss, mAP@50) committed as images, best weights published as a GitHub release*

**3.5 — Evaluation & error analysis**
- [ ] Publish the headline table: **mAP@50 and mAP@50-95, per class, pretrained-baseline vs fine-tuned** — including where fine-tuning *doesn't* help
- [ ] Confusion matrix across the 7 classes (where does `fielder` bleed into `batsman`?)
- [ ] Honest failure gallery: motion-blurred balls, crowd occlusion, night-game white ball vs floodlights, umpire/keeper confusion
  - *Done when: a reader can tell exactly how good the model is, per class, and where it breaks*

**3.6 — Ship it**
- [ ] Swap the app's weights to the fine-tuned model behind a `CRICVISION_WEIGHTS` env var (pretrained stays the default fallback)
- [ ] Upload weights + dataset card to Hugging Face Hub; deploy the app to HF Spaces
  - *Done when: the live demo detects `batsman` vs `bowler` — something COCO fundamentally cannot do*

### Phase 4 — Cricket intelligence
- [ ] Shot classification (drive/pull/cut) from pose keypoints
- [ ] Auto-highlights: detect boundaries/wickets from detection + scoreboard OCR
- [ ] Win-probability model fed by extracted match state

## Why this project

Anyone can call a hosted LLM API. Training and evaluating your own vision model — data collection, labeling, fine-tuning, mAP, error analysis — is the proof of real ML fundamentals. Phase 1 ships a working product; Phase 3 is where the depth lives.

# CricVision 🏏

[![CI](https://github.com/eklavya2201/cricvision/actions/workflows/ci.yml/badge.svg)](https://github.com/eklavya2201/cricvision/actions/workflows/ci.yml)

**Real computer vision on cricket footage.** Upload a match photo and CricVision runs YOLO object detection locally — every player and the ball boxed with confidences, rendered in a scoreboard-style UI. No API keys, no cloud inference.

> Unlike my LLM projects ([InterviewOps](https://github.com/eklavya2201/InterviewOPS), [TokenMeter](https://github.com/eklavya2201/Token-Meter), [agenttrace](https://github.com/eklavya2201/agenttrace)) which orchestrate hosted models, CricVision is the *classical ML* track: model weights on disk, tensor inference on your CPU, and a roadmap that ends in fine-tuning on my own labeled dataset.

## How it works

```
photo upload ──► YOLOv8n (COCO-pretrained, runs locally)
                   │  classes filtered: person → "player", sports ball → "ball"
                   ▼
        boxes + confidences + annotated frame ──► scoreboard UI
```

- **Model**: `yolov8n` (nano) — 6 MB of weights, fast enough for CPU inference (~1s per image)
- **Classes**: COCO `person` and `sports ball` map cleanly onto cricket frames out of the box
- Uploads are validated (type, size ≤ 8 MB) and never stored — inference is stateless

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
| `GET /api/health` | `{status, weights}` |

## Tests

5 API tests (valid shape on real inference, empty/garbage/oversize rejection) run on every push — CI performs actual YOLO inference on a generated image.

```bash
pip install -r backend/requirements-dev.txt
cd backend && python -m pytest tests -q
```

## Roadmap: pretrained → fine-tuned

### Phase 1 — Working detector ✅ (shipped)
- [x] YOLOv8n inference API + annotated output
- [x] Scoreboard UI with drag-and-drop upload
- [x] Test suite + CI (real inference in CI)

### Phase 2 — Video & tracking
- [ ] Short-clip upload: per-frame detection stitched into an annotated video
- [ ] Ball trajectory tracking across frames (ByteTrack), speed estimate from frame delta
- [ ] Player heatmap over a fixed-camera clip

### Phase 3 — My own dataset (the real ML)
- [ ] Collect frames from broadcast footage; label with Label Studio: `batsman / bowler / keeper / umpire / fielder / ball / stumps`
- [ ] Fine-tune YOLOv8 on the custom classes; publish training curves + mAP table vs the pretrained baseline
- [ ] Honest error analysis: where the model fails (motion blur, crowd occlusion, white-ball vs red-ball)

### Phase 4 — Cricket intelligence
- [ ] Shot classification (drive/pull/cut) from pose keypoints
- [ ] Auto-highlights: detect boundaries/wickets from detection + scoreboard OCR
- [ ] Win-probability model fed by extracted match state

## Why this project

Anyone can call a hosted LLM API. Training and evaluating your own vision model — data collection, labeling, fine-tuning, mAP, error analysis — is the proof of real ML fundamentals. Phase 1 ships a working product; Phase 3 is where the depth lives.

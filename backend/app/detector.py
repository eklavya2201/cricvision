"""Player + ball detection on sports images using YOLOv8.

Default is COCO-pretrained yolov8n — class 0 (person) and class 32 (sports
ball) work on cricket footage out of the box. Set CRICVISION_WEIGHTS to a
fine-tuned checkpoint (Phase 3) and the model's own cricket classes
(batsman/bowler/wicketkeeper/umpire/fielder/ball/stumps) are used unfiltered.
"""
import base64
import io
import os
import time

from PIL import Image

DEFAULT_WEIGHTS = "yolov8n.pt"  # nano: fast enough for CPU inference
WEIGHTS = os.environ.get("CRICVISION_WEIGHTS", DEFAULT_WEIGHTS)
IS_FINETUNED = WEIGHTS != DEFAULT_WEIGHTS

COCO_CLASSES = {0: "player", 32: "ball"}
# COCO needs filtering to the two meaningful classes; fine-tuned weights are all-cricket
CLASS_FILTER = None if IS_FINETUNED else list(COCO_CLASSES)
BALL_LABELS = {"ball", "sports ball"}
NON_PLAYER_LABELS = BALL_LABELS | {"stumps"}
CONF_THRESHOLD = 0.25

_model = None


def get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO  # deferred: heavy import
        _model = YOLO(WEIGHTS)
    return _model


def label_for(cls: int, names: dict) -> str:
    if IS_FINETUNED:
        return names.get(cls, str(cls))
    return COCO_CLASSES.get(cls, str(cls))


def detect(image_bytes: bytes) -> dict:
    start = time.perf_counter()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # cap size for CPU latency; YOLO letterboxes internally anyway
    image.thumbnail((1280, 1280))

    result = get_model().predict(
        image, conf=CONF_THRESHOLD, classes=CLASS_FILTER, verbose=False
    )[0]

    detections = []
    for box in result.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = (round(v) for v in box.xyxy[0].tolist())
        detections.append({
            "label": label_for(cls, result.names),
            "confidence": round(float(box.conf[0]), 3),
            "box": [x1, y1, x2, y2],
        })

    annotated_rgb = result.plot()[..., ::-1]  # BGR -> RGB
    buf = io.BytesIO()
    Image.fromarray(annotated_rgb).save(buf, format="JPEG", quality=88)

    return {
        "detections": detections,
        "players": sum(1 for d in detections if d["label"] not in NON_PLAYER_LABELS),
        "balls": sum(1 for d in detections if d["label"] in BALL_LABELS),
        "latency_ms": int((time.perf_counter() - start) * 1000),
        "image_size": list(image.size),
        "annotated_jpeg_b64": base64.b64encode(buf.getvalue()).decode(),
    }

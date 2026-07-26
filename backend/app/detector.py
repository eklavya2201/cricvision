"""Player + ball detection on sports images using pretrained YOLOv8.

MVP uses COCO-pretrained weights — class 0 (person) and class 32 (sports ball)
work on cricket footage out of the box. Fine-tuning on hand-labeled cricket
data (batsman/bowler/keeper/umpire, ball, stumps) is the roadmap.
"""
import base64
import io
import time

from PIL import Image

WEIGHTS = "yolov8n.pt"  # nano: fast enough for CPU inference
CLASSES = {0: "player", 32: "ball"}
CONF_THRESHOLD = 0.25

_model = None


def get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO  # deferred: heavy import
        _model = YOLO(WEIGHTS)
    return _model


def detect(image_bytes: bytes) -> dict:
    start = time.perf_counter()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # cap size for CPU latency; YOLO letterboxes internally anyway
    image.thumbnail((1280, 1280))

    result = get_model().predict(
        image, conf=CONF_THRESHOLD, classes=list(CLASSES), verbose=False
    )[0]

    detections = []
    for box in result.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = (round(v) for v in box.xyxy[0].tolist())
        detections.append({
            "label": CLASSES.get(cls, str(cls)),
            "confidence": round(float(box.conf[0]), 3),
            "box": [x1, y1, x2, y2],
        })

    annotated_rgb = result.plot()[..., ::-1]  # BGR -> RGB
    buf = io.BytesIO()
    Image.fromarray(annotated_rgb).save(buf, format="JPEG", quality=88)

    return {
        "detections": detections,
        "players": sum(1 for d in detections if d["label"] == "player"),
        "balls": sum(1 for d in detections if d["label"] == "ball"),
        "latency_ms": int((time.perf_counter() - start) * 1000),
        "image_size": list(image.size),
        "annotated_jpeg_b64": base64.b64encode(buf.getvalue()).decode(),
    }

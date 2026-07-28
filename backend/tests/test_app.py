import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def _jpeg_bytes(size=(320, 240), color=(90, 140, 90)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def test_health():
    r = client.get("/api/health").json()
    assert r["status"] == "ok"
    assert r["weights"].endswith(".pt")


def test_weights_env_override(monkeypatch):
    import importlib

    from app import detector

    monkeypatch.setenv("CRICVISION_WEIGHTS", "models/cricket_best.pt")
    importlib.reload(detector)
    assert detector.WEIGHTS == "models/cricket_best.pt"
    assert detector.IS_FINETUNED
    assert detector.CLASS_FILTER is None  # fine-tuned classes are all cricket
    assert detector.label_for(0, {0: "batsman"}) == "batsman"

    monkeypatch.delenv("CRICVISION_WEIGHTS")
    importlib.reload(detector)
    assert detector.WEIGHTS == detector.DEFAULT_WEIGHTS
    assert not detector.IS_FINETUNED
    assert detector.CLASS_FILTER == [0, 32]
    assert detector.label_for(0, {}) == "player"


def test_detect_on_blank_image_returns_valid_shape():
    r = client.post("/api/detect", files={"file": ("pitch.jpg", _jpeg_bytes(), "image/jpeg")})
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data["detections"], list)
    assert data["players"] >= 0 and data["balls"] >= 0
    assert data["latency_ms"] > 0
    assert data["annotated_jpeg_b64"]  # annotated image always returned
    for d in data["detections"]:
        assert d["label"] in ("player", "ball")
        assert 0 <= d["confidence"] <= 1
        assert len(d["box"]) == 4


def test_empty_upload_rejected():
    r = client.post("/api/detect", files={"file": ("x.jpg", b"", "image/jpeg")})
    assert r.status_code == 400


def test_garbage_bytes_rejected():
    r = client.post("/api/detect", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400


def test_oversize_upload_rejected():
    big = b"0" * (8 * 1024 * 1024 + 1)
    r = client.post("/api/detect", files={"file": ("big.jpg", big, "image/jpeg")})
    assert r.status_code == 413

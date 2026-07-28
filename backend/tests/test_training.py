"""Tests for the Phase-3 training tooling (repo-root /training scripts)."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))

import agreement  # noqa: E402
import extract_frames  # noqa: E402


def _write_clip(path: Path, scenes: int, frames_per_scene: int = 30, fps: int = 10):
    """Each scene is a visually distinct textured frame repeated frames_per_scene times."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    rng = np.random.default_rng(seed=7)
    for s in range(scenes):
        frame = rng.integers(0, 255, (120, 160, 3), np.uint8)
        frame = cv2.GaussianBlur(frame, (9, 9), 0)
        for _ in range(frames_per_scene):
            w.write(frame)
    w.release()


def test_extract_dedups_repeated_scenes(tmp_path):
    clip = tmp_path / "match1.mp4"
    _write_clip(clip, scenes=3)  # 90 frames @10fps, sampled every 1s = 9 samples
    out = tmp_path / "raw"

    stats = extract_frames.extract([clip], out, every_s=1.0, max_distance=6)

    assert stats["kept"] == 3  # one frame per distinct scene survives
    assert stats["skipped_duplicates"] == 6
    assert len(list(out.glob("*.jpg"))) == 3


def test_rerun_dedups_against_provenance(tmp_path):
    clip = tmp_path / "match1.mp4"
    _write_clip(clip, scenes=2)
    out = tmp_path / "raw"

    first = extract_frames.extract([clip], out, every_s=1.0, max_distance=6)
    again = extract_frames.extract([clip], out, every_s=1.0, max_distance=6)

    assert first["kept"] == 2
    assert again["kept"] == 0  # everything already in the provenance log
    assert again["total_frames"] == 2


def _write_yolo(path: Path, lines: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def test_agreement_perfect_match(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _write_yolo(a / "f1.txt", ["0 0.5 0.5 0.2 0.4", "5 0.1 0.1 0.05 0.05"])
    _write_yolo(b / "f1.txt", ["0 0.5 0.5 0.2 0.4", "5 0.1 0.1 0.05 0.05"])

    r = agreement.compute(a, b, min_iou=0.5)
    assert r["overall_f1"] == 1.0
    assert r["per_class"]["batsman"]["f1"] == 1.0
    assert r["per_class"]["ball"]["f1"] == 1.0
    assert r["per_class"]["umpire"]["f1"] is None  # class never labeled


def test_agreement_penalizes_class_disagreement_and_misses(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    # same box, different class (batsman vs fielder) + a box only pass A saw
    _write_yolo(a / "f1.txt", ["0 0.5 0.5 0.2 0.4", "6 0.8 0.8 0.1 0.2"])
    _write_yolo(b / "f1.txt", ["4 0.5 0.5 0.2 0.4"])

    r = agreement.compute(a, b, min_iou=0.5)
    assert r["overall_f1"] == 0.0  # IoU match exists but classes differ; stumps unmatched
    assert r["per_class"]["batsman"] == {"a": 1, "b": 0, "f1": 0.0}
    assert r["per_class"]["fielder"] == {"a": 0, "b": 1, "f1": 0.0}


def test_agreement_requires_iou_overlap(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _write_yolo(a / "f1.txt", ["1 0.2 0.2 0.1 0.1"])
    _write_yolo(b / "f1.txt", ["1 0.7 0.7 0.1 0.1"])  # same class, disjoint boxes

    r = agreement.compute(a, b, min_iou=0.5)
    assert r["per_class"]["bowler"]["f1"] == 0.0


def test_provenance_rows_have_source_and_timestamp(tmp_path):
    clip = tmp_path / "odi_2026.mp4"
    _write_clip(clip, scenes=2)
    out = tmp_path / "raw"
    extract_frames.extract([clip], out, every_s=1.0, max_distance=6)

    rows = extract_frames.load_provenance(out)
    assert len(rows) == 2
    for row in rows:
        assert row["source"] == "odi_2026.mp4"
        assert float(row["timestamp_s"]) >= 0
        assert len(row["dhash"]) == 16
        assert (out / row["file"]).exists()

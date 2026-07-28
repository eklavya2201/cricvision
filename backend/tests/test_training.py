"""Tests for the Phase-3 training tooling (repo-root /training scripts)."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))

import agreement  # noqa: E402
import baseline_eval  # noqa: E402
import extract_frames  # noqa: E402
import split_dataset  # noqa: E402


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


def _fake_pool(tmp_path, matches: dict[str, int]):
    """Fake data/raw + labels: {match_name: frame_count}. Every 2nd frame gets a label."""
    import csv

    images, labels = tmp_path / "raw", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    rows = []
    i = 0
    for match, count in matches.items():
        for k in range(count):
            name = f"{match}_{k:03d}.jpg"
            (images / name).write_bytes(b"\xff\xd8fake")
            rows.append({"file": name, "source": f"{match}.mp4",
                         "timestamp_s": f"{k * 2}.00", "dhash": f"{i:016x}"})
            if k % 2 == 0:
                (labels / f"{match}_{k:03d}.txt").write_text("0 0.5 0.5 0.2 0.4\n5 0.1 0.1 0.05 0.05")
            i += 1
    with open(images / "provenance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=extract_frames.PROVENANCE_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return images, labels


def test_split_never_crosses_matches_and_hits_ratios(tmp_path):
    images, labels = _fake_pool(tmp_path, {"m1": 40, "m2": 30, "m3": 12, "m4": 10, "m5": 8})
    out = tmp_path / "dataset"

    result = split_dataset.build(images, labels, out)

    # every frame of a match landed in exactly one split dir
    for match, split in result["assignment"].items():
        stem = match.removesuffix(".mp4")
        for other in ("train", "val", "test"):
            found = list((out / "images" / other).glob(f"{stem}_*.jpg"))
            assert bool(found) == (other == split)

    stats = result["stats"]
    total = sum(s["frames"] for s in stats.values())
    assert total == 100
    assert stats["train"]["frames"] >= 50  # 70% target, match-granular
    assert stats["val"]["frames"] > 0 and stats["test"]["frames"] > 0

    yaml = (out / "dataset.yaml").read_text()
    assert "6: stumps" in yaml and "0: batsman" in yaml


def test_split_counts_classes_and_keeps_background_frames(tmp_path):
    images, labels = _fake_pool(tmp_path, {"m1": 4, "m2": 4, "m3": 4})
    out = tmp_path / "dataset"

    result = split_dataset.build(images, labels, out)

    stats = result["stats"]
    # 6 labeled frames (every 2nd of 12), each with 1 batsman + 1 ball
    assert sum(s["classes"][0] for s in stats.values()) == 6
    assert sum(s["classes"][5] for s in stats.values()) == 6
    # unlabeled (background) frames still copied as images
    total_images = sum(len(list((out / "images" / s).glob("*.jpg")))
                       for s in ("train", "val", "test"))
    assert total_images == 12


def test_split_refuses_fewer_than_three_matches(tmp_path):
    images, labels = _fake_pool(tmp_path, {"m1": 5, "m2": 5})
    import pytest

    with pytest.raises(SystemExit):
        split_dataset.build(images, labels, tmp_path / "dataset")


def test_baseline_collapse_maps_players_and_ball_drops_stumps():
    text = "\n".join([
        "0 0.5 0.5 0.2 0.4",   # batsman -> person
        "4 0.3 0.3 0.1 0.2",   # fielder -> person
        "5 0.1 0.1 0.05 0.05", # ball -> sports ball
        "6 0.8 0.8 0.1 0.2",   # stumps -> dropped
    ])
    out = baseline_eval.collapse_label_text(text).splitlines()
    assert out == ["0 0.5 0.5 0.2 0.4", "0 0.3 0.3 0.1 0.2", "32 0.1 0.1 0.05 0.05"]


def test_baseline_builds_collapsed_test_split(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "images" / "test").mkdir(parents=True)
    (dataset / "labels" / "test").mkdir(parents=True)
    (dataset / "images" / "test" / "f1.jpg").write_bytes(b"\xff\xd8fake")
    (dataset / "labels" / "test" / "f1.txt").write_text("2 0.5 0.5 0.2 0.4\n6 0.8 0.8 0.1 0.2")
    (dataset / "images" / "test" / "bg.jpg").write_bytes(b"\xff\xd8fake")  # background frame

    coco_names = {0: "person", 32: "sports ball"}
    yaml_path = baseline_eval.build_collapsed_split(dataset, tmp_path / "collapsed", coco_names)

    collapsed = (tmp_path / "collapsed" / "labels" / "test" / "f1.txt").read_text()
    assert collapsed == "0 0.5 0.5 0.2 0.4"  # keeper -> person, stumps gone
    assert (tmp_path / "collapsed" / "images" / "test" / "bg.jpg").exists()
    yaml = yaml_path.read_text()
    assert "0: person" in yaml and "32: sports ball" in yaml

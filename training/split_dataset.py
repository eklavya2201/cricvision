"""Phase 3.3: build the YOLO dataset with a leakage-free split.

    python split_dataset.py --images ../data/raw --labels ../data/labels --out ../data/dataset

Frames are grouped by source match (the ``source`` column of provenance.csv)
and whole matches are assigned to train/val/test targeting 70/20/10 **by frame
count** — frames from one match never cross splits, because near-identical
frames on both sides of the split would inflate mAP.

Outputs the standard YOLO layout, ``dataset.yaml``, and a markdown class-count
table per split (paste into the README).
"""
import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from extract_frames import load_provenance

CLASSES = ["batsman", "bowler", "wicketkeeper", "umpire", "fielder", "ball", "stumps"]
SPLITS = {"train": 0.70, "val": 0.20, "test": 0.10}


def assign_matches(frames_by_match: dict[str, list[str]]) -> dict[str, str]:
    """Greedy: biggest matches first, each to the split furthest below its target."""
    total = sum(len(v) for v in frames_by_match.values())
    filled = {s: 0 for s in SPLITS}
    assignment = {}
    for match, frames in sorted(frames_by_match.items(), key=lambda kv: -len(kv[1])):
        split = max(SPLITS, key=lambda s: SPLITS[s] - filled[s] / total)
        assignment[match] = split
        filled[split] += len(frames)
    return assignment


def count_classes(label_path: Path) -> list[int]:
    counts = [0] * len(CLASSES)
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if parts:
                counts[int(parts[0])] += 1
    return counts


def build(images_dir: Path, labels_dir: Path, out_dir: Path) -> dict:
    rows = load_provenance(images_dir)
    if not rows:
        sys.exit(f"no provenance.csv in {images_dir} — run extract_frames.py first")

    frames_by_match = defaultdict(list)
    for r in rows:
        frames_by_match[r["source"]].append(r["file"])
    if len(frames_by_match) < 3:
        sys.exit(f"only {len(frames_by_match)} source matches — need at least 3 "
                 "(one per split), collect more footage")

    assignment = assign_matches(frames_by_match)

    stats = {s: {"frames": 0, "classes": [0] * len(CLASSES)} for s in SPLITS}
    for match, files in frames_by_match.items():
        split = assignment[match]
        img_out = out_dir / "images" / split
        lbl_out = out_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for f in files:
            label = labels_dir / (Path(f).stem + ".txt")
            shutil.copy2(images_dir / f, img_out / f)
            if label.exists():  # background-only frames legitimately have no label file
                shutil.copy2(label, lbl_out / label.name)
            stats[split]["frames"] += 1
            for c, n in enumerate(count_classes(label)):
                stats[split]["classes"][c] += n

    (out_dir / "dataset.yaml").write_text(
        f"path: {out_dir.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)),
        encoding="utf-8")

    return {"assignment": assignment, "stats": stats}


def markdown_table(stats: dict) -> str:
    head = "| split | frames | " + " | ".join(CLASSES) + " |"
    sep = "|" + "---|" * (len(CLASSES) + 2)
    body = [f"| {s} | {v['frames']} | " + " | ".join(map(str, v["classes"])) + " |"
            for s, v in stats.items()]
    return "\n".join([head, sep, *body])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--images", type=Path, default=Path("../data/raw"))
    p.add_argument("--labels", type=Path, default=Path("../data/labels"))
    p.add_argument("--out", type=Path, default=Path("../data/dataset"))
    args = p.parse_args(argv)

    result = build(args.images, args.labels, args.out)
    print("match -> split:")
    for match, split in sorted(result["assignment"].items()):
        print(f"  {match}: {split}")
    print(f"\n{markdown_table(result['stats'])}")
    print(f"\ndataset.yaml written to {args.out / 'dataset.yaml'}")


if __name__ == "__main__":
    main()

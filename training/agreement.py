"""Phase 3.2: measure annotation agreement between two YOLO-format label exports.

    python agreement.py export_a/labels export_b/labels [--iou 0.5]

Compares the double-labeled sample (same .txt filenames in both dirs).
Boxes are greedily matched by IoU; per-class agreement is the F1 between the
two passes: 2*matches / (boxes_in_a + boxes_in_b). Classes under 90% need
their rule tightened in LABELING_GUIDE.md and a re-label.
"""
import argparse
import sys
from pathlib import Path

CLASSES = ["batsman", "bowler", "wicketkeeper", "umpire", "fielder", "ball", "stumps"]
THRESHOLD = 0.90


def read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    """YOLO format: class cx cy w h (normalized)."""
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            boxes.append((int(parts[0]), *map(float, parts[1:5])))
    return boxes


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1, bx2, by2 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[3] * a[4] + b[3] * b[4] - inter
    return inter / union if union > 0 else 0.0


def match_frame(boxes_a, boxes_b, min_iou: float):
    """Greedy best-IoU matching (class-agnostic, so class disagreements count against F1)."""
    pairs = sorted(((iou(a, b), i, j) for i, a in enumerate(boxes_a)
                    for j, b in enumerate(boxes_b)), reverse=True)
    used_a, used_b, matches = set(), set(), []
    for score, i, j in pairs:
        if score < min_iou:
            break
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            matches.append((i, j))
    return matches


def compute(dir_a: Path, dir_b: Path, min_iou: float) -> dict:
    files = sorted(set(p.name for p in dir_a.glob("*.txt"))
                   & set(p.name for p in dir_b.glob("*.txt")))
    if not files:
        sys.exit("no common .txt label files between the two dirs")

    count_a = [0] * len(CLASSES)
    count_b = [0] * len(CLASSES)
    agreed = [0] * len(CLASSES)  # matched by IoU AND same class

    for name in files:
        boxes_a = read_labels(dir_a / name)
        boxes_b = read_labels(dir_b / name)
        for c, *_ in boxes_a:
            count_a[c] += 1
        for c, *_ in boxes_b:
            count_b[c] += 1
        for i, j in match_frame(boxes_a, boxes_b, min_iou):
            if boxes_a[i][0] == boxes_b[j][0]:
                agreed[boxes_a[i][0]] += 1

    per_class = {}
    for c, name in enumerate(CLASSES):
        total = count_a[c] + count_b[c]
        per_class[name] = {"a": count_a[c], "b": count_b[c],
                           "f1": round(2 * agreed[c] / total, 3) if total else None}
    total_boxes = sum(count_a) + sum(count_b)
    overall = round(2 * sum(agreed) / total_boxes, 3) if total_boxes else None
    return {"frames": len(files), "overall_f1": overall, "per_class": per_class}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dir_a", type=Path)
    p.add_argument("dir_b", type=Path)
    p.add_argument("--iou", type=float, default=0.5)
    args = p.parse_args(argv)

    r = compute(args.dir_a, args.dir_b, args.iou)
    print(f"{r['frames']} double-labeled frames — overall agreement F1: {r['overall_f1']}\n")
    print(f"{'class':<14}{'pass A':>8}{'pass B':>8}{'F1':>8}")
    for name, s in r["per_class"].items():
        f1 = "-" if s["f1"] is None else f"{s['f1']:.3f}"
        flag = "  <-- re-label" if s["f1"] is not None and s["f1"] < THRESHOLD else ""
        print(f"{name:<14}{s['a']:>8}{s['b']:>8}{f1:>8}{flag}")


if __name__ == "__main__":
    main()

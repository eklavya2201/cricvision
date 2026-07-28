"""Phase 3.4: baseline — COCO-pretrained YOLOv8n on the cricket test split.

    python baseline_eval.py --dataset ../data/dataset [--model yolov8n.pt]

This is the number fine-tuning has to beat. The pretrained model can't know
cricket roles, so ground truth is collapsed to what COCO *can* see:
batsman/bowler/wicketkeeper/umpire/fielder -> person(0), ball -> sports ball(32),
stumps dropped (COCO has no such class). A temp collapsed copy of the test
split is built and `model.val()` reports mAP@50 / mAP@50-95 per class.
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from split_dataset import CLASSES  # noqa: F401 — same 7-class order everywhere

PERSON, SPORTS_BALL = 0, 32
COLLAPSE = {0: PERSON, 1: PERSON, 2: PERSON, 3: PERSON, 4: PERSON, 5: SPORTS_BALL}
# class 6 (stumps) has no COCO equivalent — dropped from the baseline ground truth


def collapse_label_text(text: str) -> str:
    """Rewrite a YOLO label file's classes into COCO indices; drop stumps lines."""
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 5 and int(parts[0]) in COLLAPSE:
            out.append(" ".join([str(COLLAPSE[int(parts[0])]), *parts[1:5]]))
    return "\n".join(out)


def build_collapsed_split(dataset_dir: Path, out_dir: Path, coco_names: dict) -> Path:
    """Copy the test split with collapsed labels + an 80-class COCO dataset.yaml."""
    img_src = dataset_dir / "images" / "test"
    lbl_src = dataset_dir / "labels" / "test"
    if not img_src.is_dir():
        sys.exit(f"no test split at {img_src} — run split_dataset.py first")

    img_out = out_dir / "images" / "test"
    lbl_out = out_dir / "labels" / "test"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for img in img_src.iterdir():
        shutil.copy2(img, img_out / img.name)
        lbl = lbl_src / (img.stem + ".txt")
        if lbl.exists():
            (lbl_out / lbl.name).write_text(collapse_label_text(lbl.read_text()))

    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(
        f"path: {out_dir.resolve().as_posix()}\n"
        "train: images/test\nval: images/test\ntest: images/test\n\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in coco_names.items()),
        encoding="utf-8")
    return yaml_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", type=Path, default=Path("../data/dataset"))
    p.add_argument("--model", default="yolov8n.pt")
    args = p.parse_args(argv)

    from ultralytics import YOLO
    model = YOLO(args.model)

    with tempfile.TemporaryDirectory(prefix="cricvision_baseline_") as tmp:
        yaml_path = build_collapsed_split(args.dataset, Path(tmp), model.names)
        metrics = model.val(data=str(yaml_path), split="test", verbose=False)

    print(f"\nBaseline: {args.model} on collapsed cricket test split")
    print(f"mAP@50    = {metrics.box.map50:.3f}")
    print(f"mAP@50-95 = {metrics.box.map:.3f}")
    for idx, cls_map50, cls_map in zip(metrics.box.ap_class_index,
                                       metrics.box.ap50, metrics.box.ap):
        print(f"  {model.names[int(idx)]:<14} mAP@50={cls_map50:.3f}  mAP@50-95={cls_map:.3f}")
    print("\nRecord these numbers in the README before fine-tuning.")


if __name__ == "__main__":
    main()

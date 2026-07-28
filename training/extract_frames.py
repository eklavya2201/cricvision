"""Phase 3.1: sample frames from match footage for the labeling pool.

    python extract_frames.py CLIP [CLIP ...] --out ../data/raw [--every 2.0]

One frame every ``--every`` seconds, deduplicated by perceptual hash (dHash,
Hamming distance) against every frame already kept — including frames from
earlier runs, whose hashes are reloaded from the provenance log. Each kept
frame gets a row in ``provenance.csv``: file, source clip, timestamp, hash.

Only needs cv2 + numpy (already in backend requirements).
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2

HASH_BITS = 64          # dHash on an 8x8 difference grid
DEFAULT_DISTANCE = 6    # Hamming distance at or below which frames count as duplicates
PROVENANCE = "provenance.csv"
PROVENANCE_FIELDS = ["file", "source", "timestamp_s", "dhash"]


def dhash(frame) -> int:
    """64-bit difference hash: robust to compression/brightness, cheap to compare."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = 0
    for row in small[:, 1:] > small[:, :-1]:
        for b in row:
            bits = (bits << 1) | int(b)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def load_provenance(out_dir: Path) -> list[dict]:
    path = out_dir / PROVENANCE
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def extract(clips: list[Path], out_dir: Path, every_s: float, max_distance: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_provenance(out_dir)
    seen_hashes = [int(r["dhash"], 16) for r in rows]

    kept = skipped = 0
    for clip in clips:
        cap = cv2.VideoCapture(str(clip))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps < 1:
            fps = 25.0
        step = max(1, round(fps * every_s))

        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if n % step == 0:
                h = dhash(frame)
                if any(hamming(h, s) <= max_distance for s in seen_hashes):
                    skipped += 1
                else:
                    seen_hashes.append(h)
                    ts = n / fps
                    name = f"{clip.stem}_{ts:08.2f}s.jpg"
                    cv2.imwrite(str(out_dir / name), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    rows.append({"file": name, "source": clip.name,
                                 "timestamp_s": f"{ts:.2f}", "dhash": f"{h:016x}"})
                    kept += 1
            n += 1
        cap.release()

    with open(out_dir / PROVENANCE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PROVENANCE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return {"kept": kept, "skipped_duplicates": skipped, "total_frames": len(rows)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("clips", nargs="+", type=Path, help="video files to sample")
    p.add_argument("--out", type=Path, default=Path("../data/raw"))
    p.add_argument("--every", type=float, default=2.0, help="seconds between samples")
    p.add_argument("--hash-distance", type=int, default=DEFAULT_DISTANCE,
                   help="max Hamming distance to count as duplicate")
    args = p.parse_args(argv)

    missing = [c for c in args.clips if not c.exists()]
    if missing:
        sys.exit(f"clip not found: {', '.join(map(str, missing))}")

    stats = extract(args.clips, args.out, args.every, args.hash_distance)
    print(f"kept {stats['kept']} frames ({stats['skipped_duplicates']} near-duplicates "
          f"skipped) — {stats['total_frames']} total in {args.out}")


if __name__ == "__main__":
    main()

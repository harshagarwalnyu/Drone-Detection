"""Download a HuggingFace drone dataset and convert it to YOLO format."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def yolo_label(bbox: list[float], width: int, height: int) -> str:
    x, y, w, h = (float(v) for v in bbox)
    cx = (x + w / 2) / width
    cy = (y + h / 2) / height
    return f"0 {cx:.6f} {cy:.6f} {w / width:.6f} {h / height:.6f}"


def make_dirs(root: Path, split: str) -> tuple[Path, Path]:
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    return img_dir, lbl_dir


def write_split(dataset: Dataset, split: str, output: Path, limit: int | None) -> int:
    img_dir, lbl_dir = make_dirs(output, split)
    rows = dataset.select(range(min(limit, len(dataset)))) if limit else dataset

    for count, sample in enumerate(rows, 1):
        w, h = int(sample["width"]), int(sample["height"])
        stem = f"{split}_{int(sample['image_id']):06d}"

        sample["image"].save(img_dir / f"{stem}.jpg")

        lines = [yolo_label(bb, w, h) for bb in sample["objects"]["bbox"]]
        (lbl_dir / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

    return count


def add_negatives(neg_dir: Path, output: Path, split: str, start: int) -> int:
    if not neg_dir.exists():
        raise FileNotFoundError(f"Negative dir not found: {neg_dir}")

    img_dir, lbl_dir = make_dirs(output, split)
    added = 0

    for added, src in enumerate(
        sorted(p for p in neg_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES), 1
    ):
        stem = f"{split}_negative_{start + added:06d}"
        shutil.copy2(src, img_dir / f"{stem}{src.suffix.lower()}")
        (lbl_dir / f"{stem}.txt").write_text("", encoding="utf-8")

    return added


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert HF drone dataset to YOLO format.")
    parser.add_argument("--dataset-id", default="pathikg/drone-detection-dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("data/pathikg_drone_detection"))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="test")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--negative-dir", type=Path, default=None)
    args = parser.parse_args()

    ds = load_dataset(args.dataset_id)
    if not isinstance(ds, DatasetDict):
        raise TypeError(f"Expected DatasetDict, got {type(ds)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_n = write_split(ds[args.train_split], "train", args.output_dir, args.limit_train)
    val_n = write_split(ds[args.val_split], "val", args.output_dir, args.limit_val)
    neg_n = add_negatives(args.negative_dir, args.output_dir, "train", train_n) if args.negative_dir else 0

    yaml_path = args.output_dir / "data.yaml"
    yaml_path.write_text("\n".join([
        f"path: {args.output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "names:",
        "  0: drone",
        "",
    ]), encoding="utf-8")

    summary = {"dataset_id": args.dataset_id, "train": train_n, "val": val_n, "negatives": neg_n}
    (args.output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

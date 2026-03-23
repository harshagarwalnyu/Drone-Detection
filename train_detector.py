"""Fine-tune a YOLO model on a prepared drone dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on drone data.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("yolov8n.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="drone_detector")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Dataset config not found: {args.data}")

    model = YOLO(str(args.model))
    model.train(
        data=str(args.data),
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, project=args.project, name=args.name,
        workers=args.workers, pretrained=True, single_cls=True, patience=10,
    )
    print(f"Done. Checkpoint: {Path(args.project) / args.name / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""CLI entry point — subcommands for the full drone pipeline."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def cmd_detect(argv: list[str]) -> int:
    from pipeline import detection_cli
    return detection_cli(argv)


def cmd_track(argv: list[str]) -> int:
    from pipeline import render_cli
    return render_cli(argv)


def cmd_upload(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Build Parquet dataset from detection frames.")
    parser.add_argument("--detections-dir", type=Path, default=Path("detections"))
    parser.add_argument("--output-parquet", type=Path, default=Path("artifacts/detections.parquet"))
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args(argv)

    if not args.detections_dir.exists():
        raise FileNotFoundError(f"Not found: {args.detections_dir}")

    images = sorted(p for p in args.detections_dir.glob("*.jpg") if p.is_file())
    if not images:
        raise RuntimeError(f"No .jpg files in {args.detections_dir}")

    def read(p: Path):
        return p.stem.split("_frame_", 1)[0], p.name, p.read_bytes()

    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(read, p): i for i, p in enumerate(images)}
        results = [None] * len(images)
        for f in as_completed(futs):
            results[futs[f]] = f.result()

    videos, names, blobs = zip(*results)
    img_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    table = pa.table({
        "video": pa.array(videos),
        "frame_name": pa.array(names),
        "image": pa.array([{"bytes": b, "path": n} for b, n in zip(blobs, names)], type=img_type),
    })

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(args.output_parquet), compression="snappy")
    print(f"Wrote {len(images)} frames to {args.output_parquet}")

    if args.repo_id:
        from datasets import Dataset
        Dataset.from_parquet(str(args.output_parquet)).push_to_hub(args.repo_id, private=args.private)
        print(f"Pushed to https://huggingface.co/datasets/{args.repo_id}")

    return 0


COMMANDS = {
    "detect": (cmd_detect, "Task 1: save detection frames"),
    "track":  (cmd_track,  "Task 2: detection + tracking + video output"),
    "upload": (cmd_upload,  "Package detections into HF Parquet dataset"),
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("Usage: python main.py {detect|track|upload} [options]")
        print()
        for name, (_, desc) in COMMANDS.items():
            print(f"  {name:10s} {desc}")
        print()
        print("Default (no subcommand): runs 'track'")
        if len(sys.argv) >= 2 and sys.argv[1] not in COMMANDS:
            return cmd_track(sys.argv[1:])
        return cmd_track([])

    cmd_name = sys.argv[1]
    fn, _ = COMMANDS[cmd_name]
    return fn(sys.argv[2:])


if __name__ == "__main__":
    raise SystemExit(main())

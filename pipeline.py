"""Detection + Kalman tracking pipeline for drone videos."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
from tqdm import tqdm

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

from ultralytics import YOLO

from tracker import BBox, DroneTracker, TrackerState

VIDEO_EXTENSIONS = {".mp4"}
SUMMARY_PATH = Path("artifacts/pipeline_summary.json")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PipelineConfig:
    videos_dir: Path
    frames_dir: Path
    detections_dir: Path
    output_videos_dir: Path
    summary_path: Path
    weights: Path
    fps: float = 10.0
    conf: float = 0.10
    imgsz: int = 1280
    max_missing_frames: int = 50
    trajectory_points: int = 300
    overwrite_frames: bool = False
    overwrite_outputs: bool = False
    render_output_videos: bool = True
    device: str | None = None
    half: bool = False
    drone_class_name: str | None = "drone"
    drone_class_id: int | None = None
    max_frames: int | None = None
    augment: bool = True
    max_box_area_ratio: float | None = 0.03
    max_box_width_ratio: float | None = 0.15
    max_box_height_ratio: float | None = 0.15


@dataclass(slots=True)
class FrameDetection:
    bbox: BBox
    confidence: float
    class_id: int
    class_name: str


@dataclass(slots=True)
class VideoSummary:
    video_name: str
    input_video: str
    frame_dir: str
    detection_frames: int
    sampled_frames: int
    output_video: str | None
    detector_weights: str
    detector_class_ids: list[int]
    rendered_frames: int | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_writable(path: Path) -> bool:
    if path.exists():
        return path.is_dir() and os.access(path, os.W_OK | os.X_OK)
    return os.access(path.parent, os.W_OK | os.X_OK)


def writable_dir(requested: Path) -> Path:
    if is_writable(requested):
        return requested
    fallback = Path("artifacts") / requested.name
    ensure_dir(fallback)
    return fallback


def canonical_name(name: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", name.lower())
    return "".join(tokens[1:] if tokens[:1] == ["drone"] else tokens)


def find_frame_dir(video: Path, root: Path) -> Path:
    direct = root / video.stem
    if direct.exists():
        return direct

    if root.exists():
        matches = [
            d for d in root.iterdir()
            if d.is_dir() and canonical_name(d.name) == canonical_name(video.stem)
        ]
        if len(matches) == 1:
            return matches[0]

    base = root if is_writable(root) else writable_dir(root)
    return base / video.stem


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(video: Path, dest: Path, fps: float, overwrite: bool) -> list[Path]:
    existing = sorted(dest.glob("*.jpg"))
    if existing and not overwrite:
        return existing

    ensure_dir(dest)
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video),
        "-vf", f"fps={fps}",
        str(dest / "frame_%06d.jpg"),
    ], check=True)

    return sorted(dest.glob("*.jpg"))


def discover_videos(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)


# ---------------------------------------------------------------------------
# Class resolution
# ---------------------------------------------------------------------------

def _norm(name: str) -> str:
    return name.strip().lower().replace("-", " ").replace("_", " ")


def resolve_drone_class(model: YOLO, name: str | None, cid: int | None) -> list[int]:
    names = model.names
    labels = (
        {i: v for i, v in enumerate(names)}
        if isinstance(names, list)
        else {int(k): v for k, v in names.items()}
    )

    if cid is not None:
        if cid not in labels:
            raise ValueError(f"class {cid} not in model: {labels}")
        return [cid]

    if name:
        ids = []
        for req in (s.strip() for s in name.split(",") if s.strip()):
            target = _norm(req)
            exact = [i for i, v in labels.items() if _norm(v) == target]
            if exact:
                ids.extend(exact)
            else:
                partial = [i for i, v in labels.items() if target in _norm(v)]
                if partial:
                    ids.extend(partial)
                else:
                    raise ValueError(f"'{req}' not found in model: {labels}")
        if ids:
            return sorted(set(ids))

    auto = [i for i, v in labels.items()
            if any(t in _norm(v) for t in ("drone", "uav", "quadcopter"))]
    if auto:
        return auto

    if len(labels) == 1:
        return [next(iter(labels))]

    raise ValueError(f"Cannot infer drone class from: {labels}")


# ---------------------------------------------------------------------------
# Detection selection with geometry filtering
# ---------------------------------------------------------------------------

def _box_ok(bbox: BBox, shape: tuple[int, int],
            max_area: float | None, max_w: float | None, max_h: float | None) -> bool:
    ih, iw = shape
    if iw <= 0 or ih <= 0:
        return False

    bw = max(0.0, bbox[2] - bbox[0])
    bh = max(0.0, bbox[3] - bbox[1])
    if bw <= 0 or bh <= 0:
        return False

    if max_area and (bw * bh) / (iw * ih) > max_area:
        return False
    if max_w and bw / iw > max_w:
        return False
    if max_h and bh / ih > max_h:
        return False
    return True


def pick_detection(result, class_ids: set[int],
                   max_area: float | None, max_w: float | None, max_h: float | None
                   ) -> FrameDetection | None:
    if not len(result.boxes):
        return None

    names = result.names if isinstance(result.names, dict) else dict(enumerate(result.names))
    best = None

    for box in result.boxes:
        cls = int(box.cls.item())
        if cls not in class_ids:
            continue

        conf = float(box.conf.item())
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        bbox = (x1, y1, x2, y2)

        if not _box_ok(bbox, result.orig_shape, max_area, max_w, max_h):
            continue

        candidate = FrameDetection(bbox, conf, cls, str(names[cls]))
        if best is None or conf > best.confidence:
            best = candidate

    return best


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_label(frame, text: str, x: int, y: int) -> None:
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    y = max(y, th + bl + 4)
    cv2.rectangle(frame, (x, y - th - bl - 6), (x + tw + 8, y + 2), (0, 0, 0), -1)
    cv2.putText(frame, text, (x + 4, y - bl - 2), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def draw_overlays(frame, det: FrameDetection | None, state: TrackerState,
                  trail_len: int, frame_idx: int, total: int) -> None:
    h, w = frame.shape[:2]

    # Bounding box or predicted crosshair
    if det:
        ix1, iy1 = int(det.bbox[0]), int(det.bbox[1])
        ix2, iy2 = int(det.bbox[2]), int(det.bbox[3])
        cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (0, 255, 0), 2)

        corner = max(10, min(ix2 - ix1, iy2 - iy1) // 4)
        for cx, cy, dx, dy in [
            (ix1, iy1, 1, 1), (ix2, iy1, -1, 1),
            (ix1, iy2, 1, -1), (ix2, iy2, -1, -1),
        ]:
            cv2.line(frame, (cx, cy), (cx + dx * corner, cy), (0, 255, 0), 3)
            cv2.line(frame, (cx, cy), (cx, cy + dy * corner), (0, 255, 0), 3)

        draw_label(frame, f"drone {det.confidence:.2f}", ix1, iy1)

    elif state.predicted and state.estimated_center:
        cx, cy = state.estimated_center
        for i in range(0, 30, 6):
            cv2.line(frame, (cx - 30 + i, cy), (cx - 27 + i, cy), (0, 165, 255), 1)
            cv2.line(frame, (cx + i, cy), (cx + i + 3, cy), (0, 165, 255), 1)
            cv2.line(frame, (cx, cy - 30 + i), (cx, cy - 27 + i), (0, 165, 255), 1)
            cv2.line(frame, (cx, cy + i), (cx, cy + i + 3), (0, 165, 255), 1)

    # Trajectory with gradient
    for seg in state.trajectory_segments:
        pts = seg[-trail_len:]
        n = len(pts)
        if n < 2:
            continue
        for i in range(1, n):
            t = i / n
            alpha = 0.2 + 0.8 * t
            color = (int(255 * alpha), int(140 * alpha), 0)
            cv2.line(frame, pts[i - 1], pts[i], color, max(1, int(1 + 2 * t)))

    # Center marker
    if state.estimated_center:
        if state.predicted:
            cv2.circle(frame, state.estimated_center, 8, (0, 165, 255), 2)
            cv2.circle(frame, state.estimated_center, 3, (0, 165, 255), -1)
        else:
            cv2.circle(frame, state.estimated_center, 6, (0, 0, 255), -1)
            cv2.circle(frame, state.estimated_center, 8, (255, 255, 255), 1)

    # HUD
    draw_label(frame, f"Frame {frame_idx}/{total}", w - 220, 30)

    if state.predicted:
        label, color = f"PREDICTED (miss {state.missing_frames})", (0, 165, 255)
    elif state.active:
        label, color = "TRACKING", (0, 255, 0)
    else:
        label, color = "NO TRACK", (0, 0, 255)

    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (tw, th), bl = cv2.getTextSize(label, font, scale, thick)
    cv2.rectangle(frame, (10, 30 - th - bl - 6), (10 + tw + 12, 32), color, -1)
    cv2.putText(frame, label, (16, 30 - bl - 2), font, scale, (0, 0, 0), thick, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Video composition
# ---------------------------------------------------------------------------

def compose_video(frame_dir: Path, output: Path, fps: float) -> None:
    ensure_dir(output.parent)
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%06d.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(output),
    ], check=True)


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------

def process_video(video: Path, model: YOLO, cfg: PipelineConfig,
                  class_ids: list[int]) -> VideoSummary:
    name = video.stem
    frame_dir = find_frame_dir(video, cfg.frames_dir)
    frames = extract_frames(video, frame_dir, cfg.fps, cfg.overwrite_frames)

    if cfg.max_frames:
        frames = frames[:cfg.max_frames]
    if not frames:
        raise RuntimeError(f"No frames for {video}")

    tracker = DroneTracker(fps=cfg.fps, max_missing_frames=cfg.max_missing_frames)
    det_count = 0
    rendered = 0
    out_path: Path | None = None

    predict_kwargs = dict(
        verbose=False, conf=cfg.conf, imgsz=cfg.imgsz,
        classes=class_ids, half=cfg.half, augment=cfg.augment,
    )
    if cfg.device:
        predict_kwargs["device"] = cfg.device

    id_set = set(class_ids)

    with tempfile.TemporaryDirectory(prefix=f"{name}_") as tmp:
        render_dir = Path(tmp)

        for idx, fpath in enumerate(tqdm(frames, desc=name, unit="fr"), 1):
            result = model.predict(source=str(fpath), **predict_kwargs)[0]

            det = pick_detection(
                result, id_set,
                cfg.max_box_area_ratio, cfg.max_box_width_ratio, cfg.max_box_height_ratio,
            )
            state = tracker.step(det.bbox if det else None)
            should_render = det is not None or state.predicted

            if should_render:
                frame = result.orig_img.copy()
                draw_overlays(frame, det, state, cfg.trajectory_points, idx, len(frames))

                if det:
                    det_count += 1
                    cv2.imwrite(str(cfg.detections_dir / f"{name}_{fpath.name}"), frame)

                if cfg.render_output_videos:
                    rendered += 1
                    cv2.imwrite(str(render_dir / f"frame_{rendered:06d}.jpg"), frame)

            del result

        if cfg.render_output_videos and rendered > 0:
            out_path = cfg.output_videos_dir / f"{name}.mp4"
            if out_path.exists() and not cfg.overwrite_outputs:
                raise FileExistsError(f"{out_path} exists; use --overwrite-outputs")
            compose_video(render_dir, out_path, cfg.fps)

    return VideoSummary(
        video_name=name,
        input_video=str(video),
        frame_dir=str(frame_dir),
        detection_frames=det_count,
        sampled_frames=len(frames),
        output_video=str(out_path) if out_path else None,
        detector_weights=str(cfg.weights),
        detector_class_ids=class_ids,
        rendered_frames=rendered if cfg.render_output_videos else None,
    )


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(config: PipelineConfig) -> list[VideoSummary]:
    if not config.videos_dir.exists():
        raise FileNotFoundError(f"Videos dir missing: {config.videos_dir}")
    if not config.weights.exists():
        raise FileNotFoundError(f"Weights missing: {config.weights}")

    config.detections_dir = writable_dir(config.detections_dir)
    ensure_dir(config.detections_dir)

    if config.render_output_videos:
        config.output_videos_dir = writable_dir(config.output_videos_dir)
        ensure_dir(config.output_videos_dir)

    videos = discover_videos(config.videos_dir)
    if not videos:
        raise FileNotFoundError(f"No .mp4 files in {config.videos_dir}")

    model = YOLO(str(config.weights))
    class_ids = resolve_drone_class(model, config.drone_class_name, config.drone_class_id)

    summaries = []
    for v in videos:
        summaries.append(process_video(v, model, config, class_ids))
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    ensure_dir(config.summary_path.parent)
    config.summary_path.write_text(
        json.dumps([asdict(s) for s in summaries], indent=2), encoding="utf-8"
    )
    return summaries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--videos-dir", type=Path, default=Path("videos"))
    parser.add_argument("--frames-dir", type=Path, default=Path("frames"))
    parser.add_argument("--detections-dir", type=Path, default=Path("detections"))
    parser.add_argument("--output-videos-dir", type=Path, default=Path("output_videos"))
    parser.add_argument("--summary-path", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--max-missing-frames", type=int, default=50)
    parser.add_argument("--trajectory-points", type=int, default=300)
    parser.add_argument("--device", default=None)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--drone-class-name", default="drone")
    parser.add_argument("--drone-class-id", type=int, default=None)
    parser.add_argument("--max-box-area-ratio", type=float, default=0.03)
    parser.add_argument("--max-box-width-ratio", type=float, default=0.15)
    parser.add_argument("--max-box-height-ratio", type=float, default=0.15)
    parser.add_argument("--overwrite-frames", action="store_true")
    parser.add_argument("--overwrite-outputs", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)


def config_from_args(args, render: bool) -> PipelineConfig:
    return PipelineConfig(
        videos_dir=args.videos_dir, frames_dir=args.frames_dir,
        detections_dir=args.detections_dir, output_videos_dir=args.output_videos_dir,
        summary_path=args.summary_path, weights=args.weights,
        fps=args.fps, conf=args.conf, imgsz=args.imgsz,
        max_missing_frames=args.max_missing_frames,
        trajectory_points=args.trajectory_points,
        overwrite_frames=args.overwrite_frames, overwrite_outputs=args.overwrite_outputs,
        render_output_videos=render, device=args.device, half=args.half,
        augment=args.augment, drone_class_name=args.drone_class_name,
        drone_class_id=args.drone_class_id, max_frames=args.max_frames,
        max_box_area_ratio=args.max_box_area_ratio,
        max_box_width_ratio=args.max_box_width_ratio,
        max_box_height_ratio=args.max_box_height_ratio,
    )


def detection_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Task 1: save drone detection frames.")
    add_args(parser)
    cfg = config_from_args(parser.parse_args(argv), render=False)
    for s in run_pipeline(cfg):
        print(f"{s.video_name}: {s.sampled_frames} sampled, {s.detection_frames} detections")
    return 0


def render_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Task 2: detection + tracking + video output.")
    add_args(parser)
    cfg = config_from_args(parser.parse_args(argv), render=True)
    for s in run_pipeline(cfg):
        print(f"{s.video_name}: {s.detection_frames} detections, "
              f"{s.rendered_frames} rendered, output={s.output_video}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return render_cli(argv)

# Assignment 3: UAV Drone Detection and Tracking

## Overview

A complete multi-object detection and tracking pipeline for drones in video. The system uses a fine-tuned YOLOv8 detector to locate drones frame-by-frame, then a Kalman filter to maintain smooth trajectory tracking — including through frames where the detector misses the drone entirely.

## Dataset Choice

**Dataset:** [`pathikg/drone-detection-dataset`](https://huggingface.co/datasets/pathikg/drone-detection-dataset) on Hugging Face

- **Single-class (`drone`)** — not a broad multi-class aerial benchmark
- **54.1k total images** (51.4k train, 2.63k test) — large enough for robust fine-tuning
- **Schema:** `image`, `width`, `height`, `image_id`, `objects` with `[x, y, w, h]` bounding boxes
- Already in Parquet-backed HF format, matching the assignment's Parquet deliverable requirement

### Why this dataset over alternatives

The assignment requires detecting **the drone itself** — not analyzing imagery captured from a drone. `pathikg/drone-detection-dataset` is ground-view footage where drones are the labeled targets. VisDrone, while mentioned in the assignment, is primarily an aerial perspective benchmark not built around ground-view drone detection.

### Optional augmentation

For the strongest detector, the pipeline supports adding hard-negative images (sky, birds, empty backgrounds) via `prepare_dataset.py --negative-dir`, which creates empty YOLO labels for background images to reduce false positives.

## Detector: YOLO11x (Ultralytics)

The pipeline uses the [Ultralytics](https://docs.ultralytics.com/) deep learning framework for drone detection. The current checkpoint uses **YOLO11x** — the extra-large variant of YOLO11 (released September 2024) — from [`doguilmak/Drone-Detection-YOLOv11x`](https://huggingface.co/doguilmak/Drone-Detection-YOLOv11x), which was fine-tuned specifically on drone imagery. YOLO11x achieves **precision 0.922, recall 0.831, mAP@50 0.905** on the validation set with ~8.9ms inference latency, representing the state of the art for single-class drone detection. The Ultralytics framework also supports the newer YOLO26 (January 2026), but no drone-specific YOLO26 checkpoint is publicly available yet.

Key inference settings:
- **Confidence threshold:** 0.10 (aggressive to catch small/distant drones)
- **Image size:** 1280px (high resolution for small object detection)
- **Frame sampling:** 10 FPS (high temporal resolution for smooth tracking)
- **Test-time augmentation (TTA):** Enabled — runs multi-scale inference which significantly improves detection of small drones at varying distances
- **Bounding box geometry filters:** Rejects false-positive detections where the box exceeds 3% of frame area, 15% of frame width, or 15% of frame height — drones are small objects, so any oversized detection is almost certainly a false positive (e.g., treeline or sky region)

The detector automatically resolves the "drone" class from the model's label map using fuzzy matching on `drone`, `uav`, and `quadcopter`. For single-class checkpoints, it falls back to using the only available class.

## Kalman Filter Design

The tracker uses [`filterpy.kalman.KalmanFilter`](https://filterpy.readthedocs.io/) with a constant-velocity motion model.

### State vector and matrices

| Component | Value | Description |
|-----------|-------|-------------|
| **State** `x` | `[x, y, vx, vy]` | 2D pixel position + velocity |
| **Measurement** `z` | `[x, y]` | Bounding box center from detector |
| **Transition** `F` | Constant-velocity with `dt = 1/fps` | Predicts next position from current position + velocity |
| **Observation** `H` | `[[1,0,0,0], [0,1,0,0]]` | Extracts position from state |
| **Measurement noise** `R` | `4.0 * I` | Low — trusts detector position tightly |
| **Initial covariance** `P` | `diag([100, 100, 25, 25])` | Moderate uncertainty at initialization |
| **Process noise** `Q` | `Q_discrete_white_noise(var=15.0)` | Higher — allows for fast drone maneuvering between frames |

### Predict-update cycle (per frame)

1. **Detection found + filter exists:** Call `predict()` to advance state, then `update()` with the measured bounding box center. Reset missing-frame counter to 0.
2. **Detection found + no filter:** Initialize a new `KalmanFilter` with the detection center as the starting state. Begin a new trajectory segment.
3. **No detection + filter active:** Call `predict()` only — the filter extrapolates the drone's position using its velocity estimate. Increment the missing-frame counter.
4. **No detection + exceeded `max_missing_frames`:** The tracker goes inactive. If a detection reappears later, a new filter is initialized and a new trajectory segment begins.

### Missed-detection handling

The tracker bridges detection gaps of up to **50 consecutive frames** (5 seconds at 10 FPS). During these gaps:

- The Kalman filter continues predicting the drone's position using its learned velocity
- The estimated position is added to the trajectory, maintaining a continuous track
- The output video renders these **prediction-only frames** with a distinct "PREDICTED" status badge and a hollow orange circle (vs. the solid red circle for measured positions)
- If the drone reappears within the gap window, the filter smoothly incorporates the new measurement
- If the gap exceeds the threshold, the tracker terminates and starts a fresh segment on the next detection — this prevents runaway predictions from drifting indefinitely

This approach means the output video shows the Kalman filter working in real time: you can see it maintain the track through brief occlusions and detector misses.

## Output Video Visualization

Each output frame includes:

- **Green bounding box** with confidence score when a detection is present
- **Trajectory polyline** with a fade effect (older points dimmer, recent points brighter) showing the drone's path history
- **Center point:** solid red circle (measured) or hollow orange circle (predicted)
- **Status badge:** green "TRACKING" when detection is active, orange "PREDICTED (miss N)" during gaps, red "NO TRACK" when lost
- **Frame counter** showing current position in the video

## Failure Cases

- **Tiny drones at long range** — even with TTA and low confidence threshold, extremely distant drones may be missed
- **Bright haze / low contrast** — compression artifacts and atmospheric conditions reduce detector confidence
- **Birds and aircraft** — the main semantic confusers; hard-negative training images help but don't eliminate this
- **Very long dropouts** (>50 frames / 5 seconds) — force the tracker to terminate and reinitialize, creating a trajectory segment break
- **Fast lateral maneuvers** — the constant-velocity model can lag behind sudden direction changes, though the process noise parameter helps compensate

## Usage

All commands go through `main.py`:

```bash
# Task 1: save detection frames only
python main.py detect --weights best.pt

# Task 2: detection + Kalman tracking + output videos (default)
python main.py track --weights best.pt --overwrite-outputs

# Package detections as HF Parquet dataset
python main.py upload --detections-dir detections

# Push to Hugging Face
python main.py upload --detections-dir detections --repo-id YOUR_USER/drone-detections
```

### Training (optional — a pre-trained checkpoint is used)

```bash
# 1. Prepare dataset
python prepare_dataset.py --dataset-id pathikg/drone-detection-dataset --output-dir data/drone

# 2. Fine-tune
python train_detector.py --data data/drone/data.yaml --epochs 30 --device cuda
```

## Deliverables

- **Hugging Face dataset:** [HarshAgarwalNYU/Assignment3Drone](https://huggingface.co/datasets/HarshAgarwalNYU/Assignment3Drone)
- **Output video 1 (YouTube):** https://youtu.be/yjuZ6CPiang
- **Output video 2 (YouTube):** https://youtu.be/XuKAufi5Ngc

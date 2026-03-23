"""Kalman-filter-based single-drone tracker using filterpy."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter

Point = tuple[int, int]
BBox = tuple[float, float, float, float]


@dataclass(slots=True)
class TrackerState:
    estimated_center: Point | None
    active: bool
    missing_frames: int
    trajectory_segments: list[list[Point]]
    predicted: bool = False


def bbox_center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def make_kalman_filter(center: tuple[float, float], dt: float) -> KalmanFilter:
    """4-state constant-velocity Kalman filter: [x, y, vx, vy]."""
    kf = KalmanFilter(dim_x=4, dim_z=2)

    kf.F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1],
    ], dtype=float)

    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=float)

    kf.R = np.eye(2) * 4.0
    kf.P = np.diag([100.0, 100.0, 25.0, 25.0])
    kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=15.0, block_size=2, order_by_dim=False)
    kf.x = np.array([[center[0]], [center[1]], [0], [0]], dtype=float)

    return kf


@dataclass(slots=True)
class DroneTracker:
    fps: float
    max_missing_frames: int = 50
    kf: KalmanFilter | None = None
    missing_frames: int = 0
    trajectory_segments: list[list[Point]] = field(default_factory=lambda: [[]])

    def _start_segment(self) -> None:
        if self.trajectory_segments and self.trajectory_segments[-1]:
            self.trajectory_segments.append([])

    def _record(self, x: float, y: float) -> None:
        if not self.trajectory_segments:
            self.trajectory_segments.append([])
        self.trajectory_segments[-1].append((round(x), round(y)))

    def _center(self) -> Point | None:
        if self.kf is None:
            return None
        return (round(float(self.kf.x[0, 0])), round(float(self.kf.x[1, 0])))

    def step(self, detection: BBox | None) -> TrackerState:
        measurement = bbox_center(detection) if detection else None

        if measurement is not None:
            if self.kf is None or self.missing_frames > self.max_missing_frames:
                self._start_segment()
                self.kf = make_kalman_filter(measurement, dt=1.0 / self.fps)
            else:
                self.kf.predict()
                self.kf.update(np.array([[measurement[0]], [measurement[1]]], dtype=float))
            self.missing_frames = 0
        elif self.kf is not None:
            self.kf.predict()
            self.missing_frames += 1

        center = self._center()
        active = self.kf is not None and self.missing_frames <= self.max_missing_frames

        if center and active:
            self._record(center[0], center[1])

        return TrackerState(
            estimated_center=center,
            active=active,
            missing_frames=self.missing_frames,
            trajectory_segments=self.trajectory_segments,
            predicted=measurement is None and active,
        )

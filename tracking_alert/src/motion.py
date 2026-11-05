from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, Optional, Tuple

import numpy as np


@dataclass
class MotionState:
    track_id: int
    frame_idx: int
    x: float
    y: float
    vx: float
    vy: float
    ax: float
    ay: float
    speed: float
    accel_mag: float
    age: int


class MotionEstimator:
    """객체 중심점 궤적으로 속도/가속도를 추정한다."""

    def __init__(self, ema_alpha: float = 0.4, history: int = 30) -> None:
        self.ema_alpha = float(np.clip(ema_alpha, 0.0, 1.0))
        self.history = history
        self._smooth_xy: Dict[int, Tuple[float, float]] = {}
        self._prev_xy: Dict[int, Tuple[float, float]] = {}
        self._prev_v: Dict[int, Tuple[float, float]] = {}
        self._age: Dict[int, int] = defaultdict(int)
        self._traj: Dict[int, Deque[Tuple[int, float, float]]] = defaultdict(
            lambda: deque(maxlen=self.history)
        )

    def reset(self) -> None:
        self._smooth_xy.clear()
        self._prev_xy.clear()
        self._prev_v.clear()
        self._age.clear()
        self._traj.clear()

    def update(
        self,
        frame_idx: int,
        detections: Iterable[Tuple[int, float, float]],
    ) -> Dict[int, MotionState]:
        """
        detections: (track_id, cx, cy)
        """
        alive = set()
        states: Dict[int, MotionState] = {}

        for track_id, cx, cy in detections:
            alive.add(track_id)
            sx, sy = self._smooth(track_id, cx, cy)
            self._age[track_id] += 1
            self._traj[track_id].append((frame_idx, sx, sy))

            prev_xy = self._prev_xy.get(track_id)
            if prev_xy is None:
                vx = vy = ax = ay = 0.0
            else:
                vx = sx - prev_xy[0]
                vy = sy - prev_xy[1]
                prev_v = self._prev_v.get(track_id)
                if prev_v is None:
                    ax = ay = 0.0
                else:
                    ax = vx - prev_v[0]
                    ay = vy - prev_v[1]
                self._prev_v[track_id] = (vx, vy)

            self._prev_xy[track_id] = (sx, sy)
            speed = float(np.hypot(vx, vy))
            accel_mag = float(np.hypot(ax, ay))
            states[track_id] = MotionState(
                track_id=track_id,
                frame_idx=frame_idx,
                x=sx,
                y=sy,
                vx=vx,
                vy=vy,
                ax=ax,
                ay=ay,
                speed=speed,
                accel_mag=accel_mag,
                age=self._age[track_id],
            )

        self._drop_missing(alive)
        return states

    def trajectory(self, track_id: int) -> list[Tuple[int, float, float]]:
        return list(self._traj.get(track_id, []))

    def _smooth(self, track_id: int, cx: float, cy: float) -> Tuple[float, float]:
        prev = self._smooth_xy.get(track_id)
        if prev is None:
            self._smooth_xy[track_id] = (cx, cy)
            return cx, cy
        a = self.ema_alpha
        sx = a * prev[0] + (1.0 - a) * cx
        sy = a * prev[1] + (1.0 - a) * cy
        self._smooth_xy[track_id] = (sx, sy)
        return sx, sy

    def _drop_missing(self, alive: set[int]) -> None:
        for store in (
            self._smooth_xy,
            self._prev_xy,
            self._prev_v,
            self._age,
            self._traj,
        ):
            for tid in list(store.keys()):
                if tid not in alive:
                    del store[tid]

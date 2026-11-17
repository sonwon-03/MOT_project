from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, Optional, Tuple, Union

import numpy as np

# (track_id, cx, cy) 또는 (track_id, cx, cy, scale)
Detection = Union[
    Tuple[int, float, float],
    Tuple[int, float, float, float],
]


def bbox_scale(bbox_xyxy: Tuple[float, float, float, float]) -> float:
    """원근 보정용 크기: √(폭×높이)."""
    x1, y1, x2, y2 = bbox_xyxy
    w = max(0.0, float(x2) - float(x1))
    h = max(0.0, float(y2) - float(y1))
    return float(np.sqrt(max(1.0, w * h)))


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
    # 박스 크기(√면적). 원근 보정용
    scale: float = 0.0
    # 작아질 때(+): 멀어짐/전진, 커질 때(+): 다가옴
    v_depth: float = 0.0


class MotionEstimator:
    """객체 중심점 + 박스 크기 변화로 속도/가속도를 추정한다.

    화면상 중심이 거의 안 움직여도 박스가 작아지면 앞으로(멀어짐) 가는 것으로 본다.
    """

    def __init__(
        self,
        ema_alpha: float = 0.4,
        history: int = 30,
        depth_gain: float = 1.0,
        use_size_depth: bool = True,
    ) -> None:
        self.ema_alpha = float(np.clip(ema_alpha, 0.0, 1.0))
        self.history = history
        self.depth_gain = float(max(0.0, depth_gain))
        self.use_size_depth = bool(use_size_depth)
        self._smooth_xy: Dict[int, Tuple[float, float]] = {}
        self._smooth_scale: Dict[int, float] = {}
        self._prev_xy: Dict[int, Tuple[float, float]] = {}
        self._prev_scale: Dict[int, float] = {}
        self._prev_v: Dict[int, Tuple[float, float]] = {}
        self._age: Dict[int, int] = defaultdict(int)
        self._traj: Dict[int, Deque[Tuple[int, float, float]]] = defaultdict(
            lambda: deque(maxlen=self.history)
        )

    def reset(self) -> None:
        self._smooth_xy.clear()
        self._smooth_scale.clear()
        self._prev_xy.clear()
        self._prev_scale.clear()
        self._prev_v.clear()
        self._age.clear()
        self._traj.clear()

    def update(
        self,
        frame_idx: int,
        detections: Iterable[Detection],
    ) -> Dict[int, MotionState]:
        """
        detections: (track_id, cx, cy) 또는 (track_id, cx, cy, scale)
        scale은 보통 √(w*h). 없으면 원근(깊이) 보정은 0.
        """
        alive = set()
        states: Dict[int, MotionState] = {}

        for item in detections:
            if len(item) >= 4:
                track_id, cx, cy, raw_scale = int(item[0]), float(item[1]), float(item[2]), float(item[3])
            else:
                track_id, cx, cy = int(item[0]), float(item[1]), float(item[2])
                raw_scale = 0.0

            alive.add(track_id)
            sx, sy = self._smooth(track_id, cx, cy)
            scale = self._smooth_s(track_id, raw_scale) if raw_scale > 0 else 0.0
            self._age[track_id] += 1
            self._traj[track_id].append((frame_idx, sx, sy))

            prev_xy = self._prev_xy.get(track_id)
            prev_scale = self._prev_scale.get(track_id)
            if prev_xy is None:
                vx = vy = ax = ay = 0.0
                v_depth = 0.0
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

                v_depth = 0.0
                if (
                    self.use_size_depth
                    and self.depth_gain > 0
                    and scale > 1.0
                    and prev_scale is not None
                    and prev_scale > 1.0
                ):
                    # 상대 크기 변화. 작아짐(+) → 멀어짐/전진
                    rel = (prev_scale - scale) / prev_scale
                    # 픽셀 속도 단위에 가깝게: 상대변화 * 이전 크기 * 게인
                    v_depth = float(rel * prev_scale * self.depth_gain)

            self._prev_xy[track_id] = (sx, sy)
            if scale > 0:
                self._prev_scale[track_id] = scale

            plane = float(np.hypot(vx, vy))
            # 깊이 성분은 부호 있는 값의 절댓값을 속도에 합산
            # (전진=작아짐, 접근=커짐 모두 "움직임")
            speed = float(np.hypot(plane, abs(v_depth)))
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
                scale=scale,
                v_depth=v_depth,
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

    def _smooth_s(self, track_id: int, scale: float) -> float:
        prev = self._smooth_scale.get(track_id)
        if prev is None or prev <= 0:
            self._smooth_scale[track_id] = scale
            return scale
        a = self.ema_alpha
        s = a * prev + (1.0 - a) * scale
        self._smooth_scale[track_id] = s
        return s

    def _drop_missing(self, alive: set[int]) -> None:
        for store in (
            self._smooth_xy,
            self._smooth_scale,
            self._prev_xy,
            self._prev_scale,
            self._prev_v,
            self._age,
            self._traj,
        ):
            for tid in list(store.keys()):
                if tid not in alive:
                    del store[tid]

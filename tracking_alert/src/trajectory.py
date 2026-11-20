"""객체별 이동 궤적 기록 · 이동 방향/거리 계산 · 경로 시각화."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

_DIRS = ["→ E", "↘ SE", "↓ S", "↙ SW", "← W", "↖ NW", "↑ N", "↗ NE"]
_DIRS_ASCII = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]


def track_color(track_id: int) -> Tuple[int, int, int]:
    """ID마다 고정된 BGR 색 (궤적 구분용)."""
    rng = np.random.default_rng(int(track_id) * 7919 + 17)
    hue = int(rng.integers(0, 180))
    hsv = np.uint8([[[hue, 200, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


@dataclass
class TrackPath:
    track_id: int
    label: str = ""
    points: Deque[Tuple[int, float, float]] = field(default_factory=lambda: deque(maxlen=600))
    total_distance: float = 0.0  # 누적 이동 거리(px)
    first_frame: int = 0
    last_frame: int = 0
    start_xy: Tuple[float, float] = (0.0, 0.0)

    @property
    def displacement(self) -> Tuple[float, float]:
        """시작점 → 현재 위치 변위 (dx, dy)."""
        if not self.points:
            return (0.0, 0.0)
        _, x, y = self.points[-1]
        return (x - self.start_xy[0], y - self.start_xy[1])

    def heading_deg(self, window: int = 10) -> Optional[float]:
        """최근 window 프레임의 이동 방향(도). 화면 기준 0°=오른쪽, 90°=아래."""
        if len(self.points) < 2:
            return None
        pts = list(self.points)[-window:]
        dx, dy = pts[-1][1] - pts[0][1], pts[-1][2] - pts[0][2]
        if math.hypot(dx, dy) < 2.0:
            return None  # 거의 정지
        return math.degrees(math.atan2(dy, dx)) % 360.0

    def direction(self, ascii_only: bool = False) -> str:
        deg = self.heading_deg()
        if deg is None:
            return "-" if ascii_only else "정지"
        idx = int(((deg + 22.5) % 360) // 45)
        return (_DIRS_ASCII if ascii_only else _DIRS)[idx]

    def avg_speed(self) -> float:
        """평균 속도 (px/프레임)."""
        frames = max(1, self.last_frame - self.first_frame)
        return self.total_distance / frames


class TrajectoryStore:
    """track_id → TrackPath. 추적이 끊겨도 기록은 max_idle 프레임 동안 유지."""

    def __init__(self, max_points: int = 600, max_idle: int = 150) -> None:
        self.max_points = max_points
        self.max_idle = max_idle
        self.paths: Dict[int, TrackPath] = {}

    def reset(self) -> None:
        self.paths.clear()

    def update(self, frame_idx: int, tracked: Iterable) -> None:
        for obj in tracked:
            cx, cy = obj.center
            path = self.paths.get(obj.track_id)
            if path is None:
                path = TrackPath(
                    track_id=obj.track_id,
                    label=getattr(obj, "label", ""),
                    points=deque(maxlen=self.max_points),
                    first_frame=frame_idx,
                    start_xy=(cx, cy),
                )
                self.paths[obj.track_id] = path
            elif path.points:
                _, px, py = path.points[-1]
                path.total_distance += math.hypot(cx - px, cy - py)
            path.points.append((frame_idx, cx, cy))
            path.last_frame = frame_idx
        stale = [tid for tid, p in self.paths.items() if frame_idx - p.last_frame > self.max_idle]
        for tid in stale:
            del self.paths[tid]

    def summary(self, active_ids: Optional[set] = None) -> List[dict]:
        rows = []
        for tid, p in sorted(self.paths.items()):
            if active_ids is not None and tid not in active_ids:
                continue
            dx, dy = p.displacement
            rows.append(
                {
                    "track_id": tid,
                    "label": p.label,
                    "frames": p.last_frame - p.first_frame + 1,
                    "distance_px": round(p.total_distance, 1),
                    "displacement_px": round(math.hypot(dx, dy), 1),
                    "direction": p.direction(),
                    "avg_speed": round(p.avg_speed(), 2),
                }
            )
        return rows

    def draw(self, img: np.ndarray, active_ids: Optional[set] = None, tail: int = 60) -> None:
        """현재 보이는 객체의 최근 tail 프레임 경로를 ID 색으로 그린다."""
        for tid, p in self.paths.items():
            if active_ids is not None and tid not in active_ids:
                continue
            pts = list(p.points)[-tail:]
            if len(pts) < 2:
                continue
            arr = np.array([[int(x), int(y)] for _, x, y in pts], dtype=np.int32)
            color = track_color(tid)
            n = len(arr)
            for i in range(1, n):
                # 오래된 점일수록 얇게
                th = 1 + int(2 * i / n)
                cv2.line(img, tuple(arr[i - 1]), tuple(arr[i]), color, th, cv2.LINE_AA)
            cv2.circle(img, tuple(arr[-1]), 3, color, -1, cv2.LINE_AA)
            x, y = arr[-1]
            cv2.putText(
                img,
                f"{p.direction(ascii_only=True)} {p.total_distance:.0f}px",
                (int(x) + 6, int(y) + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

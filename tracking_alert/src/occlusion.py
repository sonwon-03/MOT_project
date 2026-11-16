from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from src.motion import MotionState
from src.tracker import TrackedObject


@dataclass
class PredictedObject:
    """가림 구간에서 등속 직선으로 예측한 객체. 사고 판정에는 사용하지 않는다."""

    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    label: str
    cls_id: int
    vx: float
    vy: float
    missing_frames: int
    hold_frames: int

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


class OcclusionPredictor:
    """
    검출이 끊긴 객체를 직전 속도로 잠시 예측한다.

    - 사고 판정 입력에는 넣지 않는다.
    - hold_seconds 동안만 유지하고, 다시 나타나면 실제 박스로 교체된다.
    """

    def __init__(self, hold_seconds: float = 1.0, fps_hint: float = 20.0) -> None:
        self.hold_seconds = max(0.0, float(hold_seconds))
        self.fps_hint = max(1.0, float(fps_hint))
        self._last: Dict[int, dict] = {}

    @property
    def hold_frames(self) -> int:
        return max(1, int(round(self.hold_seconds * self.fps_hint)))

    def reset(self) -> None:
        self._last.clear()

    def update(
        self,
        frame_idx: int,
        tracked: Iterable[TrackedObject],
        states: Dict[int, MotionState],
    ) -> List[PredictedObject]:
        if self.hold_seconds <= 0:
            self._last.clear()
            return []

        alive: set[int] = set()
        for obj in tracked:
            alive.add(obj.track_id)
            st = states.get(obj.track_id)
            x1, y1, x2, y2 = obj.bbox_xyxy
            self._last[obj.track_id] = {
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "label": obj.label,
                "cls_id": obj.cls_id,
                "vx": float(st.vx) if st is not None else 0.0,
                "vy": float(st.vy) if st is not None else 0.0,
                "last_frame": frame_idx,
            }

        predicted: List[PredictedObject] = []
        max_miss = self.hold_frames
        for tid, info in list(self._last.items()):
            if tid in alive:
                continue
            miss = frame_idx - int(info["last_frame"])
            if miss <= 0:
                continue
            if miss > max_miss:
                del self._last[tid]
                continue

            x1, y1, x2, y2 = info["bbox"]
            dx = info["vx"] * miss
            dy = info["vy"] * miss
            predicted.append(
                PredictedObject(
                    track_id=tid,
                    bbox_xyxy=(x1 + dx, y1 + dy, x2 + dx, y2 + dy),
                    label=info["label"],
                    cls_id=info["cls_id"],
                    vx=info["vx"],
                    vy=info["vy"],
                    missing_frames=miss,
                    hold_frames=max_miss,
                )
            )
        return predicted

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple

import cv2
import numpy as np

from src.tracker import TrackedObject


@dataclass
class _Agent:
    track_id: int
    x: float
    y: float
    vx: float
    vy: float
    w: float = 60
    h: float = 40
    label: str = "car"
    cls_id: int = 2


def generate_collision_scene(
    width: int = 960,
    height: int = 540,
    fps: int = 30,
) -> Iterator[Tuple[np.ndarray, List[TrackedObject]]]:
    """
    YOLO 없이 로직 검증용 합성 장면.
    - ID1: 좌→우 이동 후 충돌하며 정지
    - ID2: 우→좌 이동 후 충돌하며 정지
    - ID3: 상단에서 낙하(하방 가속)
    """
    agents = [
        _Agent(1, 80, 270, 6.0, 0.0, label="car", cls_id=2),
        _Agent(2, 880, 270, -6.0, 0.0, label="car", cls_id=2),
        _Agent(3, 480, 60, 0.0, 1.0, w=36, h=70, label="person", cls_id=0),
    ]

    total = fps * 8
    for frame_idx in range(total):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (35, 35, 35)
        cv2.line(frame, (0, 300), (width, 300), (70, 70, 70), 2)

        # 약 2.5초 지점에서 충돌
        if frame_idx == int(fps * 2.5):
            agents[0].vx = 0.0
            agents[1].vx = 0.0
            agents[0].x = 450
            agents[1].x = 520

        # 낙하 가속
        if frame_idx > fps:
            agents[2].vy += 0.35
        if agents[2].y > height - 80:
            agents[2].y = height - 80
            agents[2].vy = 0.0

        tracked: List[TrackedObject] = []
        for a in agents:
            # 충돌 전까지만 등속 적분
            if not (frame_idx >= int(fps * 2.5) and a.track_id in (1, 2)):
                a.x += a.vx
                a.y += a.vy
            else:
                # 정지 유지 (약간의 박스 노이즈)
                a.x += np.random.uniform(-0.3, 0.3)
                a.y += np.random.uniform(-0.3, 0.3)

            x1, y1 = a.x - a.w / 2, a.y - a.h / 2
            x2, y2 = a.x + a.w / 2, a.y + a.h / 2
            color = (80, 180, 255) if a.cls_id == 2 else (180, 220, 100)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
            cv2.putText(
                frame,
                f"sim-{a.label}",
                (int(x1), int(y1) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            tracked.append(
                TrackedObject(
                    track_id=a.track_id,
                    bbox_xyxy=(x1, y1, x2, y2),
                    conf=0.99,
                    cls_id=a.cls_id,
                    label=a.label,
                )
            )

        cv2.putText(
            frame,
            f"SYNTHETIC frame={frame_idx}",
            (12, height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        yield frame, tracked

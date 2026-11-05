from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class EventConfig:
    """사고 의심 이벤트 판정 설정."""

    # 속도/가속도는 픽셀/프레임, 픽셀/프레임^2 단위
    accel_threshold: float = 8.0
    stop_speed: float = 2.0
    # 이 속도 이상으로 달리다가 줄어야만 사고 후보가 된다.
    min_move_speed: float = 5.0
    baseline_frames: int = 8
    drop_ratio: float = 0.5
    relative_tol_pct: float = 25.0
    absolute_tol: float = 3.0
    confirm_frames: int = 5
    min_track_age: int = 8
    cooldown_frames: int = 30
    proximity_px: float = 80.0

    # 비교용 ON/OFF
    use_accel_spike: bool = True
    use_post_state_confirm: bool = True
    use_proximity_tag: bool = True

    # 좌표 스무딩 (0~1, 클수록 이전값 비중↑)
    ema_alpha: float = 0.4

    def apply_sensitivity(self, level: int) -> None:
        """1은 둔감, 5는 민감. 3이 기본 판정 기준이다."""
        level = min(5, max(1, int(level)))
        presets = {
            1: (0.70, 8.0, 14.0, 8),
            2: (0.60, 6.5, 11.0, 6),
            3: (0.50, 5.0, 8.0, 5),
            4: (0.40, 3.5, 6.0, 4),
            5: (0.30, 2.0, 4.0, 3),
        }
        drop_ratio, min_move_speed, accel_threshold, confirm_frames = presets[level]
        self.drop_ratio = drop_ratio
        self.min_move_speed = min_move_speed
        self.accel_threshold = accel_threshold
        self.confirm_frames = confirm_frames


@dataclass
class AppConfig:
    model_name: str = "yolov8n.pt"
    tracker_cfg: str = "bytetrack.yaml"
    conf: float = 0.35
    iou: float = 0.5
    classes: tuple[int, ...] = (0, 2, 3, 5, 7)  # person, car, motorcycle, bus, truck
    device: str | None = None
    output_dir: str = "outputs"
    event: EventConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.event is None:
            self.event = EventConfig()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

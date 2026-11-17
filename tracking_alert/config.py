from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class EventConfig:
    """사고 의심 이벤트 판정 설정."""

    # 속도/가속도는 픽셀/프레임, 픽셀/프레임^2 단위
    # FAR=0 유지 튜닝 최적값 (Acc≈78.4%, Recall≈65.2%)
    accel_threshold: float = 3.0
    stop_speed: float = 2.0
    # 이 속도 이상으로 달리다가 줄어야만 사고 후보가 된다.
    min_move_speed: float = 3.5
    baseline_frames: int = 8
    drop_ratio: float = 0.25
    relative_tol_pct: float = 25.0
    absolute_tol: float = 3.0
    confirm_frames: int = 4
    min_track_age: int = 5
    cooldown_frames: int = 30
    # 충돌 태그: 박스 일부가 겹칠 때. 0=엄격 겹침, >0이면 그 픽셀만큼 박스를 키워 판정
    proximity_px: float = 0.0

    # 비교용 ON/OFF
    use_accel_spike: bool = True
    use_post_state_confirm: bool = True
    use_proximity_tag: bool = True

    # 코너 선회 오탐 억제: 최근 헤딩이 이만큼 바뀌고 속도가 유지되면 사고로 보지 않음
    turn_heading_deg: float = 18.0
    turn_speed_keep_ratio: float = 0.55
    # 확정 시에는 더 강하게 감속된 상태만 인정 (선회 중 잠깐 느려진 경우 제외)
    confirm_drop_ratio: float = 0.55

    # 좌표 스무딩 (0~1, 클수록 이전값 비중↑)
    ema_alpha: float = 0.4

    def apply_sensitivity(self, level: int) -> None:
        """1은 둔감, 5는 민감. 3이 기본 판정 기준이다."""
        level = min(5, max(1, int(level)))
        # (drop_ratio, min_move_speed, accel_threshold, confirm_frames)
        # 3 = FAR=0 튜닝 최적 근처
        presets = {
            1: (0.50, 5.0, 8.0, 6),
            2: (0.35, 4.0, 5.0, 5),
            3: (0.25, 3.5, 3.0, 4),
            4: (0.22, 3.0, 2.5, 3),
            5: (0.20, 2.5, 2.0, 3),
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
    conf: float = 0.05
    iou: float = 0.5
    classes: tuple[int, ...] = (0, 2, 3, 5, 7)  # person, car, motorcycle, bus, truck
    device: str | None = None
    output_dir: str = "outputs"
    # 가림 시 등속 예측 유지 시간(초). 0이면 기능 끔.
    occlusion_hold_sec: float = 1.0
    # 저화질/소형 차량 검출 강화 (입력 확대 + 고해상도 추론)
    small_object_boost: bool = True
    imgsz: int = 1280
    # 사고 의심 확정 시 시스템 알림음
    sound_alert: bool = True
    event: EventConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.event is None:
            self.event = EventConfig()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

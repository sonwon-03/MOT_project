from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import EventConfig
from src.event_detector import AccidentSuspectDetector
from src.motion import MotionEstimator


def test_collision_stop_triggers_alert() -> None:
    cfg = EventConfig(
        accel_threshold=5.0,
        stop_speed=2.0,
        confirm_frames=3,
        min_track_age=3,
        cooldown_frames=100,
        use_accel_spike=True,
        use_post_state_confirm=True,
        use_proximity_tag=True,
        ema_alpha=0.0,
    )
    motion = MotionEstimator(ema_alpha=0.0)
    det = AccidentSuspectDetector(cfg)

    def boxes(x1: float, x2: float) -> dict:
        # 정지 시 중심 간격(~156px)에서도 박스 일부가 겹치게 한다.
        return {
            1: (x1 - 90, 160.0, x1 + 90, 240.0),
            2: (x2 - 90, 160.0, x2 + 90, 240.0),
        }

    # 등속 접근
    xs1 = [100 + i * 8 for i in range(10)]
    xs2 = [400 - i * 8 for i in range(10)]
    events = []
    for i, (x1, x2) in enumerate(zip(xs1, xs2)):
        states = motion.update(i, [(1, x1, 200.0), (2, x2, 200.0)])
        events.extend(det.update(i, states, bboxes=boxes(x1, x2)))

    # 충돌 후 정지 (박스 겹침)
    for j in range(8):
        i = 10 + j
        x1, x2 = xs1[-1], xs2[-1]
        states = motion.update(i, [(1, x1, 200.0), (2, x2, 200.0)])
        events.extend(det.update(i, states, bboxes=boxes(x1, x2)))

    assert events, "충돌 후 정지에서 사고 의심 이벤트가 발생해야 합니다"
    assert any(e.category == "accident_suspect" for e in events)
    assert any(e.tag == "collision_suspect" for e in events), events
    print("crash ok:", [(e.track_id, e.tag, e.reason) for e in events])


def test_parked_jitter_does_not_alert() -> None:
    cfg = EventConfig(
        accel_threshold=5.0,
        stop_speed=2.0,
        confirm_frames=3,
        min_track_age=3,
        min_move_speed=5.0,
        cooldown_frames=100,
        ema_alpha=0.0,
    )
    motion = MotionEstimator(ema_alpha=0.0)
    det = AccidentSuspectDetector(cfg)
    xs = [100, 101, 100, 102, 100, 101, 100, 118, 100, 101, 100, 100, 101, 100]
    events = []
    for i, x in enumerate(xs):
        states = motion.update(i, [(7, float(x), 200.0)])
        events.extend(det.update(i, states))
    assert not events, events
    print("parked ok")


def test_corner_turn_does_not_alert() -> None:
    """등속 코너 선회는 사고 의심이 나면 안 된다."""
    cfg = EventConfig(
        accel_threshold=3.0,
        stop_speed=2.0,
        min_move_speed=3.5,
        drop_ratio=0.25,
        confirm_frames=3,
        min_track_age=3,
        cooldown_frames=100,
        use_accel_spike=True,
        use_post_state_confirm=True,
        ema_alpha=0.0,
        turn_heading_deg=28.0,
        turn_speed_keep_ratio=0.55,
        confirm_drop_ratio=0.55,
    )
    motion = MotionEstimator(ema_alpha=0.0)
    det = AccidentSuspectDetector(cfg)
    events = []
    # 직선 주행 후 1/4원 선회 (속도 대략 유지)
    speed = 8.0
    x, y = 100.0, 300.0
    for i in range(12):
        x += speed
        states = motion.update(i, [(3, x, y)])
        events.extend(det.update(i, states))
    cx, cy = x, y - 60.0
    for j in range(16):
        i = 12 + j
        ang = -math.pi / 2.0 * (j + 1) / 16.0
        x = cx + 60.0 * math.cos(ang)
        y = cy - 60.0 * math.sin(ang)
        states = motion.update(i, [(3, x, y)])
        events.extend(det.update(i, states))
    assert not events, events
    print("corner ok")


if __name__ == "__main__":
    test_collision_stop_triggers_alert()
    test_parked_jitter_does_not_alert()
    test_corner_turn_does_not_alert()

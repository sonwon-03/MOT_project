from __future__ import annotations

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

    # 등속 접근
    xs1 = [100 + i * 8 for i in range(10)]
    xs2 = [400 - i * 8 for i in range(10)]
    events = []
    for i, (x1, x2) in enumerate(zip(xs1, xs2)):
        states = motion.update(i, [(1, x1, 200.0), (2, x2, 200.0)])
        events.extend(det.update(i, states))

    # 충돌 후 정지
    for j in range(8):
        i = 10 + j
        states = motion.update(i, [(1, xs1[-1], 200.0), (2, xs2[-1], 200.0)])
        events.extend(det.update(i, states))

    assert events, "충돌 후 정지에서 사고 의심 이벤트가 발생해야 합니다"
    assert any(e.category == "accident_suspect" for e in events)
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


if __name__ == "__main__":
    test_collision_stop_triggers_alert()
    test_parked_jitter_does_not_alert()

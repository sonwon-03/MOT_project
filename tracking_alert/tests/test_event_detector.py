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
        ema_alpha=0.0,
    )
    motion = MotionEstimator(ema_alpha=0.0)
    det = AccidentSuspectDetector(cfg)

    def boxes(x1: float, x2: float) -> dict:
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

    # 충돌 후 정지
    for j in range(8):
        i = 10 + j
        x1, x2 = xs1[-1], xs2[-1]
        states = motion.update(i, [(1, x1, 200.0), (2, x2, 200.0)])
        events.extend(det.update(i, states, bboxes=boxes(x1, x2)))

    assert events, "충돌 후 정지에서 사고 의심 이벤트가 발생해야 합니다"
    assert any(e.category == "accident_suspect" for e in events)
    assert any(e.tag == "accident_suspect" for e in events), events
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


def test_corner_turn_constant_speed_still_quiet() -> None:
    """선회 필터 없이도, 속도가 유지되면 급감속 후보가 안 생긴다."""
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
    )
    motion = MotionEstimator(ema_alpha=0.0)
    det = AccidentSuspectDetector(cfg)
    events = []
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
    print("corner constant-speed ok")


def test_headon_bounce_still_alerts() -> None:
    """정면충돌처럼 방향이 뒤집혀도 정지하면 알림이 나야 한다."""
    cfg = EventConfig()
    cfg.apply_sensitivity(5)
    cfg.use_post_state_confirm = True
    cfg.ema_alpha = 0.0
    motion = MotionEstimator(ema_alpha=0.0)
    det = AccidentSuspectDetector(cfg)

    def boxes(x1: float, x2: float) -> dict:
        return {
            1: (x1 - 80, 160.0, x1 + 80, 240.0),
            2: (x2 - 80, 160.0, x2 + 80, 240.0),
        }

    events = []
    for i in range(12):
        x1 = 50 + i * 10
        x2 = 450 - i * 10
        states = motion.update(i, [(1, float(x1), 200.0), (2, float(x2), 200.0)])
        events.extend(det.update(i, states, bboxes=boxes(x1, x2)))

    seq1 = [170, 178, 165, 162, 160, 160, 160, 160]
    seq2 = [280, 272, 285, 288, 290, 290, 290, 290]
    for j, (x1, x2) in enumerate(zip(seq1, seq2)):
        i = 12 + j
        states = motion.update(i, [(1, float(x1), 200.0), (2, float(x2), 200.0)])
        events.extend(det.update(i, states, bboxes=boxes(x1, x2)))

    assert events, "정면충돌 후 정지에서 알림이 필요합니다"
    print("headon bounce ok:", [(e.track_id, e.tag) for e in events])


def test_sudden_interior_loss_alerts() -> None:
    """화면 안쪽에서 감속 직후 추적이 끊기면 사고 후보로 알린다."""
    cfg = EventConfig()
    cfg.apply_sensitivity(5)
    cfg.use_post_state_confirm = True
    cfg.ema_alpha = 0.0
    motion = MotionEstimator(ema_alpha=0.0)
    det = AccidentSuspectDetector(cfg)
    events = []
    # 고속 접근
    for i in range(10):
        x1 = 120 + i * 12
        x2 = 420 - i * 12
        states = motion.update(i, [(9, float(x1), 200.0), (10, float(x2), 200.0)])
        events.extend(
            det.update(
                i,
                states,
                bboxes={
                    9: (x1 - 50, 160, x1 + 50, 240),
                    10: (x2 - 50, 160, x2 + 50, 240),
                },
                frame_shape=(400, 640),
            )
        )
    # 충돌 직전 급감속
    for j, step in enumerate((4, 1, 0)):
        i = 10 + j
        x1 = 120 + 9 * 12 + step
        x2 = 420 - 9 * 12 - step
        states = motion.update(i, [(9, float(x1), 200.0), (10, float(x2), 200.0)])
        events.extend(
            det.update(
                i,
                states,
                bboxes={
                    9: (x1 - 50, 160, x1 + 50, 240),
                    10: (x2 - 50, 160, x2 + 50, 240),
                },
                frame_shape=(400, 640),
            )
        )
    # 추적 손실
    for j in range(5):
        i = 13 + j
        events.extend(det.update(i, {}, bboxes={}, frame_shape=(400, 640)))
    assert events, events
    print("sudden loss ok:", [(e.track_id, e.reason) for e in events])


def test_lone_track_flicker_does_not_alert() -> None:
    """혼자 주행 중 추적만 깜빡인 경우는 사고로 보지 않는다."""
    cfg = EventConfig()
    cfg.apply_sensitivity(5)
    cfg.use_post_state_confirm = True
    cfg.ema_alpha = 0.0
    motion = MotionEstimator(ema_alpha=0.0)
    det = AccidentSuspectDetector(cfg)
    events = []
    for i in range(12):
        x = 200 + i * 10
        states = motion.update(i, [(9, float(x), 200.0)])
        events.extend(
            det.update(
                i,
                states,
                bboxes={9: (x - 40, 160, x + 40, 240)},
                frame_shape=(800, 450),  # 세로에 가까운 흔들림 상황
            )
        )
    for j in range(5):
        events.extend(det.update(12 + j, {}, bboxes={}, frame_shape=(800, 450)))
    assert not events, events
    print("lone flicker ok")


def test_track_lost_after_drop_confirms() -> None:
    """감속 후보 상태에서 추적이 끊겨도 사고로 확정한다."""
    cfg = EventConfig()
    cfg.apply_sensitivity(5)
    cfg.use_post_state_confirm = True
    cfg.ema_alpha = 0.0
    motion = MotionEstimator(ema_alpha=0.0)
    det = AccidentSuspectDetector(cfg)
    events = []
    for i in range(12):
        x = 100 + i * 8
        states = motion.update(i, [(5, float(x), 200.0)])
        events.extend(det.update(i, states, bboxes={5: (x - 40, 160, x + 40, 240)}))
    # 급정지
    for j in range(2):
        i = 12 + j
        x = 100 + 11 * 8
        states = motion.update(i, [(5, float(x), 200.0)])
        events.extend(det.update(i, states, bboxes={5: (x - 40, 160, x + 40, 240)}))
    assert 5 in det._pending
    # 추적 손실
    for j in range(5):
        i = 14 + j
        events.extend(det.update(i, {}, bboxes={}))
    assert events, events
    print("track lost confirm ok:", [(e.track_id, e.reason) for e in events])


def test_size_shrink_counts_as_forward_motion() -> None:
    """중심이 고정돼도 박스가 작아지면 속도 > 0 (전진/멀어짐)."""
    motion = MotionEstimator(ema_alpha=0.0, depth_gain=1.0)
    cx, cy = 320.0, 240.0
    # 점점 작아지는 √(w*h)
    scales = [120.0, 110.0, 100.0, 90.0, 80.0]
    last_speed = 0.0
    for i, s in enumerate(scales):
        states = motion.update(i, [(1, cx, cy, s)])
        last_speed = states[1].speed
    assert last_speed > 1.0, last_speed
    assert states[1].v_depth > 0.0, states[1].v_depth
    print("size shrink forward ok:", round(last_speed, 2), round(states[1].v_depth, 2))


if __name__ == "__main__":
    test_collision_stop_triggers_alert()
    test_parked_jitter_does_not_alert()
    test_corner_turn_constant_speed_still_quiet()
    test_headon_bounce_still_alerts()
    test_sudden_interior_loss_alerts()
    test_lone_track_flicker_does_not_alert()
    test_track_lost_after_drop_confirms()
    test_size_shrink_counts_as_forward_motion()

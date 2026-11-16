from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.motion import MotionEstimator
from src.occlusion import OcclusionPredictor
from src.tracker import TrackedObject


def test_occlusion_predicts_then_expires() -> None:
    motion = MotionEstimator(ema_alpha=0.0)
    occ = OcclusionPredictor(hold_seconds=1.0, fps_hint=10.0)  # 10 frames

    # 5프레임 동안 우측으로 이동
    for i in range(5):
        obj = TrackedObject(
            track_id=1,
            bbox_xyxy=(100 + i * 10, 100, 160 + i * 10, 160),
            conf=0.9,
            cls_id=2,
            label="car",
        )
        states = motion.update(i, [(1, *obj.center)])
        preds = occ.update(i, [obj], states)
        assert preds == []

    # 검출 끊김 → 예측 시작 (hold=10프레임)
    for i in range(5, 15):
        states = motion.update(i, [])
        preds = occ.update(i, [], states)
        assert len(preds) == 1, i
        assert preds[0].track_id == 1
        assert preds[0].bbox_xyxy[0] > 140

    # hold 초과 → 제거
    states = motion.update(15, [])
    preds = occ.update(15, [], states)
    assert preds == []

    print("occlusion ok")


if __name__ == "__main__":
    test_occlusion_predicts_then_expires()

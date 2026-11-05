from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np

from src.event_detector import SuspectEvent
from src.motion import MotionState
from src.tracker import TrackedObject


def draw_overlay(
    frame: np.ndarray,
    tracked: Iterable[TrackedObject],
    states: Dict[int, MotionState],
    events: List[SuspectEvent],
    recent_alerts: List[SuspectEvent],
    toggles: dict,
) -> np.ndarray:
    out = frame.copy()

    for obj in tracked:
        x1, y1, x2, y2 = map(int, obj.bbox_xyxy)
        st = states.get(obj.track_id)
        color = (60, 200, 60)
        for ev in recent_alerts:
            if ev.track_id == obj.track_id:
                color = (0, 0, 255)
                break

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"ID {obj.track_id} {obj.label}"
        if st is not None:
            label += f" v={st.speed:.1f} a={st.accel_mag:.1f}"
        cv2.putText(
            out,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
        if st is not None:
            p1 = (int(st.x), int(st.y))
            p2 = (int(st.x + st.vx * 5), int(st.y + st.vy * 5))
            cv2.arrowedLine(out, p1, p2, (255, 180, 0), 2, tipLength=0.3)

    y0 = 28
    lines = [
        "[1] accel spike: " + ("ON" if toggles.get("use_accel_spike") else "OFF"),
        "[2] post confirm: " + ("ON" if toggles.get("use_post_state_confirm") else "OFF"),
        "[3] proximity tag: " + ("ON" if toggles.get("use_proximity_tag") else "OFF"),
        "q: quit | s: save frame",
    ]
    for i, text in enumerate(lines):
        cv2.putText(
            out,
            text,
            (12, y0 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )

    if recent_alerts:
        latest = recent_alerts[-1]
        msg = (
            f"ALERT ID={latest.track_id} {latest.tag} "
            f"frame={latest.frame_idx}"
        )
        cv2.rectangle(out, (0, out.shape[0] - 40), (out.shape[1], out.shape[0]), (0, 0, 180), -1)
        cv2.putText(
            out,
            msg,
            (12, out.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    for ev in events:
        cv2.circle(out, (int(ev.x), int(ev.y)), 16, (0, 0, 255), 2)

    return out

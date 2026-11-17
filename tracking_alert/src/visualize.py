from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np

from src.event_detector import SuspectEvent
from src.motion import MotionState
from src.occlusion import PredictedObject
from src.tracker import TrackedObject


def _draw_dashed_rect(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    gap: int = 8,
) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    for x in range(x1, x2, gap * 2):
        cv2.line(img, (x, y1), (min(x + gap, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + gap, x2), y2), color, thickness)
    for y in range(y1, y2, gap * 2):
        cv2.line(img, (x1, y), (x1, min(y + gap, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + gap, y2)), color, thickness)


# BGR
NORMAL_BGR = (60, 200, 60)
ALERT_BGR = (0, 0, 255)
PREDICT_BGR = (0, 200, 255)
VELOCITY_BGR = (255, 180, 0)


def draw_legend_bar(
    img: np.ndarray,
    y_bottom: int | None = None,
    scale: float = 1.0,
) -> int:
    """화면 하단에 테두리 의미 범례를 그리고, 사용한 바 높이를 반환한다."""
    h, w = img.shape[:2]
    bar_h = max(36, int(42 * scale))
    if y_bottom is None:
        y_bottom = h
    y0 = max(0, y_bottom - bar_h)
    cv2.rectangle(img, (0, y0), (w, y_bottom), (24, 24, 24), -1)

    items = [
        (NORMAL_BGR, "solid", "Normal track"),
        (ALERT_BGR, "solid", "Accident alert"),
        (PREDICT_BGR, "dash", "Occlusion predict"),
        (VELOCITY_BGR, "arrow", "Velocity"),
    ]
    x = 12
    box = max(14, int(16 * scale))
    font = 0.45 * scale
    thick = max(1, int(round(scale)))
    cy = y0 + bar_h // 2
    for color, style, text in items:
        if style == "solid":
            cv2.rectangle(
                img,
                (x, cy - box // 2),
                (x + box, cy + box // 2),
                color,
                max(2, thick),
            )
        elif style == "dash":
            _draw_dashed_rect(
                img,
                (x, cy - box // 2),
                (x + box, cy + box // 2),
                color,
                thickness=max(2, thick),
                gap=4,
            )
        else:
            cv2.arrowedLine(
                img,
                (x, cy),
                (x + box, cy),
                color,
                max(2, thick),
                tipLength=0.4,
            )
        x += box + 6
        cv2.putText(
            img,
            text,
            (x, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        text_w = int(cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font, 1)[0][0])
        x += text_w + 18
        if x > w - 80:
            break
    return bar_h


def draw_overlay(
    frame: np.ndarray,
    tracked: Iterable[TrackedObject],
    states: Dict[int, MotionState],
    events: List[SuspectEvent],
    recent_alerts: List[SuspectEvent],
    toggles: dict,
    predicted: Optional[Iterable[PredictedObject]] = None,
) -> np.ndarray:
    out = frame.copy()
    alert_ids = {ev.track_id for ev in recent_alerts}
    tracked_list = list(tracked)
    # 박스 과다 시 그리기 비용 제한 (알림 대상은 항상 표시)
    if len(tracked_list) > 80:
        alerts = [o for o in tracked_list if o.track_id in alert_ids]
        others = sorted(
            (o for o in tracked_list if o.track_id not in alert_ids),
            key=lambda o: o.conf,
            reverse=True,
        )
        tracked_list = alerts + others[: max(0, 80 - len(alerts))]

    for obj in tracked_list:
        x1, y1, x2, y2 = map(int, obj.bbox_xyxy)
        st = states.get(obj.track_id)
        is_alert = obj.track_id in alert_ids
        color = ALERT_BGR if is_alert else NORMAL_BGR
        thickness = 3 if is_alert else 2

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        # 저신뢰 박스는 사각형만 (텍스트/화살표 생략 → 속도↑)
        if not is_alert and obj.conf < 0.15:
            continue
        label = f"ID {obj.track_id} {obj.label}"
        if is_alert:
            label = "ALERT " + label
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
            cv2.arrowedLine(out, p1, p2, VELOCITY_BGR, 2, tipLength=0.3)

    for obj in predicted or []:
        x1, y1, x2, y2 = map(int, obj.bbox_xyxy)
        color = PREDICT_BGR
        _draw_dashed_rect(out, (x1, y1), (x2, y2), color, thickness=2)
        remain = max(0, obj.hold_frames - obj.missing_frames)
        label = f"ID {obj.track_id} PREDICT ({remain}f)"
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
        cx, cy = obj.center
        p1 = (int(cx), int(cy))
        p2 = (int(cx + obj.vx * 5), int(cy + obj.vy * 5))
        cv2.arrowedLine(out, p1, p2, color, 2, tipLength=0.3)

    y0 = 28
    hold = toggles.get("occlusion_hold_sec", 0)
    lines = [
        "[1] accel spike: " + ("ON" if toggles.get("use_accel_spike") else "OFF"),
        "[2] post confirm: " + ("ON" if toggles.get("use_post_state_confirm") else "OFF"),
        "[3] proximity tag: " + ("ON" if toggles.get("use_proximity_tag") else "OFF"),
        f"occlusion hold: {hold}s",
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

    legend_h = draw_legend_bar(out, y_bottom=out.shape[0], scale=1.0)
    if recent_alerts:
        latest = recent_alerts[-1]
        msg = (
            f"ALERT ID={latest.track_id} {latest.tag} "
            f"frame={latest.frame_idx}"
        )
        alert_h = 36
        y1 = out.shape[0] - legend_h - alert_h
        cv2.rectangle(
            out,
            (0, y1),
            (out.shape[1], out.shape[0] - legend_h),
            ALERT_BGR,
            -1,
        )
        cv2.putText(
            out,
            msg,
            (12, y1 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    for ev in events:
        cv2.circle(out, (int(ev.x), int(ev.y)), 16, ALERT_BGR, 2)

    return out

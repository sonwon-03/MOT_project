from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from config import EventConfig
from src.motion import MotionState


@dataclass
class SuspectEvent:
    frame_idx: int
    track_id: int
    category: str  # accident_suspect
    tag: str  # collision_suspect | fall_suspect | accident_suspect
    x: float
    y: float
    speed: float
    accel_mag: float
    reason: str
    nearby_ids: list[int] = field(default_factory=list)


class AccidentSuspectDetector:
    """
    움직이던 객체가 속도를 크게 잃은 뒤 그 상태가 유지되면 사고 의심으로 본다.

    정지 객체의 박스 흔들림은 가속도만 크고 직전 속도가 낮아서 제외한다.
    코너 선회처럼 방향만 바뀌고 속도가 유지되는 경우도 제외한다.
    """

    def __init__(self, cfg: EventConfig) -> None:
        self.cfg = cfg
        self._pending: Dict[int, dict] = {}
        self._cooldown_until: Dict[int, int] = {}
        self._speed_hist: Dict[int, Deque[float]] = defaultdict(
            lambda: deque(maxlen=max(cfg.baseline_frames * 2, 12))
        )
        self._vel_hist: Dict[int, Deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=8)
        )
        self._bboxes: Dict[int, tuple[float, float, float, float]] = {}
        self.events: List[SuspectEvent] = []

    def reset(self) -> None:
        self._pending.clear()
        self._cooldown_until.clear()
        self._speed_hist.clear()
        self._vel_hist.clear()
        self._bboxes.clear()
        self.events.clear()

    def update(
        self,
        frame_idx: int,
        states: Dict[int, MotionState],
        bboxes: Optional[Dict[int, tuple[float, float, float, float]]] = None,
    ) -> List[SuspectEvent]:
        self._bboxes = dict(bboxes or {})
        fired: List[SuspectEvent] = []
        alive_ids = set(states.keys())

        for tid in list(self._pending.keys()):
            if tid not in alive_ids:
                del self._pending[tid]
        for tid in list(self._speed_hist.keys()):
            if tid not in alive_ids:
                del self._speed_hist[tid]
        for tid in list(self._vel_hist.keys()):
            if tid not in alive_ids:
                del self._vel_hist[tid]

        for tid, st in states.items():
            self._vel_hist[tid].append((st.vx, st.vy))

            if st.age < self.cfg.min_track_age:
                self._speed_hist[tid].append(st.speed)
                continue
            if frame_idx < self._cooldown_until.get(tid, -1):
                self._speed_hist[tid].append(st.speed)
                continue

            if tid in self._pending:
                event = self._advance_pending(frame_idx, st, states)
                self._speed_hist[tid].append(st.speed)
                if event is not None:
                    fired.append(event)
                    self.events.append(event)
                    self._cooldown_until[tid] = frame_idx + self.cfg.cooldown_frames
                    self._pending.pop(tid, None)
                continue

            baseline = self._baseline_speed(tid)
            self._speed_hist[tid].append(st.speed)
            if self._is_likely_turn(tid, st, baseline):
                continue
            if not self._is_crash_drop(st, baseline):
                continue

            self._pending[tid] = {
                "start_frame": frame_idx,
                "pre_speed": baseline,
                "peak_accel": st.accel_mag,
                "stable_count": 0,
            }
            if self.cfg.use_post_state_confirm:
                continue

            event = self._make_event(
                frame_idx,
                st,
                states,
                reason=(
                    f"speed_drop {baseline:.2f}->{st.speed:.2f}, "
                    f"accel={st.accel_mag:.2f}"
                ),
            )
            fired.append(event)
            self.events.append(event)
            self._cooldown_until[tid] = frame_idx + self.cfg.cooldown_frames
            self._pending.pop(tid, None)

        return fired

    def _advance_pending(
        self,
        frame_idx: int,
        st: MotionState,
        states: Dict[int, MotionState],
    ) -> Optional[SuspectEvent]:
        pending = self._pending[st.track_id]
        pending["peak_accel"] = max(pending["peak_accel"], st.accel_mag)

        # 선회 중이면 후보 폐기
        if self._is_likely_turn(st.track_id, st, pending["pre_speed"]):
            self._pending.pop(st.track_id, None)
            return None

        if self._is_post_state_stable(st, pending):
            pending["stable_count"] += 1
        else:
            pending["stable_count"] = 0
            # 너무 오래 불안정하면 후보 폐기 (프레임 튐 등)
            if frame_idx - pending["start_frame"] > self.cfg.confirm_frames * 3:
                self._pending.pop(st.track_id, None)
                return None

        if pending["stable_count"] >= self.cfg.confirm_frames:
            return self._make_event(
                frame_idx,
                st,
                states,
                reason=(
                    f"speed_drop {pending['pre_speed']:.2f}->{st.speed:.2f}, "
                    f"stable={pending['stable_count']}f, "
                    f"accel={pending['peak_accel']:.2f}"
                ),
            )
        return None

    def _baseline_speed(self, track_id: int) -> Optional[float]:
        hist = self._speed_hist.get(track_id)
        need = max(4, self.cfg.baseline_frames // 2)
        if hist is None or len(hist) < need:
            return None
        vals = sorted(list(hist)[-self.cfg.baseline_frames :])
        return vals[len(vals) // 2]

    def _heading_change_deg(self, track_id: int) -> float:
        hist = list(self._vel_hist.get(track_id, []))
        angles: list[float] = []
        for vx, vy in hist:
            if math.hypot(vx, vy) < 1.0:
                continue
            angles.append(math.atan2(vy, vx))
        if len(angles) < 3:
            return 0.0
        total = 0.0
        for a0, a1 in zip(angles, angles[1:]):
            d = a1 - a0
            while d > math.pi:
                d -= 2.0 * math.pi
            while d < -math.pi:
                d += 2.0 * math.pi
            total += d
        return abs(math.degrees(total))

    def _is_likely_turn(
        self, track_id: int, st: MotionState, baseline: Optional[float]
    ) -> bool:
        """방향이 크게 바뀌는데 속도가 어느 정도 유지되면 코너 선회로 본다."""
        if baseline is None or baseline < self.cfg.min_move_speed * 0.8:
            return False
        keep = float(self.cfg.turn_speed_keep_ratio)
        if st.speed < baseline * keep:
            return False
        if st.speed < self.cfg.stop_speed * 2.0:
            return False
        return self._heading_change_deg(track_id) >= float(self.cfg.turn_heading_deg)

    def _is_crash_drop(self, st: MotionState, baseline: Optional[float]) -> bool:
        if baseline is None or baseline < self.cfg.min_move_speed:
            return False
        dropped = st.speed <= baseline * (1.0 - self.cfg.drop_ratio)
        if not dropped:
            return False
        if not self.cfg.use_accel_spike:
            return True
        sharp_drop = (baseline - st.speed) >= self.cfg.min_move_speed
        return st.accel_mag >= self.cfg.accel_threshold or sharp_drop

    def _is_post_state_stable(self, st: MotionState, pending: dict) -> bool:
        # 사고 확정은 선회 중 잠깐 느려진 수준이 아니라, 충분히 감속된 상태만 인정
        confirm_drop = max(float(self.cfg.drop_ratio), float(self.cfg.confirm_drop_ratio))
        limit = pending["pre_speed"] * (1.0 - confirm_drop) + self.cfg.absolute_tol
        return st.speed <= max(limit, self.cfg.stop_speed + self.cfg.absolute_tol)

    def _make_event(
        self,
        frame_idx: int,
        st: MotionState,
        states: Dict[int, MotionState],
        reason: str,
    ) -> SuspectEvent:
        nearby = self._nearby_ids(st, states) if self.cfg.use_proximity_tag else []
        if nearby:
            tag = "collision_suspect"
        elif st.vy > abs(st.vx) and st.ay > 0:
            # 화면 좌표 y↓ 증가를 낙하 쪽 힌트로만 사용
            tag = "fall_suspect"
        else:
            tag = "accident_suspect"

        return SuspectEvent(
            frame_idx=frame_idx,
            track_id=st.track_id,
            category="accident_suspect",
            tag=tag,
            x=st.x,
            y=st.y,
            speed=st.speed,
            accel_mag=st.accel_mag,
            reason=reason,
            nearby_ids=nearby,
        )

    def _nearby_ids(
        self, st: MotionState, states: Dict[int, MotionState]
    ) -> list[int]:
        """박스 일부라도 겹치면 충돌 근접으로 본다. 박스가 없으면 태그를 붙이지 않는다."""
        mine = self._bboxes.get(st.track_id)
        if mine is None:
            return []
        margin = max(0.0, float(self.cfg.proximity_px))
        ids: list[int] = []
        for other in states.values():
            if other.track_id == st.track_id:
                continue
            other_box = self._bboxes.get(other.track_id)
            if other_box is None:
                continue
            if self._boxes_overlap(mine, other_box, margin=margin):
                ids.append(other.track_id)
        return ids

    @staticmethod
    def _boxes_overlap(
        a: tuple[float, float, float, float],
        b: tuple[float, float, float, float],
        margin: float = 0.0,
    ) -> bool:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        if margin > 0:
            ax1 -= margin
            ay1 -= margin
            ax2 += margin
            ay2 += margin
            bx1 -= margin
            by1 -= margin
            bx2 += margin
            by2 += margin
        return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1

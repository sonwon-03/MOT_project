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
    tag: str  # accident_suspect | fall_suspect
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
    근접(박스 겹침) 충돌 태그는 쓰지 않는다. 원근/세로 영상에서 오탐이 많기 때문.
    정면충돌 직후 추적이 잠깐 끊겨도 후보를 바로 버리지 않는다.
    """

    def __init__(self, cfg: EventConfig) -> None:
        self.cfg = cfg
        self._pending: Dict[int, dict] = {}
        self._cooldown_until: Dict[int, int] = {}
        self._speed_hist: Dict[int, Deque[float]] = defaultdict(
            lambda: deque(maxlen=max(cfg.baseline_frames * 2, 12))
        )
        self._bboxes: Dict[int, tuple[float, float, float, float]] = {}
        self._last_states: Dict[int, MotionState] = {}
        self._last_bboxes: Dict[int, tuple[float, float, float, float]] = {}
        self._frame_shape: Optional[tuple[int, int]] = None
        self.events: List[SuspectEvent] = []

    def reset(self) -> None:
        self._pending.clear()
        self._cooldown_until.clear()
        self._speed_hist.clear()
        self._bboxes.clear()
        self._last_states.clear()
        self._last_bboxes.clear()
        self._frame_shape = None
        self.events.clear()

    def update(
        self,
        frame_idx: int,
        states: Dict[int, MotionState],
        bboxes: Optional[Dict[int, tuple[float, float, float, float]]] = None,
        frame_shape: Optional[tuple[int, int]] = None,
    ) -> List[SuspectEvent]:
        self._bboxes = dict(bboxes or {})
        if frame_shape is not None:
            self._frame_shape = frame_shape
        fired: List[SuspectEvent] = []
        alive_ids = set(states.keys())

        # 추적 끊김: 충돌 직후 흔한 일이라 바로 폐기하지 않고 정지로 간주해 확정 시도
        for tid in list(self._pending.keys()):
            if tid in alive_ids:
                continue
            event = self._advance_missing(frame_idx, tid)
            if event is not None:
                fired.append(event)
                self.events.append(event)
                self._cooldown_until[tid] = frame_idx + self.cfg.cooldown_frames
                self._pending.pop(tid, None)

        # 고속 객체가 화면 안쪽에서 갑자기 사라짐:
        # 직전 감속이 있을 때만 후보로 본다. (근접 겹침 조건은 사용하지 않음)
        for tid, prev in self._last_states.items():
            if tid in alive_ids or tid in self._pending:
                continue
            if frame_idx < self._cooldown_until.get(tid, -1):
                continue
            baseline = self._baseline_speed(tid)
            if baseline is None or baseline < self.cfg.min_move_speed:
                continue
            if not self._is_interior(prev.x, prev.y):
                continue
            if not self._recently_decelerating(tid, baseline):
                continue
            stopped = MotionState(
                track_id=prev.track_id,
                frame_idx=frame_idx,
                x=prev.x,
                y=prev.y,
                vx=0.0,
                vy=0.0,
                ax=-prev.vx,
                ay=-prev.vy,
                speed=0.0,
                accel_mag=float(math.hypot(prev.vx, prev.vy)),
                age=prev.age,
            )
            self._pending[tid] = {
                "start_frame": frame_idx,
                "pre_speed": baseline,
                "peak_accel": max(prev.accel_mag, stopped.accel_mag),
                "stable_count": 0,
                "missing": 0,
                "last_st": stopped,
                "sudden_loss": True,
            }

        for tid in list(self._speed_hist.keys()):
            if tid not in alive_ids and tid not in self._pending:
                del self._speed_hist[tid]

        for tid, st in states.items():
            if st.age < self.cfg.min_track_age:
                self._speed_hist[tid].append(st.speed)
                continue
            if frame_idx < self._cooldown_until.get(tid, -1):
                self._speed_hist[tid].append(st.speed)
                continue

            if tid in self._pending:
                self._pending[tid]["missing"] = 0
                self._pending[tid]["last_st"] = st
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
            if not self._is_crash_drop(st, baseline):
                continue

            self._pending[tid] = {
                "start_frame": frame_idx,
                "pre_speed": baseline,
                "peak_accel": st.accel_mag,
                "stable_count": 0,
                "missing": 0,
                "last_st": st,
            }
            if self.cfg.use_post_state_confirm:
                continue

            event = self._make_event(
                frame_idx,
                st,
                reason=(
                    f"speed_drop {baseline:.2f}->{st.speed:.2f}, "
                    f"accel={st.accel_mag:.2f}"
                ),
            )
            fired.append(event)
            self.events.append(event)
            self._cooldown_until[tid] = frame_idx + self.cfg.cooldown_frames
            self._pending.pop(tid, None)

        self._last_states = dict(states)
        self._last_bboxes = dict(self._bboxes)
        return fired

    def _is_interior(self, x: float, y: float) -> bool:
        """화면 가장자리로 빠져나간 소실은 충돌 후보에서 제외."""
        if self._frame_shape is None:
            return True
        h, w = self._frame_shape
        mx = w * 0.08
        my = h * 0.08
        return mx <= x <= (w - mx) and my <= y <= (h - my)

    def _recently_decelerating(self, track_id: int, baseline: float) -> bool:
        hist = list(self._speed_hist.get(track_id, []))
        if len(hist) < 3 or baseline <= 0:
            return False
        latest = hist[-1]
        prev = hist[-3]
        return latest <= baseline * (1.0 - self.cfg.drop_ratio) or latest <= prev * 0.7

    def _advance_missing(self, frame_idx: int, tid: int) -> Optional[SuspectEvent]:
        pending = self._pending.get(tid)
        if pending is None:
            return None
        pending["missing"] = int(pending.get("missing", 0)) + 1
        st = pending.get("last_st")
        if st is None or not self._is_post_state_stable(st, pending):
            if pending["missing"] > self.cfg.confirm_frames * 3:
                self._pending.pop(tid, None)
            return None

        pending["stable_count"] = int(pending.get("stable_count", 0)) + 1
        need = self.cfg.confirm_frames
        if pending["stable_count"] >= need:
            return self._make_event(
                frame_idx,
                st,
                reason=(
                    f"speed_drop {pending['pre_speed']:.2f}->{st.speed:.2f}, "
                    f"track_lost_confirm={pending['stable_count']}f, "
                    f"accel={pending['peak_accel']:.2f}"
                ),
            )
        if pending["missing"] > self.cfg.confirm_frames * 3:
            self._pending.pop(tid, None)
        return None

    def _advance_pending(
        self,
        frame_idx: int,
        st: MotionState,
        states: Dict[int, MotionState],
    ) -> Optional[SuspectEvent]:
        pending = self._pending[st.track_id]
        pending["peak_accel"] = max(pending["peak_accel"], st.accel_mag)

        if self._is_post_state_stable(st, pending):
            pending["stable_count"] += 1
        else:
            pending["stable_count"] = 0
            if frame_idx - pending["start_frame"] > self.cfg.confirm_frames * 3:
                self._pending.pop(st.track_id, None)
                return None

        need = self.cfg.confirm_frames
        if pending["stable_count"] >= need:
            return self._make_event(
                frame_idx,
                st,
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

    def _is_crash_drop(self, st: MotionState, baseline: Optional[float]) -> bool:
        if baseline is None or baseline < self.cfg.min_move_speed:
            return False
        drop_ratio = float(self.cfg.drop_ratio)
        dropped = st.speed <= baseline * (1.0 - drop_ratio)
        if not dropped:
            return False
        if not self.cfg.use_accel_spike:
            return True
        sharp_drop = (baseline - st.speed) >= self.cfg.min_move_speed
        return st.accel_mag >= self.cfg.accel_threshold or sharp_drop

    def _is_post_state_stable(self, st: MotionState, pending: dict) -> bool:
        near_stop = (
            st.speed <= self.cfg.stop_speed + self.cfg.absolute_tol
            and st.speed <= pending["pre_speed"] * (1.0 - float(self.cfg.drop_ratio)) + 1.0
        )
        if near_stop:
            return True
        confirm_drop = max(float(self.cfg.drop_ratio), float(self.cfg.confirm_drop_ratio))
        limit = pending["pre_speed"] * (1.0 - confirm_drop) + 1.0
        return st.speed <= limit

    def _make_event(
        self,
        frame_idx: int,
        st: MotionState,
        reason: str,
    ) -> SuspectEvent:
        if st.vy > abs(st.vx) and st.ay > 0:
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
            nearby_ids=[],
        )

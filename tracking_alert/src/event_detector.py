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
    코너 선회 제외 필터는 쓰지 않는다. (박스 크기 원근 보정으로 전진/선회 속도 구분)
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
        # 세로/저화질에서 추적 깜빡임이 흔해서, 근처 다른 차 또는 직전 감속이 있을 때만 후보로 본다.
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
            nearby_ids = self._nearby_from_last(tid, prev)
            decelerating = self._recently_decelerating(tid, baseline)
            if not nearby_ids and not decelerating:
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
                "had_proximity": bool(nearby_ids),
                "nearby_ids": nearby_ids,
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
                nearby = self._nearby_ids(st, states) if self.cfg.use_proximity_tag else []
                if nearby:
                    self._pending[tid]["had_proximity"] = True
                    self._pending[tid]["nearby_ids"] = nearby
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

            nearby = self._nearby_ids(st, states) if self.cfg.use_proximity_tag else []
            self._pending[tid] = {
                "start_frame": frame_idx,
                "pre_speed": baseline,
                "peak_accel": st.accel_mag,
                "stable_count": 0,
                "missing": 0,
                "last_st": st,
                "had_proximity": bool(nearby),
                "nearby_ids": nearby,
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
                nearby_ids=nearby,
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
        # 세로 영상은 상하로 出入이 많아 가장자리 마진을 더 둔다
        portrait = h > w
        mx = w * (0.12 if portrait else 0.08)
        my = h * (0.12 if portrait else 0.08)
        return mx <= x <= (w - mx) and my <= y <= (h - my)

    def _recently_decelerating(self, track_id: int, baseline: float) -> bool:
        hist = list(self._speed_hist.get(track_id, []))
        if len(hist) < 3 or baseline <= 0:
            return False
        latest = hist[-1]
        prev = hist[-3]
        return latest <= baseline * (1.0 - self.cfg.drop_ratio) or latest <= prev * 0.7

    def _nearby_from_last(self, track_id: int, st: MotionState) -> list[int]:
        """직전 프레임 기준으로 가까운 다른 추적 ID."""
        mine = self._last_bboxes.get(track_id)
        ids: list[int] = []
        margin = max(12.0, float(self.cfg.proximity_px))
        for oid, other in self._last_states.items():
            if oid == track_id:
                continue
            other_box = self._last_bboxes.get(oid)
            if mine is not None and other_box is not None:
                if self._boxes_overlap(mine, other_box, margin=margin):
                    ids.append(oid)
                    continue
            # 박스 없거나 안 겹쳐도 중심이 가까우면 충돌 가능성으로 본다
            if math.hypot(st.x - other.x, st.y - other.y) <= 140.0:
                ids.append(oid)
        return ids

    def _advance_missing(self, frame_idx: int, tid: int) -> Optional[SuspectEvent]:
        pending = self._pending.get(tid)
        if pending is None:
            return None
        pending["missing"] = int(pending.get("missing", 0)) + 1
        st = pending.get("last_st")
        # 사라지기 직전에도 감속/정지 상태였을 때만 추적 손실을 확정으로 인정
        if st is None or not self._is_post_state_stable(st, pending):
            if pending["missing"] > self.cfg.confirm_frames * 3:
                self._pending.pop(tid, None)
            return None

        pending["stable_count"] = int(pending.get("stable_count", 0)) + 1
        need = self.cfg.confirm_frames
        if pending.get("had_proximity"):
            need = max(1, self.cfg.confirm_frames - 1)
        if pending["stable_count"] >= need:
            return self._make_event(
                frame_idx,
                st,
                {st.track_id: st},
                reason=(
                    f"speed_drop {pending['pre_speed']:.2f}->{st.speed:.2f}, "
                    f"track_lost_confirm={pending['stable_count']}f, "
                    f"accel={pending['peak_accel']:.2f}"
                ),
                nearby_ids=list(pending.get("nearby_ids") or []),
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
        if pending.get("had_proximity"):
            need = max(1, self.cfg.confirm_frames - 1)
        if pending["stable_count"] >= need:
            return self._make_event(
                frame_idx,
                st,
                states,
                reason=(
                    f"speed_drop {pending['pre_speed']:.2f}->{st.speed:.2f}, "
                    f"stable={pending['stable_count']}f, "
                    f"accel={pending['peak_accel']:.2f}"
                ),
                nearby_ids=list(pending.get("nearby_ids") or []),
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
        # 세로 영상은 원근 때문에 픽셀 속도가 출렁여서 더 큰 감속만 사고로 본다
        if self._frame_shape is not None:
            h, w = self._frame_shape
            if h > w:
                drop_ratio = max(drop_ratio, 0.35)
        dropped = st.speed <= baseline * (1.0 - drop_ratio)
        if not dropped:
            return False
        if not self.cfg.use_accel_spike:
            return True
        sharp_drop = (baseline - st.speed) >= self.cfg.min_move_speed
        return st.accel_mag >= self.cfg.accel_threshold or sharp_drop

    def _is_post_state_stable(self, st: MotionState, pending: dict) -> bool:
        # 거의 정지했거나, 충돌 근접 후 분명히 느려졌으면 유지로 본다.
        # absolute_tol을 비율 한도에 크게 더하지 않아 코너 진입 감속 오확정을 줄인다.
        near_stop = (
            st.speed <= self.cfg.stop_speed + self.cfg.absolute_tol
            and st.speed <= pending["pre_speed"] * (1.0 - float(self.cfg.drop_ratio)) + 1.0
        )
        if near_stop:
            return True
        confirm_drop = max(float(self.cfg.drop_ratio), float(self.cfg.confirm_drop_ratio))
        if pending.get("had_proximity"):
            confirm_drop = min(confirm_drop, max(float(self.cfg.drop_ratio), 0.35))
        limit = pending["pre_speed"] * (1.0 - confirm_drop) + 1.0
        return st.speed <= limit

    def _make_event(
        self,
        frame_idx: int,
        st: MotionState,
        states: Dict[int, MotionState],
        reason: str,
        nearby_ids: Optional[list[int]] = None,
    ) -> SuspectEvent:
        if nearby_ids is None:
            nearby = self._nearby_ids(st, states) if self.cfg.use_proximity_tag else []
        else:
            nearby = list(nearby_ids)
            if self.cfg.use_proximity_tag and not nearby:
                nearby = self._nearby_ids(st, states)
        if nearby:
            tag = "collision_suspect"
        elif st.vy > abs(st.vx) and st.ay > 0:
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

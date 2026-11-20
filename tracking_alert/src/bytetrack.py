"""ByteTrack 직접 구현 + 개선(외형 특징 융합, 유실 트랙 재연결).

원 논문: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every
Detection Box" (ECCV 2022)

처리 과정 (update 1회 = 1프레임)
  1) 검출을 신뢰도로 high(>= track_high_thresh) / low(track_low_thresh ~ high)로 나눈다.
  2) 기존 트랙(Tracked + Lost)의 다음 위치를 칼만 필터로 예측한다.
  3) 1차 연관: high 검출 ↔ 전체 트랙, 비용 = 1 - IoU (fuse_score면 IoU×점수)
     [개선 A] appearance_weight > 0 이면 색 히스토그램 거리와 가중합한다.
  4) 2차 연관: low 검출 ↔ 1차에서 남은 Tracked 트랙 (IoU만, 임계 0.5)
     → 가려져 점수가 떨어진 객체도 ID를 유지한다 (ByteTrack 핵심 아이디어).
  5) 미확정(새로 생긴) 트랙 ↔ 남은 high 검출 (임계 0.7), 실패하면 제거.
  6) 매칭 안 된 트랙은 Lost, track_buffer 프레임 지나면 Removed.
  7) 남은 high 검출 중 new_track_thresh 이상이면 새 트랙.
     [개선 B] 새 트랙을 만들기 전에, IoU로는 안 붙었던 Lost/최근 Removed 트랙과
     "예측 중심 거리 + 크기 비율 + 외형"으로 다시 비교해 같은 객체면 예전 ID로 재연결한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# ─────────────────────────── 칼만 필터 ───────────────────────────


class KalmanFilterXYAH:
    """상태 [cx, cy, a(가로/세로), h, vcx, vcy, va, vh] 등속 모델 (원 ByteTrack과 동일 구성)."""

    def __init__(self) -> None:
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        h = measurement[3]
        std = [
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            1e-2,
            2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
            1e-5,
            10 * self._std_weight_velocity * h,
        ]
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        h = mean[3]
        std_pos = [self._std_weight_position * h, self._std_weight_position * h, 1e-2, self._std_weight_position * h]
        std_vel = [self._std_weight_velocity * h, self._std_weight_velocity * h, 1e-5, self._std_weight_velocity * h]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        h = mean[3]
        std = [self._std_weight_position * h, self._std_weight_position * h, 1e-1, self._std_weight_position * h]
        innovation_cov = np.diag(np.square(std))
        proj_mean = self._update_mat @ mean
        proj_cov = self._update_mat @ covariance @ self._update_mat.T + innovation_cov
        kalman_gain = np.linalg.solve(proj_cov, (covariance @ self._update_mat.T).T).T
        new_mean = mean + (measurement - proj_mean) @ kalman_gain.T
        new_cov = covariance - kalman_gain @ proj_cov @ kalman_gain.T
        return new_mean, new_cov


_KF = KalmanFilterXYAH()

# ─────────────────────────── 보조 함수 ───────────────────────────


def xyxy_to_xyah(b: np.ndarray) -> np.ndarray:
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h], dtype=float)


def xyah_to_xyxy(m: np.ndarray) -> np.ndarray:
    cx, cy, a, h = m[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=float)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4) b: (M,4) xyxy → (N,M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def linear_assignment(cost: np.ndarray, thresh: float):
    """헝가리안 매칭. cost > thresh 인 쌍은 버린다."""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    rows, cols = linear_sum_assignment(cost)
    matches, used_r, used_c = [], set(), set()
    for r, c in zip(rows, cols):
        if cost[r, c] <= thresh:
            matches.append((r, c))
            used_r.add(r)
            used_c.add(c)
    un_r = [r for r in range(cost.shape[0]) if r not in used_r]
    un_c = [c for c in range(cost.shape[1]) if c not in used_c]
    return matches, un_r, un_c


def color_histogram(frame: Optional[np.ndarray], box: np.ndarray) -> Optional[np.ndarray]:
    """[개선 A] 박스 안쪽(가장자리 15% 제외) HSV 히스토그램 (H 16 × S 8), L1 정규화."""
    if frame is None:
        return None
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = int(max(0, x1 + 0.15 * bw)), int(min(w_img, x2 - 0.15 * bw))
    y1, y2 = int(max(0, y1 + 0.15 * bh)), int(min(h_img, y2 - 0.15 * bh))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten()
    s = hist.sum()
    return hist / s if s > 0 else None


def appearance_distance(f1: Optional[np.ndarray], f2: Optional[np.ndarray]) -> float:
    """Bhattacharyya 거리 0(같음)~1(다름). 특징이 없으면 중립값 0.5."""
    if f1 is None or f2 is None:
        return 0.5
    bc = float(np.sum(np.sqrt(f1 * f2)))
    return float(np.sqrt(max(0.0, 1.0 - bc)))


# ─────────────────────────── 트랙 ───────────────────────────

TRACKED, LOST, REMOVED = 1, 2, 3


class STrack:
    _next_id = 0

    def __init__(self, box: np.ndarray, score: float, cls_id: int, feature: Optional[np.ndarray]):
        self.box = np.asarray(box, dtype=float)
        self.score = float(score)
        self.cls_id = int(cls_id)
        self.feature = feature
        self.mean = None
        self.cov = None
        self.track_id = 0
        self.state = TRACKED
        self.is_activated = False
        self.frame_id = 0
        self.start_frame = 0
        self.tracklet_len = 0
        self.lost_since = 0  # Lost가 된 프레임

    @classmethod
    def next_id(cls) -> int:
        cls._next_id += 1
        return cls._next_id

    @property
    def pred_box(self) -> np.ndarray:
        return self.box if self.mean is None else xyah_to_xyxy(self.mean)

    def predict(self) -> None:
        mean = self.mean.copy()
        if self.state != TRACKED:
            mean[7] = 0  # 잃어버린 트랙은 높이 변화 속도를 멈춤 (원 구현과 동일)
        self.mean, self.cov = _KF.predict(mean, self.cov)

    def activate(self, frame_id: int, track_id: Optional[int] = None) -> None:
        self.track_id = track_id if track_id is not None else self.next_id()
        self.mean, self.cov = _KF.initiate(xyxy_to_xyah(self.box))
        self.state = TRACKED
        self.is_activated = frame_id == 1
        self.frame_id = self.start_frame = frame_id
        self.tracklet_len = 0

    def _absorb(self, det: "STrack", frame_id: int, feat_momentum: float) -> None:
        self.mean, self.cov = _KF.update(self.mean, self.cov, xyxy_to_xyah(det.box))
        self.box = det.box
        self.score = det.score
        self.cls_id = det.cls_id
        if det.feature is not None:
            if self.feature is None:
                self.feature = det.feature
            else:  # 외형 특징 지수이동평균
                f = feat_momentum * self.feature + (1 - feat_momentum) * det.feature
                self.feature = f / max(f.sum(), 1e-9)
        self.frame_id = frame_id
        self.state = TRACKED
        self.is_activated = True

    def update(self, det: "STrack", frame_id: int, feat_momentum: float = 0.9) -> None:
        self.tracklet_len += 1
        self._absorb(det, frame_id, feat_momentum)

    def re_activate(self, det: "STrack", frame_id: int, feat_momentum: float = 0.9) -> None:
        self.tracklet_len = 0
        self._absorb(det, frame_id, feat_momentum)


# ─────────────────────────── 트래커 ───────────────────────────


@dataclass
class ByteTrackParams:
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    track_buffer: int = 30
    match_thresh: float = 0.8
    fuse_score: bool = True
    # ── 개선 A: 외형 특징 융합 (0이면 끔 = 원 ByteTrack) ──
    appearance_weight: float = 0.0
    # ── 개선 B: 유실 트랙 재연결 (False면 끔 = 원 ByteTrack) ──
    relink: bool = False
    relink_max_gap: int = 90       # 사라진 뒤 이 프레임 안이면 재연결 후보
    relink_dist: float = 1.5       # 예측 중심 거리 / 박스 높이 상한
    relink_size_ratio: float = 0.6  # 높이 비 하한 (작은/큰 쪽)
    relink_app_thresh: float = 0.45  # 외형 거리 상한

    @classmethod
    def from_dict(cls, d: dict) -> "ByteTrackParams":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in (d or {}).items() if k in keys})


class ByteTracker:
    def __init__(self, params: Optional[ByteTrackParams] = None, frame_rate: float = 30.0) -> None:
        self.p = params or ByteTrackParams()
        self.frame_id = 0
        # 버퍼는 30fps 기준 값을 실제 fps에 맞춤 (원 구현과 동일)
        self.max_time_lost = int(round(frame_rate / 30.0 * self.p.track_buffer))
        self.tracked: List[STrack] = []
        self.lost: List[STrack] = []
        self.removed: List[STrack] = []
        self.relink_count = 0
        self.need_features = self.p.appearance_weight > 0 or self.p.relink

    def reset(self) -> None:
        self.frame_id = 0
        self.tracked, self.lost, self.removed = [], [], []
        self.relink_count = 0

    # ---- 비용 ----
    def _cost(self, tracks: List[STrack], dets: List[STrack], use_app: bool) -> np.ndarray:
        if not tracks or not dets:
            return np.zeros((len(tracks), len(dets)))
        iou = iou_matrix(np.array([t.pred_box for t in tracks]), np.array([d.box for d in dets]))
        sim = iou * np.array([d.score for d in dets])[None, :] if self.p.fuse_score else iou
        cost = 1.0 - sim
        w = self.p.appearance_weight
        if use_app and w > 0:
            app = np.array([[appearance_distance(t.feature, d.feature) for d in dets] for t in tracks])
            # IoU가 0이면(겹침 없음) 외형이 비슷해도 매칭하지 않음 → 순간이동 방지
            cost = np.where(iou > 0, (1 - w) * cost + w * app, 1.0)
        return cost

    # ---- 메인 ----
    def update(self, dets: np.ndarray, frame: Optional[np.ndarray] = None) -> List[STrack]:
        """dets: (N, 6) [x1, y1, x2, y2, score, cls]. 반환: 이번 프레임에 활성화된 트랙."""
        p = self.p
        self.frame_id += 1
        fid = self.frame_id
        dets = np.asarray(dets, dtype=float).reshape(-1, 6)
        scores = dets[:, 4]

        def make(rows):
            return [
                STrack(r[:4], r[4], int(r[5]), color_histogram(frame, r[:4]) if self.need_features else None)
                for r in rows
            ]

        high = make(dets[scores >= p.track_high_thresh])
        low = make(dets[(scores > p.track_low_thresh) & (scores < p.track_high_thresh)])

        unconfirmed = [t for t in self.tracked if not t.is_activated]
        confirmed = [t for t in self.tracked if t.is_activated]
        pool = confirmed + [t for t in self.lost if t not in confirmed]
        for t in pool:
            t.predict()

        activated, refound, lost_now, removed_now = [], [], [], []

        # 3) 1차 연관: high ↔ 전체
        m, u_trk, u_det = linear_assignment(self._cost(pool, high, use_app=True), p.match_thresh)
        for ti, di in m:
            t, d = pool[ti], high[di]
            if t.state == TRACKED:
                t.update(d, fid)
                activated.append(t)
            else:
                t.re_activate(d, fid)
                refound.append(t)

        # 4) 2차 연관: low ↔ 남은 Tracked (IoU만)
        remain = [pool[i] for i in u_trk if pool[i].state == TRACKED]
        if remain and low:
            iou = iou_matrix(np.array([t.pred_box for t in remain]), np.array([d.box for d in low]))
            m2, u_trk2, _ = linear_assignment(1.0 - iou, 0.5)
        else:
            m2, u_trk2 = [], list(range(len(remain)))
        for ti, di in m2:
            remain[ti].update(low[di], fid)
            activated.append(remain[ti])
        for ti in u_trk2:
            t = remain[ti]
            if t.state != LOST:
                t.state = LOST
                t.lost_since = fid
                lost_now.append(t)

        # 5) 미확정 트랙 ↔ 남은 high
        left = [high[i] for i in u_det]
        m3, u_unc, u_det3 = linear_assignment(self._cost(unconfirmed, left, use_app=False), 0.7)
        for ti, di in m3:
            unconfirmed[ti].update(left[di], fid)
            activated.append(unconfirmed[ti])
        for ti in u_unc:
            unconfirmed[ti].state = REMOVED
            removed_now.append(unconfirmed[ti])

        # 7) 새 트랙 (+ 개선 B 재연결)
        newcomers = [left[i] for i in u_det3 if left[i].score >= p.new_track_thresh]
        if p.relink and newcomers:
            newcomers = self._relink(newcomers, refound, fid)
        for d in newcomers:
            d.activate(fid)
            activated.append(d)

        # 6) 오래 잃어버린 트랙 제거
        for t in self.lost:
            if t.state == LOST and fid - t.frame_id > self.max_time_lost:
                t.state = REMOVED
                removed_now.append(t)

        # 목록 정리
        self.tracked = [t for t in self.tracked if t.state == TRACKED]
        self.tracked = _join(self.tracked, activated)
        self.tracked = _join(self.tracked, refound)
        self.lost = [t for t in self.lost if t.state == LOST and t not in self.tracked]
        self.lost = _join(self.lost, [t for t in lost_now if t.state == LOST])
        self.removed.extend(removed_now)
        # 재연결 후보로 쓸 최근 제거 트랙만 남김
        self.removed = [t for t in self.removed if t.state == REMOVED and fid - t.frame_id <= p.relink_max_gap][-200:]
        self.tracked, self.lost = _remove_duplicates(self.tracked, self.lost)
        return [t for t in self.tracked if t.is_activated]

    def _relink(self, newcomers: List[STrack], refound: List[STrack], fid: int) -> List[STrack]:
        """[개선 B] 새로 보이는 검출을 IoU 매칭에 실패한 Lost/최근 Removed 트랙과 다시 비교."""
        p = self.p
        cands = [
            t for t in self.lost + self.removed
            if t.state in (LOST, REMOVED) and t.is_activated and t.tracklet_len >= 3
            and 0 < fid - t.frame_id <= p.relink_max_gap
        ]
        if not cands:
            return newcomers
        big = 1e6
        cost = np.full((len(cands), len(newcomers)), big)
        for i, t in enumerate(cands):
            pb = t.pred_box
            ph = max(pb[3] - pb[1], 1.0)
            pc = np.array([(pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2])
            for j, d in enumerate(newcomers):
                dh = max(d.box[3] - d.box[1], 1.0)
                ratio = min(ph, dh) / max(ph, dh)
                if ratio < p.relink_size_ratio:
                    continue
                dc = np.array([(d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2])
                dist = float(np.linalg.norm(pc - dc)) / max(ph, dh)
                if dist > p.relink_dist:
                    continue
                app = appearance_distance(t.feature, d.feature)
                if app > p.relink_app_thresh:
                    continue
                gap = (fid - t.frame_id) / max(p.relink_max_gap, 1)
                cost[i, j] = 0.5 * dist / p.relink_dist + 0.4 * app / p.relink_app_thresh + 0.1 * gap
        m, _, u_new = linear_assignment(cost, big / 2)
        for ci, ni in m:
            t, d = cands[ci], newcomers[ni]
            # 오래 예측만 하던 칼만 상태는 부정확하므로 새 검출 위치에서 다시 시작
            t.mean, t.cov = _KF.initiate(xyxy_to_xyah(d.box))
            t.re_activate(d, fid)
            if t in self.removed:
                self.removed.remove(t)
            refound.append(t)
            self.relink_count += 1
        return [newcomers[i] for i in u_new]


def _join(a: List[STrack], b: List[STrack]) -> List[STrack]:
    seen = {t.track_id for t in a}
    out = list(a)
    for t in b:
        if t.track_id not in seen:
            seen.add(t.track_id)
            out.append(t)
    return out


def _remove_duplicates(tracked: List[STrack], lost: List[STrack]):
    """Tracked와 Lost가 거의 같은 위치(IoU>0.85)면 짧은 쪽을 버린다 (원 구현과 동일)."""
    if not tracked or not lost:
        return tracked, lost
    iou = iou_matrix(np.array([t.pred_box for t in tracked]), np.array([t.pred_box for t in lost]))
    drop_t, drop_l = set(), set()
    for i, j in zip(*np.where(iou > 0.85)):
        ti = tracked[i].frame_id - tracked[i].start_frame
        tj = lost[j].frame_id - lost[j].start_frame
        if ti > tj:
            drop_l.add(j)
        else:
            drop_t.add(i)
    return (
        [t for i, t in enumerate(tracked) if i not in drop_t],
        [t for j, t in enumerate(lost) if j not in drop_l],
    )

"""MOT 추적 성능 지표: HOTA(직접 구현), MOTA·IDF1·ID Switch·MOTP(motmetrics).

GT 형식: MOTChallenge gt.txt  frame,id,x,y,w,h,flag,class,visibility
 - 평가 대상: class==1(pedestrian) & flag==1
 - distractor(2 사람이 탄 차량, 7 정지한 사람, 8 distractor, 12 반사): 이 박스와 겹치는
   예측은 FP로 세지 않고 제거 (MOTChallenge 공식 평가와 같은 처리)
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.bytetrack import iou_matrix

DISTRACTOR_CLASSES = {2, 7, 8, 12}
# frame → (ids ndarray, boxes xyxy ndarray)
FrameData = Dict[int, Tuple[np.ndarray, np.ndarray]]


def load_mot_gt(gt_path: Path, min_visibility: float = 0.0) -> Tuple[FrameData, FrameData]:
    """(평가 GT, distractor GT) 반환."""
    gt, dis = defaultdict(list), defaultdict(list)
    arr = np.loadtxt(gt_path, delimiter=",", ndmin=2)
    for r in arr:
        f, tid, x, y, w, h, flag, cls, vis = r[:9]
        box = [x, y, x + w, y + h]
        if int(cls) == 1 and int(flag) == 1 and vis >= min_visibility:
            gt[int(f)].append((int(tid), box))
        elif int(cls) in DISTRACTOR_CLASSES or (int(cls) == 1 and int(flag) == 0):
            dis[int(f)].append((int(tid), box))
    return _pack(gt), _pack(dis)


def load_mot_results(path: Path) -> FrameData:
    out = defaultdict(list)
    if Path(path).stat().st_size == 0:
        return {}
    arr = np.loadtxt(path, delimiter=",", ndmin=2)
    for r in arr:
        f, tid, x, y, w, h = r[:6]
        out[int(f)].append((int(tid), [x, y, x + w, y + h]))
    return _pack(out)


def _pack(d) -> FrameData:
    return {
        f: (np.array([i for i, _ in v], dtype=int), np.array([b for _, b in v], dtype=float).reshape(-1, 4))
        for f, v in d.items()
    }


def remove_distractor_matches(pred: FrameData, gt: FrameData, dis: FrameData, thr: float = 0.5) -> FrameData:
    """GT와 먼저 매칭되지 않은 예측 중 distractor와 IoU>=thr이면 제거."""
    out = {}
    for f, (pids, pboxes) in pred.items():
        keep = np.ones(len(pids), dtype=bool)
        if f in dis and len(pids):
            gids, gboxes = gt.get(f, (np.zeros(0, int), np.zeros((0, 4))))
            matched = set()
            if len(gids):
                iou = iou_matrix(gboxes, pboxes)
                r, c = linear_sum_assignment(-iou)
                matched = {cc for rr, cc in zip(r, c) if iou[rr, cc] >= thr}
            iou_d = iou_matrix(dis[f][1], pboxes)
            r, c = linear_sum_assignment(-iou_d)
            for rr, cc in zip(r, c):
                if iou_d[rr, cc] >= thr and cc not in matched:
                    keep[cc] = False
        out[f] = (pids[keep], pboxes[keep])
    return out


# ───────────────────────────── HOTA ─────────────────────────────

ALPHAS = np.arange(0.05, 0.99, 0.05)


def hota_raw(gt: FrameData, pred: FrameData) -> dict:
    """한 시퀀스의 HOTA 누적값 (여러 시퀀스 합산 가능). TrackEval hota.py와 같은 절차."""
    frames = sorted(set(gt) | set(pred))
    g_map, p_map = {}, {}
    for f in frames:
        for i in gt.get(f, (np.zeros(0, int),))[0]:
            g_map.setdefault(int(i), len(g_map))
        for i in pred.get(f, (np.zeros(0, int),))[0]:
            p_map.setdefault(int(i), len(p_map))
    ng, npd = max(1, len(g_map)), max(1, len(p_map))
    potential = np.zeros((ng, npd))
    g_count, p_count = np.zeros((ng, 1)), np.zeros((1, npd))
    cache = []
    # 1단계: 전역 정렬 점수 계산용 누적
    for f in frames:
        gids, gb = gt.get(f, (np.zeros(0, int), np.zeros((0, 4))))
        pids, pb = pred.get(f, (np.zeros(0, int), np.zeros((0, 4))))
        gi = np.array([g_map[int(i)] for i in gids], dtype=int)
        pi = np.array([p_map[int(i)] for i in pids], dtype=int)
        sim = iou_matrix(gb, pb)
        cache.append((gi, pi, sim))
        if len(gi) and len(pi):
            denom = sim.sum(0)[None, :] + sim.sum(1)[:, None] - sim
            sim_iou = np.where(denom > 1e-10, sim / np.maximum(denom, 1e-10), 0.0)
            potential[gi[:, None], pi[None, :]] += sim_iou
        g_count[gi] += 1
        p_count[0, pi] += 1
    gas = potential / np.maximum(g_count + p_count - potential, 1e-10)

    na = len(ALPHAS)
    tp, fn, fp, loc = np.zeros(na), np.zeros(na), np.zeros(na), np.zeros(na)
    matches = [np.zeros((ng, npd)) for _ in range(na)]
    # 2단계: 프레임별 매칭
    for gi, pi, sim in cache:
        if len(gi) == 0:
            fp += len(pi)
            continue
        if len(pi) == 0:
            fn += len(gi)
            continue
        score = gas[gi[:, None], pi[None, :]] * sim
        r, c = linear_sum_assignment(-score)
        for a, alpha in enumerate(ALPHAS):
            ok = sim[r, c] >= alpha - np.finfo(float).eps
            mr, mc = r[ok], c[ok]
            n = len(mr)
            tp[a] += n
            fn[a] += len(gi) - n
            fp[a] += len(pi) - n
            loc[a] += sim[mr, mc].sum()
            matches[a][gi[mr], pi[mc]] += 1
    ass_a = np.zeros(na)
    for a in range(na):
        mcnt = matches[a]
        ass = mcnt / np.maximum(1, g_count + p_count - mcnt)
        ass_a[a] = (mcnt * ass).sum()  # 나중에 TP로 나눔 (시퀀스 합산 위해 분자 유지)
    return {"tp": tp, "fn": fn, "fp": fp, "loc": loc, "ass_num": ass_a}


def hota_finalize(raw: dict) -> dict:
    tp, fn, fp = raw["tp"], raw["fn"], raw["fp"]
    det_a = tp / np.maximum(1, tp + fn + fp)
    ass_a = raw["ass_num"] / np.maximum(1, tp)
    hota = np.sqrt(det_a * ass_a)
    loc_a = np.maximum(1e-10, raw["loc"]) / np.maximum(1e-10, tp)
    return {
        "HOTA": float(hota.mean()),
        "DetA": float(det_a.mean()),
        "AssA": float(ass_a.mean()),
        "LocA": float(loc_a.mean()),
    }


def hota_merge(raws: List[dict]) -> dict:
    return {k: sum(r[k] for r in raws) for k in raws[0]}


# ─────────────────────── CLEAR / ID 지표 (motmetrics) ───────────────────────


def clear_id_metrics(gt: FrameData, pred: FrameData):
    """motmetrics 누적기 반환 (IoU 0.5 기준)."""
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=False)
    for f in sorted(set(gt) | set(pred)):
        gids, gb = gt.get(f, (np.zeros(0, int), np.zeros((0, 4))))
        pids, pb = pred.get(f, (np.zeros(0, int), np.zeros((0, 4))))
        iou = iou_matrix(gb, pb)
        dist = np.where(iou >= 0.5, 1.0 - iou, np.nan)
        acc.update(gids.tolist(), pids.tolist(), dist, frameid=f)
    return acc


METRIC_NAMES = [
    "mota", "idf1", "num_switches", "motp", "num_false_positives", "num_misses",
    "mostly_tracked", "mostly_lost", "num_fragmentations", "idp", "idr", "precision", "recall",
]


def summarize_accumulators(accs: list, names: list) -> dict:
    """시퀀스별 + 전체(OVERALL) 지표 dict."""
    import motmetrics as mm

    mh = mm.metrics.create()
    df = mh.compute_many(accs, metrics=METRIC_NAMES, names=names, generate_overall=True)
    out = {}
    for name, row in df.iterrows():
        out[name] = {
            "MOTA": float(row["mota"]),
            "IDF1": float(row["idf1"]),
            "IDSW": int(row["num_switches"]),
            # motmetrics의 motp는 (1-IoU) 평균 → IoU 평균으로 바꿔 표기
            "MOTP": float(1.0 - row["motp"]) if not np.isnan(row["motp"]) else 0.0,
            "FP": int(row["num_false_positives"]),
            "FN": int(row["num_misses"]),
            "MT": int(row["mostly_tracked"]),
            "ML": int(row["mostly_lost"]),
            "Frag": int(row["num_fragmentations"]),
            "Precision": float(row["precision"]),
            "Recall": float(row["recall"]),
        }
    return out

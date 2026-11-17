"""
오탐(FP)=0 유지하면서 Accuracy/Recall을 올리는 판정 파라미터를 찾는다.

1) 영상별 궤적을 한 번 추출해 캐시
2) EventConfig 조합만 빠르게 재채점
"""

from __future__ import annotations

import json
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import AppConfig, EventConfig
from eval_accuracy import VideoResult, default_workers, list_videos, summarize
from src.event_detector import AccidentSuspectDetector
from src.motion import MotionEstimator, MotionState, bbox_scale
from src.tracker import YoloByteTracker

CACHE_DIR = ROOT / "outputs" / "tune_cache"
OUT_PATH = ROOT / "outputs" / "tune_far0_results.json"


def extract_motion(video_path: Path, conf: float) -> list[dict[int, dict]]:
    cfg = AppConfig(conf=conf, device=0)
    tracker = YoloByteTracker(
        model_name=cfg.model_name,
        tracker_cfg=cfg.tracker_cfg,
        conf=cfg.conf,
        iou=cfg.iou,
        classes=cfg.classes,
        device=cfg.device,
    )
    motion = MotionEstimator(ema_alpha=cfg.event.ema_alpha)
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"open failed: {video_path}")
    frames: list[dict[int, dict]] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        tracked = tracker.track_frame(frame)
        dets = [
            (o.track_id, *o.center, bbox_scale(o.bbox_xyxy)) for o in tracked
        ]
        states = motion.update(frame_idx, dets)
        frames.append(
            {
                tid: {
                    "track_id": st.track_id,
                    "frame_idx": st.frame_idx,
                    "x": st.x,
                    "y": st.y,
                    "vx": st.vx,
                    "vy": st.vy,
                    "ax": st.ax,
                    "ay": st.ay,
                    "speed": st.speed,
                    "accel_mag": st.accel_mag,
                    "age": st.age,
                }
                for tid, st in states.items()
            }
        )
        frame_idx += 1
    cap.release()
    return frames


def score_cached(
    frames: list[dict[int, dict]],
    label: str,
    path: str,
    event: EventConfig,
) -> VideoResult:
    detector = AccidentSuspectDetector(event)
    alerts = []
    for frame_idx, packed in enumerate(frames):
        states = {
            tid: MotionState(**vals) for tid, vals in packed.items()
        }
        fired = detector.update(frame_idx, states)
        alerts.extend(fired)
    predicted = len(alerts) > 0
    correct = (label == "accident" and predicted) or (
        label == "normal" and not predicted
    )
    return VideoResult(
        path=path,
        label=label,
        predicted_accident=predicted,
        alert_count=len(alerts),
        first_alert_frame=alerts[0].frame_idx if alerts else None,
        frames=len(frames),
        correct=correct,
        alerts=[],
    )


def build_cache(conf: float, workers: int) -> list[tuple[str, Path, Path]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [("accident", p) for p in list_videos(ROOT / "eval_data" / "accident")] + [
        ("normal", p) for p in list_videos(ROOT / "eval_data" / "normal")
    ]
    out_jobs: list[tuple[str, Path, Path]] = []

    def one(label: str, path: Path) -> tuple[str, Path, Path]:
        cache_path = CACHE_DIR / f"conf{conf:.2f}_{label}_{path.stem}.pkl"
        if not cache_path.exists():
            print(f"[cache] {label}: {path.name}", flush=True)
            frames = extract_motion(path, conf)
            with cache_path.open("wb") as f:
                pickle.dump(frames, f)
        else:
            print(f"[cache-hit] {label}: {path.name}", flush=True)
        return label, path, cache_path

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(one, label, path) for label, path in jobs]
        for fut in as_completed(futs):
            out_jobs.append(fut.result())
    # stable order
    order = {str(p): i for i, (_, p) in enumerate(jobs)}
    out_jobs.sort(key=lambda x: order[str(x[1])])
    return out_jobs


def load_all(cache_jobs: list[tuple[str, Path, Path]]):
    loaded = []
    for label, path, cache_path in cache_jobs:
        with cache_path.open("rb") as f:
            frames = pickle.load(f)
        loaded.append((label, str(path), frames))
    return loaded


def make_event(
    sensitivity: int | None = None,
    drop_ratio: float | None = None,
    min_move_speed: float | None = None,
    accel_threshold: float | None = None,
    confirm_frames: int | None = None,
    min_track_age: int | None = None,
    absolute_tol: float | None = None,
    use_post_state_confirm: bool = True,
    use_accel_spike: bool = True,
) -> EventConfig:
    event = EventConfig()
    if sensitivity is not None:
        event.apply_sensitivity(sensitivity)
    if drop_ratio is not None:
        event.drop_ratio = drop_ratio
    if min_move_speed is not None:
        event.min_move_speed = min_move_speed
    if accel_threshold is not None:
        event.accel_threshold = accel_threshold
    if confirm_frames is not None:
        event.confirm_frames = confirm_frames
    if min_track_age is not None:
        event.min_track_age = min_track_age
    if absolute_tol is not None:
        event.absolute_tol = absolute_tol
    event.use_post_state_confirm = use_post_state_confirm
    event.use_accel_spike = use_accel_spike
    return event


def evaluate_event(loaded, event: EventConfig) -> dict:
    results = [
        score_cached(frames, label, path, deepcopy(event))
        for label, path, frames in loaded
    ]
    metrics = summarize(results)
    fn = [
        Path(r.path).name
        for r in results
        if r.label == "accident" and not r.predicted_accident
    ]
    fp = [
        Path(r.path).name
        for r in results
        if r.label == "normal" and r.predicted_accident
    ]
    return {"metrics": metrics, "fn": fn, "fp": fp, "event": asdict(event)}


def param_grid() -> list[EventConfig]:
    cfgs: list[EventConfig] = []
    # sensitivity presets
    for s in (3, 4, 5):
        cfgs.append(make_event(sensitivity=s))
        cfgs.append(make_event(sensitivity=s, use_post_state_confirm=False))

    # finer grid around more sensitive settings
    for drop_ratio, min_move, accel, confirm, age in product(
        [0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
        [2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
        [2, 3, 4, 5],
        [5, 6, 8],
    ):
        cfgs.append(
            make_event(
                drop_ratio=drop_ratio,
                min_move_speed=min_move,
                accel_threshold=accel,
                confirm_frames=confirm,
                min_track_age=age,
            )
        )
    # de-dupe by key fields
    uniq: dict[tuple, EventConfig] = {}
    for e in cfgs:
        key = (
            round(e.drop_ratio, 3),
            round(e.min_move_speed, 3),
            round(e.accel_threshold, 3),
            e.confirm_frames,
            e.min_track_age,
            e.use_post_state_confirm,
            e.use_accel_spike,
            round(e.absolute_tol, 3),
        )
        uniq[key] = e
    return list(uniq.values())


def main() -> None:
    conf = 0.25
    workers = min(2, default_workers())
    print(f"[tune] conf={conf} cache_workers={workers}", flush=True)
    cache_jobs = build_cache(conf=conf, workers=workers)
    loaded = load_all(cache_jobs)
    print(f"[tune] videos={len(loaded)} grid scoring...", flush=True)

    grid = param_grid()
    print(f"[tune] candidates={len(grid)}", flush=True)
    rows = []
    for i, event in enumerate(grid, start=1):
        row = evaluate_event(loaded, event)
        row["id"] = i
        rows.append(row)
        if i % 50 == 0 or i == len(grid):
            print(f"[tune] scored {i}/{len(grid)}", flush=True)

    far0 = [r for r in rows if r["metrics"]["fp"] == 0]
    far0.sort(
        key=lambda r: (
            r["metrics"]["accuracy"],
            r["metrics"]["recall"],
            r["metrics"]["f1"],
        ),
        reverse=True,
    )

    best = far0[0] if far0 else None
    target = [r for r in far0 if r["metrics"]["accuracy"] >= 0.70]
    payload = {
        "conf": conf,
        "total_candidates": len(rows),
        "far0_count": len(far0),
        "target70_far0_count": len(target),
        "best_far0": best,
        "top10_far0": far0[:10],
        "baseline_sens3": next(
            (
                r
                for r in rows
                if abs(r["event"]["drop_ratio"] - 0.5) < 1e-6
                and abs(r["event"]["min_move_speed"] - 5.0) < 1e-6
                and abs(r["event"]["accel_threshold"] - 8.0) < 1e-6
                and r["event"]["confirm_frames"] == 5
                and r["event"]["use_post_state_confirm"]
            ),
            None,
        ),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n===== BEST FAR=0 =====", flush=True)
    if best is None:
        print("no FAR=0 config found", flush=True)
    else:
        m = best["metrics"]
        e = best["event"]
        print(
            f"Acc={m['accuracy']:.4f} Recall={m['recall']:.4f} "
            f"FP={m['fp']} FN={m['fn']} TP={m['tp']} TN={m['tn']}",
            flush=True,
        )
        print(
            "params:",
            {
                "drop_ratio": e["drop_ratio"],
                "min_move_speed": e["min_move_speed"],
                "accel_threshold": e["accel_threshold"],
                "confirm_frames": e["confirm_frames"],
                "min_track_age": e["min_track_age"],
                "use_post_state_confirm": e["use_post_state_confirm"],
            },
            flush=True,
        )
        print("FN:", best["fn"], flush=True)
    print(f"[save] {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()

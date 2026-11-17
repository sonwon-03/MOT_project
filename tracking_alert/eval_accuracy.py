"""
사고 영상 / 정상 영상으로 사고 의심 알림의 영상 단위 정확도를 평가한다.

폴더 구조:
  eval_data/
    accident/   # 사고가 있는 영상
    normal/     # 사고가 없는 영상

실행 예:
  python eval_accuracy.py --data eval_data --sensitivity 3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import cv2

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import AppConfig, EventConfig
from src.event_detector import AccidentSuspectDetector, SuspectEvent
from src.motion import MotionEstimator, bbox_scale
from src.occlusion import OcclusionPredictor
from src.tracker import YoloByteTracker
from src.visualize import draw_overlay

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def default_workers(requested: Optional[int] = None) -> int:
    cpu = os.cpu_count() or 4
    auto = max(2, min(4, cpu))
    if requested is None or requested <= 0:
        return auto
    return max(1, min(int(requested), 8))


@dataclass
class VideoResult:
    path: str
    label: str  # accident | normal
    predicted_accident: bool
    alert_count: int
    first_alert_frame: Optional[int]
    frames: int
    correct: bool
    alerts: list[dict]


def list_videos(folder: Path, recursive: bool = False) -> List[Path]:
    if not folder.exists():
        return []
    if recursive:
        files = [
            p
            for p in sorted(folder.rglob("*"))
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        ]
    else:
        files = [
            p
            for p in sorted(folder.iterdir())
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        ]
    return files


def evaluate_video(
    video_path: Path,
    label: str,
    cfg: AppConfig,
    max_frames: Optional[int] = None,
    on_frame: Optional[Callable[[dict], None]] = None,
    stop_event=None,
    visual_stride: int = 1,
    visual_max_width: int = 720,
) -> VideoResult:
    tracker = YoloByteTracker(
        model_name=cfg.model_name,
        tracker_cfg=cfg.tracker_cfg,
        conf=cfg.conf,
        iou=cfg.iou,
        classes=cfg.classes,
        device=cfg.device,
        small_object_boost=cfg.small_object_boost,
        imgsz=cfg.imgsz,
    )
    motion = MotionEstimator(ema_alpha=cfg.event.ema_alpha)
    detector = AccidentSuspectDetector(cfg.event)
    want_visual = on_frame is not None
    occlusion = (
        OcclusionPredictor(
            hold_seconds=cfg.occlusion_hold_sec,
            fps_hint=30.0,
        )
        if want_visual
        else None
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

    total_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if occlusion is not None and fps > 1:
        occlusion.fps_hint = fps

    alerts: List[SuspectEvent] = []
    recent = deque(maxlen=8)
    frame_idx = 0
    stride = max(1, int(visual_stride))
    toggles = {
        "use_accel_spike": cfg.event.use_accel_spike,
        "use_post_state_confirm": cfg.event.use_post_state_confirm,
        "use_proximity_tag": cfg.event.use_proximity_tag,
        "occlusion_hold_sec": cfg.occlusion_hold_sec,
    }

    while True:
        if stop_event is not None and stop_event.is_set():
            break
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break
        tracked = tracker.track_frame(frame)
        dets = [
            (o.track_id, *o.center, bbox_scale(o.bbox_xyxy)) for o in tracked
        ]
        states = motion.update(frame_idx, dets)
        bboxes = {o.track_id: tuple(map(float, o.bbox_xyxy)) for o in tracked}
        fired = detector.update(
            frame_idx,
            states,
            bboxes=bboxes,
            frame_shape=frame.shape[:2],
        )
        if fired:
            alerts.extend(fired)
            recent.extend(fired)

        if want_visual and (
            fired or frame_idx % stride == 0 or frame_idx + 1 == total_hint
        ):
            predicted_objs = (
                occlusion.update(frame_idx, tracked, states) if occlusion else []
            )
            vis = draw_overlay(
                frame,
                tracked,
                states,
                fired,
                list(recent),
                toggles,
                predicted=predicted_objs,
            )
            is_accident = label == "accident"
            truth_text = "ACCIDENT" if is_accident else "NORMAL"
            truth_bg = (0, 0, 200) if is_accident else (0, 140, 60)
            badge_w = 170
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 34), (20, 20, 20), -1)
            cv2.rectangle(vis, (0, 0), (badge_w, 34), truth_bg, -1)
            cv2.putText(
                vis,
                truth_text,
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            banner = (
                f"{video_path.name}  frame {frame_idx}"
                + (f"/{total_hint}" if total_hint > 0 else "")
                + f"  alerts={len(alerts)}"
            )
            cv2.putText(
                vis,
                banner,
                (badge_w + 12, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            max_w = max(160, int(visual_max_width))
            if vis.shape[1] > max_w:
                scale = max_w / vis.shape[1]
                vis = cv2.resize(
                    vis,
                    (max_w, max(1, int(vis.shape[0] * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            on_frame(
                {
                    "vis": vis,
                    "frame_idx": frame_idx,
                    "total_hint": total_hint,
                    "alert_count": len(alerts),
                    "label": label,
                    "name": video_path.name,
                    "tracked": len(tracked),
                }
            )

        frame_idx += 1

    cap.release()
    predicted = len(alerts) > 0
    correct = (label == "accident" and predicted) or (
        label == "normal" and not predicted
    )
    return VideoResult(
        path=str(video_path),
        label=label,
        predicted_accident=predicted,
        alert_count=len(alerts),
        first_alert_frame=alerts[0].frame_idx if alerts else None,
        frames=frame_idx,
        correct=correct,
        alerts=[
            {
                "frame_idx": a.frame_idx,
                "track_id": a.track_id,
                "tag": a.tag,
                "reason": a.reason,
            }
            for a in alerts
        ],
    )


def evaluate_jobs_parallel(
    jobs: Sequence[Tuple[str, Path]],
    cfg: AppConfig,
    workers: int = 0,
    max_frames: Optional[int] = None,
    on_frame: Optional[Callable[[dict], None]] = None,
    on_job_start: Optional[Callable[[dict], None]] = None,
    on_job_done: Optional[Callable[[VideoResult, int, int], None]] = None,
    stop_event=None,
    visual_stride: int = 2,
    visual_max_width: int = 720,
) -> List[VideoResult]:
    """여러 영상을 워커 스레드로 동시에 평가한다."""
    n_workers = default_workers(workers)
    total = len(jobs)
    if total == 0:
        return []

    results_by_index: dict[int, VideoResult] = {}

    def run_one(job_index: int, label: str, path: Path) -> Tuple[int, VideoResult]:
        if stop_event is not None and stop_event.is_set():
            return job_index, VideoResult(
                path=str(path),
                label=label,
                predicted_accident=False,
                alert_count=0,
                first_alert_frame=None,
                frames=0,
                correct=(label == "normal"),
                alerts=[],
            )
        if on_job_start is not None:
            on_job_start(
                {
                    "index": job_index,
                    "total": total,
                    "label": label,
                    "name": path.name,
                }
            )

        def frame_cb(info: dict) -> None:
            if on_frame is None:
                return
            payload = dict(info)
            payload["job_index"] = job_index
            payload["job_total"] = total
            on_frame(payload)

        local_cfg = deepcopy(cfg)
        result = evaluate_video(
            path,
            label,
            local_cfg,
            max_frames=max_frames,
            on_frame=frame_cb if on_frame is not None else None,
            stop_event=stop_event,
            visual_stride=visual_stride,
            visual_max_width=visual_max_width,
        )
        return job_index, result

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(run_one, i, label, path): i
            for i, (label, path) in enumerate(jobs, start=1)
        }
        for fut in as_completed(futures):
            job_index, result = fut.result()
            if result.frames == 0 and stop_event is not None and stop_event.is_set():
                continue
            results_by_index[job_index] = result
            if on_job_done is not None:
                on_job_done(result, job_index, total)

    return [results_by_index[i] for i in sorted(results_by_index)]


def summarize(results: Iterable[VideoResult]) -> dict:
    rows = list(results)
    tp = sum(1 for r in rows if r.label == "accident" and r.predicted_accident)
    fn = sum(1 for r in rows if r.label == "accident" and not r.predicted_accident)
    fp = sum(1 for r in rows if r.label == "normal" and r.predicted_accident)
    tn = sum(1 for r in rows if r.label == "normal" and not r.predicted_accident)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    accuracy = (tp + tn) / len(rows) if rows else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0  # false alarm rate on normal

    return {
        "count": len(rows),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "false_alarm_rate": round(far, 4),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="사고/정상 영상 정확도 평가")
    p.add_argument(
        "--data",
        default="eval_data",
        help="accident/ normal/ 하위 폴더가 있는 데이터 루트",
    )
    p.add_argument("--sensitivity", type=int, default=3, help="사고 판정 민감도 1~5")
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--workers", type=int, default=0, help="병렬 워커 수 (0=자동)")
    p.add_argument("--output", default="outputs/eval_accuracy")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data)
    accident_dir = data_root / "accident"
    normal_dir = data_root / "normal"
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    accident_videos = list_videos(accident_dir)
    normal_videos = list_videos(normal_dir)
    if not accident_videos and not normal_videos:
        print(
            f"[안내] {data_root} 아래에 accident/ 와 normal/ 폴더를 만들고 "
            "mp4를 넣은 뒤 다시 실행하세요."
        )
        print("예:")
        print(f"  {accident_dir}\\crash1.mp4")
        print(f"  {normal_dir}\\traffic1.mp4")
        return

    cfg = AppConfig(
        model_name=args.model,
        conf=args.conf,
        output_dir=str(out_dir),
        event=EventConfig(),
    )
    cfg.event.apply_sensitivity(args.sensitivity)

    jobs = [("accident", p) for p in accident_videos] + [
        ("normal", p) for p in normal_videos
    ]
    workers = default_workers(args.workers)
    print(
        f"[eval] accident={len(accident_videos)} normal={len(normal_videos)} "
        f"sensitivity={args.sensitivity} workers={workers}"
    )

    def on_job_done(result: VideoResult, job_index: int, total: int) -> None:
        mark = "O" if result.correct else "X"
        print(
            f"[{job_index}/{total}] {result.label}: {Path(result.path).name} "
            f"-> pred={'accident' if result.predicted_accident else 'normal'} "
            f"alerts={result.alert_count} first={result.first_alert_frame} [{mark}]"
        )

    results = evaluate_jobs_parallel(
        jobs,
        cfg,
        workers=workers,
        max_frames=args.max_frames,
        on_job_done=on_job_done,
        visual_stride=1,
    )

    metrics = summarize(results)
    print("\n===== 결과 =====")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    json_path = out_dir / "summary.json"
    csv_path = out_dir / "per_video.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "videos": [asdict(r) for r in results],
                "settings": {
                    "sensitivity": args.sensitivity,
                    "conf": args.conf,
                    "model": args.model,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "label",
                "predicted_accident",
                "alert_count",
                "first_alert_frame",
                "frames",
                "correct",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "path": r.path,
                    "label": r.label,
                    "predicted_accident": r.predicted_accident,
                    "alert_count": r.alert_count,
                    "first_alert_frame": r.first_alert_frame,
                    "frames": r.frames,
                    "correct": r.correct,
                }
            )
    print(f"[save] {json_path}")
    print(f"[save] {csv_path}")


if __name__ == "__main__":
    main()

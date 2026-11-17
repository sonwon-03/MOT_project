from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import winsound

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import AppConfig, EventConfig
from src.event_detector import AccidentSuspectDetector
from src.event_logger import EventLogger
from src.motion import MotionEstimator
from src.occlusion import OcclusionPredictor
from src.synthetic import generate_collision_scene
from src.visualize import draw_overlay


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="추적 기반 사고 의심 이벤트 신고 프로토타입"
    )
    p.add_argument(
        "--source",
        default="synthetic",
        help="synthetic | webcam | screen | 영상경로",
    )
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--device", default=None)
    p.add_argument("--output", default="outputs")
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--save-video", action="store_true")
    p.add_argument(
        "--monitor",
        type=int,
        default=1,
        help="screen 모드 모니터 번호 (1=주 모니터)",
    )
    p.add_argument(
        "--region",
        default=None,
        help="screen 캡처 영역 left,top,width,height (유튜브 창만 지정 권장)",
    )
    p.add_argument(
        "--max-width",
        type=int,
        default=1280,
        help="screen 입력 리사이즈 최대 가로",
    )
    p.add_argument(
        "--window",
        action="store_true",
        help="screen 모드에서 별도 결과 창을 연다. 기본은 화면 위 투명 오버레이",
    )
    p.add_argument(
        "--ui",
        action="store_true",
        help="영역 선택 후 시작 버튼을 눌러 추적하는 제어 창을 연다",
    )
    return p.parse_args()


def parse_region(text: Optional[str]) -> Optional[tuple[int, int, int, int]]:
    if not text:
        return None
    parts = [int(x.strip()) for x in text.split(",")]
    if len(parts) != 4:
        raise ValueError("--region 형식: left,top,width,height")
    return parts[0], parts[1], parts[2], parts[3]


def build_config(args: argparse.Namespace) -> AppConfig:
    return AppConfig(
        model_name=args.model,
        conf=args.conf,
        device=args.device,
        output_dir=args.output,
        event=EventConfig(),
    )


def play_alert() -> None:
    """사고 의심 확정 시 시스템 알림음을 비동기로 재생한다."""
    winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS | winsound.SND_ASYNC)


def toggle(cfg: EventConfig, name: str) -> None:
    setattr(cfg, name, not getattr(cfg, name))
    print(f"[toggle] {name} -> {getattr(cfg, name)}")


def handle_keys(key: int, cfg: AppConfig, frame_idx: int, vis: np.ndarray) -> bool:
    """q면 True(종료)."""
    if key == ord("q"):
        return True
    if key == ord("1"):
        toggle(cfg.event, "use_accel_spike")
    elif key == ord("2"):
        toggle(cfg.event, "use_post_state_confirm")
    elif key == ord("3"):
        toggle(cfg.event, "use_proximity_tag")
    elif key == ord("s"):
        snap = Path(cfg.output_dir) / f"frame_{frame_idx:05d}.jpg"
        cv2.imwrite(str(snap), vis)
        print(f"[save] {snap}")
    return False


def run_tracked_stream(
    cfg: AppConfig,
    frame_iter: Callable[[], Optional[np.ndarray]],
    show: bool,
    save_video: bool,
    out_name: str,
    wait_ms: int = 1,
    fps_hint: float = 0.0,
    overlay=None,
    stop_event=None,
    on_alert: Callable | None = None,
    on_visual: Callable | None = None,
) -> None:
    """프레임 수/FPS 상한 없이 들어오는 프레임을 전부 처리한다.

    fps_hint <= 0 이면 실제 처리 속도를 측정해 녹화·가림 타이밍에 쓴다.
    """
    from src.tracker import YoloByteTracker

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
    # fps_hint>0: 영상 메타 FPS / <=0: 측정 전까지 임시값
    known_fps = float(fps_hint) if float(fps_hint) > 1.0 else None
    occlusion = OcclusionPredictor(
        hold_seconds=cfg.occlusion_hold_sec,
        fps_hint=known_fps if known_fps is not None else 30.0,
    )
    logger = EventLogger(cfg.output_dir)
    recent = deque(maxlen=8)

    writer = None
    out_path = Path(cfg.output_dir) / out_name
    frame_idx = 0
    pending_vis: list[np.ndarray] = []
    t0 = time.perf_counter()
    write_fps = known_fps

    def _ensure_writer(sample: np.ndarray) -> None:
        nonlocal writer, write_fps
        if writer is not None:
            return
        if write_fps is None:
            elapsed = time.perf_counter() - t0
            # 하한만 두고 상한(30 등)은 두지 않음
            write_fps = max(1.0, (frame_idx + 1) / max(elapsed, 1e-3))
            occlusion.fps_hint = write_fps
        h, w = sample.shape[:2]
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(write_fps),
            (w, h),
        )
        for buffered in pending_vis:
            writer.write(buffered)
        pending_vis.clear()

    while True:
        if stop_event is not None and stop_event.is_set():
            break
        frame = frame_iter()
        if frame is None:
            break

        tracked = tracker.track_frame(frame)
        dets = [(o.track_id, *o.center) for o in tracked]
        states = motion.update(frame_idx, dets)
        bboxes = {o.track_id: tuple(map(float, o.bbox_xyxy)) for o in tracked}
        # 사고 판정은 실제 검출만 사용. 가림 예측은 표시 전용.
        events = detector.update(
            frame_idx,
            states,
            bboxes=bboxes,
            frame_shape=frame.shape[:2],
        )
        predicted = occlusion.update(frame_idx, tracked, states)
        if events:
            logger.log(events)
            recent.extend(events)
            if cfg.sound_alert:
                play_alert()
            for e in events:
                print(
                    f"[ALERT] frame={e.frame_idx} id={e.track_id} "
                    f"tag={e.tag} reason={e.reason}"
                )
                if on_alert is not None:
                    on_alert(e)

        toggles = {
            "use_accel_spike": cfg.event.use_accel_spike,
            "use_post_state_confirm": cfg.event.use_post_state_confirm,
            "use_proximity_tag": cfg.event.use_proximity_tag,
            "occlusion_hold_sec": cfg.occlusion_hold_sec,
        }
        vis = draw_overlay(
            frame, tracked, states, events, list(recent), toggles, predicted=predicted
        )

        if save_video:
            if writer is None and write_fps is None and frame_idx < 12:
                pending_vis.append(vis)
            else:
                _ensure_writer(vis)
                if writer is not None:
                    writer.write(vis)

        if on_visual is not None:
            fh, fw = frame.shape[:2]
            on_visual(
                tracked, states, events, list(recent), toggles, fw, fh, vis, predicted
            )

        if overlay is not None:
            fh, fw = frame.shape[:2]
            overlay.present(
                tracked,
                states,
                events,
                list(recent),
                toggles,
                scale_x=overlay.width / max(fw, 1),
                scale_y=overlay.height / max(fh, 1),
                predicted=predicted,
            )
            quit_loop = False
            for key_name in overlay.poll_keys():
                if key_name == "q":
                    quit_loop = True
                    break
                handle_keys(ord(key_name), cfg, frame_idx, vis)
            if quit_loop:
                break
        elif show:
            cv2.imshow("tracking_alert", vis)
            key = cv2.waitKey(wait_ms) & 0xFF
            if handle_keys(key, cfg, frame_idx, vis):
                break

        frame_idx += 1

    if save_video and writer is None and pending_vis:
        _ensure_writer(pending_vis[-1])
    if writer is not None:
        writer.release()
        print(f"[save] {out_path}  fps={write_fps}")
    cv2.destroyAllWindows()
    print(f"[done] events={len(detector.events)} log={logger.csv_path}")


def run_synthetic(cfg: AppConfig, show: bool, save_video: bool) -> None:
    motion = MotionEstimator(ema_alpha=cfg.event.ema_alpha)
    detector = AccidentSuspectDetector(cfg.event)
    logger = EventLogger(cfg.output_dir)
    recent = deque(maxlen=8)

    writer = None
    out_path = Path(cfg.output_dir) / "synthetic_demo.mp4"

    for frame_idx, (frame, tracked) in enumerate(generate_collision_scene()):
        dets = [(o.track_id, *o.center) for o in tracked]
        states = motion.update(frame_idx, dets)
        bboxes = {o.track_id: tuple(map(float, o.bbox_xyxy)) for o in tracked}
        events = detector.update(
            frame_idx,
            states,
            bboxes=bboxes,
            frame_shape=frame.shape[:2],
        )
        if events:
            logger.log(events)
            recent.extend(events)
            if cfg.sound_alert:
                play_alert()
            for e in events:
                print(
                    f"[ALERT] frame={e.frame_idx} id={e.track_id} "
                    f"tag={e.tag} reason={e.reason}"
                )

        toggles = {
            "use_accel_spike": cfg.event.use_accel_spike,
            "use_post_state_confirm": cfg.event.use_post_state_confirm,
            "use_proximity_tag": cfg.event.use_proximity_tag,
        }
        vis = draw_overlay(frame, tracked, states, events, list(recent), toggles)

        if save_video:
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(
                    str(out_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    30.0,  # synthetic 장면 생성 FPS와 동일
                    (w, h),
                )
            writer.write(vis)

        if not show:
            continue

        cv2.imshow("tracking_alert", vis)
        key = cv2.waitKey(1) & 0xFF
        if handle_keys(key, cfg, frame_idx, vis):
            break

    if writer is not None:
        writer.release()
        print(f"[save] {out_path}")
    cv2.destroyAllWindows()
    print(f"[done] events={len(detector.events)} log={logger.csv_path}")


def run_video(
    cfg: AppConfig,
    source: str,
    show: bool,
    save_video: bool,
    stop_event=None,
    on_alert: Callable | None = None,
    on_visual: Callable | None = None,
    out_name: str = "tracked_alert.mp4",
) -> Path | None:
    cap_source: str | int = 0 if source == "webcam" else source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {source}")

    # 원본 FPS 그대로 사용 (상한 없음). 메타가 없으면 처리 속도 측정.
    meta_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fps_hint = meta_fps if meta_fps > 1.0 else 0.0
    out_path = Path(cfg.output_dir) / out_name if save_video else None

    def frame_iter() -> Optional[np.ndarray]:
        ok, frame = cap.read()
        return frame if ok else None

    try:
        run_tracked_stream(
            cfg,
            frame_iter=frame_iter,
            show=show,
            save_video=save_video,
            out_name=out_name,
            wait_ms=1,
            fps_hint=float(fps_hint),
            stop_event=stop_event,
            on_alert=on_alert,
            on_visual=on_visual,
        )
    finally:
        cap.release()
    if out_path is not None and out_path.exists():
        return out_path
    return None


def run_screen(
    cfg: AppConfig,
    monitor: int,
    region: Optional[tuple[int, int, int, int]],
    max_width: int,
    show: bool,
    save_video: bool,
    use_overlay: bool,
    stop_event=None,
    on_alert: Callable | None = None,
    on_visual: Callable | None = None,
    before_grab: Callable | None = None,
    after_grab: Callable | None = None,
) -> None:
    from src.overlay_win import ScreenOverlay, enable_dpi_awareness
    from src.screen_capture import ScreenCapture

    enable_dpi_awareness()
    cap = ScreenCapture(monitor=monitor, region=region, max_width=max_width)
    overlay = None
    print(
        f"[screen] region=({cap.region.left},{cap.region.top},"
        f"{cap.region.width}x{cap.region.height}) max_width={max_width}"
    )
    if use_overlay and show:
        overlay = ScreenOverlay(
            cap.region.left, cap.region.top, cap.region.width, cap.region.height
        )
        print("[overlay] 유튜브 화면 위에 박스만 표시합니다. 클릭은 그대로 통과합니다.")
        print("[overlay] F1/F2/F3 조건 토글 | F12 저장 | F9 종료")
    elif show:
        print("[window] 별도 결과 창을 사용합니다.")

    def frame_iter() -> Optional[np.ndarray]:
        if before_grab is not None:
            before_grab()
        try:
            ok, frame = cap.read()
        finally:
            if after_grab is not None:
                after_grab()
        return frame if ok else None

    try:
        run_tracked_stream(
            cfg,
            frame_iter=frame_iter,
            show=show and overlay is None,
            save_video=save_video,
            out_name="screen_alert.mp4",
            wait_ms=1,
            fps_hint=0.0,  # 고정 20fps 한도 없음 → 실제 처리 속도 측정
            overlay=overlay,
            stop_event=stop_event,
            on_alert=on_alert,
            on_visual=on_visual,
        )
    finally:
        if overlay is not None:
            overlay.close()
        cap.close()


def main() -> None:
    args = parse_args()
    if args.ui:
        from ui_app import main as launch_ui

        launch_ui()
        return
    cfg = build_config(args)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    if args.source == "synthetic":
        run_synthetic(cfg, show=not args.no_show, save_video=args.save_video)
    elif args.source == "screen":
        run_screen(
            cfg,
            monitor=args.monitor,
            region=parse_region(args.region),
            max_width=args.max_width,
            show=not args.no_show,
            save_video=args.save_video,
            use_overlay=not args.window,
        )
    else:
        run_video(
            cfg,
            source=args.source,
            show=not args.no_show,
            save_video=args.save_video,
        )


if __name__ == "__main__":
    main()

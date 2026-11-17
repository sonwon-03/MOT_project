from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional

import cv2
import numpy as np

_CONFIGS = Path(__file__).resolve().parents[1] / "configs"
# 병렬 평가에서 GPU 추론이 동시에 겹치면 화면이 멈추는 경우가 있어 직렬화
_INFER_LOCK = Lock()


@dataclass
class TrackedObject:
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    conf: float
    cls_id: int
    label: str

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _tracker_yaml_for_conf(conf: float) -> str:
    """YOLO conf보다 높은 new_track_thresh 때문에 약한 검출이 버려지지 않게 맞춤."""
    conf = float(max(0.001, min(0.9, conf)))
    high = conf
    low = max(0.01, conf * 0.5)
    new = conf
    _CONFIGS.mkdir(parents=True, exist_ok=True)
    path = _CONFIGS / f"bytetrack_conf{conf:.2f}.yaml"
    path.write_text(
        "\n".join(
            [
                "tracker_type: bytetrack",
                f"track_high_thresh: {high:.3f}",
                f"track_low_thresh: {low:.3f}",
                f"new_track_thresh: {new:.3f}",
                "track_buffer: 60",
                "match_thresh: 0.7",
                "fuse_score: True",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return str(path)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _round_imgsz(value: int) -> int:
    return int(max(640, min(1280, (int(value) + 31) // 32 * 32)))


class YoloByteTracker:
    """Ultralytics YOLO + ByteTrack 래퍼."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        tracker_cfg: str = "bytetrack.yaml",
        conf: float = 0.1,
        iou: float = 0.5,
        classes: Optional[tuple[int, ...]] = (0, 2, 3, 5, 7),
        device: Optional[str] = None,
        small_object_boost: bool = True,
        imgsz: int = 1280,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        if tracker_cfg in ("bytetrack.yaml", "botsort.yaml") or not Path(tracker_cfg).is_file():
            self.tracker_cfg = _tracker_yaml_for_conf(conf)
        else:
            self.tracker_cfg = tracker_cfg
        self.conf = conf
        self.iou = iou
        self.classes = list(classes) if classes is not None else None
        self.device = device
        self.small_object_boost = bool(small_object_boost)
        self.imgsz = _round_imgsz(imgsz)
        self.names = self.model.names
        self._fallback: dict[int, tuple[float, float, float, float]] = {}
        self._next_fb_id = 10000

    def _prepare_frame(self, frame: np.ndarray) -> tuple[np.ndarray, float, int]:
        """저해상도/소형 객체를 위해 확대 후 추론 해상도를 고른다. 박스는 이후 원본 좌표로 환산."""
        h, w = frame.shape[:2]
        scale = 1.0
        view = frame
        if self.small_object_boost:
            # 짧은 변이 720 미만이면 키워서 작은 차량 픽셀을 키운다
            short = min(h, w)
            target_short = 720
            if short < target_short:
                scale = target_short / float(short)
                view = cv2.resize(
                    frame,
                    (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                    interpolation=cv2.INTER_CUBIC,
                )
            vh, vw = view.shape[:2]
            imgsz = _round_imgsz(max(self.imgsz, max(vh, vw), 1280))
        else:
            imgsz = _round_imgsz(max(960, min(h, w)))
        return view, scale, imgsz

    def track_frame(self, frame: np.ndarray) -> list[TrackedObject]:
        view, scale, imgsz = self._prepare_frame(frame)
        with _INFER_LOCK:
            results = self.model.track(
                source=view,
                persist=True,
                tracker=self.tracker_cfg,
                conf=self.conf,
                iou=self.iou,
                classes=self.classes,
                device=self.device,
                imgsz=imgsz,
                verbose=False,
            )
        if not results:
            self._fallback.clear()
            return []

        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            self._fallback.clear()
            return []

        boxes = r0.boxes.xyxy.cpu().numpy()
        if scale != 1.0:
            boxes = boxes / scale
        confs = r0.boxes.conf.cpu().numpy()
        clss = r0.boxes.cls.cpu().numpy().astype(int)

        if r0.boxes.id is not None:
            ids = r0.boxes.id.cpu().numpy().astype(int)
            tracked: list[TrackedObject] = []
            self._fallback.clear()
            for bbox, tid, conf, cls_id in zip(boxes, ids, confs, clss):
                bb = tuple(map(float, bbox.tolist()))
                self._fallback[int(tid)] = bb
                tracked.append(
                    TrackedObject(
                        track_id=int(tid),
                        bbox_xyxy=bb,
                        conf=float(conf),
                        cls_id=int(cls_id),
                        label=str(self.names.get(int(cls_id), cls_id)),
                    )
                )
            if tracked:
                self._next_fb_id = max(self._next_fb_id, max(t.track_id for t in tracked) + 1)
            return tracked

        # ByteTrack이 ID를 못 붙인 프레임: IoU로 임시 연결해 급정지를 놓치지 않음
        return self._match_fallback(boxes, confs, clss)

    def _match_fallback(
        self,
        boxes: np.ndarray,
        confs: np.ndarray,
        clss: np.ndarray,
    ) -> list[TrackedObject]:
        used_prev: set[int] = set()
        tracked: list[TrackedObject] = []
        new_fallback: dict[int, tuple[float, float, float, float]] = {}

        dets = [
            (tuple(map(float, b.tolist())), float(c), int(cls))
            for b, c, cls in zip(boxes, confs, clss)
        ]
        # 높은 점수 검출부터 매칭
        dets.sort(key=lambda x: x[1], reverse=True)

        for bb, conf, cls_id in dets:
            best_id = None
            best_iou = 0.3
            for tid, prev in self._fallback.items():
                if tid in used_prev:
                    continue
                ov = _iou(bb, prev)
                if ov > best_iou:
                    best_iou = ov
                    best_id = tid
            if best_id is None:
                best_id = self._next_fb_id
                self._next_fb_id += 1
            used_prev.add(best_id)
            new_fallback[best_id] = bb
            tracked.append(
                TrackedObject(
                    track_id=best_id,
                    bbox_xyxy=bb,
                    conf=conf,
                    cls_id=cls_id,
                    label=str(self.names.get(cls_id, cls_id)),
                )
            )

        self._fallback = new_fallback
        return tracked

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_CONFIGS = Path(__file__).resolve().parents[1] / "configs"


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
    conf = float(max(0.01, min(0.9, conf)))
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
                "track_buffer: 45",
                "match_thresh: 0.8",
                "fuse_score: True",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return str(path)


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
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        # 기본 ultralytics yaml(new_track=0.25)은 낮은 conf와 안 맞아서 conf에 맞춰 생성
        if tracker_cfg in ("bytetrack.yaml", "botsort.yaml") or not Path(tracker_cfg).is_file():
            self.tracker_cfg = _tracker_yaml_for_conf(conf)
        else:
            self.tracker_cfg = tracker_cfg
        self.conf = conf
        self.iou = iou
        self.classes = list(classes) if classes is not None else None
        self.device = device
        self.names = self.model.names

    def track_frame(self, frame: np.ndarray) -> list[TrackedObject]:
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_cfg,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        r0 = results[0]
        if r0.boxes is None or r0.boxes.id is None:
            return []

        boxes = r0.boxes.xyxy.cpu().numpy()
        ids = r0.boxes.id.cpu().numpy().astype(int)
        confs = r0.boxes.conf.cpu().numpy()
        clss = r0.boxes.cls.cpu().numpy().astype(int)

        tracked: list[TrackedObject] = []
        for bbox, tid, conf, cls_id in zip(boxes, ids, confs, clss):
            tracked.append(
                TrackedObject(
                    track_id=int(tid),
                    bbox_xyxy=tuple(map(float, bbox.tolist())),
                    conf=float(conf),
                    cls_id=int(cls_id),
                    label=str(self.names.get(int(cls_id), cls_id)),
                )
            )
        return tracked

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np


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


class YoloByteTracker:
    """Ultralytics YOLO + ByteTrack 래퍼."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        tracker_cfg: str = "bytetrack.yaml",
        conf: float = 0.35,
        iou: float = 0.5,
        classes: Optional[tuple[int, ...]] = (0, 2, 3, 5, 7),
        device: Optional[str] = None,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_name)
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

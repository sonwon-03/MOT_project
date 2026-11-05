from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ScreenRegion:
    left: int
    top: int
    width: int
    height: int

    def as_mss(self) -> dict:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


class ScreenCapture:
    """PC 화면을 OpenCV BGR 프레임으로 캡처한다."""

    def __init__(
        self,
        monitor: int = 1,
        region: Optional[Tuple[int, int, int, int]] = None,
        max_width: int = 1280,
    ) -> None:
        import mss

        self._mss = mss.mss()
        self.max_width = max_width

        if region is not None:
            left, top, width, height = region
            self.region = ScreenRegion(left, top, width, height)
        else:
            mon = self._mss.monitors[monitor]
            self.region = ScreenRegion(
                left=mon["left"],
                top=mon["top"],
                width=mon["width"],
                height=mon["height"],
            )

    def read(self) -> tuple[bool, np.ndarray]:
        shot = self._mss.grab(self.region.as_mss())
        frame = np.array(shot)[:, :, :3]  # BGRA -> BGR
        frame = np.ascontiguousarray(frame)
        if self.max_width and frame.shape[1] > self.max_width:
            scale = self.max_width / frame.shape[1]
            frame = cv2.resize(
                frame,
                (self.max_width, int(frame.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
        return True, frame

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self.read()
            if not ok:
                break
            yield frame

    def close(self) -> None:
        self._mss.close()

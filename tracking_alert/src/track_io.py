"""추적 결과 저장/불러오기.

저장 형식
  - <name>_tracks.csv : frame, track_id, x1, y1, x2, y2, conf, cls_id, label (프레임은 1부터, MOT 형식과 동일)
  - <name>_tracks.mot.txt : MOTChallenge 형식 frame,id,x,y,w,h,conf,-1,-1,-1 (평가 도구 호환)
  - <name>_meta.json : 원본 영상 경로, 모델 가중치, 추적기 설정, 처리 성능(FPS 등)
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

FIELDS = ["frame", "track_id", "x1", "y1", "x2", "y2", "conf", "cls_id", "label"]


@dataclass
class LoadedTrack:
    track_id: int
    bbox_xyxy: tuple
    conf: float
    cls_id: int
    label: str

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox_xyxy
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


class TrackWriter:
    """프레임마다 추적 결과를 CSV + MOT txt로 기록한다."""

    def __init__(self, out_dir: Path, name: str = "last") -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = out_dir / f"{name}_tracks.csv"
        self.mot_path = out_dir / f"{name}_tracks.mot.txt"
        self.meta_path = out_dir / f"{name}_meta.json"
        self._csv_f = self.csv_path.open("w", newline="", encoding="utf-8-sig")
        self._csv = csv.writer(self._csv_f)
        self._csv.writerow(FIELDS)
        self._mot_f = self.mot_path.open("w", encoding="utf-8")
        self.rows = 0

    def write(self, frame_idx: int, tracked: Iterable) -> None:
        f = frame_idx + 1  # MOT 형식은 1부터
        for o in tracked:
            x1, y1, x2, y2 = o.bbox_xyxy
            self._csv.writerow(
                [f, o.track_id, round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1),
                 round(o.conf, 3), o.cls_id, o.label]
            )
            self._mot_f.write(
                f"{f},{o.track_id},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},{o.conf:.3f},-1,-1,-1\n"
            )
            self.rows += 1

    def close(self, meta: Optional[dict] = None) -> None:
        self._csv_f.close()
        self._mot_f.close()
        if meta is not None:
            self.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_tracks(csv_path: Path) -> Dict[int, List[LoadedTrack]]:
    """CSV를 읽어 {프레임 인덱스(0부터): [LoadedTrack, ...]} 로 반환."""
    by_frame: Dict[int, List[LoadedTrack]] = defaultdict(list)
    with Path(csv_path).open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            by_frame[int(row["frame"]) - 1].append(
                LoadedTrack(
                    track_id=int(row["track_id"]),
                    bbox_xyxy=(float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])),
                    conf=float(row["conf"]),
                    cls_id=int(row["cls_id"]),
                    label=row.get("label", ""),
                )
            )
    return dict(by_frame)


def load_meta(csv_path: Path) -> dict:
    meta_path = Path(str(csv_path).replace("_tracks.csv", "_meta.json"))
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}

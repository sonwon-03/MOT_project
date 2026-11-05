from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from src.event_detector import SuspectEvent


class EventLogger:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "suspect_events.csv"
        self.json_path = self.output_dir / "suspect_events.json"
        self._rows: list[dict] = []
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "frame_idx",
                        "track_id",
                        "category",
                        "tag",
                        "x",
                        "y",
                        "speed",
                        "accel_mag",
                        "nearby_ids",
                        "reason",
                    ],
                )
                writer.writeheader()

    def log(self, events: Iterable[SuspectEvent]) -> None:
        rows = []
        for e in events:
            row = {
                "frame_idx": e.frame_idx,
                "track_id": e.track_id,
                "category": e.category,
                "tag": e.tag,
                "x": round(e.x, 2),
                "y": round(e.y, 2),
                "speed": round(e.speed, 3),
                "accel_mag": round(e.accel_mag, 3),
                "nearby_ids": ",".join(map(str, e.nearby_ids)),
                "reason": e.reason,
            }
            rows.append(row)
            self._rows.append(row)

        if not rows:
            return

        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writerows(rows)

        with self.json_path.open("w", encoding="utf-8") as f:
            json.dump(self._rows, f, ensure_ascii=False, indent=2)

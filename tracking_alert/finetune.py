"""YOLOv8n을 MOT16 학습 분할(사람)로 Fine-tuning 한다.

  python mot_eval.py prepare   # 먼저 학습 데이터 생성
  python finetune.py --epochs 10

결과
  weights/yolov8n_mot16.pt     : 학습된 가중치 (GUI 모델 선택 / mot_eval.py 비교에 사용)
  weights/yolov8n_mot16.json   : 학습 설정 + 학습 전/후 검증 mAP
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_YAML = ROOT / "datasets" / "mot16_yolo" / "data.yaml"
WEIGHTS_DIR = ROOT / "weights"
OUT_PT = WEIGHTS_DIR / "yolov8n_mot16.pt"


def val_map(model_path: str, coco_person: bool) -> dict:
    """검증 분할 mAP. COCO 원본 모델은 person(class 0)만 평가하도록 classes=[0]."""
    from ultralytics import YOLO

    m = YOLO(model_path)
    kw = {"classes": [0]} if coco_person else {}
    r = m.val(data=str(DATA_YAML), imgsz=640, batch=8, device="cpu", workers=0, verbose=False,
              plots=False, project=str(ROOT / "outputs" / "finetune"), name="val", exist_ok=True, **kw)
    return {"mAP50": float(r.box.map50), "mAP50-95": float(r.box.map), "precision": float(r.box.mp),
            "recall": float(r.box.mr)}


def main() -> None:
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    if not DATA_YAML.exists():
        raise SystemExit("먼저 python mot_eval.py prepare 를 실행하세요.")

    print("[finetune] 학습 전 검증 mAP (COCO 사전학습, person만)")
    before = val_map(a.base, coco_person=True)
    print(f"  {before}")

    t0 = time.time()
    model = YOLO(a.base)
    model.train(
        data=str(DATA_YAML),
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        device="cpu",
        workers=2,
        lr0=0.002,          # 사전학습 특징 유지를 위해 낮은 학습률
        warmup_epochs=1,
        freeze=10,          # 백본(앞 10개 층) 고정: 데이터가 적고 CPU라 헤드 위주로 조정
        mosaic=0.5,
        close_mosaic=2,
        patience=5,
        plots=True,
        project=str(ROOT / "outputs" / "finetune"),
        name="train",
        exist_ok=True,
        verbose=False,
    )
    best = Path(model.trainer.best)
    WEIGHTS_DIR.mkdir(exist_ok=True)
    shutil.copyfile(best, OUT_PT)
    minutes = (time.time() - t0) / 60

    print("[finetune] 학습 후 검증 mAP")
    after = val_map(str(OUT_PT), coco_person=False)
    print(f"  {after}")
    meta = {
        "base": a.base, "epochs": a.epochs, "imgsz": a.imgsz, "batch": a.batch, "lr0": 0.002,
        "freeze": 10, "device": "cpu", "train_minutes": round(minutes, 1),
        "data": "MOT16 train(02,04,13) / val(10,11), 8프레임 간격 샘플, person 1클래스",
        "val_before": before, "val_after": after, "weights": str(OUT_PT.relative_to(ROOT)),
        "created": time.strftime("%Y-%m-%d %H:%M"),
    }
    OUT_PT.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[finetune] 저장: {OUT_PT}  ({minutes:.1f}분)")


if __name__ == "__main__":
    main()

"""MOT16 공개 데이터셋으로 추적 성능을 평가·튜닝한다.

사용법
  python mot_eval.py prepare            # 압축 해제 + 학습/검증/테스트 분할 + YOLO 학습 데이터 변환
  python mot_eval.py tune               # 검증(val) 세트에서 추적 파라미터 격자 탐색 → configs/tuned_tracker.json
  python mot_eval.py eval               # 테스트(test) 세트에서 모델·추적기 비교 → outputs/mot_eval/summary.json
  python mot_eval.py all                # prepare → tune → eval

데이터 분할 (시퀀스 단위로 나눠 같은 장면이 학습·평가에 섞이지 않게 함)
  train : MOT16-02, 04, 13   → YOLO Fine-tuning 학습
  val   : MOT16-10, 11       → Fine-tuning 검증 + 추적 파라미터 튜닝
  test  : MOT16-05, 09       → 최종 성능 비교 (튜닝·학습에 쓰지 않음)

속도: CPU에서도 돌도록 시퀀스마다 YOLO 검출 결과를 한 번만 계산해 캐시하고(outputs/mot_eval/cache),
추적기 설정만 바꿔 가며 재사용한다. FPS = 1000 / (검출 ms + 추적 ms).
"""

from __future__ import annotations

import argparse
import configparser
import csv
import itertools
import json
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bytetrack import ByteTracker, ByteTrackParams
from src.mot_metrics import (
    clear_id_metrics,
    hota_finalize,
    hota_merge,
    hota_raw,
    load_mot_gt,
    remove_distractor_matches,
    summarize_accumulators,
)
from src.tracker import BASELINE_PARAMS, IMPROVED_PARAMS, TUNED_PATH, YoloByteTracker, params_for_conf

DATA_DIR = ROOT / "datasets"
ZIP_PATH = DATA_DIR / "MOT16.zip"
MOT_DIR = DATA_DIR / "MOT16" / "train"
YOLO_DIR = DATA_DIR / "mot16_yolo"
OUT_DIR = ROOT / "outputs" / "mot_eval"
CACHE_DIR = OUT_DIR / "cache"
WEIGHTS_DIR = ROOT / "weights"
FINETUNED = WEIGHTS_DIR / "yolov8n_mot16.pt"

SPLITS = {
    "train": ["MOT16-02", "MOT16-04", "MOT16-13"],
    "val": ["MOT16-10", "MOT16-11"],
    "test": ["MOT16-05", "MOT16-09"],
}
PERSON = (0,)


# ───────────────────────────── 준비 ─────────────────────────────


def seq_info(seq: str) -> dict:
    cp = configparser.ConfigParser()
    cp.read(MOT_DIR / seq / "seqinfo.ini")
    s = cp["Sequence"]
    return {
        "frames": int(s["seqLength"]),
        "fps": float(s["frameRate"]),
        "w": int(s["imWidth"]),
        "h": int(s["imHeight"]),
        "img_dir": MOT_DIR / seq / s.get("imDir", "img1"),
    }


def prepare(stride: int = 8) -> None:
    """MOT16 학습 시퀀스 압축 해제 → 분할 정보 저장 → YOLO 형식 변환."""
    if not (MOT_DIR / "MOT16-02").exists():
        if not ZIP_PATH.exists():
            raise SystemExit(f"{ZIP_PATH} 가 없습니다. https://motchallenge.net/data/MOT16.zip 을 받으세요.")
        print("[prepare] 압축 해제 (train 시퀀스만)")
        with zipfile.ZipFile(ZIP_PATH) as zf:
            members = [m for m in zf.namelist() if m.startswith("train/")]
            for m in members:
                zf.extract(m, DATA_DIR / "MOT16")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "split.json").write_text(json.dumps(SPLITS, indent=2), encoding="utf-8")

    # YOLO 학습용: train/val 시퀀스에서 stride 프레임마다 이미지 + 라벨(class 0 person)
    stats = {}
    for split in ("train", "val"):
        img_out = YOLO_DIR / "images" / split
        lbl_out = YOLO_DIR / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        n_img = n_box = 0
        for seq in SPLITS[split]:
            info = seq_info(seq)
            gt, _ = load_mot_gt(MOT_DIR / seq / "gt" / "gt.txt", min_visibility=0.25)
            for f in range(1, info["frames"] + 1, stride):
                src = info["img_dir"] / f"{f:06d}.jpg"
                dst = img_out / f"{seq}_{f:06d}.jpg"
                if not dst.exists():
                    # 1080p 원본을 그대로 쓰면 학습이 느려 긴 변 960으로 축소
                    img = cv2.imread(str(src))
                    s = 960 / max(img.shape[:2])
                    if s < 1:
                        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                lines = []
                if f in gt:
                    for x1, y1, x2, y2 in gt[f][1]:
                        x1, x2 = np.clip([x1, x2], 0, info["w"])
                        y1, y2 = np.clip([y1, y2], 0, info["h"])
                        if x2 - x1 < 2 or y2 - y1 < 2:
                            continue
                        cx, cy = (x1 + x2) / 2 / info["w"], (y1 + y2) / 2 / info["h"]
                        bw, bh = (x2 - x1) / info["w"], (y2 - y1) / info["h"]
                        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                (lbl_out / f"{seq}_{f:06d}.txt").write_text("\n".join(lines), encoding="utf-8")
                n_img += 1
                n_box += len(lines)
        stats[split] = {"images": n_img, "boxes": n_box}
    (YOLO_DIR / "data.yaml").write_text(
        f"path: {YOLO_DIR.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n",
        encoding="utf-8",
    )
    (OUT_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[prepare] 분할={SPLITS}\n[prepare] YOLO 데이터 {stats}")


# ───────────────────────────── 검출 캐시 ─────────────────────────────


def detection_cache(seq: str, model: str, boost: bool, iou: float, conf: float = 0.005) -> dict:
    """시퀀스 전체 검출을 한 번 계산해 저장. {frame: dets(N,6)}, 평균 검출 ms."""
    tag = f"{seq}_{Path(model).stem}_{'boost' if boost else '640'}_iou{iou}"
    path = CACHE_DIR / f"{tag}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return {"dets": z["dets"].item(), "det_ms": float(z["det_ms"])}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    info = seq_info(seq)
    det = YoloByteTracker(
        model_name=str(model), conf=conf, iou=iou, classes=PERSON,
        small_object_boost=boost, imgsz=1280 if boost else 640,
        tracker_params={"track_low_thresh": conf},
    )
    if not boost:
        det._prepare_frame = lambda fr: (fr, 1.0, 640)  # 기본 640 추론 (확대 없음)
    dets, times = {}, []
    for f in range(1, info["frames"] + 1):
        frame = cv2.imread(str(info["img_dir"] / f"{f:06d}.jpg"))
        dets[f] = det.detect(frame).astype(np.float32)
        times.append(det.last_infer_ms)
        if f % 100 == 0:
            print(f"  [cache] {tag} {f}/{info['frames']}  {np.mean(times):.0f}ms/frame")
    det_ms = float(np.mean(times[5:])) if len(times) > 10 else float(np.mean(times))
    np.savez_compressed(path, dets=np.array(dets, dtype=object), det_ms=det_ms)
    return {"dets": dets, "det_ms": det_ms}


# ───────────────────────────── 추적 실행 · 채점 ─────────────────────────────


def run_tracker_on_cache(seq: str, cache: dict, params: dict, need_frames: bool) -> tuple[dict, float, int]:
    """캐시된 검출로 직접 구현 ByteTrack 실행. (예측 FrameData, 평균 추적 ms, 재연결 횟수)."""
    info = seq_info(seq)
    trk = ByteTracker(ByteTrackParams.from_dict(params), frame_rate=info["fps"])
    pred, times = {}, []
    for f in range(1, info["frames"] + 1):
        frame = cv2.imread(str(info["img_dir"] / f"{f:06d}.jpg")) if need_frames else None
        dets = cache["dets"].get(f, np.zeros((0, 6)))
        if len(dets) > 120:
            dets = dets[np.argsort(-dets[:, 4])[:120]]
        t0 = time.perf_counter()
        tracks = trk.update(dets, frame)
        times.append((time.perf_counter() - t0) * 1000)
        if tracks:
            pred[f] = (
                np.array([t.track_id for t in tracks], dtype=int),
                np.array([t.box for t in tracks], dtype=float),
            )
    return pred, float(np.mean(times)), trk.relink_count


def run_ultralytics(seq: str, model: str) -> tuple[dict, float]:
    """비교 기준선: Ultralytics 내장 ByteTrack (model.track, 기본 bytetrack.yaml, conf 0.25)."""
    from ultralytics import YOLO

    info = seq_info(seq)
    m = YOLO(str(model))
    pred, times = {}, []
    for f in range(1, info["frames"] + 1):
        frame = cv2.imread(str(info["img_dir"] / f"{f:06d}.jpg"))
        t0 = time.perf_counter()
        r = m.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.25, iou=0.7,
                    classes=list(PERSON), imgsz=640, verbose=False)[0]
        times.append((time.perf_counter() - t0) * 1000)
        if r.boxes is not None and r.boxes.id is not None:
            pred[f] = (r.boxes.id.int().cpu().numpy(), r.boxes.xyxy.cpu().numpy())
    return pred, float(np.mean(times[5:]))


def score(seqs: list, preds: dict) -> dict:
    """preds: {seq: FrameData} → {'OVERALL': {...}, seq: {...}}"""
    accs, raws, names = [], [], []
    per_seq_hota = {}
    for seq in seqs:
        gt, dis = load_mot_gt(MOT_DIR / seq / "gt" / "gt.txt")
        p = remove_distractor_matches(preds[seq], gt, dis)
        accs.append(clear_id_metrics(gt, p))
        raw = hota_raw(gt, p)
        raws.append(raw)
        per_seq_hota[seq] = hota_finalize(raw)
        names.append(seq)
    res = summarize_accumulators(accs, names)
    for seq in seqs:
        res[seq].update(per_seq_hota[seq])
    res["OVERALL"].update(hota_finalize(hota_merge(raws)))
    return res


def write_mot(path: Path, pred: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for fr in sorted(pred):
            for tid, (x1, y1, x2, y2) in zip(*pred[fr]):
                f.write(f"{fr},{tid},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},1,-1,-1,-1\n")


# ───────────────────────────── 튜닝 ─────────────────────────────

TUNE_GRID = {
    "track_high_thresh": [0.3, 0.4, 0.5],
    "new_track_thresh": [0.4, 0.5, 0.6],
    "track_buffer": [30, 60],
    "match_thresh": [0.7, 0.8, 0.9],
}


def tune(model: str = "yolov8n.pt") -> dict:
    """검증 세트에서 (1) 기본 파라미터 격자 탐색 → (2) 개선 A/B 세기 탐색. HOTA 최대 조합 저장."""
    seqs = SPLITS["val"]
    caches = {s: detection_cache(s, model, boost=False, iou=0.7) for s in seqs}
    log = []

    def evaluate(params: dict, need_frames: bool) -> dict:
        preds = {s: run_tracker_on_cache(s, caches[s], params, need_frames)[0] for s in seqs}
        return score(seqs, preds)["OVERALL"]

    best, best_p = None, None
    keys = list(TUNE_GRID)
    combos = list(itertools.product(*TUNE_GRID.values()))
    print(f"[tune] 1단계 기본 파라미터 {len(combos)}개 조합 (val={seqs})")
    for i, vals in enumerate(combos, 1):
        p = dict(BASELINE_PARAMS, **dict(zip(keys, vals)))
        if p["new_track_thresh"] < p["track_high_thresh"]:
            continue
        r = evaluate(p, need_frames=False)
        log.append({"stage": 1, "params": p, **{k: r[k] for k in ("HOTA", "IDF1", "MOTA", "IDSW")}})
        if best is None or r["HOTA"] > best["HOTA"]:
            best, best_p = r, p
            print(f"  [{i}/{len(combos)}] HOTA={r['HOTA']:.4f} IDF1={r['IDF1']:.4f} MOTA={r['MOTA']:.4f} IDSW={r['IDSW']}  {dict(zip(keys, vals))}")
    stage1 = dict(best_p)
    print("[tune] 2단계 개선 A(외형 가중치) / B(재연결) 탐색")
    for w, rl, gap in itertools.product([0.0, 0.2, 0.3, 0.5], [False, True], [60, 120]):
        if not rl and gap != 60:
            continue
        p = dict(stage1, appearance_weight=w, relink=rl, relink_max_gap=gap)
        r = evaluate(p, need_frames=(w > 0 or rl))
        log.append({"stage": 2, "params": p, **{k: r[k] for k in ("HOTA", "IDF1", "MOTA", "IDSW")}})
        print(f"  w={w} relink={rl} gap={gap}: HOTA={r['HOTA']:.4f} IDF1={r['IDF1']:.4f} IDSW={r['IDSW']}")
        if r["HOTA"] > best["HOTA"]:
            best, best_p = r, p
    TUNED_PATH.parent.mkdir(parents=True, exist_ok=True)
    TUNED_PATH.write_text(
        json.dumps({"params": best_p, "val_metrics": best, "val_seqs": seqs, "model": model}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (OUT_DIR / "tune_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[tune] 최적: {best_p}\n       val HOTA={best['HOTA']:.4f} IDF1={best['IDF1']:.4f} → {TUNED_PATH}")
    return best_p


# ───────────────────────────── 최종 비교 ─────────────────────────────


def evaluate_test() -> dict:
    seqs = SPLITS["test"]
    tuned = json.loads(TUNED_PATH.read_text(encoding="utf-8"))["params"] if TUNED_PATH.exists() else IMPROVED_PARAMS
    tuned_base = {k: v for k, v in tuned.items() if k in BASELINE_PARAMS}

    # (이름, 모델, 검출 방식, 추적 파라미터 or "ultralytics")
    configs = [
        ("① Ultralytics ByteTrack (라이브러리 기본)", "yolov8n.pt", None, "ultralytics"),
        ("② 직접 구현 ByteTrack (기본값)", "yolov8n.pt", "640", BASELINE_PARAMS),
        ("③ ② + 외형 특징 융합(A)", "yolov8n.pt", "640", dict(BASELINE_PARAMS, appearance_weight=0.3)),
        ("④ ② + 유실 트랙 재연결(B)", "yolov8n.pt", "640", dict(BASELINE_PARAMS, relink=True)),
        ("⑤ ② + A + B", "yolov8n.pt", "640", dict(BASELINE_PARAMS, appearance_weight=0.3, relink=True)),
        ("⑥ 튜닝 파라미터 (A/B 제외)", "yolov8n.pt", "640", tuned_base),
        ("⑦ 튜닝 + A/B (최종 개선)", "yolov8n.pt", "640", tuned),
        ("⑧ 기존 tracking_alert 설정 (conf 0.01, 1280 확대)", "yolov8n.pt", "boost", params_for_conf(0.01)),
        ("⑨ YOLOv8s + 최종 개선", "yolov8s.pt", "640", tuned),
    ]
    if FINETUNED.exists():
        configs.append(("⑩ Fine-tuned YOLOv8n(MOT16) + 최종 개선", str(FINETUNED), "640", tuned))

    rows, results_dir = [], OUT_DIR / "results"
    for name, model, det_mode, params in configs:
        print(f"[eval] {name}")
        preds, det_ms, trk_ms, relinks = {}, [], [], 0
        for seq in seqs:
            if params == "ultralytics":
                p, ms = run_ultralytics(seq, model)
                preds[seq] = p
                det_ms.append(ms)
                trk_ms.append(0.0)
            else:
                boost = det_mode == "boost"
                cache = detection_cache(seq, model, boost=boost, iou=0.5 if boost else 0.7)
                p, ms, rl = run_tracker_on_cache(
                    seq, cache, params,
                    need_frames=params.get("appearance_weight", 0) > 0 or params.get("relink", False),
                )
                preds[seq] = p
                det_ms.append(cache["det_ms"])
                trk_ms.append(ms)
                relinks += rl
            write_mot(results_dir / name.split(" ")[0] / f"{seq}.txt", preds[seq])
        res = score(seqs, preds)
        d, t = float(np.mean(det_ms)), float(np.mean(trk_ms))
        o = res["OVERALL"]
        row = {
            "name": name, "model": Path(model).name,
            "params": params if params != "ultralytics" else "ultralytics bytetrack.yaml",
            "HOTA": o["HOTA"], "DetA": o["DetA"], "AssA": o["AssA"], "MOTA": o["MOTA"],
            "IDF1": o["IDF1"], "IDSW": o["IDSW"], "FP": o["FP"], "FN": o["FN"],
            "MT": o["MT"], "ML": o["ML"], "Frag": o["Frag"],
            "det_ms": d, "track_ms": t, "ms_per_frame": d + t, "FPS": 1000.0 / max(d + t, 1e-6),
            "relinks": relinks,
            "per_seq": {s: res[s] for s in seqs},
        }
        rows.append(row)
        print(f"   HOTA={o['HOTA']:.3f} MOTA={o['MOTA']:.3f} IDF1={o['IDF1']:.3f} IDSW={o['IDSW']} "
              f"FPS={row['FPS']:.1f} ({d:.0f}+{t:.1f}ms) relinks={relinks}")

    summary = {
        "dataset": "MOT16 (train split 공개 GT)", "split": SPLITS, "test_seqs": seqs,
        "created": time.strftime("%Y-%m-%d %H:%M"), "device": "CPU",
        "rows": rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (OUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        cols = ["name", "model", "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN", "MT", "ML",
                "Frag", "det_ms", "track_ms", "ms_per_frame", "FPS", "relinks"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in cols})
    print(f"[eval] 저장: {OUT_DIR / 'summary.json'}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prepare", "tune", "eval", "all"])
    a = ap.parse_args()
    if a.cmd in ("prepare", "all"):
        prepare()
    if a.cmd in ("tune", "all"):
        tune()
    if a.cmd in ("eval", "all"):
        evaluate_test()


if __name__ == "__main__":
    main()

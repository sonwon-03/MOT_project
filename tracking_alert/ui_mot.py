"""GUI 확장: 모델/추적기 선택, 실시간 성능·궤적 표시, 추적 결과 불러오기, MOT 성능 비교 화면.

ui_app.StudioApp 가 이 믹스인을 상속해 사용한다.
"""

from __future__ import annotations

import contextlib
import io
import json
import threading
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import cv2

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
MOT_OUT = ROOT / "outputs" / "mot_eval"

TRACKER_CHOICES = {
    "직접 구현 ByteTrack · 사고 감지용 (conf 연동)": ("custom", "conf"),
    "직접 구현 ByteTrack · 최종 개선 (튜닝 + 외형 + 재연결)": ("custom", "tuned"),
    "직접 구현 ByteTrack · 개선 (외형 + 재연결)": ("custom", "improved"),
    "직접 구현 ByteTrack · 기본값 (원 논문)": ("custom", "baseline"),
    "Ultralytics 내장 ByteTrack (비교 기준)": ("ultralytics", None),
}
DEFAULT_TRACKER = next(iter(TRACKER_CHOICES))


class _Stream(io.TextIOBase):
    """mot_eval 출력(print)을 GUI 로그로 넘긴다."""

    def __init__(self, put) -> None:
        self._put = put
        self._buf = ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._put(line)
        return len(s)


class MotUIMixin:
    # ─────────────── 캡처 화면: 모델 / 추적기 / 궤적 ───────────────
    def _model_choices(self) -> list[str]:
        items = ["yolov8n.pt", "yolov8s.pt"]
        if WEIGHTS_DIR.exists():
            items += [f"weights/{p.name}" for p in sorted(WEIGHTS_DIR.glob("*.pt"))]
        extra = getattr(self, "_custom_model", None)
        if extra and extra not in items:
            items.append(extra)
        return items

    def _build_model_tracker_section(self, parent: tk.Frame, panel: str, fg: str) -> None:
        saved = (self._settings or {}).get("cap", {})

        def label(text: str) -> None:
            tk.Label(parent, text=text, bg=panel, fg="#c5ccd6", font=("Malgun Gothic", 10)).pack(
                anchor="w", padx=14, pady=(12, 4)
            )

        label("검출 모델")
        row = tk.Frame(parent, bg=panel)
        row.pack(fill="x", padx=14)
        self.model_var = tk.StringVar(value=str(saved.get("model", "yolov8n.pt")))
        self.model_box = ttk.Combobox(row, textvariable=self.model_var, values=self._model_choices(), state="readonly")
        self.model_box.pack(side="left", fill="x", expand=True)
        self.model_box.bind("<Button-1>", lambda _e: self.model_box.configure(values=self._model_choices()))
        tk.Button(row, text="파일…", command=self._browse_model, bg="#3a4150", fg=fg, relief="flat",
                  padx=6).pack(side="left", padx=(6, 0))
        tk.Label(parent, text="weights/ 의 Fine-tuned 모델은 사람(person) 전용", bg=panel, fg="#9aa3b2",
                 font=("Malgun Gothic", 8)).pack(anchor="w", padx=14)

        label("추적 알고리즘")
        self.tracker_choice_var = tk.StringVar(value=str(saved.get("tracker", DEFAULT_TRACKER)))
        if self.tracker_choice_var.get() not in TRACKER_CHOICES:
            self.tracker_choice_var.set(DEFAULT_TRACKER)
        ttk.Combobox(parent, textvariable=self.tracker_choice_var, values=list(TRACKER_CHOICES),
                     state="readonly").pack(fill="x", padx=14)
        self.trajectory_var = tk.BooleanVar(value=bool(saved.get("draw_trajectory", True)))
        tk.Checkbutton(parent, text="이동 궤적 표시 (방향·거리)", variable=self.trajectory_var, bg=panel, fg=fg,
                       selectcolor="#101216", activebackground=panel, activeforeground=fg,
                       font=("Malgun Gothic", 10)).pack(anchor="w", padx=14, pady=(4, 0))

    def _browse_model(self) -> None:
        path = filedialog.askopenfilename(title="YOLO 가중치 선택", initialdir=str(WEIGHTS_DIR if WEIGHTS_DIR.exists() else ROOT),
                                          filetypes=[("PyTorch weights", "*.pt")])
        if path:
            self._custom_model = path
            self.model_box.configure(values=self._model_choices())
            self.model_var.set(path)

    def _apply_model_tracker(self, cfg) -> None:
        from src.tracker import BASELINE_PARAMS, IMPROVED_PARAMS, load_tuned_params

        model = self.model_var.get() if hasattr(self, "model_var") else "yolov8n.pt"
        if model.startswith("weights/"):
            model = str(ROOT / model)
        cfg.model_name = model
        backend, preset = TRACKER_CHOICES.get(self.tracker_choice_var.get(), ("custom", "conf"))
        cfg.tracker_backend = backend
        if preset == "baseline":
            cfg.tracker_params = dict(BASELINE_PARAMS)
        elif preset == "improved":
            cfg.tracker_params = dict(IMPROVED_PARAMS)
        elif preset == "tuned":
            cfg.tracker_params = load_tuned_params() or dict(IMPROVED_PARAMS)
        else:
            cfg.tracker_params = None  # conf 연동 (기존 사고 감지 방식)
        cfg.draw_trajectory = bool(self.trajectory_var.get())

    def _mot_settings(self) -> dict:
        if not hasattr(self, "model_var"):
            return {}
        return {
            "model": self.model_var.get(),
            "tracker": self.tracker_choice_var.get(),
            "draw_trajectory": bool(self.trajectory_var.get()),
        }

    def _apply_mot_settings(self, block: dict) -> None:
        if not hasattr(self, "model_var"):
            return
        self.model_var.set(str(block.get("model", "yolov8n.pt")))
        t = str(block.get("tracker", DEFAULT_TRACKER))
        self.tracker_choice_var.set(t if t in TRACKER_CHOICES else DEFAULT_TRACKER)
        self.trajectory_var.set(bool(block.get("draw_trajectory", True)))

    # ─────────────── 캡처 화면: 실시간 성능 + 궤적 목록 + 결과 불러오기 ───────────────
    def _build_stats_section(self, parent: tk.Frame, panel: str, fg: str) -> None:
        tk.Label(parent, text="실시간 처리 성능", bg=panel, fg="#c5ccd6", font=("Malgun Gothic", 10)).pack(
            anchor="w", padx=14, pady=(8, 2)
        )
        self.perf_var = tk.StringVar(value="FPS -  ·  프레임당 - ms")
        tk.Label(parent, textvariable=self.perf_var, bg=panel, fg="#80ffff", font=("Consolas", 9),
                 justify="left", anchor="w").pack(fill="x", padx=14)
        tk.Label(parent, text="이동 궤적 (ID · 방향 · 누적거리 · 평균속도)", bg=panel, fg="#c5ccd6",
                 font=("Malgun Gothic", 10)).pack(anchor="w", padx=14, pady=(8, 2))
        self.traj_list = tk.Listbox(parent, bg="#101216", fg="#f3f5f7", height=6, highlightthickness=0,
                                    activestyle="none", font=("Consolas", 9))
        self.traj_list.pack(fill="x", padx=14)
        tk.Button(parent, text="추적 결과 불러오기 (CSV → 재생)", command=self._load_track_results, bg="#3a6ea5",
                  fg="white", relief="flat", font=("Malgun Gothic", 10), pady=4).pack(fill="x", padx=14, pady=(8, 0))
        tk.Label(parent, text="추적 후 outputs/review/last_tracks.csv · last_meta.json 자동 저장", bg=panel,
                 fg="#9aa3b2", font=("Malgun Gothic", 8)).pack(anchor="w", padx=14)

    def _on_stats_message(self, s: dict) -> None:
        if hasattr(self, "perf_var"):
            self.perf_var.set(
                f"FPS {s['fps']:.1f}  ·  프레임당 {s['ms']:.0f} ms\n"
                f"검출 {s['det_ms']:.0f} ms · 추적 {s['trk_ms']:.1f} ms\n"
                f"현재 {s['objects']}개 · 누적 ID {s['unique_ids']}개 · frame {s['frame']}"
            )
        if hasattr(self, "traj_list"):
            self.traj_list.delete(0, tk.END)
            for r in s.get("trajectories", []):
                self.traj_list.insert(
                    tk.END,
                    f"ID{r['track_id']:<4} {r['direction']:<5} {r['distance_px']:>7.0f}px  v={r['avg_speed']:.1f}",
                )

    def _load_track_results(self) -> None:
        if self._busy():
            return
        path = filedialog.askopenfilename(
            title="추적 결과 CSV 선택", initialdir=str(ROOT / "outputs" / "review"),
            filetypes=[("Tracks CSV", "*_tracks.csv"), ("CSV", "*.csv")],
        )
        if not path:
            return
        from src.track_io import load_meta

        meta = load_meta(Path(path))
        src = meta.get("source", "")
        if not src or not Path(src).exists():
            src = filedialog.askopenfilename(title="원본 영상 선택 (결과를 그릴 영상)",
                                             filetypes=[("Video", "*.mp4;*.avi;*.mov;*.mkv"), ("All", "*.*")])
            if not src:
                return
        self.log.insert(0, f"결과 불러오는 중: {Path(path).name}")
        threading.Thread(target=self._render_replay, args=(Path(path), Path(src), meta), daemon=True).start()

    def _render_replay(self, csv_path: Path, video: Path, meta: dict) -> None:
        """저장된 추적 결과를 원본 영상에 다시 그려 재생용 영상을 만든다 (재추적 없음)."""
        try:
            from src.track_io import load_tracks
            from src.trajectory import TrajectoryStore, track_color

            tracks = load_tracks(csv_path)
            cap = cv2.VideoCapture(str(video))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            out = ROOT / "outputs" / "review" / "replay.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            writer, store, f = None, TrajectoryStore(), 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                objs = tracks.get(f, [])
                store.update(f, objs)
                for o in objs:
                    x1, y1, x2, y2 = map(int, o.bbox_xyxy)
                    c = track_color(o.track_id)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
                    cv2.putText(frame, f"ID {o.track_id} {o.label}", (x1, max(16, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
                store.draw(frame, {o.track_id for o in objs})
                cv2.putText(frame, f"REPLAY  frame {f + 1}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2, cv2.LINE_AA)
                if writer is None:
                    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                             (frame.shape[1], frame.shape[0]))
                writer.write(frame)
                f += 1
            cap.release()
            if writer is not None:
                writer.release()
            perf = meta.get("performance", {})
            self.messages.put(("replay_ready", {"path": str(out), "csv": str(csv_path), "frames": f,
                                                "ids": len({t.track_id for v in tracks.values() for t in v}),
                                                "perf": perf, "model": meta.get("model", "?")}))
        except Exception:
            self.messages.put(("error", traceback.format_exc()))

    # ─────────────── 메시지 처리 훅 ───────────────
    def _handle_mot_message(self, kind: str, payload) -> bool:
        if kind == "stats":
            self._on_stats_message(payload)
            return True
        if kind == "replay_ready":
            self._open_review(Path(payload["path"]), [])
            self.log.insert(
                0,
                f"불러옴: {Path(payload['csv']).name}  {payload['frames']}f  ID {payload['ids']}개  "
                f"모델={Path(str(payload['model'])).name}  평균 {payload['perf'].get('avg_fps', '?')} FPS",
            )
            return True
        if kind == "mot_log":
            if hasattr(self, "mot_log"):
                self.mot_log.insert(tk.END, payload)
                self.mot_log.see(tk.END)
            return True
        if kind == "mot_done":
            self._mot_running = False
            if hasattr(self, "mot_run_btns"):
                for b in self.mot_run_btns:
                    b.configure(state="normal")
            self._refresh_mot_table()
            return True
        return False

    # ─────────────── MOT 성능 비교 화면 ───────────────
    def _show_mot_compare(self) -> None:
        if self._busy():
            return
        self.mode = "mot"
        self._stop_preview()
        self._clear_container()
        if getattr(self, "mot_frame", None) is None:
            self._build_mot_compare()
        self.mot_frame.pack(fill="both", expand=True)
        self._refresh_mot_table()

    def _build_mot_compare(self) -> None:
        bg, panel, fg = "#1b1d21", "#262a31", "#f3f5f7"
        self._mot_running = False
        self.mot_frame = tk.Frame(self.container, bg=bg)
        header = tk.Frame(self.mot_frame, bg=bg)
        header.pack(fill="x", padx=16, pady=(14, 6))
        tk.Button(header, text="← 메인", command=self._back_to_home, bg="#3a4150", fg=fg, relief="flat",
                  padx=10, pady=4, font=("Malgun Gothic", 10)).pack(side="left")
        tk.Label(header, text="MOT 추적 성능 비교 (MOT16 공개 데이터셋)", bg=bg, fg=fg,
                 font=("Malgun Gothic", 16, "bold")).pack(side="left", padx=12)

        self.mot_info_var = tk.StringVar(value="")
        tk.Label(self.mot_frame, textvariable=self.mot_info_var, bg=bg, fg="#9aa3b2", justify="left",
                 anchor="w", font=("Malgun Gothic", 9)).pack(fill="x", padx=18)

        cols = ("name", "model", "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN", "Frag", "FPS", "ms")
        widths = (330, 120, 60, 60, 60, 60, 60, 55, 60, 60, 50, 55, 60)
        style = ttk.Style()
        style.configure("Mot.Treeview", rowheight=24, font=("Malgun Gothic", 9))
        style.configure("Mot.Treeview.Heading", font=("Malgun Gothic", 9, "bold"))
        table_wrap = tk.Frame(self.mot_frame, bg=bg)
        table_wrap.pack(fill="x", padx=16, pady=(8, 4))
        self.mot_table = ttk.Treeview(table_wrap, columns=cols, show="headings", height=10, style="Mot.Treeview")
        for c, w in zip(cols, widths):
            self.mot_table.heading(c, text=c)
            self.mot_table.column(c, width=w, anchor="w" if c in ("name", "model") else "e")
        self.mot_table.pack(fill="x")
        self.mot_table.bind("<<TreeviewSelect>>", lambda _e: self._show_mot_detail())

        mid = tk.Frame(self.mot_frame, bg=bg)
        mid.pack(fill="both", expand=True, padx=16, pady=6)
        self.mot_chart = tk.Canvas(mid, bg=panel, highlightthickness=0, height=260)
        self.mot_chart.pack(side="left", fill="both", expand=True)
        self.mot_chart.bind("<Configure>", lambda _e: self._draw_mot_chart())
        right = tk.Frame(mid, bg=panel, width=430)
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)
        btns = tk.Frame(right, bg=panel)
        btns.pack(fill="x", padx=10, pady=(10, 6))
        self.mot_run_btns = []
        for text, cmd, color in (
            ("① 데이터 준비", "prepare", "#3a4150"),
            ("② 파라미터 튜닝 (val)", "tune", "#3a4150"),
            ("③ 성능 비교 실행 (test)", "eval", "#2f9e44"),
        ):
            b = tk.Button(btns, text=text, command=lambda c=cmd: self._run_mot_job(c), bg=color, fg="white",
                          relief="flat", padx=6, pady=4, font=("Malgun Gothic", 9))
            b.pack(side="left", padx=(0, 6))
            self.mot_run_btns.append(b)
        tk.Button(btns, text="새로고침", command=self._refresh_mot_table, bg="#3a4150", fg="white",
                  relief="flat", padx=6, pady=4, font=("Malgun Gothic", 9)).pack(side="left")
        self.mot_detail_var = tk.StringVar(value="표에서 행을 고르면 시퀀스별 결과가 보입니다")
        tk.Label(right, textvariable=self.mot_detail_var, bg=panel, fg="#c5ccd6", justify="left", anchor="nw",
                 font=("Consolas", 9), wraplength=410).pack(fill="x", padx=10)
        self.mot_log = tk.Listbox(right, bg="#101216", fg="#f3f5f7", highlightthickness=0, activestyle="none",
                                  font=("Consolas", 8))
        self.mot_log.pack(fill="both", expand=True, padx=10, pady=(6, 10))

    def _refresh_mot_table(self) -> None:
        if not hasattr(self, "mot_table"):
            return
        self.mot_table.delete(*self.mot_table.get_children())
        self._mot_rows = []
        info = []
        summary_path = MOT_OUT / "summary.json"
        if summary_path.exists():
            s = json.loads(summary_path.read_text(encoding="utf-8"))
            self._mot_rows = s["rows"]
            info.append(f"테스트 시퀀스 {', '.join(s['test_seqs'])}  ·  분할 train {s['split']['train']} / "
                        f"val {s['split']['val']}  ·  {s['device']}  ·  {s['created']}")
            best_h = max(r["HOTA"] for r in self._mot_rows)
            for i, r in enumerate(self._mot_rows):
                self.mot_table.insert(
                    "", tk.END, iid=str(i),
                    values=(r["name"], r["model"], f"{r['HOTA']:.3f}", f"{r['DetA']:.3f}", f"{r['AssA']:.3f}",
                            f"{r['MOTA']:.3f}", f"{r['IDF1']:.3f}", r["IDSW"], r["FP"], r["FN"], r["Frag"],
                            f"{r['FPS']:.1f}", f"{r['ms_per_frame']:.0f}"),
                    tags=("best",) if r["HOTA"] == best_h else (),
                )
            self.mot_table.tag_configure("best", background="#d8f5dd")
        else:
            info.append("아직 결과가 없습니다. ① → ② → ③ 순서로 실행하세요 (CPU 기준 수십 분).")
        tuned = ROOT / "configs" / "tuned_tracker.json"
        if tuned.exists():
            t = json.loads(tuned.read_text(encoding="utf-8"))
            p = t["params"]
            info.append(
                f"튜닝 결과: high={p.get('track_high_thresh')} new={p.get('new_track_thresh')} "
                f"buffer={p.get('track_buffer')} match={p.get('match_thresh')} 외형={p.get('appearance_weight', 0)} "
                f"재연결={p.get('relink', False)}  (val HOTA {t['val_metrics']['HOTA']:.3f})"
            )
        ft = WEIGHTS_DIR / "yolov8n_mot16.json"
        if ft.exists():
            m = json.loads(ft.read_text(encoding="utf-8"))
            info.append(
                f"Fine-tuning: {m['epochs']} epoch, {m['train_minutes']}분 · val mAP50 "
                f"{m['val_before']['mAP50']:.3f} → {m['val_after']['mAP50']:.3f}"
            )
        self.mot_info_var.set("\n".join(info))
        self._draw_mot_chart()

    def _show_mot_detail(self) -> None:
        sel = self.mot_table.selection()
        if not sel or not getattr(self, "_mot_rows", None):
            return
        r = self._mot_rows[int(sel[0])]
        lines = [r["name"], f"검출 {r['det_ms']:.0f}ms + 추적 {r['track_ms']:.1f}ms = {r['ms_per_frame']:.0f}ms "
                            f"({r['FPS']:.1f} FPS) · 재연결 {r.get('relinks', 0)}회", ""]
        for seq, m in r["per_seq"].items():
            lines.append(f"{seq}: HOTA {m['HOTA']:.3f} MOTA {m['MOTA']:.3f} IDF1 {m['IDF1']:.3f} "
                         f"IDSW {m['IDSW']} MT {m['MT']} ML {m['ML']}")
        self.mot_detail_var.set("\n".join(lines))

    def _draw_mot_chart(self) -> None:
        c = getattr(self, "mot_chart", None)
        rows = getattr(self, "_mot_rows", [])
        if c is None:
            return
        c.delete("all")
        w, h = max(c.winfo_width(), 300), max(c.winfo_height(), 200)
        if not rows:
            c.create_text(w / 2, h / 2, text="결과 없음", fill="#9aa3b2", font=("Malgun Gothic", 11))
            return
        metrics = (("HOTA", "#4c9be8"), ("IDF1", "#3ddc84"), ("MOTA", "#f0b429"))
        left, top, bottom = 40, 30, 40
        plot_h = h - top - bottom
        group_w = (w - left - 10) / len(rows)
        bar_w = max(4, group_w / (len(metrics) + 1))
        c.create_text(left, 12, text="지표 비교 (높을수록 좋음)", fill="#c5ccd6", anchor="w", font=("Malgun Gothic", 9))
        for k, (m, color) in enumerate(metrics):
            c.create_rectangle(w - 210 + k * 70, 6, w - 198 + k * 70, 18, fill=color, outline="")
            c.create_text(w - 194 + k * 70, 12, text=m, fill="#c5ccd6", anchor="w", font=("Malgun Gothic", 8))
        for g in (0, 0.25, 0.5, 0.75, 1.0):
            y = top + plot_h * (1 - g)
            c.create_line(left, y, w - 10, y, fill="#3a4150")
            c.create_text(left - 6, y, text=f"{g:.2f}", fill="#9aa3b2", anchor="e", font=("Consolas", 8))
        for i, r in enumerate(rows):
            x0 = left + i * group_w + bar_w / 2
            for k, (m, color) in enumerate(metrics):
                v = max(0.0, min(1.0, float(r[m])))
                x = x0 + k * bar_w
                c.create_rectangle(x, top + plot_h * (1 - v), x + bar_w - 2, top + plot_h, fill=color, outline="")
            c.create_text(x0 + bar_w * 1.5, h - bottom + 14, text=r["name"].split(" ")[0], fill="#f3f5f7",
                          font=("Malgun Gothic", 10, "bold"))

    def _run_mot_job(self, cmd: str) -> None:
        if getattr(self, "_mot_running", False):
            return
        self._mot_running = True
        for b in self.mot_run_btns:
            b.configure(state="disabled")
        self.mot_log.delete(0, tk.END)
        self.mot_log.insert(tk.END, f"[{cmd}] 시작 — CPU에서는 수 분~수십 분 걸립니다")

        def work() -> None:
            import mot_eval

            stream = _Stream(lambda line: self.messages.put(("mot_log", line)))
            try:
                with contextlib.redirect_stdout(stream):
                    if cmd == "prepare":
                        mot_eval.prepare()
                    elif cmd == "tune":
                        mot_eval.tune()
                    else:
                        mot_eval.evaluate_test()
                self.messages.put(("mot_log", f"[{cmd}] 완료"))
            except BaseException as exc:  # SystemExit 포함
                self.messages.put(("mot_log", f"[{cmd}] 실패: {exc}"))
                self.messages.put(("mot_log", traceback.format_exc().splitlines()[-1]))
            finally:
                self.messages.put(("mot_done", cmd))

        threading.Thread(target=work, daemon=True).start()

"""메인 화면에서 캡처 모드 / 성능평가 모드를 골라 실행한다."""

from __future__ import annotations

import csv
import json
import queue
import sys
import threading
import time
import traceback
import tkinter as tk
from dataclasses import asdict
from pathlib import Path
from tkinter import filedialog, ttk

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import AppConfig, EventConfig
from eval_accuracy import (
    default_workers,
    evaluate_jobs_parallel,
    list_videos,
    summarize,
)
from main import run_screen, run_video
from src.overlay_win import ScreenOverlay, enable_dpi_awareness

SETTINGS_PATH = ROOT / "user_settings.json"


def list_monitors() -> list[dict]:
    import mss

    with mss.mss() as sct:
        mons = []
        for index, mon in enumerate(sct.monitors):
            if index == 0:
                continue
            mons.append(
                {
                    "index": index,
                    "left": mon["left"],
                    "top": mon["top"],
                    "width": mon["width"],
                    "height": mon["height"],
                }
            )
        return mons


def pick_region(parent: tk.Misc, mon: dict) -> tuple[int, int, int, int] | None:
    result: dict = {}
    parent.withdraw()
    win = tk.Toplevel(parent)
    win.overrideredirect(True)
    win.geometry(f"{mon['width']}x{mon['height']}+{mon['left']}+{mon['top']}")
    win.attributes("-topmost", True)
    win.attributes("-alpha", 0.35)
    win.configure(bg="#111111")
    canvas = tk.Canvas(win, bg="#111111", highlightthickness=0, cursor="crosshair")
    canvas.pack(fill="both", expand=True)
    canvas.create_text(
        mon["width"] // 2,
        36,
        text="추적할 영상 영역을 드래그하세요. Esc 는 취소",
        fill="white",
        font=("Malgun Gothic", 16),
    )
    state = {"x0": 0, "y0": 0, "rect": None}

    def down(event: tk.Event) -> None:
        state["x0"], state["y0"] = event.x, event.y
        if state["rect"] is not None:
            canvas.delete(state["rect"])
        state["rect"] = canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline="#3dff7a", width=3
        )

    def drag(event: tk.Event) -> None:
        if state["rect"] is not None:
            canvas.coords(state["rect"], state["x0"], state["y0"], event.x, event.y)

    def up(event: tk.Event) -> None:
        x1, y1 = min(state["x0"], event.x), min(state["y0"], event.y)
        x2, y2 = max(state["x0"], event.x), max(state["y0"], event.y)
        width, height = x2 - x1, y2 - y1
        if width >= 40 and height >= 40:
            result["region"] = (mon["left"] + x1, mon["top"] + y1, width, height)
        win.destroy()

    def cancel(_event: tk.Event | None = None) -> None:
        win.destroy()

    canvas.bind("<ButtonPress-1>", down)
    canvas.bind("<B1-Motion>", drag)
    canvas.bind("<ButtonRelease-1>", up)
    win.bind("<Escape>", cancel)
    win.focus_force()
    parent.wait_window(win)
    parent.deiconify()
    parent.lift()
    return result.get("region")


class StudioApp:
    def __init__(self) -> None:
        enable_dpi_awareness()
        self.monitors = list_monitors()
        if not self.monitors:
            raise RuntimeError("사용할 모니터를 찾지 못했습니다.")

        self.root = tk.Tk()
        self.root.title("Tracking Alert")
        self.root.geometry("1080x720")
        self.root.minsize(900, 620)
        self.root.configure(bg="#1b1d21")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.mode: str | None = None
        self.region: tuple[int, int, int, int] | None = None
        self.video_path: Path | None = None
        self.tracking = False
        self.evaluating = False
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.messages: queue.Queue[tuple] = queue.Queue()
        self.preview_image: tk.PhotoImage | None = None
        self.eval_preview_image: tk.PhotoImage | None = None
        self.overlay: ScreenOverlay | None = None
        self._visual = None
        self._eval_visuals: dict[int, dict] = {}
        self._eval_focus_job: int | None = None
        self._visual_lock = threading.Lock()
        self._eval_visual_lock = threading.Lock()
        self._preview_job: str | None = None
        self._eval_results_lock = threading.Lock()

        self.container = tk.Frame(self.root, bg="#1b1d21")
        self.container.pack(fill="both", expand=True)

        self.home_frame: tk.Frame | None = None
        self.capture_frame: tk.Frame | None = None
        self.eval_frame: tk.Frame | None = None
        self._settings = self._load_settings_file()

        self._build_home()
        self._show_home()
        self._poll_messages()

    # ── 공통 스타일 ──────────────────────────────────────────
    def _clear_container(self) -> None:
        for child in self.container.winfo_children():
            child.pack_forget()

    def _show_home(self) -> None:
        if self._busy():
            return
        self._stop_preview()
        self._close_overlay()
        self.mode = None
        self._clear_container()
        if self.home_frame is None:
            self._build_home()
        self.home_frame.pack(fill="both", expand=True)

    def _show_capture_screen(self) -> None:
        self._open_capture("화면")

    def _show_capture_video(self) -> None:
        self._open_capture("영상 파일")

    def _open_capture(self, source: str) -> None:
        self.mode = "capture"
        self._clear_container()
        if self.capture_frame is None:
            self._build_capture()
        self.source_var.set(source)
        self._on_source_change()
        if source == "영상 파일":
            self._refresh_video_library()
            if hasattr(self, "capture_title"):
                self.capture_title.configure(text="영상 파일 모드")
        else:
            if hasattr(self, "capture_title"):
                self.capture_title.configure(text="화면 캡처 모드")
        self.capture_frame.pack(fill="both", expand=True)
        self._set_status("대기", "#8b93a1")
        self._schedule_preview()

    def _show_eval(self) -> None:
        self.mode = "eval"
        self._stop_preview()
        self._clear_container()
        if self.eval_frame is None:
            self._build_eval()
        self.eval_frame.pack(fill="both", expand=True)
        accident = list_videos(ROOT / "eval_data" / "accident")
        normal = list_videos(ROOT / "eval_data" / "normal")
        if hasattr(self, "eval_info_var"):
            self.eval_info_var.set(
                f"accident {len(accident)}개  ·  normal {len(normal)}개"
            )
        self._set_eval_status("대기", "#8b93a1")

    # ── 메인 화면 ────────────────────────────────────────────
    def _build_home(self) -> None:
        bg = "#1b1d21"
        self.home_frame = tk.Frame(self.container, bg=bg)

        center = tk.Frame(self.home_frame, bg=bg)
        center.place(relx=0.5, rely=0.48, anchor="center")

        tk.Label(
            center,
            text="Tracking Alert",
            bg=bg,
            fg="#f3f5f7",
            font=("Malgun Gothic", 28, "bold"),
        ).pack()
        tk.Label(
            center,
            text="모드를 선택하세요",
            bg=bg,
            fg="#9aa3b2",
            font=("Malgun Gothic", 12),
        ).pack(pady=(8, 28))

        btn_row = tk.Frame(center, bg=bg)
        btn_row.pack()

        tk.Button(
            btn_row,
            text="화면 캡처",
            command=self._show_capture_screen,
            bg="#2f9e44",
            fg="white",
            activebackground="#27863a",
            activeforeground="white",
            relief="flat",
            font=("Malgun Gothic", 13, "bold"),
            width=14,
            pady=18,
        ).pack(side="left", padx=8)

        tk.Button(
            btn_row,
            text="영상 파일",
            command=self._show_capture_video,
            bg="#2f9e44",
            fg="white",
            activebackground="#27863a",
            activeforeground="white",
            relief="flat",
            font=("Malgun Gothic", 13, "bold"),
            width=14,
            pady=18,
        ).pack(side="left", padx=8)

        tk.Button(
            btn_row,
            text="성능평가",
            command=self._show_eval,
            bg="#3a6ea5",
            fg="white",
            activebackground="#315f8c",
            activeforeground="white",
            relief="flat",
            font=("Malgun Gothic", 13, "bold"),
            width=14,
            pady=18,
        ).pack(side="left", padx=8)

        tk.Label(
            center,
            text="화면 캡처  ·  영상 파일/저장소  ·  eval_data 정확도",
            bg=bg,
            fg="#6f7886",
            font=("Malgun Gothic", 9),
        ).pack(pady=(26, 0))

        settings_row = tk.Frame(center, bg=bg)
        settings_row.pack(pady=(18, 0))
        tk.Button(
            settings_row,
            text="현재 설정 저장",
            command=self._save_settings,
            bg="#3a4150",
            fg="#f3f5f7",
            relief="flat",
            padx=12,
            pady=6,
            font=("Malgun Gothic", 10),
        ).pack(side="left", padx=4)
        tk.Button(
            settings_row,
            text="설정 불러오기",
            command=self._apply_settings_to_ui,
            bg="#3a4150",
            fg="#f3f5f7",
            relief="flat",
            padx=12,
            pady=6,
            font=("Malgun Gothic", 10),
        ).pack(side="left", padx=4)
        self.settings_status_var = tk.StringVar(
            value="저장됨" if SETTINGS_PATH.exists() else "기본값 사용 중"
        )
        tk.Label(
            center,
            textvariable=self.settings_status_var,
            bg=bg,
            fg="#6f7886",
            font=("Malgun Gothic", 8),
        ).pack(pady=(8, 0))

    # ── 캡처 모드 ────────────────────────────────────────────
    def _build_capture(self) -> None:
        bg = "#1b1d21"
        panel = "#262a31"
        fg = "#f3f5f7"
        self.capture_frame = tk.Frame(self.container, bg=bg)

        header = tk.Frame(self.capture_frame, bg=bg)
        header.pack(fill="x", padx=16, pady=(14, 6))
        tk.Button(
            header,
            text="← 메인",
            command=self._back_to_home,
            bg="#3a4150",
            fg=fg,
            relief="flat",
            padx=10,
            pady=4,
            font=("Malgun Gothic", 10),
        ).pack(side="left")
        self.capture_title = tk.Label(
            header,
            text="캡처 모드",
            bg=bg,
            fg=fg,
            font=("Malgun Gothic", 16, "bold"),
        )
        self.capture_title.pack(side="left", padx=12)
        self.status = tk.Label(
            header,
            text="대기",
            bg="#2c313a",
            fg="#8b93a1",
            padx=12,
            pady=4,
            font=("Malgun Gothic", 10),
        )
        self.status.pack(side="right")

        body = tk.Frame(self.capture_frame, bg=bg)
        body.pack(fill="both", expand=True, padx=16, pady=8)

        preview_wrap = tk.Frame(body, bg=panel)
        preview_wrap.pack(side="left", fill="both", expand=True)
        self.preview_title = tk.Label(
            preview_wrap,
            text="미리보기",
            bg=panel,
            fg="#9aa3b2",
            font=("Malgun Gothic", 10),
        )
        self.preview_title.pack(anchor="w", padx=12, pady=(10, 4))
        tk.Label(
            preview_wrap,
            text=(
                "초록 실선: 정상 추적   ·   빨간 실선: 사고 의심   ·   "
                "주황 점선: 가림 예측   ·   화살표: 속도 방향"
            ),
            bg=panel,
            fg="#c5ccd6",
            font=("Malgun Gothic", 9),
            anchor="w",
            justify="left",
        ).pack(fill="x", padx=12, pady=(0, 4))

        # 하단 검토 바를 먼저 고정해 영상이 설정을 밀어내지 않게 한다.
        self.review_bar = tk.Frame(preview_wrap, bg=panel)
        self.review_info_var = tk.StringVar(value="추적 후 검토 영상이 여기에 열립니다")
        tk.Label(
            self.review_bar,
            textvariable=self.review_info_var,
            bg=panel,
            fg="#c5ccd6",
            font=("Malgun Gothic", 9),
            anchor="w",
        ).pack(fill="x", padx=12, pady=(0, 4))
        self.review_frame_var = tk.IntVar(value=0)
        self.review_scale = tk.Scale(
            self.review_bar,
            from_=0,
            to=0,
            orient="horizontal",
            variable=self.review_frame_var,
            bg=panel,
            fg=fg,
            highlightthickness=0,
            troughcolor="#101216",
            showvalue=True,
            command=self._on_review_seek,
        )
        self.review_scale.pack(fill="x", padx=12)
        ctrl = tk.Frame(self.review_bar, bg=panel)
        ctrl.pack(fill="x", padx=12, pady=(4, 10))
        self.review_play_btn = tk.Button(
            ctrl,
            text="재생",
            command=self._toggle_review_play,
            bg="#3a4150",
            fg=fg,
            relief="flat",
            padx=10,
            pady=3,
        )
        self.review_play_btn.pack(side="left")
        tk.Button(
            ctrl,
            text="이전 알림",
            command=lambda: self._jump_review_alert(-1),
            bg="#3a4150",
            fg=fg,
            relief="flat",
            padx=8,
            pady=3,
        ).pack(side="left", padx=(10, 0))
        tk.Button(
            ctrl,
            text="다음 알림",
            command=lambda: self._jump_review_alert(1),
            bg="#c0392b",
            fg="white",
            relief="flat",
            padx=8,
            pady=3,
        ).pack(side="left", padx=(6, 0))
        self.review_alert_var = tk.StringVar(value="알림 0개")
        tk.Label(
            ctrl,
            textvariable=self.review_alert_var,
            bg=panel,
            fg="#f3f5f7",
            font=("Malgun Gothic", 9),
        ).pack(side="right")

        speed_row = tk.Frame(self.review_bar, bg=panel)
        speed_row.pack(fill="x", padx=12, pady=(0, 10))
        tk.Label(
            speed_row,
            text="배속",
            bg=panel,
            fg="#9aa3b2",
            font=("Malgun Gothic", 9),
        ).pack(side="left")
        saved_speed = 1.0
        try:
            raw = str((self._settings or {}).get("review_speed", "1.0")).replace("x", "")
            saved_speed = float(raw)
        except Exception:
            saved_speed = 1.0
        self.review_speed_var = tk.DoubleVar(value=max(0.05, min(2.0, saved_speed)))
        self.review_speed_label = tk.Label(
            speed_row,
            text=f"{self.review_speed_var.get():.2f}x",
            bg=panel,
            fg="#f3f5f7",
            width=6,
            font=("Malgun Gothic", 9, "bold"),
        )
        self.review_speed_label.pack(side="right")
        self.review_speed_scale = tk.Scale(
            speed_row,
            from_=0.05,
            to=2.0,
            resolution=0.05,
            orient="horizontal",
            variable=self.review_speed_var,
            bg=panel,
            fg=fg,
            highlightthickness=0,
            troughcolor="#101216",
            showvalue=False,
            command=self._on_review_speed_change,
        )
        self.review_speed_scale.pack(side="left", fill="x", expand=True, padx=(8, 8))
        for preset, text in ((0.25, "0.25"), (0.5, "0.5"), (1.0, "1.0"), (1.5, "1.5")):
            tk.Button(
                speed_row,
                text=text,
                command=lambda v=preset: self._set_review_speed(v),
                bg="#3a4150",
                fg=fg,
                relief="flat",
                padx=4,
                pady=1,
                font=("Malgun Gothic", 8),
            ).pack(side="right", padx=2)

        # review_bar는 영상 모드에서 side=bottom 으로 붙이고, 미리보기는 남은 영역만 사용
        self.preview_host = tk.Frame(preview_wrap, bg="#101216")
        self.preview_host.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.preview_host.pack_propagate(False)
        self.preview = tk.Label(self.preview_host, bg="#101216")
        self.preview.place(relx=0, rely=0, relwidth=1, relheight=1)

        self._review_active = False
        self._review_playing = False
        self._review_path: Path | None = None
        self._review_cap = None
        self._review_fps = 30.0
        self._review_total = 0
        self._review_frame = 0
        self._review_alerts: list[dict] = []
        self._review_job: str | None = None
        self._session_alerts: list = []

        side = tk.Frame(body, bg=panel, width=340)
        side.pack(side="right", fill="y", padx=(12, 0))
        side.pack_propagate(False)

        side_canvas = tk.Canvas(side, bg=panel, highlightthickness=0, width=320)
        side_scroll = ttk.Scrollbar(side, orient="vertical", command=side_canvas.yview)
        side_canvas.configure(yscrollcommand=side_scroll.set)
        side_scroll.pack(side="right", fill="y")
        side_canvas.pack(side="left", fill="both", expand=True)
        side_inner = tk.Frame(side_canvas, bg=panel)
        side_window = side_canvas.create_window((0, 0), window=side_inner, anchor="nw")

        def _sync_side(_event=None) -> None:
            side_canvas.configure(scrollregion=side_canvas.bbox("all"))
            side_canvas.itemconfigure(side_window, width=side_canvas.winfo_width())

        side_inner.bind("<Configure>", _sync_side)
        side_canvas.bind("<Configure>", _sync_side)

        def _side_wheel(event) -> None:
            side_canvas.yview_scroll(int(-event.delta / 120), "units")

        side_canvas.bind("<Enter>", lambda _e: side_canvas.bind_all("<MouseWheel>", _side_wheel))
        side_canvas.bind("<Leave>", lambda _e: side_canvas.unbind_all("<MouseWheel>"))
        side = side_inner  # 이하 위젯은 스크롤 가능한 내부에 배치

        def label(text: str) -> None:
            tk.Label(
                side, text=text, bg=panel, fg="#c5ccd6", font=("Malgun Gothic", 10)
            ).pack(anchor="w", padx=14, pady=(12, 4))

        self.source_var = tk.StringVar(value="화면")
        self.source_host = tk.Frame(side, bg=panel)
        self.source_host.pack(fill="x")

        self.screen_frame = tk.Frame(self.source_host, bg=panel)
        self.screen_frame.pack(fill="x")
        tk.Label(
            self.screen_frame,
            text="모니터",
            bg=panel,
            fg="#c5ccd6",
            font=("Malgun Gothic", 10),
        ).pack(anchor="w", padx=14, pady=(12, 4))
        self.monitor_var = tk.StringVar()
        names = [
            f"{m['index']}: {m['width']}x{m['height']} ({m['left']},{m['top']})"
            for m in self.monitors
        ]
        self.monitor_var.set(names[0])
        self.monitor_box = ttk.Combobox(
            self.screen_frame,
            textvariable=self.monitor_var,
            values=names,
            state="readonly",
        )
        self.monitor_box.pack(fill="x", padx=14)
        self.monitor_box.bind("<<ComboboxSelected>>", lambda _e: self._reset_region())

        tk.Label(
            self.screen_frame,
            text="캡처 영역",
            bg=panel,
            fg="#c5ccd6",
            font=("Malgun Gothic", 10),
        ).pack(anchor="w", padx=14, pady=(12, 4))
        self.region_var = tk.StringVar(value="모니터 전체")
        tk.Label(
            self.screen_frame,
            textvariable=self.region_var,
            bg=panel,
            fg=fg,
            anchor="w",
            font=("Malgun Gothic", 10),
        ).pack(fill="x", padx=14)
        row = tk.Frame(self.screen_frame, bg=panel)
        row.pack(fill="x", padx=14, pady=(6, 0))
        tk.Button(
            row,
            text="영역 선택",
            command=self._choose_region,
            bg="#3a4150",
            fg=fg,
            relief="flat",
            padx=8,
            pady=4,
        ).pack(side="left")
        tk.Button(
            row,
            text="전체",
            command=self._reset_region,
            bg="#3a4150",
            fg=fg,
            relief="flat",
            padx=8,
            pady=4,
        ).pack(side="left", padx=(6, 0))

        self.file_frame = tk.Frame(self.source_host, bg=panel)
        tk.Label(
            self.file_frame,
            text="선택한 영상",
            bg=panel,
            fg="#c5ccd6",
            font=("Malgun Gothic", 10),
        ).pack(anchor="w", padx=14, pady=(12, 4))
        self.video_var = tk.StringVar(value="선택 안 됨")
        tk.Label(
            self.file_frame,
            textvariable=self.video_var,
            bg=panel,
            fg=fg,
            anchor="w",
            font=("Malgun Gothic", 9),
            wraplength=300,
            justify="left",
        ).pack(fill="x", padx=14)
        open_row = tk.Frame(self.file_frame, bg=panel)
        open_row.pack(fill="x", padx=14, pady=(6, 0))
        tk.Button(
            open_row,
            text="파일에서 열기",
            command=self._choose_video,
            bg="#3a4150",
            fg=fg,
            relief="flat",
            padx=8,
            pady=4,
        ).pack(side="left")
        tk.Button(
            open_row,
            text="목록 새로고침",
            command=self._refresh_video_library,
            bg="#3a4150",
            fg=fg,
            relief="flat",
            padx=8,
            pady=4,
        ).pack(side="left", padx=(6, 0))

        tk.Label(
            self.file_frame,
            text="저장소 영상 (더블클릭으로 선택)",
            bg=panel,
            fg="#c5ccd6",
            font=("Malgun Gothic", 10),
        ).pack(anchor="w", padx=14, pady=(12, 4))
        self.library_list = tk.Listbox(
            self.file_frame,
            bg="#101216",
            fg="#f3f5f7",
            height=5,
            highlightthickness=0,
            activestyle="none",
            font=("Malgun Gothic", 9),
            exportselection=False,
        )
        self.library_list.pack(fill="x", padx=14)
        self.library_list.bind("<Double-Button-1>", self._on_library_select)
        self._library_paths: list[Path] = []
        tk.Button(
            self.file_frame,
            text="저장소에서 선택",
            command=self._pick_library_video,
            bg="#3a6ea5",
            fg="white",
            relief="flat",
            padx=8,
            pady=4,
        ).pack(anchor="w", padx=14, pady=(6, 0))
        tk.Label(
            self.file_frame,
            text="폴더: video_library / eval_data",
            bg=panel,
            fg="#9aa3b2",
            font=("Malgun Gothic", 8),
        ).pack(anchor="w", padx=14, pady=(4, 0))

        self._build_shared_settings(side, panel, fg, prefix="cap")

        self.start_btn = tk.Button(
            side,
            text="추적 시작",
            command=self._toggle,
            bg="#2f9e44",
            fg="white",
            relief="flat",
            font=("Malgun Gothic", 12, "bold"),
            pady=8,
        )
        self.start_btn.pack(fill="x", padx=14, pady=(14, 6))
        tk.Button(
            side,
            text="현재 설정 저장",
            command=self._save_settings,
            bg="#3a4150",
            fg=fg,
            relief="flat",
            font=("Malgun Gothic", 10),
            pady=4,
        ).pack(fill="x", padx=14, pady=(0, 6))

        tk.Label(
            side, text="로그", bg=panel, fg="#c5ccd6", font=("Malgun Gothic", 10)
        ).pack(anchor="w", padx=14, pady=(8, 4))
        self.log = tk.Listbox(
            side,
            bg="#101216",
            fg="#f3f5f7",
            height=10,
            highlightthickness=0,
            activestyle="none",
            font=("Consolas", 9),
        )
        self.log.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        self._on_source_change()
        self._apply_settings_to_ui(prefix="cap")

    # ── 성능평가 모드 ────────────────────────────────────────
    def _build_eval(self) -> None:
        bg = "#1b1d21"
        panel = "#262a31"
        fg = "#f3f5f7"
        self.eval_frame = tk.Frame(self.container, bg=bg)

        header = tk.Frame(self.eval_frame, bg=bg)
        header.pack(fill="x", padx=16, pady=(14, 6))
        tk.Button(
            header,
            text="← 메인",
            command=self._back_to_home,
            bg="#3a4150",
            fg=fg,
            relief="flat",
            padx=10,
            pady=4,
            font=("Malgun Gothic", 10),
        ).pack(side="left")
        tk.Label(
            header,
            text="성능평가 모드",
            bg=bg,
            fg=fg,
            font=("Malgun Gothic", 16, "bold"),
        ).pack(side="left", padx=12)
        self.eval_status = tk.Label(
            header,
            text="대기",
            bg="#2c313a",
            fg="#8b93a1",
            padx=12,
            pady=4,
            font=("Malgun Gothic", 10),
        )
        self.eval_status.pack(side="right")

        body = tk.Frame(self.eval_frame, bg=bg)
        body.pack(fill="both", expand=True, padx=16, pady=8)

        # 고정폭 설정칸을 먼저 배치해 미리보기 이미지가 설정을 덮지 않게 한다.
        side = tk.Frame(body, bg=panel, width=340)
        side.pack(side="right", fill="y", padx=(12, 0))
        side.pack_propagate(False)

        side_canvas = tk.Canvas(side, bg=panel, highlightthickness=0, width=320)
        side_scroll = ttk.Scrollbar(side, orient="vertical", command=side_canvas.yview)
        side_canvas.configure(yscrollcommand=side_scroll.set)
        side_scroll.pack(side="right", fill="y")
        side_canvas.pack(side="left", fill="both", expand=True)
        side_inner = tk.Frame(side_canvas, bg=panel)
        side_window = side_canvas.create_window((0, 0), window=side_inner, anchor="nw")

        def _sync_eval_side(_event=None) -> None:
            side_canvas.configure(scrollregion=side_canvas.bbox("all"))
            side_canvas.itemconfigure(side_window, width=side_canvas.winfo_width())

        side_inner.bind("<Configure>", _sync_eval_side)
        side_canvas.bind("<Configure>", _sync_eval_side)

        def _eval_side_wheel(event) -> None:
            side_canvas.yview_scroll(int(-event.delta / 120), "units")

        side_canvas.bind(
            "<Enter>", lambda _e: side_canvas.bind_all("<MouseWheel>", _eval_side_wheel)
        )
        side_canvas.bind(
            "<Leave>", lambda _e: side_canvas.unbind_all("<MouseWheel>")
        )
        side = side_inner

        preview_wrap = tk.Frame(body, bg=panel)
        preview_wrap.pack(side="left", fill="both", expand=True)
        title_row = tk.Frame(preview_wrap, bg=panel)
        title_row.pack(fill="x", padx=12, pady=(10, 4))
        self.eval_preview_title = tk.Label(
            title_row,
            text="평가 미리보기",
            bg=panel,
            fg="#9aa3b2",
            font=("Malgun Gothic", 10),
        )
        self.eval_preview_title.pack(side="left")
        self.eval_truth_badge = tk.Label(
            title_row,
            text="정답 대기",
            bg="#3a4150",
            fg="#f3f5f7",
            padx=12,
            pady=3,
            font=("Malgun Gothic", 11, "bold"),
        )
        self.eval_truth_badge.pack(side="right")

        # 하단 지표를 먼저 고정하고, 미리보기는 남은 영역만 사용
        eval_bottom = tk.Frame(preview_wrap, bg=panel)
        eval_bottom.pack(side="bottom", fill="x")
        self.eval_progress_var = tk.StringVar(value="평가를 시작하면 영상이 여기에 재생됩니다")
        tk.Label(
            eval_bottom,
            textvariable=self.eval_progress_var,
            bg=panel,
            fg=fg,
            font=("Malgun Gothic", 10),
            anchor="w",
            justify="left",
        ).pack(fill="x", padx=12, pady=(0, 4))

        self.eval_metrics_var = tk.StringVar(value="진행 중 지표: -")
        tk.Label(
            eval_bottom,
            textvariable=self.eval_metrics_var,
            bg=panel,
            fg="#c5ccd6",
            font=("Malgun Gothic", 10),
            anchor="w",
        ).pack(fill="x", padx=12, pady=(0, 12))

        self.eval_preview_host = tk.Frame(preview_wrap, bg="#101216")
        self.eval_preview_host.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.eval_preview_host.pack_propagate(False)
        self.eval_preview = tk.Label(self.eval_preview_host, bg="#101216")
        self.eval_preview.place(relx=0, rely=0, relwidth=1, relheight=1)

        tk.Label(
            side,
            text="평가 데이터",
            bg=panel,
            fg="#c5ccd6",
            font=("Malgun Gothic", 10),
        ).pack(anchor="w", padx=14, pady=(16, 4))
        accident = list_videos(ROOT / "eval_data" / "accident")
        normal = list_videos(ROOT / "eval_data" / "normal")
        self.eval_info_var = tk.StringVar(
            value=f"accident {len(accident)}개  ·  normal {len(normal)}개"
        )
        tk.Label(
            side,
            textvariable=self.eval_info_var,
            bg=panel,
            fg=fg,
            font=("Malgun Gothic", 11),
            anchor="w",
        ).pack(fill="x", padx=14)
        tk.Label(
            side,
            text="eval_data/accident, normal 폴더 기준",
            bg=panel,
            fg="#9aa3b2",
            font=("Malgun Gothic", 9),
        ).pack(anchor="w", padx=14, pady=(4, 0))

        self._build_shared_settings(side, panel, fg, prefix="eval")

        tk.Label(
            side,
            text="병렬 워커 수",
            bg=panel,
            fg="#c5ccd6",
            font=("Malgun Gothic", 10),
        ).pack(anchor="w", padx=14, pady=(12, 4))
        saved_workers = (self._settings or {}).get("eval", {}).get("workers")
        self.eval_workers_var = tk.IntVar(
            value=int(saved_workers) if saved_workers else default_workers()
        )
        tk.Scale(
            side,
            from_=1,
            to=8,
            resolution=1,
            orient="horizontal",
            variable=self.eval_workers_var,
            bg=panel,
            fg=fg,
            highlightthickness=0,
            troughcolor="#101216",
        ).pack(fill="x", padx=14)
        tk.Label(
            side,
            text="여러 영상을 동시에 평가합니다",
            bg=panel,
            fg="#9aa3b2",
            font=("Malgun Gothic", 9),
        ).pack(anchor="w", padx=14)

        btn_row = tk.Frame(side, bg=panel)
        btn_row.pack(fill="x", padx=14, pady=(18, 8))
        self.eval_btn = tk.Button(
            btn_row,
            text="평가 시작",
            command=self._run_eval,
            bg="#3a6ea5",
            fg="white",
            relief="flat",
            font=("Malgun Gothic", 12, "bold"),
            pady=8,
        )
        self.eval_btn.pack(side="left", fill="x", expand=True)
        self.eval_stop_btn = tk.Button(
            btn_row,
            text="중지",
            command=self._stop_eval,
            bg="#3a4150",
            fg=fg,
            relief="flat",
            font=("Malgun Gothic", 11),
            pady=8,
            state="disabled",
        )
        self.eval_stop_btn.pack(side="left", padx=(8, 0))
        tk.Button(
            side,
            text="현재 설정 저장",
            command=self._save_settings,
            bg="#3a4150",
            fg=fg,
            relief="flat",
            font=("Malgun Gothic", 10),
            pady=4,
        ).pack(fill="x", padx=14, pady=(0, 6))

        tk.Label(
            side, text="평가 로그", bg=panel, fg="#c5ccd6", font=("Malgun Gothic", 10)
        ).pack(anchor="w", padx=14, pady=(4, 4))
        self.eval_log = tk.Listbox(
            side,
            bg="#101216",
            fg="#f3f5f7",
            height=10,
            highlightthickness=0,
            activestyle="none",
            font=("Consolas", 9),
        )
        self.eval_log.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        self._apply_settings_to_ui(prefix="eval")

    def _build_shared_settings(
        self, parent: tk.Frame, panel: str, fg: str, prefix: str
    ) -> None:
        def label(text: str) -> None:
            tk.Label(
                parent, text=text, bg=panel, fg="#c5ccd6", font=("Malgun Gothic", 10)
            ).pack(anchor="w", padx=14, pady=(12, 4))

        saved = (self._settings or {}).get(prefix, {})
        label("사고 판정 민감도")
        sens = tk.IntVar(value=int(saved.get("sensitivity", 3)))
        conf = tk.DoubleVar(value=float(saved.get("conf", 0.05)))
        occ = tk.DoubleVar(value=float(saved.get("occlusion_sec", 1.0)))
        spike = tk.BooleanVar(value=bool(saved.get("spike", True)))
        confirm = tk.BooleanVar(value=bool(saved.get("confirm", True)))
        prox = tk.BooleanVar(value=bool(saved.get("prox", True)))

        tk.Scale(
            parent,
            from_=1,
            to=5,
            resolution=1,
            orient="horizontal",
            variable=sens,
            bg=panel,
            fg=fg,
            highlightthickness=0,
            troughcolor="#101216",
        ).pack(fill="x", padx=14)
        tk.Label(
            parent,
            text="1 둔감    ·    5 민감",
            bg=panel,
            fg="#9aa3b2",
            font=("Malgun Gothic", 9),
        ).pack(anchor="w", padx=14)

        label("검출 신뢰도")
        tk.Scale(
            parent,
            from_=0.01,
            to=0.8,
            resolution=0.01,
            orient="horizontal",
            variable=conf,
            bg=panel,
            fg=fg,
            highlightthickness=0,
            troughcolor="#101216",
        ).pack(fill="x", padx=14)
        tk.Label(
            parent,
            text="낮출수록 더 많이 잡힘  ·  너무 낮으면 잡음↑",
            bg=panel,
            fg="#9aa3b2",
            font=("Malgun Gothic", 9),
        ).pack(anchor="w", padx=14)

        label("가림 예측 유지 시간(초)")
        tk.Scale(
            parent,
            from_=0.0,
            to=3.0,
            resolution=0.5,
            orient="horizontal",
            variable=occ,
            bg=panel,
            fg=fg,
            highlightthickness=0,
            troughcolor="#101216",
        ).pack(fill="x", padx=14)
        tk.Label(
            parent,
            text="0이면 끔  ·  점선 박스로 표시",
            bg=panel,
            fg="#9aa3b2",
            font=("Malgun Gothic", 9),
        ).pack(anchor="w", padx=14)

        for text, var in (
            ("가속도 급변", spike),
            ("급변 후 상태 유지", confirm),
            ("근접하면 충돌 태그", prox),
        ):
            tk.Checkbutton(
                parent,
                text=text,
                variable=var,
                bg=panel,
                fg=fg,
                selectcolor="#101216",
                activebackground=panel,
                activeforeground=fg,
                font=("Malgun Gothic", 10),
            ).pack(anchor="w", padx=14)

        if prefix == "cap":
            self.sensitivity_var = sens
            self.conf_var = conf
            self.occlusion_sec_var = occ
            self.spike_var = spike
            self.confirm_var = confirm
            self.prox_var = prox
        else:
            self.eval_sensitivity_var = sens
            self.eval_conf_var = conf
            self.eval_occlusion_sec_var = occ
            self.eval_spike_var = spike
            self.eval_confirm_var = confirm
            self.eval_prox_var = prox

    # ── 설정 저장 / 불러오기 ─────────────────────────────────
    def _default_settings(self) -> dict:
        return {
            "cap": {
                "sensitivity": 3,
                "conf": 0.05,
                "occlusion_sec": 1.0,
                "spike": True,
                "confirm": True,
                "prox": True,
            },
            "eval": {
                "sensitivity": 3,
                "conf": 0.05,
                "occlusion_sec": 1.0,
                "spike": True,
                "confirm": True,
                "prox": True,
                "workers": default_workers(),
            },
            "review_speed": 1.0,
        }

    def _load_settings_file(self) -> dict:
        data = self._default_settings()
        if not SETTINGS_PATH.exists():
            return data
        try:
            loaded = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                for key in ("cap", "eval"):
                    if isinstance(loaded.get(key), dict):
                        data[key].update(loaded[key])
                if loaded.get("review_speed") is not None:
                    try:
                        data["review_speed"] = float(
                            str(loaded["review_speed"]).replace("x", "")
                        )
                    except Exception:
                        pass
        except Exception as exc:
            print(f"[settings] load failed: {exc}")
        return data

    def _collect_settings(self) -> dict:
        data = self._default_settings()
        if hasattr(self, "sensitivity_var"):
            data["cap"] = {
                "sensitivity": int(self.sensitivity_var.get()),
                "conf": float(self.conf_var.get()),
                "occlusion_sec": float(self.occlusion_sec_var.get()),
                "spike": bool(self.spike_var.get()),
                "confirm": bool(self.confirm_var.get()),
                "prox": bool(self.prox_var.get()),
            }
        if hasattr(self, "eval_sensitivity_var"):
            data["eval"] = {
                "sensitivity": int(self.eval_sensitivity_var.get()),
                "conf": float(self.eval_conf_var.get()),
                "occlusion_sec": float(self.eval_occlusion_sec_var.get()),
                "spike": bool(self.eval_spike_var.get()),
                "confirm": bool(self.eval_confirm_var.get()),
                "prox": bool(self.eval_prox_var.get()),
                "workers": int(self.eval_workers_var.get())
                if hasattr(self, "eval_workers_var")
                else default_workers(),
            }
        elif "eval" in self._settings:
            data["eval"] = dict(self._settings["eval"])
        if hasattr(self, "review_speed_var"):
            data["review_speed"] = float(self.review_speed_var.get())
        elif self._settings.get("review_speed") is not None:
            data["review_speed"] = self._settings["review_speed"]
        # 아직 안 연 모드 설정은 기존 저장값 유지
        if not hasattr(self, "sensitivity_var") and "cap" in self._settings:
            data["cap"] = dict(self._settings["cap"])
        return data

    def _save_settings(self) -> None:
        data = self._collect_settings()
        try:
            SETTINGS_PATH.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._settings = data
            msg = f"설정 저장 완료  →  {SETTINGS_PATH.name}"
            if hasattr(self, "settings_status_var"):
                self.settings_status_var.set(msg)
            if self.mode == "capture" and hasattr(self, "log"):
                self.log.insert(0, msg)
            elif self.mode == "eval" and hasattr(self, "eval_log"):
                self.eval_log.insert(0, msg)
        except Exception as exc:
            err = f"설정 저장 실패: {exc}"
            if hasattr(self, "settings_status_var"):
                self.settings_status_var.set(err)
            if self.mode == "capture" and hasattr(self, "log"):
                self.log.insert(0, err)
            elif self.mode == "eval" and hasattr(self, "eval_log"):
                self.eval_log.insert(0, err)

    def _apply_settings_to_ui(self, prefix: str | None = None) -> None:
        data = self._settings or self._load_settings_file()
        self._settings = data
        targets = [prefix] if prefix else ["cap", "eval"]
        for key in targets:
            block = data.get(key) or {}
            if key == "cap" and hasattr(self, "sensitivity_var"):
                self.sensitivity_var.set(int(block.get("sensitivity", 3)))
                self.conf_var.set(float(block.get("conf", 0.05)))
                self.occlusion_sec_var.set(float(block.get("occlusion_sec", 1.0)))
                self.spike_var.set(bool(block.get("spike", True)))
                self.confirm_var.set(bool(block.get("confirm", True)))
                self.prox_var.set(bool(block.get("prox", True)))
            if key == "eval" and hasattr(self, "eval_sensitivity_var"):
                self.eval_sensitivity_var.set(int(block.get("sensitivity", 3)))
                self.eval_conf_var.set(float(block.get("conf", 0.05)))
                self.eval_occlusion_sec_var.set(float(block.get("occlusion_sec", 1.0)))
                self.eval_spike_var.set(bool(block.get("spike", True)))
                self.eval_confirm_var.set(bool(block.get("confirm", True)))
                self.eval_prox_var.set(bool(block.get("prox", True)))
                if hasattr(self, "eval_workers_var") and "workers" in block:
                    self.eval_workers_var.set(int(block["workers"]))
        if hasattr(self, "review_speed_var") and data.get("review_speed") is not None:
            try:
                speed = float(str(data["review_speed"]).replace("x", ""))
                self._set_review_speed(speed)
            except Exception:
                pass
        if hasattr(self, "settings_status_var"):
            self.settings_status_var.set(
                f"불러옴  ·  {SETTINGS_PATH.name}"
                if SETTINGS_PATH.exists()
                else "기본값 사용 중"
            )
        if prefix is None:
            if self.mode == "capture" and hasattr(self, "log"):
                self.log.insert(0, "설정 불러오기 완료")
            elif self.mode == "eval" and hasattr(self, "eval_log"):
                self.eval_log.insert(0, "설정 불러오기 완료")

    # ── 캡처 모드 동작 ───────────────────────────────────────
    def _back_to_home(self) -> None:
        if self._busy():
            if self.mode == "capture" and hasattr(self, "log"):
                self.log.insert(0, "작업 중에는 메인으로 돌아갈 수 없습니다")
            elif self.mode == "eval" and hasattr(self, "eval_log"):
                self.eval_log.insert(0, "평가 중에는 메인으로 돌아갈 수 없습니다")
            return
        self._close_review()
        self._show_home()

    def _on_source_change(self) -> None:
        if self.source_var.get() == "화면":
            self.file_frame.pack_forget()
            self.screen_frame.pack(fill="x")
            if hasattr(self, "review_bar"):
                self.review_bar.pack_forget()
            self._close_review()
        else:
            self.screen_frame.pack_forget()
            self.file_frame.pack(fill="x")
            if hasattr(self, "review_bar"):
                self.review_bar.pack(side="bottom", fill="x", padx=0, pady=(0, 4))

    def _selected_monitor(self) -> dict:
        index = int(self.monitor_var.get().split(":", 1)[0])
        for mon in self.monitors:
            if mon["index"] == index:
                return mon
        return self.monitors[0]

    def _reset_region(self) -> None:
        self.region = None
        self.region_var.set("모니터 전체")

    def _choose_region(self) -> None:
        if self.tracking or self.evaluating:
            return
        picked = pick_region(self.root, self._selected_monitor())
        if picked is None:
            return
        self.region = picked
        left, top, width, height = picked
        self.region_var.set(f"{width}x{height}  @ {left},{top}")

    def _choose_video(self) -> None:
        if self.tracking or self.evaluating:
            return
        path = filedialog.askopenfilename(
            title="추적할 영상 선택",
            filetypes=[("Video", "*.mp4;*.avi;*.mov;*.mkv"), ("All", "*.*")],
        )
        if not path:
            return
        self._set_video_path(Path(path), source_label="파일")

    def _library_roots(self) -> list[tuple[str, Path]]:
        return [
            ("library", ROOT / "video_library"),
            ("accident", ROOT / "eval_data" / "accident"),
            ("normal", ROOT / "eval_data" / "normal"),
        ]

    def _refresh_video_library(self) -> None:
        if not hasattr(self, "library_list"):
            return
        self.library_list.delete(0, tk.END)
        self._library_paths = []
        for tag, folder in self._library_roots():
            folder.mkdir(parents=True, exist_ok=True)
            for path in list_videos(folder):
                self._library_paths.append(path)
                self.library_list.insert(tk.END, f"[{tag}] {path.name}")
        if not self._library_paths:
            self.library_list.insert(tk.END, "(저장소에 영상 없음)")

    def _pick_library_video(self) -> None:
        if self.tracking or self.evaluating:
            return
        if not hasattr(self, "library_list"):
            return
        sel = self.library_list.curselection()
        if not sel:
            if hasattr(self, "log"):
                self.log.insert(0, "저장소에서 영상을 먼저 선택하세요")
            return
        index = int(sel[0])
        if index < 0 or index >= len(self._library_paths):
            return
        self._set_video_path(self._library_paths[index], source_label="저장소")

    def _on_library_select(self, _event=None) -> None:
        self._pick_library_video()

    def _set_video_path(self, path: Path, source_label: str = "") -> None:
        self.video_path = path
        prefix = f"{source_label}: " if source_label else ""
        self.video_var.set(f"{prefix}{path.name}")
        if hasattr(self, "log"):
            self.log.insert(0, f"영상 선택: {path}")

    def _set_status(self, text: str, color: str) -> None:
        if hasattr(self, "status"):
            self.status.configure(text=text, fg=color)

    def _set_eval_status(self, text: str, color: str) -> None:
        if hasattr(self, "eval_status"):
            self.eval_status.configure(text=text, fg=color)

    def _make_cfg_from(
        self,
        sensitivity: int,
        conf: float,
        occlusion_sec: float,
        spike: bool,
        confirm: bool,
        prox: bool,
        output_dir: str = "outputs",
    ) -> AppConfig:
        cfg = AppConfig(
            conf=conf,
            output_dir=output_dir,
            occlusion_hold_sec=occlusion_sec,
            event=EventConfig(
                use_accel_spike=spike,
                use_post_state_confirm=confirm,
                use_proximity_tag=prox,
            ),
        )
        cfg.event.apply_sensitivity(sensitivity)
        return cfg

    def _make_cfg(self) -> AppConfig:
        return self._make_cfg_from(
            int(self.sensitivity_var.get()),
            float(self.conf_var.get()),
            float(self.occlusion_sec_var.get()),
            bool(self.spike_var.get()),
            bool(self.confirm_var.get()),
            bool(self.prox_var.get()),
        )

    def _make_eval_cfg(self) -> AppConfig:
        return self._make_cfg_from(
            int(self.eval_sensitivity_var.get()),
            float(self.eval_conf_var.get()),
            float(self.eval_occlusion_sec_var.get()),
            bool(self.eval_spike_var.get()),
            bool(self.eval_confirm_var.get()),
            bool(self.eval_prox_var.get()),
            output_dir="outputs/eval_accuracy",
        )

    def _stop_preview(self) -> None:
        if self._preview_job is not None:
            try:
                self.root.after_cancel(self._preview_job)
            except Exception:
                pass
            self._preview_job = None

    def _schedule_preview(self) -> None:
        self._stop_preview()
        self._preview_job = self.root.after(120, self._update_preview)

    def _show_bgr_on(
        self,
        widget: tk.Label,
        frame: np.ndarray,
        image_attr: str,
        letterbox: bool = False,
        size_widget: tk.Misc | None = None,
    ) -> None:
        size_ref = size_widget or widget
        canvas_w = max(size_ref.winfo_width(), 320)
        canvas_h = max(size_ref.winfo_height(), 180)
        scale = min(canvas_w / frame.shape[1], canvas_h / frame.shape[0])
        view = cv2.resize(
            frame,
            (max(1, int(frame.shape[1] * scale)), max(1, int(frame.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        if letterbox:
            canvas = np.full((canvas_h, canvas_w, 3), 16, dtype=np.uint8)
            x0 = (canvas_w - view.shape[1]) // 2
            y0 = (canvas_h - view.shape[0]) // 2
            canvas[y0 : y0 + view.shape[0], x0 : x0 + view.shape[1]] = view
            view = canvas
        view = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        ok, buf = cv2.imencode(".png", view)
        if not ok:
            return
        photo = tk.PhotoImage(data=buf.tobytes())
        setattr(self, image_attr, photo)
        widget.configure(image=photo)

    def _show_bgr(self, frame: np.ndarray) -> None:
        if hasattr(self, "preview"):
            # 이미지 크기가 레이아웃을 키우지 않도록 항상 호스트 영역에 맞춰 레터박스
            self._show_bgr_on(
                self.preview,
                frame,
                "preview_image",
                letterbox=True,
                size_widget=getattr(self, "preview_host", self.preview),
            )

    def _show_eval_bgr(self, frame: np.ndarray) -> None:
        if hasattr(self, "eval_preview"):
            self._show_bgr_on(
                self.eval_preview,
                frame,
                "eval_preview_image",
                letterbox=True,
                size_widget=getattr(self, "eval_preview_host", self.eval_preview),
            )

    def _update_preview(self) -> None:
        if self.mode != "capture":
            self._preview_job = None
            return
        # 검토 모드는 녹화본만 재생하므로 라이브 미리보기/추적을 돌리지 않는다.
        if self._review_active and not self.tracking:
            self._preview_job = None
            return
        if not self.tracking and not self.evaluating:
            if self.source_var.get() == "화면":
                frame = self._grab_preview()
                if frame is not None:
                    self._show_bgr(frame)
            elif self.video_path is not None and self.video_path.exists():
                cap = cv2.VideoCapture(str(self.video_path))
                ok, frame = cap.read()
                cap.release()
                if ok:
                    self._show_bgr(frame)
        self._preview_job = self.root.after(120, self._update_preview)

    def _grab_preview(self):
        import mss

        mon = self._selected_monitor()
        if self.region is None:
            box = {
                "left": mon["left"],
                "top": mon["top"],
                "width": mon["width"],
                "height": mon["height"],
            }
        else:
            left, top, width, height = self.region
            box = {"left": left, "top": top, "width": width, "height": height}
        with mss.mss() as sct:
            shot = np.array(sct.grab(box))
        frame = shot[:, :, :3]
        if frame.shape[1] > 960:
            scale = 960 / frame.shape[1]
            frame = cv2.resize(
                frame,
                (960, max(1, int(frame.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return frame

    def _busy(self) -> bool:
        return self.tracking or self.evaluating

    def _toggle(self) -> None:
        if self.evaluating:
            return
        if self.tracking:
            self._stop()
        else:
            self._start()

    def _start(self) -> None:
        cfg = self._make_cfg()
        use_screen = self.source_var.get() == "화면"
        if not use_screen:
            if self.video_path is None or not self.video_path.exists():
                self.log.insert(0, "오류: 영상 파일을 먼저 선택하세요")
                return

        self._close_review()
        self._session_alerts = []
        self.tracking = True
        self.stop_event = threading.Event()
        self.start_btn.configure(text="중지", bg="#d64545")
        self.monitor_box.configure(state="disabled")
        self.preview_title.configure(text="추적 중 · 결과 녹화")
        self._set_status("모델 준비 중", "#f0b429")

        mon = self._selected_monitor()
        if use_screen:
            if self.region is None:
                left, top, width, height = (
                    mon["left"],
                    mon["top"],
                    mon["width"],
                    mon["height"],
                )
            else:
                left, top, width, height = self.region
            try:
                self.overlay = ScreenOverlay(left, top, width, height)
                self._paint_overlay([], {}, [], [], width, height)
            except Exception as exc:
                self.tracking = False
                self.start_btn.configure(text="추적 시작", bg="#2f9e44")
                self.monitor_box.configure(state="readonly")
                self.preview_title.configure(text="미리보기")
                self._set_status("오버레이 실패", "#d64545")
                self.log.insert(0, f"오류: {exc}")
                return

        def on_visual(
            tracked, states, events, recent, toggles, fw, fh, vis, predicted
        ) -> None:
            with self._visual_lock:
                self._visual = (
                    tracked,
                    states,
                    events,
                    recent,
                    toggles,
                    fw,
                    fh,
                    vis,
                    predicted,
                )

        video_path = str(self.video_path) if self.video_path else ""
        review_dir = ROOT / "outputs" / "review"
        review_name = "last_review.mp4"

        def on_alert(event) -> None:
            self._session_alerts.append(event)
            self.messages.put(("alert", event))

        def work() -> None:
            self.messages.put(("status", "추적 중"))
            try:
                if use_screen:
                    run_screen(
                        cfg,
                        monitor=mon["index"],
                        region=self.region,
                        max_width=1280,
                        show=False,
                        save_video=False,
                        use_overlay=False,
                        stop_event=self.stop_event,
                        on_alert=on_alert,
                        on_visual=on_visual,
                    )
                    self.messages.put(("stopped", None))
                else:
                    cfg.output_dir = str(review_dir)
                    review_dir.mkdir(parents=True, exist_ok=True)
                    out_path = run_video(
                        cfg,
                        source=video_path,
                        show=False,
                        save_video=True,
                        stop_event=self.stop_event,
                        on_alert=on_alert,
                        on_visual=on_visual,
                        out_name=review_name,
                    )
                    alerts = [
                        {
                            "frame_idx": e.frame_idx,
                            "track_id": e.track_id,
                            "tag": e.tag,
                            "reason": e.reason,
                        }
                        for e in self._session_alerts
                    ]
                    meta_path = review_dir / "last_review.json"
                    with meta_path.open("w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "source": video_path,
                                "review_video": str(out_path) if out_path else None,
                                "alerts": alerts,
                            },
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )
                    self.messages.put(
                        (
                            "review_ready",
                            {
                                "path": str(out_path) if out_path else None,
                                "alerts": alerts,
                            },
                        )
                    )
                    self.messages.put(("stopped", None))
            except Exception:
                self.messages.put(("error", traceback.format_exc()))
                self.messages.put(("stopped", None))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _stop(self) -> None:
        self.stop_event.set()
        self._set_status("중지 중", "#f0b429")
        self.start_btn.configure(state="disabled")

    def _stop_eval(self) -> None:
        if not self.evaluating:
            return
        self.stop_event.set()
        self._set_eval_status("중지 중", "#f0b429")
        if hasattr(self, "eval_stop_btn"):
            self.eval_stop_btn.configure(state="disabled")

    def _run_eval(self) -> None:
        if self._busy():
            return
        data_root = ROOT / "eval_data"
        accident = list_videos(data_root / "accident")
        normal = list_videos(data_root / "normal")
        if not accident and not normal:
            self.eval_log.insert(0, "오류: eval_data/accident, normal 에 영상이 없습니다")
            return

        workers = default_workers(
            int(self.eval_workers_var.get())
            if hasattr(self, "eval_workers_var")
            else 0
        )
        self.evaluating = True
        self.stop_event = threading.Event()
        with self._eval_visual_lock:
            self._eval_visuals = {}
            self._eval_focus_job = None
        self.eval_btn.configure(state="disabled", text="평가 중...")
        if hasattr(self, "eval_stop_btn"):
            self.eval_stop_btn.configure(state="normal", bg="#d64545", fg="white")
        self._set_eval_status(f"병렬 평가 중 ×{workers}", "#f0b429")
        if hasattr(self, "eval_preview_title"):
            self.eval_preview_title.configure(text=f"병렬 평가 중  ·  워커 {workers}")
        if hasattr(self, "eval_progress_var"):
            self.eval_progress_var.set("워커가 영상을 동시에 처리합니다...")
        if hasattr(self, "eval_metrics_var"):
            self.eval_metrics_var.set("진행 중 지표: -")
        self.eval_log.insert(
            0,
            f"평가 시작  accident={len(accident)} normal={len(normal)} workers={workers}",
        )
        cfg = self._make_eval_cfg()
        sensitivity = int(self.eval_sensitivity_var.get())
        conf = float(self.eval_conf_var.get())
        jobs = [("accident", p) for p in accident] + [
            ("normal", p) for p in normal
        ]

        def on_frame(info: dict) -> None:
            job_index = int(info.get("job_index", 0))
            with self._eval_visual_lock:
                self._eval_visuals[job_index] = info

        def on_job_start(info: dict) -> None:
            self.messages.put(
                (
                    "eval_progress",
                    f"시작 [{info['index']}/{info['total']}] {info['label']}: {info['name']}",
                )
            )
            self.messages.put(("eval_job", info))

        def work() -> None:
            try:
                completed: list = []

                def on_job_done(result, job_index: int, total: int) -> None:
                    with self._eval_results_lock:
                        completed.append(result)
                        snapshot = list(completed)
                    if result.label == "accident" and not result.predicted_accident:
                        tag = "FN 미검출"
                    elif result.label == "normal" and result.predicted_accident:
                        tag = "FP 오탐"
                    elif result.correct:
                        tag = "OK"
                    else:
                        tag = "X"
                    self.messages.put(
                        (
                            "eval_progress",
                            f"완료 [{job_index}/{total}] {result.label}: "
                            f"{Path(result.path).name} -> "
                            f"{'accident' if result.predicted_accident else 'normal'} "
                            f"alerts={result.alert_count} [{tag}]",
                        )
                    )
                    self.messages.put(("eval_partial", summarize(snapshot)))
                    with self._eval_visual_lock:
                        self._eval_visuals.pop(job_index, None)

                results = evaluate_jobs_parallel(
                    jobs,
                    cfg,
                    workers=workers,
                    on_frame=on_frame,
                    on_job_start=on_job_start,
                    on_job_done=on_job_done,
                    stop_event=self.stop_event,
                    visual_stride=2,
                    visual_max_width=640,
                )

                if not results:
                    self.messages.put(("eval_done", None))
                    return

                metrics = summarize(results)
                out_dir = Path(cfg.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "metrics": metrics,
                            "videos": [asdict(r) for r in results],
                            "settings": {
                                "sensitivity": sensitivity,
                                "conf": conf,
                                "workers": workers,
                            },
                            "stopped": self.stop_event.is_set(),
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                with (out_dir / "per_video.csv").open(
                    "w", newline="", encoding="utf-8-sig"
                ) as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "path",
                            "label",
                            "predicted_accident",
                            "alert_count",
                            "first_alert_frame",
                            "frames",
                            "correct",
                        ],
                    )
                    writer.writeheader()
                    for r in results:
                        writer.writerow(
                            {
                                "path": r.path,
                                "label": r.label,
                                "predicted_accident": r.predicted_accident,
                                "alert_count": r.alert_count,
                                "first_alert_frame": r.first_alert_frame,
                                "frames": r.frames,
                                "correct": r.correct,
                            }
                        )
                fn_names = [
                    Path(r.path).name
                    for r in results
                    if r.label == "accident" and not r.predicted_accident
                ]
                fp_names = [
                    Path(r.path).name
                    for r in results
                    if r.label == "normal" and r.predicted_accident
                ]
                self.messages.put(
                    (
                        "eval_done",
                        {
                            "metrics": metrics,
                            "stopped": self.stop_event.is_set(),
                            "fn_names": fn_names,
                            "fp_names": fp_names,
                        },
                    )
                )
            except Exception:
                self.messages.put(("error", traceback.format_exc()))
                self.messages.put(("eval_done", None))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _poll_messages(self) -> None:
        while True:
            try:
                kind, payload = self.messages.get_nowait()
            except queue.Empty:
                break
            if kind == "status":
                self._set_status("추적 중", "#3ddc84")
            elif kind == "alert":
                if hasattr(self, "log"):
                    line = (
                        f"ID {payload.track_id}  {payload.tag}  "
                        f"frame {payload.frame_idx}"
                    )
                    self.log.insert(0, line)
                    self.log.see(0)
            elif kind == "review_ready":
                path = payload.get("path") if payload else None
                alerts = (payload or {}).get("alerts") or []
                if path and Path(path).exists():
                    self._open_review(Path(path), alerts)
                    if hasattr(self, "log"):
                        self.log.insert(
                            0,
                            f"검토 준비 완료  alerts={len(alerts)}  file={Path(path).name}",
                        )
                else:
                    if hasattr(self, "log"):
                        self.log.insert(0, "검토 영상 저장 실패")
            elif kind == "eval_progress":
                if hasattr(self, "eval_log"):
                    self.eval_log.insert(0, str(payload))
                    self.eval_log.see(0)
            elif kind == "eval_job":
                if hasattr(self, "eval_preview_title"):
                    self.eval_preview_title.configure(
                        text=(
                            f"평가 중  [{payload['index']}/{payload['total']}]  "
                            f"{payload['label']} · {payload['name']}"
                        )
                    )
            elif kind == "eval_partial":
                if hasattr(self, "eval_metrics_var") and payload:
                    self.eval_metrics_var.set(
                        "진행 중  Acc={:.1f}%  FN(미검출)={}  FP(오탐)={}  "
                        "Recall={:.1f}%  FAR={:.1f}%  ({}개 완료)".format(
                            payload["accuracy"] * 100,
                            payload["fn"],
                            payload["fp"],
                            payload["recall"] * 100,
                            payload["false_alarm_rate"] * 100,
                            payload["count"],
                        )
                    )
            elif kind == "eval_done":
                self.evaluating = False
                with self._eval_visual_lock:
                    self._eval_visuals = {}
                    self._eval_focus_job = None
                if hasattr(self, "eval_btn"):
                    self.eval_btn.configure(state="normal", text="평가 시작")
                if hasattr(self, "eval_stop_btn"):
                    self.eval_stop_btn.configure(
                        state="disabled", bg="#3a4150", fg="#f3f5f7"
                    )
                if hasattr(self, "eval_preview_title"):
                    self.eval_preview_title.configure(text="평가 미리보기")
                if hasattr(self, "eval_truth_badge"):
                    self.eval_truth_badge.configure(
                        text="정답 대기", bg="#3a4150", fg="#f3f5f7"
                    )
                if payload is None:
                    self._set_eval_status("평가 실패", "#d64545")
                    if hasattr(self, "eval_progress_var"):
                        self.eval_progress_var.set("평가가 중단되었거나 실패했습니다")
                else:
                    metrics = payload["metrics"]
                    stopped = payload.get("stopped", False)
                    self._set_eval_status(
                        "평가 중지됨" if stopped else "평가 완료",
                        "#f0b429" if stopped else "#3ddc84",
                    )
                    if hasattr(self, "eval_progress_var"):
                        self.eval_progress_var.set(
                            "중지됨 · 부분 결과 저장" if stopped else "전체 평가 완료"
                        )
                    fn_names = payload.get("fn_names") or []
                    fp_names = payload.get("fp_names") or []
                    if hasattr(self, "eval_metrics_var"):
                        self.eval_metrics_var.set(
                            "최종 Acc={:.1f}%  FN(미검출)={}  FP(오탐)={}  "
                            "TP={} TN={}  Recall={:.1f}%  FAR={:.1f}%".format(
                                metrics["accuracy"] * 100,
                                metrics["fn"],
                                metrics["fp"],
                                metrics["tp"],
                                metrics["tn"],
                                metrics["recall"] * 100,
                                metrics["false_alarm_rate"] * 100,
                            )
                        )
                    if hasattr(self, "eval_log"):
                        self.eval_log.insert(
                            0,
                            "평가 Acc={:.1f}%  FN={}  FP={}  TP={}  TN={}".format(
                                metrics["accuracy"] * 100,
                                metrics["fn"],
                                metrics["fp"],
                                metrics["tp"],
                                metrics["tn"],
                            ),
                        )
                        if fn_names:
                            self.eval_log.insert(
                                0,
                                "FN 미검출(사고→정상 오판): " + ", ".join(fn_names),
                            )
                        else:
                            self.eval_log.insert(0, "FN 미검출: 없음")
                        if fp_names:
                            self.eval_log.insert(
                                0,
                                "FP 오탐(정상→사고 오판): " + ", ".join(fp_names),
                            )
                        else:
                            self.eval_log.insert(0, "FP 오탐: 없음")
                        self.eval_log.insert(0, "결과 저장: outputs/eval_accuracy/")
            elif kind == "error":
                target = None
                if self.mode == "eval" and hasattr(self, "eval_log"):
                    target = self.eval_log
                elif hasattr(self, "log"):
                    target = self.log
                if target is not None:
                    target.insert(0, "오류: 작업 중 예외가 발생했습니다")
                    target.insert(1, str(payload).splitlines()[-1][:120])
                print(payload)
            elif kind == "stopped":
                self._close_overlay()
                self.tracking = False
                if hasattr(self, "start_btn"):
                    self.start_btn.configure(
                        text="추적 시작", bg="#2f9e44", state="normal"
                    )
                if hasattr(self, "monitor_box"):
                    self.monitor_box.configure(state="readonly")
                if hasattr(self, "preview_title"):
                    if self._review_active:
                        self.preview_title.configure(
                            text="검토 모드 · 녹화본 재생(추적 없음)"
                        )
                    else:
                        self.preview_title.configure(text="미리보기")
                self._set_status(
                    "검토(재생만)" if self._review_active else "대기",
                    "#3ddc84" if self._review_active else "#8b93a1",
                )
                if self._review_active and hasattr(self, "start_btn"):
                    self.start_btn.configure(text="다시 분석")
        if self.mode == "capture":
            self._paint_latest()
        elif self.mode == "eval":
            self._paint_eval_latest()
        self.root.after(50, self._poll_messages)

    def _paint_eval_latest(self) -> None:
        with self._eval_visual_lock:
            active = dict(self._eval_visuals)
            focus = self._eval_focus_job
            if not active:
                return
            # 보던 영상이 끝나면, 같은 미리보기 자리에서 다음 활성 영상으로 이어간다.
            if focus not in active:
                focus = min(active.keys())
                self._eval_focus_job = focus
            info = active[focus]
            items = [active[k] for k in sorted(active)]
        vis = info.get("vis")
        if vis is None:
            return
        self._show_eval_bgr(vis)
        label = info.get("label", "")
        is_accident = label == "accident"
        if hasattr(self, "eval_truth_badge"):
            if is_accident:
                self.eval_truth_badge.configure(
                    text="정답: 사고 영상", bg="#c0392b", fg="white"
                )
            elif label == "normal":
                self.eval_truth_badge.configure(
                    text="정답: 정상 영상", bg="#1e8449", fg="white"
                )
            else:
                self.eval_truth_badge.configure(
                    text="정답 대기", bg="#3a4150", fg="#f3f5f7"
                )
        if hasattr(self, "eval_preview_title"):
            truth_ko = "사고" if is_accident else ("정상" if label == "normal" else "?")
            self.eval_preview_title.configure(
                text=(
                    f"평가 미리보기  [{info.get('job_index', '?')}/"
                    f"{info.get('job_total', '?')}]  "
                    f"{truth_ko} · {info.get('name', '')}"
                )
            )
        parts = []
        for item in items:
            total = item.get("total_hint") or 0
            frame_txt = (
                f"{item['frame_idx']}/{total}"
                if total > 0
                else str(item["frame_idx"])
            )
            parts.append(
                f"{item.get('name', '?')}({frame_txt},a{item.get('alert_count', 0)})"
            )
        if hasattr(self, "eval_progress_var"):
            self.eval_progress_var.set(
                f"동시 {len(items)}개  ·  표시 중 {info.get('name', '')}  ·  "
                + "  |  ".join(parts)
            )

    def _open_review(self, path: Path, alerts: list[dict]) -> None:
        self._close_review(keep_bar=True)
        self._stop_preview()  # 라이브 미리보기 중단 (YOLO 추적 없음)
        with self._visual_lock:
            self._visual = None
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            if hasattr(self, "log"):
                self.log.insert(0, f"오류: 검토 영상을 열 수 없습니다 ({path.name})")
            return
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self._review_cap = cap
        self._review_path = path
        self._review_total = max(0, total - 1)
        self._review_fps = max(1.0, fps)
        self._review_alerts = sorted(alerts, key=lambda a: a.get("frame_idx", 0))
        self._review_active = True
        self._review_playing = False
        if hasattr(self, "review_bar"):
            self.review_bar.pack(side="bottom", fill="x", padx=0, pady=(0, 4))
        self.review_scale.configure(to=max(0, self._review_total))
        self.review_frame_var.set(0)
        n_alert = len(self._review_alerts)
        first = (
            f"첫 알림 frame {self._review_alerts[0]['frame_idx']}"
            if n_alert
            else "알림 없음"
        )
        self.review_info_var.set(
            f"녹화본 재생(추적 없음)  ·  {path.name}  ·  {total}f @ {fps:.1f}fps  ·  {first}"
        )
        self.review_alert_var.set(f"알림 {n_alert}개")
        self.review_play_btn.configure(text="재생")
        if hasattr(self, "preview_title"):
            self.preview_title.configure(text="검토 모드 · 녹화본 재생(추적 없음)")
        if hasattr(self, "log"):
            self.log.insert(0, "검토 시작: 녹화 파일만 재생합니다 (재추적 없음)")
        self._review_show_frame(0)
        if n_alert:
            self._jump_review_alert(1)

    def _close_review(self, keep_bar: bool = False) -> None:
        self._review_playing = False
        if self._review_job is not None:
            try:
                self.root.after_cancel(self._review_job)
            except Exception:
                pass
            self._review_job = None
        if self._review_cap is not None:
            self._review_cap.release()
            self._review_cap = None
        was_review = self._review_active
        self._review_active = False
        self._review_path = None
        self._review_alerts = []
        if hasattr(self, "review_play_btn"):
            self.review_play_btn.configure(text="재생")
        if hasattr(self, "review_info_var"):
            self.review_info_var.set("추적 후 검토 영상이 여기에 열립니다")
        if hasattr(self, "review_alert_var"):
            self.review_alert_var.set("알림 0개")
        if hasattr(self, "start_btn") and was_review and not self.tracking:
            self.start_btn.configure(text="추적 시작")
        if not keep_bar and hasattr(self, "review_bar"):
            if self.source_var.get() != "영상 파일":
                self.review_bar.pack_forget()
        # 검토 종료 후 라이브 미리보기 재개
        if was_review and self.mode == "capture" and not self.tracking:
            self._schedule_preview()

    def _review_show_frame(self, frame_idx: int, sequential: bool = False) -> None:
        if self._review_cap is None:
            return
        frame_idx = int(max(0, min(frame_idx, self._review_total)))
        # 연속 재생은 read만 하고, 점프할 때만 seek (매 프레임 seek가 느림의 원인)
        if sequential and frame_idx == self._review_frame + 1:
            ok, frame = self._review_cap.read()
        else:
            self._review_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = self._review_cap.read()
        if not ok:
            return
        self._review_frame = frame_idx
        # 재생 중 슬라이더는 가끔만 갱신해 UI 부하 감소
        if (not self._review_playing) or (frame_idx % 5 == 0) or frame_idx >= self._review_total:
            self.review_frame_var.set(frame_idx)
        alert_here = any(a.get("frame_idx") == frame_idx for a in self._review_alerts)
        if alert_here:
            cv2.rectangle(
                frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 6
            )
            cv2.putText(
                frame,
                "ALERT FRAME",
                (16, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        self._show_review_bgr(frame)
        if alert_here or (not self._review_playing) or frame_idx % 10 == 0:
            near = [
                a
                for a in self._review_alerts
                if abs(int(a.get("frame_idx", -999)) - frame_idx) <= 2
            ]
            if near:
                a = near[0]
                self.review_alert_var.set(
                    f"frame {a['frame_idx']}  ID {a['track_id']}  {a['tag']}"
                )
            else:
                self.review_alert_var.set(f"알림 {len(self._review_alerts)}개")

    def _show_review_bgr(self, frame: np.ndarray) -> None:
        """검토 재생용: JPEG로 빠르게 표시."""
        if not hasattr(self, "preview"):
            return
        size_ref = getattr(self, "preview_host", self.preview)
        canvas_w = max(size_ref.winfo_width(), 320)
        canvas_h = max(size_ref.winfo_height(), 180)
        scale = min(canvas_w / frame.shape[1], canvas_h / frame.shape[0])
        view = cv2.resize(
            frame,
            (max(1, int(frame.shape[1] * scale)), max(1, int(frame.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        canvas = np.full((canvas_h, canvas_w, 3), 16, dtype=np.uint8)
        x0 = (canvas_w - view.shape[1]) // 2
        y0 = (canvas_h - view.shape[0]) // 2
        canvas[y0 : y0 + view.shape[0], x0 : x0 + view.shape[1]] = view
        view = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        ok, buf = cv2.imencode(".png", view)
        if not ok:
            return
        photo = tk.PhotoImage(data=buf.tobytes())
        self.preview_image = photo
        self.preview.configure(image=photo)

    def _on_review_seek(self, value: str) -> None:
        if not self._review_active or self.tracking:
            return
        self._review_playing = False
        if hasattr(self, "review_play_btn"):
            self.review_play_btn.configure(text="재생")
        self._review_show_frame(int(float(value)), sequential=False)

    def _toggle_review_play(self) -> None:
        if not self._review_active or self.tracking:
            return
        if self._review_playing:
            self._review_playing = False
            self.review_play_btn.configure(text="재생")
            self.review_frame_var.set(self._review_frame)
            return
        self._review_playing = True
        self.review_play_btn.configure(text="일시정지")
        self._review_tick()

    def _review_speed_value(self) -> float:
        try:
            return max(0.05, min(2.0, float(self.review_speed_var.get())))
        except Exception:
            return 1.0

    def _on_review_speed_change(self, value: str) -> None:
        try:
            speed = max(0.05, min(2.0, float(value)))
        except Exception:
            speed = 1.0
        if hasattr(self, "review_speed_label"):
            self.review_speed_label.configure(text=f"{speed:.2f}x")

    def _set_review_speed(self, speed: float) -> None:
        speed = max(0.05, min(2.0, float(speed)))
        if hasattr(self, "review_speed_var"):
            self.review_speed_var.set(speed)
        self._on_review_speed_change(str(speed))

    def _review_tick(self) -> None:
        if not self._review_playing or not self._review_active:
            return
        started = time.perf_counter()
        nxt = self._review_frame + 1
        if nxt > self._review_total:
            self._review_playing = False
            self.review_play_btn.configure(text="재생")
            self.review_frame_var.set(self._review_frame)
            return
        self._review_show_frame(nxt, sequential=True)
        target = 1000.0 / (self._review_fps * self._review_speed_value())
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        delay = max(1, int(target - elapsed_ms))
        self._review_job = self.root.after(delay, self._review_tick)

    def _jump_review_alert(self, direction: int) -> None:
        if not self._review_active or not self._review_alerts:
            return
        self._review_playing = False
        self.review_play_btn.configure(text="재생")
        frames = [int(a["frame_idx"]) for a in self._review_alerts]
        cur = self._review_frame
        if direction > 0:
            candidates = [f for f in frames if f > cur]
            target = candidates[0] if candidates else frames[0]
        else:
            candidates = [f for f in frames if f < cur]
            target = candidates[-1] if candidates else frames[-1]
        self._review_show_frame(target, sequential=False)

    def _paint_latest(self) -> None:
        if self._review_active and not self.tracking:
            return
        with self._visual_lock:
            visual = self._visual
        if visual is None:
            return
        tracked, states, events, recent, toggles, fw, fh, vis, predicted = visual
        pred_n = len(predicted) if predicted is not None else 0
        if hasattr(self, "preview_title"):
            self.preview_title.configure(
                text=f"추적 중  ·  검출 {len(tracked)}개  ·  예측 {pred_n}개"
            )
        if hasattr(self, "preview"):
            self._show_bgr(vis)
        if self.overlay is not None:
            self._paint_overlay(
                tracked, states, events, recent, fw, fh, toggles, predicted=predicted
            )
            for key_name in self.overlay.poll_keys():
                if key_name == "q":
                    self._stop()

    def _paint_overlay(
        self, tracked, states, events, recent, fw, fh, toggles=None, predicted=None
    ) -> None:
        if self.overlay is None:
            return
        if toggles is None:
            toggles = {
                "use_accel_spike": bool(self.spike_var.get()),
                "use_post_state_confirm": bool(self.confirm_var.get()),
                "use_proximity_tag": bool(self.prox_var.get()),
                "occlusion_hold_sec": float(self.occlusion_sec_var.get()),
            }
        self.overlay.present(
            tracked,
            states,
            events,
            recent,
            toggles,
            scale_x=self.overlay.width / max(fw, 1),
            scale_y=self.overlay.height / max(fh, 1),
            predicted=predicted or [],
        )

    def _close_overlay(self) -> None:
        if self.overlay is not None:
            self.overlay.close()
            self.overlay = None
        with self._visual_lock:
            self._visual = None

    def _on_close(self) -> None:
        if self.tracking or self.evaluating:
            self.stop_event.set()
        try:
            self._save_settings()
        except Exception:
            pass
        self._stop_preview()
        self._close_review()
        self._close_overlay()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    StudioApp().run()


if __name__ == "__main__":
    main()

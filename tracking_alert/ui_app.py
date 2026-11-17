"""메인 화면에서 캡처 모드 / 성능평가 모드를 골라 실행한다."""

from __future__ import annotations

import csv
import json
import os
import queue
import shutil
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
YOUTUBE_TEMP_DIR = ROOT / "video_library" / "youtube_temp"
YOUTUBE_PERM_DIR = ROOT / "video_library" / "youtube"


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
        self.root.geometry("1400x860")
        self.root.minsize(1280, 800)
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
        self._leaving = False
        self._closing = False
        self._last_eval_paint = 0.0
        self._last_capture_paint = 0.0
        self._preview_geom: dict[str, tuple[int, int, float]] = {}
        self._youtube_temp_path: Path | None = None
        self._youtube_busy = False
        # 이전 실행에서 남은 유튜브 임시 파일 정리
        self._cleanup_youtube_temp(keep=None)

        self.container = tk.Frame(self.root, bg="#1b1d21")
        self.container.pack(fill="both", expand=True)

        self.home_frame: tk.Frame | None = None
        self.capture_frame: tk.Frame | None = None
        self.eval_frame: tk.Frame | None = None
        self._settings = self._load_settings_file()

        self._build_home()
        self._show_home()
        self.root.bind_all("<Escape>", lambda _e: self._back_to_home())
        self._poll_messages()

    # ── 공통 스타일 ──────────────────────────────────────────
    def _clear_container(self) -> None:
        for child in self.container.winfo_children():
            child.pack_forget()

    def _show_home(self) -> None:
        self._stop_preview()
        self._close_overlay()
        with self._eval_visual_lock:
            self._eval_visuals = {}
            self._eval_focus_job = None
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
            width=22,
            sliderlength=36,
            command=self._on_review_seek,
        )
        self.review_scale.pack(fill="x", padx=12, pady=(2, 4))
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
            width=18,
            sliderlength=32,
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

        side = tk.Frame(body, bg=panel, width=420)
        side.pack(side="right", fill="y", padx=(12, 0))
        side.pack_propagate(False)

        side_canvas = tk.Canvas(side, bg=panel, highlightthickness=0, width=400)
        side_scroll = ttk.Scrollbar(side, orient="vertical", command=side_canvas.yview)
        side_canvas.configure(yscrollcommand=side_scroll.set)
        side_scroll.pack(side="right", fill="y")
        side_canvas.pack(side="left", fill="both", expand=True)
        side_inner = tk.Frame(side_canvas, bg=panel)
        side_window = side_canvas.create_window((0, 0), window=side_inner, anchor="nw")

        def _sync_side(_event=None) -> None:
            side_canvas.configure(scrollregion=side_canvas.bbox("all"))
            side_canvas.itemconfigure(side_window, width=max(side_canvas.winfo_width(), 380))

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
            wraplength=380,
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
            text="유튜브 URL",
            bg=panel,
            fg="#c5ccd6",
            font=("Malgun Gothic", 10),
        ).pack(anchor="w", padx=14, pady=(12, 4))
        self.youtube_var = tk.StringVar()
        yt_row = tk.Frame(self.file_frame, bg=panel)
        yt_row.pack(fill="x", padx=14)
        self.youtube_entry = tk.Entry(
            yt_row,
            textvariable=self.youtube_var,
            bg="#101216",
            fg=fg,
            insertbackground=fg,
            relief="flat",
            font=("Malgun Gothic", 9),
        )
        self.youtube_entry.pack(side="left", fill="x", expand=True, ipady=4)
        self.youtube_entry.bind("<Return>", lambda _e: self._download_youtube())
        self.youtube_btn = tk.Button(
            yt_row,
            text="가져오기",
            command=self._download_youtube,
            bg="#c0392b",
            fg="white",
            relief="flat",
            padx=10,
            pady=4,
        )
        self.youtube_btn.pack(side="left", padx=(6, 0))
        yt_action = tk.Frame(self.file_frame, bg=panel)
        yt_action.pack(fill="x", padx=14, pady=(4, 0))
        self.youtube_status_var = tk.StringVar(value="임시로 받은 뒤, 원하면 영구 저장")
        tk.Label(
            yt_action,
            textvariable=self.youtube_status_var,
            bg=panel,
            fg="#9aa3b2",
            font=("Malgun Gothic", 8),
            wraplength=300,
            justify="left",
            anchor="w",
        ).pack(side="left", fill="x", expand=True)
        self.youtube_save_btn = tk.Button(
            yt_action,
            text="영구 저장",
            command=self._save_youtube_permanent,
            bg="#3a4150",
            fg=fg,
            relief="flat",
            padx=8,
            pady=2,
            state="disabled",
        )
        self.youtube_save_btn.pack(side="right", padx=(6, 0))

        tk.Label(
            self.file_frame,
            text="영상 목록 (임시·저장소 · 더블클릭 선택)",
            bg=panel,
            fg="#c5ccd6",
            font=("Malgun Gothic", 10),
        ).pack(anchor="w", padx=14, pady=(12, 4))
        self.library_count_var = tk.StringVar(value="영상 0개")
        tk.Label(
            self.file_frame,
            textvariable=self.library_count_var,
            bg=panel,
            fg="#9aa3b2",
            font=("Malgun Gothic", 9),
        ).pack(anchor="w", padx=14, pady=(0, 4))
        lib_wrap = tk.Frame(self.file_frame, bg=panel)
        lib_wrap.pack(fill="both", expand=False, padx=14)
        self.library_list = tk.Listbox(
            lib_wrap,
            bg="#101216",
            fg="#f3f5f7",
            height=8,
            highlightthickness=1,
            highlightbackground="#3a4150",
            activestyle="dotbox",
            font=("Malgun Gothic", 9),
            exportselection=False,
            selectmode="browse",
        )
        lib_scroll = ttk.Scrollbar(
            lib_wrap, orient="vertical", command=self.library_list.yview
        )
        self.library_list.configure(yscrollcommand=lib_scroll.set)
        self.library_list.pack(side="left", fill="both", expand=True)
        lib_scroll.pack(side="right", fill="y")
        self.library_list.bind("<Double-Button-1>", self._on_library_select)
        self.library_list.bind("<Return>", self._on_library_select)
        self._library_paths: list[Path] = []
        pick_row = tk.Frame(self.file_frame, bg=panel)
        pick_row.pack(fill="x", padx=14, pady=(6, 0))
        tk.Button(
            pick_row,
            text="저장소에서 선택",
            command=self._pick_library_video,
            bg="#3a6ea5",
            fg="white",
            relief="flat",
            padx=8,
            pady=4,
        ).pack(side="left")
        tk.Button(
            pick_row,
            text="폴더 열기",
            command=self._open_library_folder,
            bg="#3a4150",
            fg=fg,
            relief="flat",
            padx=8,
            pady=4,
        ).pack(side="left", padx=(6, 0))
        tk.Label(
            self.file_frame,
            text="video_library · eval_data · sample_videos",
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
            text="← 메인 (Esc)",
            command=self._back_to_home,
            bg="#3a4150",
            fg=fg,
            relief="flat",
            padx=12,
            pady=6,
            font=("Malgun Gothic", 11, "bold"),
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
        side = tk.Frame(body, bg=panel, width=420)
        side.pack(side="right", fill="y", padx=(12, 0))
        side.pack_propagate(False)

        side_canvas = tk.Canvas(side, bg=panel, highlightthickness=0, width=400)
        side_scroll = ttk.Scrollbar(side, orient="vertical", command=side_canvas.yview)
        side_canvas.configure(yscrollcommand=side_scroll.set)
        side_scroll.pack(side="right", fill="y")
        side_canvas.pack(side="left", fill="both", expand=True)
        side_inner = tk.Frame(side_canvas, bg=panel)
        side_window = side_canvas.create_window((0, 0), window=side_inner, anchor="nw")

        def _sync_eval_side(_event=None) -> None:
            side_canvas.configure(scrollregion=side_canvas.bbox("all"))
            side_canvas.itemconfigure(side_window, width=max(side_canvas.winfo_width(), 380))

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

        self.eval_preview_host = tk.Frame(preview_wrap, bg="#101216", height=420)
        self.eval_preview_host.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.eval_preview_host.pack_propagate(False)
        self.eval_preview = tk.Label(
            self.eval_preview_host,
            bg="#101216",
            text="평가를 시작하면 영상이 여기에 표시됩니다",
            fg="#8b93a1",
            font=("Malgun Gothic", 12),
        )
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
        # 미리보기가 보이게 기본 1. GPU에서 워커를 늘리면 멈춘 것처럼 보일 수 있음
        self.eval_workers_var = tk.IntVar(
            value=int(saved_workers) if saved_workers else 1
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
            text="미리보기 볼 때는 1 권장  ·  높이면 속도↑ 화면 멈춤 가능",
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
        occ_on = tk.BooleanVar(
            value=bool(saved.get("occlusion", saved.get("occlusion_sec", 1.0) > 0))
        )
        occ = tk.DoubleVar(
            value=float(
                saved.get("occlusion_sec", 1.0)
                if float(saved.get("occlusion_sec", 1.0)) > 0
                else 1.0
            )
        )
        spike = tk.BooleanVar(value=bool(saved.get("spike", True)))
        confirm = tk.BooleanVar(value=bool(saved.get("confirm", True)))
        prox = tk.BooleanVar(value=bool(saved.get("prox", True)))
        small_boost = tk.BooleanVar(value=bool(saved.get("small_object_boost", True)))
        sound_alert = tk.BooleanVar(value=bool(saved.get("sound_alert", True)))

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
        conf_row = tk.Frame(parent, bg=panel)
        conf_row.pack(fill="x", padx=14)
        conf_scale = tk.Scale(
            conf_row,
            from_=0.001,
            to=0.8,
            resolution=0.001,
            orient="horizontal",
            variable=conf,
            bg=panel,
            fg=fg,
            highlightthickness=0,
            troughcolor="#101216",
            showvalue=0,
        )
        conf_scale.pack(side="left", fill="x", expand=True)
        conf_entry = tk.Entry(
            conf_row,
            width=7,
            bg="#101216",
            fg=fg,
            insertbackground=fg,
            relief="flat",
            font=("Consolas", 11),
            justify="center",
        )
        conf_entry.pack(side="right", padx=(8, 0))
        conf_entry.insert(0, f"{float(conf.get()):.3f}")

        def _clamp_conf(value: float) -> float:
            return float(max(0.001, min(0.8, value)))

        def _sync_conf_entry(*_args) -> None:
            try:
                text = f"{_clamp_conf(float(conf.get())):.3f}"
            except Exception:
                return
            if conf_entry.get() != text:
                conf_entry.delete(0, "end")
                conf_entry.insert(0, text)

        def _apply_conf_entry(_event=None) -> None:
            raw = conf_entry.get().strip().replace(",", ".")
            try:
                value = _clamp_conf(float(raw))
            except Exception:
                _sync_conf_entry()
                return
            conf.set(value)
            _sync_conf_entry()

        conf.trace_add("write", lambda *_: _sync_conf_entry())
        conf_entry.bind("<Return>", _apply_conf_entry)
        conf_entry.bind("<FocusOut>", _apply_conf_entry)
        tk.Label(
            parent,
            text="슬라이더 또는 직접 입력  ·  0.001~0.8  ·  낮출수록 더 많이 잡힘",
            bg=panel,
            fg="#9aa3b2",
            font=("Malgun Gothic", 9),
        ).pack(anchor="w", padx=14)

        tk.Checkbutton(
            parent,
            text="가림 예측",
            variable=occ_on,
            bg=panel,
            fg=fg,
            selectcolor="#101216",
            activebackground=panel,
            activeforeground=fg,
            font=("Malgun Gothic", 10),
        ).pack(anchor="w", padx=14, pady=(12, 0))
        label("가림 예측 유지 시간(초)")
        occ_scale = tk.Scale(
            parent,
            from_=0.5,
            to=3.0,
            resolution=0.5,
            orient="horizontal",
            variable=occ,
            bg=panel,
            fg=fg,
            highlightthickness=0,
            troughcolor="#101216",
            state="normal" if occ_on.get() else "disabled",
        )
        occ_scale.pack(fill="x", padx=14)
        tk.Label(
            parent,
            text="켜면 주황 점선으로 가려진 위치 예측",
            bg=panel,
            fg="#9aa3b2",
            font=("Malgun Gothic", 9),
        ).pack(anchor="w", padx=14)

        def _sync_occ_scale(*_args) -> None:
            on = bool(occ_on.get())
            occ_scale.configure(state="normal" if on else "disabled")
            if on and float(occ.get()) <= 0:
                occ.set(1.0)

        occ_on.trace_add("write", _sync_occ_scale)

        for text, var in (
            ("사고 알림음", sound_alert),
            ("소형/저화질 검출 강화", small_boost),
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
        tk.Label(
            parent,
            text="사고 확정 시 Windows 알림음 · 끄면 소리만 음소거",
            bg=panel,
            fg="#9aa3b2",
            font=("Malgun Gothic", 9),
        ).pack(anchor="w", padx=14)
        tk.Label(
            parent,
            text="작은 차량·낮은 해상도에서 검출을 올립니다 (속도↓)",
            bg=panel,
            fg="#9aa3b2",
            font=("Malgun Gothic", 9),
        ).pack(anchor="w", padx=14)

        if prefix == "cap":
            self.sensitivity_var = sens
            self.conf_var = conf
            self.occlusion_var = occ_on
            self.occlusion_sec_var = occ
            self.spike_var = spike
            self.confirm_var = confirm
            self.prox_var = prox
            self.small_object_boost_var = small_boost
            self.sound_alert_var = sound_alert
        else:
            self.eval_sensitivity_var = sens
            self.eval_conf_var = conf
            self.eval_occlusion_var = occ_on
            self.eval_occlusion_sec_var = occ
            self.eval_spike_var = spike
            self.eval_confirm_var = confirm
            self.eval_prox_var = prox
            self.eval_small_object_boost_var = small_boost
            self.eval_sound_alert_var = sound_alert

    # ── 설정 저장 / 불러오기 ─────────────────────────────────
    def _default_settings(self) -> dict:
        return {
            "cap": {
                "sensitivity": 3,
                "conf": 0.05,
                "occlusion": True,
                "occlusion_sec": 1.0,
                "small_object_boost": True,
                "sound_alert": True,
                "spike": True,
                "confirm": True,
                "prox": True,
            },
            "eval": {
                "sensitivity": 3,
                "conf": 0.05,
                "occlusion": True,
                "occlusion_sec": 1.0,
                "small_object_boost": True,
                "sound_alert": False,
                "spike": True,
                "confirm": True,
                "prox": True,
                "workers": 1,
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
                "occlusion": bool(self.occlusion_var.get()),
                "occlusion_sec": float(self.occlusion_sec_var.get()),
                "small_object_boost": bool(self.small_object_boost_var.get()),
                "sound_alert": bool(self.sound_alert_var.get()),
                "spike": bool(self.spike_var.get()),
                "confirm": bool(self.confirm_var.get()),
                "prox": bool(self.prox_var.get()),
            }
        if hasattr(self, "eval_sensitivity_var"):
            data["eval"] = {
                "sensitivity": int(self.eval_sensitivity_var.get()),
                "conf": float(self.eval_conf_var.get()),
                "occlusion": bool(self.eval_occlusion_var.get()),
                "occlusion_sec": float(self.eval_occlusion_sec_var.get()),
                "small_object_boost": bool(self.eval_small_object_boost_var.get()),
                "sound_alert": bool(self.eval_sound_alert_var.get()),
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
                occ_sec = float(block.get("occlusion_sec", 1.0))
                self.occlusion_var.set(
                    bool(block.get("occlusion", occ_sec > 0))
                )
                self.occlusion_sec_var.set(occ_sec if occ_sec > 0 else 1.0)
                self.small_object_boost_var.set(
                    bool(block.get("small_object_boost", True))
                )
                self.sound_alert_var.set(bool(block.get("sound_alert", True)))
                self.spike_var.set(bool(block.get("spike", True)))
                self.confirm_var.set(bool(block.get("confirm", True)))
                self.prox_var.set(bool(block.get("prox", True)))
            if key == "eval" and hasattr(self, "eval_sensitivity_var"):
                self.eval_sensitivity_var.set(int(block.get("sensitivity", 3)))
                self.eval_conf_var.set(float(block.get("conf", 0.05)))
                occ_sec = float(block.get("occlusion_sec", 1.0))
                self.eval_occlusion_var.set(
                    bool(block.get("occlusion", occ_sec > 0))
                )
                self.eval_occlusion_sec_var.set(occ_sec if occ_sec > 0 else 1.0)
                self.eval_small_object_boost_var.set(
                    bool(block.get("small_object_boost", True))
                )
                self.eval_sound_alert_var.set(bool(block.get("sound_alert", False)))
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
    def _abort_background_work(self) -> None:
        """평가/추적 워커에 중지 신호를 보내고 UI 상태를 즉시 해제한다."""
        try:
            self.stop_event.set()
        except Exception:
            pass
        self.evaluating = False
        self.tracking = False
        with self._eval_visual_lock:
            self._eval_visuals = {}
            self._eval_focus_job = None
        with self._visual_lock:
            self._visual = None

    def _back_to_home(self, _event=None) -> None:
        if self._closing:
            return
        # 미리보기 인코딩이 메인스레드를 점유해도 바로 빠져나오게 한다.
        self._leaving = True
        self._abort_background_work()
        try:
            if hasattr(self, "eval_btn"):
                self.eval_btn.configure(state="normal", text="평가 시작")
            if hasattr(self, "eval_stop_btn"):
                self.eval_stop_btn.configure(
                    state="disabled", bg="#3a4150", fg="#f3f5f7"
                )
            if hasattr(self, "start_btn"):
                self.start_btn.configure(text="추적 시작", bg="#2f9e44", state="normal")
        except Exception:
            pass
        self._close_overlay()
        self._close_review()
        # 클릭 핸들러가 끝난 뒤 화면 전환 (UI 멈춤 완화)
        self.root.after(10, self._finish_back_to_home)

    def _finish_back_to_home(self) -> None:
        try:
            self._show_home()
        finally:
            self._leaving = False

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

    @staticmethod
    def _is_youtube_url(url: str) -> bool:
        text = (url or "").strip().lower()
        return any(
            host in text
            for host in (
                "youtube.com/",
                "youtu.be/",
                "youtube.com/shorts/",
                "m.youtube.com/",
            )
        )

    @staticmethod
    def _is_under_dir(path: Path, folder: Path) -> bool:
        try:
            path.resolve().relative_to(folder.resolve())
            return True
        except Exception:
            return False

    def _is_youtube_temp(self, path: Path | None) -> bool:
        if path is None:
            return False
        return self._is_under_dir(path, YOUTUBE_TEMP_DIR)

    def _set_youtube_save_enabled(self, enabled: bool) -> None:
        if not hasattr(self, "youtube_save_btn"):
            return
        self.youtube_save_btn.configure(state="normal" if enabled else "disabled")

    def _cleanup_youtube_temp(self, keep: Path | None = None) -> None:
        """임시 유튜브 파일을 지운다. keep만 남긴다."""
        try:
            YOUTUBE_TEMP_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            return
        keep_key = ""
        if keep is not None:
            try:
                keep_key = str(keep.resolve()).lower()
            except Exception:
                keep_key = str(keep).lower()
        for path in YOUTUBE_TEMP_DIR.iterdir():
            try:
                if not path.is_file():
                    continue
                key = str(path.resolve()).lower()
                if keep_key and key == keep_key:
                    continue
                path.unlink(missing_ok=True)
            except Exception:
                continue

    def _save_youtube_permanent(self) -> None:
        if self.tracking or self.evaluating:
            return
        src = self._youtube_temp_path
        if src is None or not src.exists() or not self._is_youtube_temp(src):
            if hasattr(self, "youtube_status_var"):
                self.youtube_status_var.set("영구 저장할 임시 영상이 없습니다")
            self._set_youtube_save_enabled(False)
            return
        try:
            YOUTUBE_PERM_DIR.mkdir(parents=True, exist_ok=True)
            dest = YOUTUBE_PERM_DIR / src.name
            if dest.exists():
                stem, suffix = dest.stem, dest.suffix
                n = 2
                while True:
                    cand = YOUTUBE_PERM_DIR / f"{stem}_{n}{suffix}"
                    if not cand.exists():
                        dest = cand
                        break
                    n += 1
            shutil.move(str(src), str(dest))
            self._youtube_temp_path = None
            self._set_video_path(dest, source_label="유튜브(영구)")
            self._set_youtube_save_enabled(False)
            self._refresh_video_library()
            if hasattr(self, "youtube_status_var"):
                self.youtube_status_var.set(f"영구 저장됨: {dest.name}")
            if hasattr(self, "log"):
                self.log.insert(0, f"유튜브 영구 저장: {dest}")
        except Exception as exc:
            if hasattr(self, "youtube_status_var"):
                self.youtube_status_var.set(f"영구 저장 실패: {str(exc)[:60]}")
            if hasattr(self, "log"):
                self.log.insert(0, f"유튜브 영구 저장 실패: {exc}")

    def _download_youtube(self) -> None:
        if self.tracking or self.evaluating:
            return
        if getattr(self, "_youtube_busy", False):
            return
        url = self.youtube_var.get().strip() if hasattr(self, "youtube_var") else ""
        if not url:
            if hasattr(self, "youtube_status_var"):
                self.youtube_status_var.set("유튜브 링크를 입력하세요")
            return
        if not self._is_youtube_url(url):
            if hasattr(self, "youtube_status_var"):
                self.youtube_status_var.set("유튜브 주소가 아닌 것 같습니다")
            return

        # 새 다운로드 전: 이전 임시 파일 정리 (현재 선택 임시본 포함 → 곧 교체)
        self._cleanup_youtube_temp(keep=None)
        self._youtube_temp_path = None
        self._set_youtube_save_enabled(False)

        out_dir = YOUTUBE_TEMP_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        self._youtube_busy = True
        if hasattr(self, "youtube_btn"):
            self.youtube_btn.configure(state="disabled", text="받는 중")
        if hasattr(self, "youtube_status_var"):
            self.youtube_status_var.set("임시 다운로드 중…")
        if hasattr(self, "log"):
            self.log.insert(0, f"유튜브 임시 다운로드 시작: {url}")

        def work() -> None:
            try:
                from yt_dlp import YoutubeDL

                outtmpl = str(out_dir / "%(title).80B [%(id)s].%(ext)s")
                ydl_opts = {
                    "format": (
                        "bestvideo[ext=mp4][vcodec^=avc1][height<=720]"
                        "/bestvideo[ext=mp4][height<=720]"
                        "/best[ext=mp4][height<=720]"
                        "/best[height<=720]"
                        "/best"
                    ),
                    "outtmpl": outtmpl,
                    "overwrites": True,
                    "noplaylist": True,
                    "quiet": True,
                    "no_warnings": True,
                }
                with YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    if info is None:
                        raise RuntimeError("영상 정보를 가져오지 못했습니다")
                    if "entries" in info:
                        entries = [e for e in info["entries"] if e]
                        if not entries:
                            raise RuntimeError("재생목록에서 영상을 찾지 못했습니다")
                        info = entries[0]
                    path = Path(ydl.prepare_filename(info))
                    if not path.exists():
                        vid = str(info.get("id", ""))
                        candidates = [
                            p
                            for p in out_dir.glob("*")
                            if p.is_file() and (not vid or vid in p.name)
                        ]
                        if not candidates:
                            raise RuntimeError("다운로드 파일을 찾지 못했습니다")
                        path = max(candidates, key=lambda p: p.stat().st_mtime)
                self.messages.put(("youtube_done", {"path": str(path), "ok": True}))
            except Exception as exc:
                self.messages.put(("youtube_done", {"ok": False, "error": str(exc)}))

        threading.Thread(target=work, daemon=True).start()

    def _library_roots(self) -> list[tuple[str, Path, bool]]:
        """(표시태그, 경로, 하위폴더 검색여부)"""
        roots = [
            ("임시", YOUTUBE_TEMP_DIR, False),
            ("library", ROOT / "video_library", True),
            ("accident", ROOT / "eval_data" / "accident", False),
            ("normal", ROOT / "eval_data" / "normal", False),
            ("sample", ROOT / "sample_videos", True),
            ("sample", ROOT.parent / "sample_videos", True),
        ]
        # MOT_project 복사본에서 실행해도 원본 eval_data를 찾게 함
        sibling = ROOT.parent / "tracking_alert"
        if sibling.resolve() != ROOT.resolve():
            roots.extend(
                [
                    ("accident", sibling / "eval_data" / "accident", False),
                    ("normal", sibling / "eval_data" / "normal", False),
                    ("library", sibling / "video_library", True),
                ]
            )
        return roots

    def _refresh_video_library(self, select_path: Path | None = None) -> None:
        if not hasattr(self, "library_list"):
            return
        self.library_list.delete(0, tk.END)
        self._library_paths = []
        seen: set[str] = set()
        temp_items: list[tuple[str, Path]] = []
        other_items: list[tuple[str, Path]] = []
        for tag, folder, recursive in self._library_roots():
            try:
                folder.mkdir(parents=True, exist_ok=True)
            except Exception:
                continue
            for path in list_videos(folder, recursive=recursive):
                key = str(path.resolve()).lower()
                if key in seen:
                    continue
                seen.add(key)
                is_temp = "youtube_temp" in {p.lower() for p in path.parts}
                # library 재귀 검색에서 임시 폴더는 [임시] 루트만 사용
                if is_temp and tag != "임시":
                    continue
                display_tag = "임시" if is_temp else tag
                if path.parent.name.lower() == "youtube" and not is_temp:
                    display_tag = "유튜브"
                item = (display_tag, path)
                if is_temp:
                    temp_items.append(item)
                else:
                    other_items.append(item)
        for display_tag, path in temp_items + other_items:
            self._library_paths.append(path)
            self.library_list.insert(tk.END, f"[{display_tag}] {path.name}")
        if hasattr(self, "library_count_var"):
            self.library_count_var.set(f"영상 {len(self._library_paths)}개")
        if not self._library_paths:
            self.library_list.insert(tk.END, "(저장소에 영상 없음)")
            if hasattr(self, "log"):
                self.log.insert(
                    0,
                    f"영상 없음. 이 폴더에 mp4를 넣으세요: {ROOT / 'video_library'}",
                )
        elif hasattr(self, "log"):
            self.log.insert(0, f"저장소 목록 새로고침: {len(self._library_paths)}개")
        if select_path is not None:
            self._highlight_library_path(select_path)

    def _highlight_library_path(self, path: Path) -> None:
        if not hasattr(self, "library_list") or not self._library_paths:
            return
        try:
            target = path.resolve()
        except Exception:
            target = path
        for i, p in enumerate(self._library_paths):
            try:
                same = p.resolve() == target
            except Exception:
                same = p == path
            if same:
                self.library_list.selection_clear(0, tk.END)
                self.library_list.selection_set(i)
                self.library_list.see(i)
                break

    def _open_library_folder(self) -> None:
        folder = ROOT / "video_library"
        folder.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(folder))  # type: ignore[attr-defined]
        except Exception as exc:
            if hasattr(self, "log"):
                self.log.insert(0, f"폴더 열기 실패: {exc}")

    def _pick_library_video(self) -> None:
        if self.tracking or self.evaluating:
            return
        if not hasattr(self, "library_list"):
            return
        sel = self.library_list.curselection()
        if not sel:
            if hasattr(self, "log"):
                self.log.insert(0, "저장소 목록에서 영상을 클릭해 고르세요")
            return
        index = int(sel[0])
        if index < 0 or index >= len(self._library_paths):
            if hasattr(self, "log"):
                self.log.insert(0, "선택 항목이 유효하지 않습니다. 새로고침 후 다시 선택하세요")
            return
        self._set_video_path(self._library_paths[index], source_label="저장소")

    def _on_library_select(self, _event=None) -> None:
        self._pick_library_video()

    def _set_video_path(self, path: Path, source_label: str = "") -> None:
        prev = self.video_path
        self.video_path = path
        if self._is_youtube_temp(path):
            # 다른 임시 영상을 고르면 이전 임시본만 정리
            self._cleanup_youtube_temp(keep=path)
            self._youtube_temp_path = path
            self._set_youtube_save_enabled(True)
            label = source_label or "유튜브(임시)"
        else:
            # 영구/다른 영상을 고르면 임시본 삭제
            if (
                prev is not None
                and self._is_youtube_temp(prev)
                and prev.resolve() != path.resolve()
            ):
                self._cleanup_youtube_temp(keep=None)
            self._youtube_temp_path = None
            self._set_youtube_save_enabled(False)
            label = source_label
        prefix = f"{label}: " if label else ""
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
        small_object_boost: bool = True,
        sound_alert: bool = True,
        output_dir: str = "outputs",
    ) -> AppConfig:
        cfg = AppConfig(
            conf=float(max(0.001, min(0.8, conf))),
            output_dir=output_dir,
            occlusion_hold_sec=occlusion_sec,
            small_object_boost=small_object_boost,
            imgsz=1280 if small_object_boost else 960,
            sound_alert=bool(sound_alert),
            event=EventConfig(
                use_accel_spike=spike,
                use_post_state_confirm=confirm,
                use_proximity_tag=prox,
            ),
        )
        cfg.event.apply_sensitivity(sensitivity)
        return cfg

    def _occlusion_hold_from_ui(self, *, eval_mode: bool = False) -> float:
        if eval_mode:
            if not bool(self.eval_occlusion_var.get()):
                return 0.0
            return max(0.5, float(self.eval_occlusion_sec_var.get()))
        if not bool(self.occlusion_var.get()):
            return 0.0
        return max(0.5, float(self.occlusion_sec_var.get()))

    def _make_cfg(self) -> AppConfig:
        return self._make_cfg_from(
            int(self.sensitivity_var.get()),
            float(self.conf_var.get()),
            self._occlusion_hold_from_ui(eval_mode=False),
            bool(self.spike_var.get()),
            bool(self.confirm_var.get()),
            bool(self.prox_var.get()),
            bool(self.small_object_boost_var.get()),
            bool(self.sound_alert_var.get()),
        )

    def _make_eval_cfg(self) -> AppConfig:
        return self._make_cfg_from(
            int(self.eval_sensitivity_var.get()),
            float(self.eval_conf_var.get()),
            self._occlusion_hold_from_ui(eval_mode=True),
            bool(self.eval_spike_var.get()),
            bool(self.eval_confirm_var.get()),
            bool(self.eval_prox_var.get()),
            bool(self.eval_small_object_boost_var.get()),
            bool(self.eval_sound_alert_var.get()),
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

    def _host_size(self, size_ref: tk.Misc, cache_key: str) -> tuple[int, int]:
        """호스트 크기. update_idletasks는 가끔만 호출해 UI가 덜 끊기게 한다."""
        now = time.perf_counter()
        cached = self._preview_geom.get(cache_key)
        if cached is not None and now - cached[2] < 0.8:
            return cached[0], cached[1]
        try:
            if cached is None:
                size_ref.update_idletasks()
        except Exception:
            pass
        canvas_w = max(int(size_ref.winfo_width()), 320)
        canvas_h = max(int(size_ref.winfo_height()), 180)
        self._preview_geom[cache_key] = (canvas_w, canvas_h, now)
        return canvas_w, canvas_h

    @staticmethod
    def _bgr_to_photo(bgr: np.ndarray) -> tk.PhotoImage | None:
        """OpenCV BGR → Tk PhotoImage.

        주의: cv2.imencode('.png', …)는 BGR 입력을 가정한다.
        RGB로 바꾼 뒤 PNG로 넣으면 R↔B가 뒤집혀 '반전 필터'처럼 보인다.
        PPM(P6)은 RGB를 그대로 쓰므로 색이 맞고 PNG 압축보다 훨씬 빠르다.
        """
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        h, w = rgb.shape[:2]
        header = f"P6 {w} {h} 255\n".encode("ascii")
        try:
            return tk.PhotoImage(data=header + rgb.tobytes())
        except tk.TclError:
            # PPM 실패 시에만 PNG (BGR 그대로 인코딩)
            ok, buf = cv2.imencode(
                ".png",
                bgr,
                [int(cv2.IMWRITE_PNG_COMPRESSION), 1],
            )
            if not ok:
                return None
            try:
                return tk.PhotoImage(data=buf.tobytes())
            except tk.TclError:
                return None

    def _letterbox_bgr(
        self, frame: np.ndarray, canvas_w: int, canvas_h: int
    ) -> np.ndarray:
        scale = min(canvas_w / frame.shape[1], canvas_h / frame.shape[0])
        nw = max(1, int(frame.shape[1] * scale))
        nh = max(1, int(frame.shape[0] * scale))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        view = cv2.resize(frame, (nw, nh), interpolation=interp)
        canvas = np.full((canvas_h, canvas_w, 3), 16, dtype=np.uint8)
        x0 = (canvas_w - nw) // 2
        y0 = (canvas_h - nh) // 2
        canvas[y0 : y0 + nh, x0 : x0 + nw] = view
        return canvas

    def _show_bgr_on(
        self,
        widget: tk.Label,
        frame: np.ndarray,
        image_attr: str,
        letterbox: bool = False,
        size_widget: tk.Misc | None = None,
    ) -> None:
        if self._leaving or self._closing:
            return
        size_ref = size_widget or widget
        canvas_w, canvas_h = self._host_size(size_ref, image_attr)
        if letterbox:
            view = self._letterbox_bgr(frame, canvas_w, canvas_h)
        else:
            scale = min(canvas_w / frame.shape[1], canvas_h / frame.shape[0])
            view = cv2.resize(
                frame,
                (
                    max(1, int(frame.shape[1] * scale)),
                    max(1, int(frame.shape[0] * scale)),
                ),
                interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
            )
        photo = self._bgr_to_photo(view)
        if photo is None:
            return
        setattr(self, image_attr, photo)
        widget.configure(image=photo, text="")

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
        if not self.evaluating and not self.stop_event.is_set():
            return
        self.stop_event.set()
        self._set_eval_status("중지 중", "#f0b429")
        if hasattr(self, "eval_stop_btn"):
            self.eval_stop_btn.configure(state="disabled")
        if hasattr(self, "eval_log"):
            self.eval_log.insert(0, "평가 중지 요청…")
        # 워커가 바로 안 끝나도 UI는 다시 조작 가능하게
        self.root.after(500, self._force_eval_idle)

    def _force_eval_idle(self) -> None:
        if self._closing or self._leaving:
            return
        if not self.evaluating:
            return
        # stop 후에도 evaluating이 남아 있으면 강제로 풀어 메인/종료가 막히지 않게 한다
        if self.stop_event.is_set():
            self.evaluating = False
            with self._eval_visual_lock:
                self._eval_visuals = {}
            if hasattr(self, "eval_btn"):
                self.eval_btn.configure(state="normal", text="평가 시작")
            self._set_eval_status("평가 중지됨", "#f0b429")
            if hasattr(self, "eval_log"):
                self.eval_log.insert(0, "평가 UI 잠금 해제 (백그라운드는 곧 종료)")

    def _run_eval(self) -> None:
        if self.tracking:
            return
        if self.evaluating:
            self.eval_log.insert(0, "이미 평가 중입니다. 중지 후 다시 시작하세요")
            return
        data_root = ROOT / "eval_data"
        accident = list_videos(data_root / "accident")
        normal = list_videos(data_root / "normal")
        if not accident and not normal:
            self.eval_log.insert(0, "오류: eval_data/accident, normal 에 영상이 없습니다")
            self.eval_log.insert(0, f"경로: {data_root}")
            return

        # 미리보기용으로는 1이 안정적. 요청값이 있으면 존중하되 GPU 폭주 방지로 상한 유지
        requested = (
            int(self.eval_workers_var.get())
            if hasattr(self, "eval_workers_var")
            else 1
        )
        workers = default_workers(requested)
        self.evaluating = True
        self.stop_event = threading.Event()
        with self._eval_visual_lock:
            self._eval_visuals = {}
            self._eval_focus_job = None
        self.eval_btn.configure(state="disabled", text="평가 중...")
        if hasattr(self, "eval_stop_btn"):
            self.eval_stop_btn.configure(state="normal", bg="#d64545", fg="white")
        self._set_eval_status(f"평가 중 ×{workers}", "#f0b429")
        if hasattr(self, "eval_preview_title"):
            self.eval_preview_title.configure(text=f"평가 중  ·  워커 {workers}")
        if hasattr(self, "eval_progress_var"):
            self.eval_progress_var.set("첫 프레임 준비 중... (모델 로딩)")
        if hasattr(self, "eval_metrics_var"):
            self.eval_metrics_var.set("진행 중 지표: -")
        if hasattr(self, "eval_preview"):
            self.eval_preview.configure(
                image="",
                text="모델 로딩 후 영상이 표시됩니다\n워커가 많으면 검게 보일 수 있어요 → 1로 낮춰보세요",
                fg="#c5ccd6",
                font=("Malgun Gothic", 11),
                justify="center",
            )
            self.eval_preview_image = None
        self.eval_log.insert(
            0,
            f"평가 시작  accident={len(accident)} normal={len(normal)} workers={workers}",
        )
        cfg = self._make_eval_cfg()
        # 미리보기 중엔 소형객체 확대가 GPU를 너무 먹을 수 있어 워커>1이면 끔
        if workers > 1:
            cfg.small_object_boost = False
            cfg.imgsz = 960
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
                    visual_stride=1 if workers == 1 else 3,
                    visual_max_width=960 if workers == 1 else 640,
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
        if self._closing:
            return
        while True:
            try:
                kind, payload = self.messages.get_nowait()
            except queue.Empty:
                break
            if self._leaving:
                continue
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
            elif kind == "youtube_done":
                self._youtube_busy = False
                if hasattr(self, "youtube_btn"):
                    self.youtube_btn.configure(state="normal", text="가져오기")
                if not payload or not payload.get("ok"):
                    err = (payload or {}).get("error", "알 수 없는 오류")
                    if hasattr(self, "youtube_status_var"):
                        self.youtube_status_var.set(f"실패: {err[:80]}")
                    self._set_youtube_save_enabled(False)
                    if hasattr(self, "log"):
                        self.log.insert(0, f"유튜브 다운로드 실패: {err}")
                else:
                    path = Path(str(payload["path"]))
                    if path.exists():
                        self._youtube_temp_path = path
                        self._set_video_path(path, source_label="유튜브(임시)")
                        self._refresh_video_library(select_path=path)
                        if hasattr(self, "youtube_status_var"):
                            self.youtube_status_var.set(
                                f"임시 저장됨: {path.name}  → 필요하면 영구 저장"
                            )
                        if hasattr(self, "log"):
                            self.log.insert(0, f"유튜브 임시 저장: {path}")
                    else:
                        if hasattr(self, "youtube_status_var"):
                            self.youtube_status_var.set("다운로드 파일을 찾지 못했습니다")
                        self._set_youtube_save_enabled(False)
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
        if not self._leaving and not self._closing:
            if self.mode == "capture":
                self._paint_latest()
            elif self.mode == "eval":
                self._paint_eval_latest()
        if not self._closing:
            self.root.after(80, self._poll_messages)

    def _paint_eval_latest(self) -> None:
        if self._leaving or self._closing or self.mode != "eval":
            return
        now = time.perf_counter()
        # 미리보기 인코딩이 클릭/종료를 먹지 않게 초당 약 4회로 제한
        if now - self._last_eval_paint < 0.25:
            return
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
        self._last_eval_paint = now
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
        """검토 재생용 빠른 표시 (색 유지 + PPM)."""
        if not hasattr(self, "preview"):
            return
        size_ref = getattr(self, "preview_host", self.preview)
        canvas_w, canvas_h = self._host_size(size_ref, "preview_image")
        view = self._letterbox_bgr(frame, canvas_w, canvas_h)
        photo = self._bgr_to_photo(view)
        if photo is None:
            return
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
        now = time.perf_counter()
        # 미리보기 과도 갱신(PNG 시절)이 추적을 더 끊기게 만들어 초당 ~12회로 제한
        paint_ui = now - self._last_capture_paint >= 0.08
        tracked, states, events, recent, toggles, fw, fh, vis, predicted = visual
        pred_n = len(predicted) if predicted is not None else 0
        if paint_ui and hasattr(self, "preview_title"):
            self.preview_title.configure(
                text=f"추적 중  ·  검출 {len(tracked)}개  ·  예측 {pred_n}개"
            )
        if paint_ui and hasattr(self, "preview"):
            self._show_bgr(vis)
            self._last_capture_paint = now
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
                "occlusion_hold_sec": self._occlusion_hold_from_ui(eval_mode=False),
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
        if self._closing:
            return
        self._closing = True
        self._leaving = True
        self._abort_background_work()
        self._stop_preview()
        try:
            self._close_review()
        except Exception:
            pass
        try:
            self._close_overlay()
        except Exception:
            pass
        # 유튜브 임시 파일은 종료 시 삭제
        try:
            self._cleanup_youtube_temp(keep=None)
        except Exception:
            pass
        # 저장은 짧게 시도하고, 실패해도 종료는 막지 않는다.
        try:
            data = self._collect_settings()
            SETTINGS_PATH.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
        # destroy가 막히면 강제 종료
        self.root.after(20, self._destroy_app)
        self.root.after(1500, self._force_exit)

    def _destroy_app(self) -> None:
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass

    def _force_exit(self) -> None:
        # 워커/Tk가 살아 있어도 프로세스 종료
        os._exit(0)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    StudioApp().run()


if __name__ == "__main__":
    main()

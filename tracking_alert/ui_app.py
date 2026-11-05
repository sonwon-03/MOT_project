"""OBS처럼 소스를 고른 뒤, 시작을 눌렀을 때만 추적한다."""

from __future__ import annotations

import queue
import sys
import threading
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import AppConfig, EventConfig
from main import run_screen
from src.overlay_win import ScreenOverlay, enable_dpi_awareness


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
        self.root.geometry("980x640")
        self.root.configure(bg="#1b1d21")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.region: tuple[int, int, int, int] | None = None
        self.tracking = False
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.messages: queue.Queue[tuple] = queue.Queue()
        self.preview_image: tk.PhotoImage | None = None
        self.overlay: ScreenOverlay | None = None
        self._visual = None
        self._visual_lock = threading.Lock()
        self._preview_job: str | None = None

        self._build()
        self._set_status("대기", "#8b93a1")
        self._schedule_preview()
        self._poll_messages()

    def _build(self) -> None:
        bg = "#1b1d21"
        panel = "#262a31"
        fg = "#f3f5f7"

        header = tk.Frame(self.root, bg=bg)
        header.pack(fill="x", padx=16, pady=(14, 6))
        tk.Label(
            header,
            text="Tracking Alert",
            bg=bg,
            fg=fg,
            font=("Malgun Gothic", 18, "bold"),
        ).pack(side="left")
        self.status = tk.Label(
            header, text="대기", bg="#2c313a", fg="#8b93a1", padx=12, pady=4,
            font=("Malgun Gothic", 10),
        )
        self.status.pack(side="right")

        body = tk.Frame(self.root, bg=bg)
        body.pack(fill="both", expand=True, padx=16, pady=8)

        preview_wrap = tk.Frame(body, bg=panel, width=620, height=520)
        preview_wrap.pack(side="left", fill="both", expand=True)
        preview_wrap.pack_propagate(False)
        self.preview_title = tk.Label(
            preview_wrap, text="미리보기  (추적 전)", bg=panel, fg="#9aa3b2",
            font=("Malgun Gothic", 10),
        )
        self.preview_title.pack(anchor="w", padx=12, pady=(10, 4))
        self.preview = tk.Label(preview_wrap, bg="#101216")
        self.preview.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        side = tk.Frame(body, bg=panel, width=300)
        side.pack(side="right", fill="y", padx=(12, 0))
        side.pack_propagate(False)

        def label(text: str) -> None:
            tk.Label(side, text=text, bg=panel, fg="#c5ccd6", font=("Malgun Gothic", 10)).pack(
                anchor="w", padx=14, pady=(14, 4)
            )

        label("모니터")
        self.monitor_var = tk.StringVar()
        names = [
            f"{m['index']}: {m['width']}x{m['height']} ({m['left']},{m['top']})"
            for m in self.monitors
        ]
        self.monitor_var.set(names[0])
        self.monitor_box = ttk.Combobox(
            side, textvariable=self.monitor_var, values=names, state="readonly"
        )
        self.monitor_box.pack(fill="x", padx=14)
        self.monitor_box.bind("<<ComboboxSelected>>", lambda _e: self._reset_region())

        label("캡처 영역")
        self.region_var = tk.StringVar(value="모니터 전체")
        tk.Label(
            side, textvariable=self.region_var, bg=panel, fg=fg, anchor="w",
            font=("Malgun Gothic", 10),
        ).pack(fill="x", padx=14)
        row = tk.Frame(side, bg=panel)
        row.pack(fill="x", padx=14, pady=(6, 0))
        tk.Button(
            row, text="영역 선택", command=self._choose_region,
            bg="#3a4150", fg=fg, relief="flat", padx=8, pady=4,
        ).pack(side="left")
        tk.Button(
            row, text="전체", command=self._reset_region,
            bg="#3a4150", fg=fg, relief="flat", padx=8, pady=4,
        ).pack(side="left", padx=(6, 0))

        label("사고 판정 민감도")
        self.sensitivity_var = tk.IntVar(value=3)
        tk.Scale(
            side, from_=1, to=5, resolution=1, orient="horizontal",
            variable=self.sensitivity_var, bg=panel, fg=fg, highlightthickness=0,
            troughcolor="#101216",
        ).pack(fill="x", padx=14)
        tk.Label(
            side, text="1 둔감    ·    5 민감", bg=panel, fg="#9aa3b2",
            font=("Malgun Gothic", 9),
        ).pack(anchor="w", padx=14)

        label("검출 신뢰도")
        self.conf_var = tk.DoubleVar(value=0.25)
        tk.Scale(
            side, from_=0.1, to=0.8, resolution=0.05, orient="horizontal",
            variable=self.conf_var, bg=panel, fg=fg, highlightthickness=0,
            troughcolor="#101216",
        ).pack(fill="x", padx=14)

        self.spike_var = tk.BooleanVar(value=True)
        self.confirm_var = tk.BooleanVar(value=True)
        self.prox_var = tk.BooleanVar(value=True)
        for text, var in (
            ("가속도 급변", self.spike_var),
            ("급변 후 상태 유지", self.confirm_var),
            ("근접하면 충돌 태그", self.prox_var),
        ):
            tk.Checkbutton(
                side, text=text, variable=var, bg=panel, fg=fg, selectcolor="#101216",
                activebackground=panel, activeforeground=fg, font=("Malgun Gothic", 10),
            ).pack(anchor="w", padx=14)

        self.start_btn = tk.Button(
            side, text="추적 시작", command=self._toggle,
            bg="#2f9e44", fg="white", relief="flat", font=("Malgun Gothic", 12, "bold"),
            pady=8,
        )
        self.start_btn.pack(fill="x", padx=14, pady=(16, 8))

        label("사고 의심 로그")
        self.log = tk.Listbox(
            side, bg="#101216", fg="#f3f5f7", height=8,
            highlightthickness=0, activestyle="none", font=("Consolas", 9),
        )
        self.log.pack(fill="both", expand=True, padx=14, pady=(0, 14))

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
        if self.tracking:
            return
        picked = pick_region(self.root, self._selected_monitor())
        if picked is None:
            return
        self.region = picked
        left, top, width, height = picked
        self.region_var.set(f"{width}x{height}  @ {left},{top}")

    def _set_status(self, text: str, color: str) -> None:
        self.status.configure(text=text, fg=color)

    def _schedule_preview(self) -> None:
        self._preview_job = self.root.after(120, self._update_preview)

    def _show_bgr(self, frame: np.ndarray) -> None:
        canvas_w = max(self.preview.winfo_width(), 320)
        canvas_h = max(self.preview.winfo_height(), 180)
        scale = min(canvas_w / frame.shape[1], canvas_h / frame.shape[0])
        view = cv2.resize(
            frame,
            (max(1, int(frame.shape[1] * scale)), max(1, int(frame.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        view = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        ok, buf = cv2.imencode(".png", view)
        if not ok:
            return
        self.preview_image = tk.PhotoImage(data=buf.tobytes())
        self.preview.configure(image=self.preview_image)

    def _update_preview(self) -> None:
        if not self.tracking:
            frame = self._grab_preview()
            if frame is not None:
                self._show_bgr(frame)
        self._schedule_preview()

    def _grab_preview(self):
        import mss
        import numpy as np

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
                frame, (960, max(1, int(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA
            )
        return frame

    def _toggle(self) -> None:
        if self.tracking:
            self._stop()
        else:
            self._start()

    def _start(self) -> None:
        self.tracking = True
        self.stop_event = threading.Event()
        self.start_btn.configure(text="중지", bg="#d64545")
        self.monitor_box.configure(state="disabled")
        self.preview_title.configure(text="추적 중")
        self._set_status("모델 준비 중", "#f0b429")
        cfg = AppConfig(
            conf=float(self.conf_var.get()),
            output_dir="outputs",
            event=EventConfig(
                use_accel_spike=bool(self.spike_var.get()),
                use_post_state_confirm=bool(self.confirm_var.get()),
                use_proximity_tag=bool(self.prox_var.get()),
            ),
        )
        cfg.event.apply_sensitivity(int(self.sensitivity_var.get()))
        mon = self._selected_monitor()
        if self.region is None:
            left, top, width, height = mon["left"], mon["top"], mon["width"], mon["height"]
        else:
            left, top, width, height = self.region
        try:
            self.overlay = ScreenOverlay(left, top, width, height)
            self._paint_overlay([], {}, [], [], width, height)
        except Exception as exc:
            self.tracking = False
            self.start_btn.configure(text="추적 시작", bg="#2f9e44")
            self.monitor_box.configure(state="readonly")
            self.preview_title.configure(text="미리보기  (추적 전)")
            self._set_status("오버레이 실패", "#d64545")
            self.log.insert(0, f"오류: {exc}")
            return

        def on_visual(tracked, states, events, recent, toggles, fw, fh, vis) -> None:
            with self._visual_lock:
                self._visual = (tracked, states, events, recent, toggles, fw, fh, vis)

        def work() -> None:
            self.messages.put(("status", "추적 중"))
            try:
                run_screen(
                    cfg,
                    monitor=mon["index"],
                    region=self.region,
                    max_width=1280,
                    show=False,
                    save_video=False,
                    use_overlay=False,
                    stop_event=self.stop_event,
                    on_alert=lambda event: self.messages.put(("alert", event)),
                    on_visual=on_visual,
                )
            except Exception:
                self.messages.put(("error", traceback.format_exc()))
            finally:
                self.messages.put(("stopped", None))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _stop(self) -> None:
        self.stop_event.set()
        self._set_status("중지 중", "#f0b429")
        self.start_btn.configure(state="disabled")

    def _poll_messages(self) -> None:
        while True:
            try:
                kind, payload = self.messages.get_nowait()
            except queue.Empty:
                break
            if kind == "status":
                self._set_status("추적 중", "#3ddc84")
            elif kind == "alert":
                line = (
                    f"ID {payload.track_id}  {payload.tag}  "
                    f"frame {payload.frame_idx}"
                )
                self.log.insert(0, line)
                self.log.see(0)
            elif kind == "error":
                self.log.insert(0, "오류: 추적 중 예외가 발생했습니다")
                self.log.insert(1, str(payload).splitlines()[-1][:120])
                print(payload)
            elif kind == "stopped":
                self._close_overlay()
                self.tracking = False
                self.start_btn.configure(text="추적 시작", bg="#2f9e44", state="normal")
                self.monitor_box.configure(state="readonly")
                self.preview_title.configure(text="미리보기  (추적 전)")
                self._set_status("대기", "#8b93a1")
        self._paint_latest()
        self.root.after(50, self._poll_messages)

    def _paint_latest(self) -> None:
        if self.overlay is None:
            return
        with self._visual_lock:
            visual = self._visual
        if visual is None:
            return
        tracked, states, events, recent, toggles, fw, fh, vis = visual
        self.preview_title.configure(text=f"추적 중  ·  검출 {len(tracked)}개")
        self._show_bgr(vis)
        self._paint_overlay(tracked, states, events, recent, fw, fh, toggles)
        for key_name in self.overlay.poll_keys():
            if key_name == "q":
                self._stop()

    def _paint_overlay(self, tracked, states, events, recent, fw, fh, toggles=None) -> None:
        if self.overlay is None:
            return
        if toggles is None:
            toggles = {
                "use_accel_spike": bool(self.spike_var.get()),
                "use_post_state_confirm": bool(self.confirm_var.get()),
                "use_proximity_tag": bool(self.prox_var.get()),
            }
        self.overlay.present(
            tracked,
            states,
            events,
            recent,
            toggles,
            scale_x=self.overlay.width / max(fw, 1),
            scale_y=self.overlay.height / max(fh, 1),
        )

    def _close_overlay(self) -> None:
        if self.overlay is not None:
            self.overlay.close()
            self.overlay = None
        with self._visual_lock:
            self._visual = None

    def _on_close(self) -> None:
        if self.tracking:
            self.stop_event.set()
        self._close_overlay()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    StudioApp().run()


if __name__ == "__main__":
    main()

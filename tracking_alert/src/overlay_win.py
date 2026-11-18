from __future__ import annotations

import ctypes
from ctypes import wintypes
from typing import Dict, Iterable, List

import cv2
import numpy as np

from src.event_detector import SuspectEvent
from src.motion import MotionState
from src.tracker import TrackedObject
from src.visualize import (
    ALERT_BGR,
    NORMAL_BGR,
    PREDICT_BGR,
    VELOCITY_BGR,
    draw_legend_bar,
)

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
kernel32 = ctypes.windll.kernel32

WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
WS_POPUP = 0x80000000
SW_SHOWNA = 8
SW_HIDE = 0
ULW_ALPHA = 0x02
AC_SRC_OVER = 0
AC_SRC_ALPHA = 1
WDA_EXCLUDEFROMCAPTURE = 0x11
PM_REMOVE = 0x0001
HWND_TOPMOST = -1
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040

VK_F1 = 0x70
VK_F2 = 0x71
VK_F3 = 0x72
VK_F9 = 0x78
VK_F12 = 0x7B


def enable_dpi_awareness() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass


class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class SIZE(ctypes.Structure):
    _fields_ = [("cx", wintypes.LONG), ("cy", wintypes.LONG)]


class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [
        ("BlendOp", ctypes.c_byte),
        ("BlendFlags", ctypes.c_byte),
        ("SourceConstantAlpha", ctypes.c_byte),
        ("AlphaFormat", ctypes.c_byte),
    ]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", BITMAPINFOHEADER),
        ("bmiColors", wintypes.DWORD * 3),
    ]


class MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", wintypes.HWND),
        ("message", wintypes.UINT),
        ("wParam", wintypes.WPARAM),
        ("lParam", wintypes.LPARAM),
        ("time", wintypes.DWORD),
        ("pt", POINT),
    ]


user32.DefWindowProcW.argtypes = [
    wintypes.HWND,
    wintypes.UINT,
    wintypes.WPARAM,
    wintypes.LPARAM,
]
user32.DefWindowProcW.restype = ctypes.c_ssize_t
user32.RegisterClassW.restype = wintypes.ATOM
user32.CreateWindowExW.argtypes = [
    wintypes.DWORD,
    wintypes.LPCWSTR,
    wintypes.LPCWSTR,
    wintypes.DWORD,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    wintypes.HWND,
    wintypes.HMENU,
    wintypes.HINSTANCE,
    wintypes.LPVOID,
]
user32.CreateWindowExW.restype = wintypes.HWND
user32.DestroyWindow.argtypes = [wintypes.HWND]
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.SetWindowPos.argtypes = [
    wintypes.HWND,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    wintypes.UINT,
]
user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
user32.SetWindowDisplayAffinity.restype = wintypes.BOOL
user32.UpdateLayeredWindow.argtypes = [
    wintypes.HWND,
    wintypes.HDC,
    ctypes.POINTER(POINT),
    ctypes.POINTER(SIZE),
    wintypes.HDC,
    ctypes.POINTER(POINT),
    wintypes.DWORD,
    ctypes.POINTER(BLENDFUNCTION),
    wintypes.DWORD,
]
user32.UpdateLayeredWindow.restype = wintypes.BOOL
user32.GetDC.argtypes = [wintypes.HWND]
user32.GetDC.restype = wintypes.HDC
user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
user32.PeekMessageW.argtypes = [
    ctypes.POINTER(MSG),
    wintypes.HWND,
    wintypes.UINT,
    wintypes.UINT,
    wintypes.UINT,
]
gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateCompatibleDC.restype = wintypes.HDC
gdi32.CreateDIBSection.argtypes = [
    wintypes.HDC,
    ctypes.POINTER(BITMAPINFO),
    wintypes.UINT,
    ctypes.POINTER(ctypes.c_void_p),
    wintypes.HANDLE,
    wintypes.DWORD,
]
gdi32.CreateDIBSection.restype = wintypes.HBITMAP
gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
gdi32.SelectObject.restype = wintypes.HGDIOBJ
gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
gdi32.DeleteDC.argtypes = [wintypes.HDC]


WNDPROC = ctypes.WINFUNCTYPE(
    ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
)


@WNDPROC
def _wnd_proc(hwnd, msg, wparam, lparam):
    return user32.DefWindowProcW(hwnd, msg, wparam, lparam)


class WNDCLASSW(ctypes.Structure):
    _fields_ = [
        ("style", wintypes.UINT),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
    ]


class KeyPoller:
    def __init__(self) -> None:
        self._down: set[int] = set()
        self._map = {
            VK_F9: "q",
            VK_F1: "1",
            VK_F2: "2",
            VK_F3: "3",
            VK_F12: "s",
        }

    def poll(self) -> list[str]:
        fired: list[str] = []
        for vk, name in self._map.items():
            down = bool(user32.GetAsyncKeyState(vk) & 0x8000)
            if down and vk not in self._down:
                fired.append(name)
            if down:
                self._down.add(vk)
            else:
                self._down.discard(vk)
        return fired


class ScreenOverlay:
    """클릭은 통과하고, 화면 위에 박스/알림만 그리는 투명 창."""

    _class_atom = 0

    def __init__(self, left: int, top: int, width: int, height: int) -> None:
        self.left = int(left)
        self.top = int(top)
        self.width = int(width)
        self.height = int(height)
        self._keys = KeyPoller()
        self.hwnd = self._create_window()

    def present(
        self,
        tracked: Iterable[TrackedObject],
        states: Dict[int, MotionState],
        events: List[SuspectEvent],
        recent_alerts: List[SuspectEvent],
        toggles: dict,
        scale_x: float,
        scale_y: float,
        predicted=None,
    ) -> None:
        bgra = _render_bgra(
            self.width,
            self.height,
            tracked,
            states,
            events,
            recent_alerts,
            toggles,
            scale_x,
            scale_y,
            predicted=predicted,
        )
        self._update(bgra)
        self._pump()

    def hide(self) -> None:
        if self.hwnd:
            user32.ShowWindow(self.hwnd, SW_HIDE)

    def show(self) -> None:
        if not self.hwnd:
            return
        user32.ShowWindow(self.hwnd, SW_SHOWNA)
        user32.SetWindowPos(
            self.hwnd,
            ctypes.c_void_p(-1),
            self.left,
            self.top,
            self.width,
            self.height,
            SWP_NOACTIVATE | SWP_SHOWWINDOW,
        )

    def poll_keys(self) -> list[str]:
        self._pump()
        return self._keys.poll()

    def close(self) -> None:
        if getattr(self, "hwnd", None):
            user32.DestroyWindow(self.hwnd)
            self.hwnd = None

    def _create_window(self) -> int:
        hinstance = kernel32.GetModuleHandleW(None)
        class_name = "TrackingAlertOverlay"
        if not ScreenOverlay._class_atom:
            wc = WNDCLASSW()
            wc.lpfnWndProc = _wnd_proc
            wc.hInstance = hinstance
            wc.lpszClassName = class_name
            ScreenOverlay._class_atom = user32.RegisterClassW(ctypes.byref(wc))
            if not ScreenOverlay._class_atom:
                raise OSError(f"오버레이 창 등록 실패: {ctypes.get_last_error()}")

        ex = (
            WS_EX_LAYERED
            | WS_EX_TRANSPARENT
            | WS_EX_TOPMOST
            | WS_EX_TOOLWINDOW
            | WS_EX_NOACTIVATE
        )
        hwnd = user32.CreateWindowExW(
            ex,
            class_name,
            "tracking_overlay",
            WS_POPUP,
            self.left,
            self.top,
            self.width,
            self.height,
            None,
            None,
            hinstance,
            None,
        )
        if not hwnd:
            raise OSError(f"오버레이 창 생성 실패: {ctypes.get_last_error()}")

        # 눈에는 보이지만 화면 캡처에는 포함되지 않게 해서, 캡처마다 숨기지 않는다.
        user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
        user32.ShowWindow(hwnd, SW_SHOWNA)
        user32.SetWindowPos(
            hwnd,
            ctypes.c_void_p(-1),
            self.left,
            self.top,
            self.width,
            self.height,
            SWP_NOACTIVATE | SWP_SHOWWINDOW,
        )
        return hwnd

    def _update(self, bgra: np.ndarray) -> None:
        height, width = bgra.shape[:2]
        hdc_screen = user32.GetDC(0)
        hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = width
        bmi.bmiHeader.biHeight = -height
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = 0

        bits = ctypes.c_void_p()
        hbitmap = gdi32.CreateDIBSection(
            hdc_screen, ctypes.byref(bmi), 0, ctypes.byref(bits), None, 0
        )
        if not hbitmap or not bits:
            gdi32.DeleteDC(hdc_mem)
            user32.ReleaseDC(0, hdc_screen)
            raise OSError("오버레이 비트맵 생성 실패")

        ctypes.memmove(bits, bgra.ctypes.data, bgra.nbytes)
        old = gdi32.SelectObject(hdc_mem, hbitmap)
        dst = POINT(self.left, self.top)
        size = SIZE(width, height)
        src = POINT(0, 0)
        blend = BLENDFUNCTION(AC_SRC_OVER, 0, 255, AC_SRC_ALPHA)
        ok = user32.UpdateLayeredWindow(
            self.hwnd,
            hdc_screen,
            ctypes.byref(dst),
            ctypes.byref(size),
            hdc_mem,
            ctypes.byref(src),
            0,
            ctypes.byref(blend),
            ULW_ALPHA,
        )
        gdi32.SelectObject(hdc_mem, old)
        gdi32.DeleteObject(hbitmap)
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(0, hdc_screen)
        if not ok:
            err = ctypes.get_last_error()
            if err:
                print(f"[overlay] UpdateLayeredWindow 실패: {err}")

    def _pump(self) -> None:
        msg = MSG()
        while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))


def _draw_dashed_rect(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    gap: int = 10,
) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    for x in range(x1, x2, gap * 2):
        cv2.line(img, (x, y1), (min(x + gap, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + gap, x2), y2), color, thickness)
    for y in range(y1, y2, gap * 2):
        cv2.line(img, (x1, y), (x1, min(y + gap, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + gap, y2)), color, thickness)


def _render_bgra(
    width: int,
    height: int,
    tracked: Iterable[TrackedObject],
    states: Dict[int, MotionState],
    events: List[SuspectEvent],
    recent_alerts: List[SuspectEvent],
    toggles: dict,
    scale_x: float,
    scale_y: float,
    predicted=None,
) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    thick = max(2, int(round(min(scale_x, scale_y))))
    cv2.rectangle(canvas, (4, 4), (width - 5, height - 5), (80, 255, 80), max(4, thick))
    font = max(0.6, 0.45 * min(scale_x, scale_y))
    alert_ids = {ev.track_id for ev in recent_alerts}

    def px(x: float) -> int:
        return int(round(x * scale_x))

    def py(y: float) -> int:
        return int(round(y * scale_y))

    for obj in tracked:
        x1, y1, x2, y2 = obj.bbox_xyxy
        is_alert = obj.track_id in alert_ids
        color = ALERT_BGR if is_alert else NORMAL_BGR
        box_thick = thick + 2 if is_alert else thick
        cv2.rectangle(canvas, (px(x1), py(y1)), (px(x2), py(y2)), color, box_thick)
        st = states.get(obj.track_id)
        label = f"ID {obj.track_id} {obj.label}"
        if is_alert:
            label = "ALERT " + label
        if st is not None:
            label += f" v={st.speed:.1f} a={st.accel_mag:.1f}"
            p1 = (px(st.x), py(st.y))
            p2 = (px(st.x + st.vx * 5), py(st.y + st.vy * 5))
            cv2.arrowedLine(canvas, p1, p2, VELOCITY_BGR, thick, tipLength=0.3)
        cv2.putText(
            canvas,
            label,
            (px(x1), max(24, py(y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font,
            color,
            max(1, thick - 1),
            cv2.LINE_AA,
        )

    for obj in predicted or []:
        x1, y1, x2, y2 = obj.bbox_xyxy
        color = PREDICT_BGR
        _draw_dashed_rect(canvas, (px(x1), py(y1)), (px(x2), py(y2)), color, max(2, thick))
        remain = max(0, obj.hold_frames - obj.missing_frames)
        label = f"ID {obj.track_id} PREDICT ({remain}f)"
        cv2.putText(
            canvas,
            label,
            (px(x1), max(24, py(y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font,
            color,
            max(1, thick - 1),
            cv2.LINE_AA,
        )

    hold = toggles.get("occlusion_hold_sec", 0)
    lines = [
        "F1 accel: " + ("ON" if toggles.get("use_accel_spike") else "OFF"),
        "F2 confirm: " + ("ON" if toggles.get("use_post_state_confirm") else "OFF"),
        f"occlusion: {hold}s",
        "F9 quit | F12 save",
    ]
    for i, text in enumerate(lines):
        cv2.putText(
            canvas,
            text,
            (16, 36 + i * int(28 * font)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font,
            (255, 255, 255),
            max(2, thick),
            cv2.LINE_AA,
        )

    legend_h = draw_legend_bar(canvas, y_bottom=height, scale=max(1.0, font))
    if recent_alerts:
        latest = recent_alerts[-1]
        msg = f"ALERT ID={latest.track_id} {latest.tag} frame={latest.frame_idx}"
        bar_h = max(48, int(36 * font))
        y1 = height - legend_h - bar_h
        cv2.rectangle(canvas, (0, y1), (width, height - legend_h), ALERT_BGR, -1)
        cv2.putText(
            canvas,
            msg,
            (16, y1 + bar_h // 2 + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            font,
            (255, 255, 255),
            max(2, thick),
            cv2.LINE_AA,
        )

    for ev in events:
        cv2.circle(canvas, (px(ev.x), py(ev.y)), max(18, thick * 8), ALERT_BGR, thick)

    # Windows layered DIB = BGRA. OpenCV 캔버스도 BGR이므로 그대로 쓴다.
    # (BGR→RGB 변환 시 빨강이 파랑으로 보인다)
    alpha = np.where(canvas.max(axis=2) > 12, 230, 0).astype(np.uint8)
    premul = (canvas.astype(np.float32) * (alpha.astype(np.float32) / 255.0)[..., None])
    bgra = np.dstack([premul.astype(np.uint8), alpha])
    return np.ascontiguousarray(bgra)

"""Frame sources: a recorded clip, or the live screen.

Both yield ``(timestamp_seconds, frame_bgr)``. Timestamps are clip-relative
for video and wall-clock-relative for the screen, which is all the engine
needs -- it only measures elapsed time.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

DOTA_WINDOW_TITLE = "Dota 2"


def _ensure_dpi_aware() -> None:
    """Opt this process out of DPI virtualisation (Windows, best effort).

    Regular python.exe declares DPI awareness in its manifest; the
    PyInstaller windowed bootloader does not. A non-aware process gets its
    coordinates silently rescaled on scaled displays, so the exe's clicks
    landed off-target while `uv run` clicked true. Making awareness
    explicit gives both the same physical-pixel view mss captures in.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        try:    # per-monitor v2, Windows 10 1703+
            ctypes.windll.user32.SetProcessDpiAwarenessContext(
                ctypes.c_void_p(-4))
        except (AttributeError, OSError):
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:   # noqa: BLE001 - already aware, or too old to matter
        pass


def _window_rect(title: str) -> tuple[int, int, int, int] | None:
    """Screen rectangle of a top-level window by exact title (Windows)."""
    if sys.platform != "win32":
        return None
    import ctypes
    import ctypes.wintypes

    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW(None, title)
    if not hwnd:
        return None
    rect = ctypes.wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    return rect.left, rect.top, rect.right, rect.bottom


def pick_monitor(rect: tuple[int, int, int, int],
                 monitors: list[dict]) -> int | None:
    """Index of the monitor overlapping a window rect the most.

    ``monitors`` is mss's list, whose entry 0 is the whole virtual desktop
    and is skipped.
    """
    best, best_overlap = None, 0
    for index, monitor in enumerate(monitors[1:], start=1):
        overlap_w = max(0, (min(rect[2], monitor["left"] + monitor["width"])
                            - max(rect[0], monitor["left"])))
        overlap_h = max(0, (min(rect[3], monitor["top"] + monitor["height"])
                            - max(rect[1], monitor["top"])))
        if overlap_w * overlap_h > best_overlap:
            best, best_overlap = index, overlap_w * overlap_h
    return best


def find_game_monitor(title: str = DOTA_WINDOW_TITLE) -> int | None:
    """Monitor index the game window is on, or None if no window found."""
    rect = _window_rect(title)
    if rect is None:
        return None
    import mss

    with mss.mss() as sct:
        return pick_monitor(rect, sct.monitors)


class VideoSource:
    """Frames from a recorded clip, sampled every ``stride`` frames."""

    def __init__(self, path: str | Path, stride: int = 15) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"clip not found: {self.path}")
        self.stride = max(1, stride)
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise RuntimeError(f"could not open clip: {self.path}")
        self.fps = self._capture.get(cv2.CAP_PROP_FPS) or 60.0
        self.frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def frames(self) -> Iterator[tuple[float, np.ndarray]]:
        index = 0
        while True:
            ok, frame = self._capture.read()
            if not ok:
                break
            index += 1
            if index % self.stride:
                continue
            yield index / self.fps, frame
        self.close()

    def frame_at(self, seconds: float) -> np.ndarray | None:
        """Grab a single frame by timestamp (used by ``calibrate``)."""
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, int(seconds * self.fps))
        ok, frame = self._capture.read()
        return frame if ok else None

    def close(self) -> None:
        self._capture.release()

    def __enter__(self) -> VideoSource:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class ScreenSource:
    """Frames grabbed from the live screen with mss."""

    # Consecutive identical grabs before the grabber is reinitialised.
    # Nothing in this game holds a frame for 0.3s; identical content that
    # long means the desktop-duplication buffer went stale (observed at a
    # round restart: 5.4s of frozen frames while a word died on screen).
    FROZEN_GRAB_LIMIT = 10

    def __init__(self, monitor: int = 1, interval: float = 0.05,
                 on_stall=None) -> None:
        try:
            import mss
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "mss is not installed. Run: uv sync"
            ) from exc
        _ensure_dpi_aware()
        self._mss = mss
        self.monitor_index = monitor
        self.interval = interval
        self.on_stall = on_stall
        with mss.mss() as sct:
            mon = sct.monitors[monitor]
            self.width, self.height = mon["width"], mon["height"]
            # Frame pixels are monitor-relative; clicks are virtual-screen
            # absolute. This origin is the difference.
            self.origin = (mon["left"], mon["top"])

    def frames(self) -> Iterator[tuple[float, np.ndarray]]:
        """Yield the freshest frame available, captured on its own thread.

        Capturing inline added ~20-40ms plus the inter-scan sleep to every
        cycle. A producer thread grabs continuously; the consumer always
        processes the newest frame and never waits on the screen.
        """
        import threading

        state = {"seq": 0, "timestamp": 0.0, "frame": None, "stop": False}
        condition = threading.Condition()
        start = time.monotonic()

        def producer() -> None:
            try:
                import ctypes

                kernel32 = ctypes.windll.kernel32
                THREAD_PRIORITY_ABOVE_NORMAL = 1
                kernel32.SetThreadPriority(kernel32.GetCurrentThread(),
                                           THREAD_PRIORITY_ABOVE_NORMAL)
            except Exception:   # noqa: BLE001 - best effort, non-Windows
                pass
            sct = self._mss.mss()
            monitor = sct.monitors[self.monitor_index]
            last_signature = None
            frozen = 0
            try:
                while not state["stop"]:
                    frame = np.asarray(sct.grab(monitor))[:, :, :3]
                    # Cheap content signature: a strided sample sum.
                    signature = int(frame[::97, ::97].sum())
                    if signature == last_signature:
                        frozen += 1
                        if frozen >= self.FROZEN_GRAB_LIMIT:
                            if self.on_stall is not None:
                                self.on_stall(
                                    "!!! capture stalled (frozen frames) "
                                    "-- reinitialising the grabber")
                            sct.close()
                            sct = self._mss.mss()
                            monitor = sct.monitors[self.monitor_index]
                            frozen = 0
                    else:
                        frozen = 0
                        last_signature = signature
                    with condition:
                        state["seq"] += 1
                        state["timestamp"] = time.monotonic() - start
                        state["frame"] = frame
                        condition.notify_all()
                    time.sleep(self.interval)
            finally:
                sct.close()

        thread = threading.Thread(target=producer, daemon=True,
                                  name="capture")
        thread.start()
        last_seen = 0
        try:
            while True:
                with condition:
                    while state["seq"] == last_seen:
                        condition.wait(0.5)
                    last_seen = state["seq"]
                    timestamp, frame = state["timestamp"], state["frame"]
                yield timestamp, frame
        finally:
            state["stop"] = True

    def grab(self) -> np.ndarray:
        """Single screenshot."""
        with self._mss.mss() as sct:
            monitor = sct.monitors[self.monitor_index]
            return np.asarray(sct.grab(monitor))[:, :, :3]

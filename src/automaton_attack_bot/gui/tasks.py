"""One-at-a-time command execution for the panel.

Commands are the CLI's own functions; they talk via ``print`` (and the
session log's ``say``, which also prints). Teeing stdout is therefore the
single tap that captures every command's output -- no per-command wiring.
"""

from __future__ import annotations

import queue
import sys
import threading


class _Tee:
    """Forward writes to the real stdout AND the event queue, line-wise."""

    def __init__(self, events: queue.Queue, echo) -> None:
        self._events = events
        self._echo = echo
        self._buffer = ""

    def write(self, text: str) -> int:
        self._echo.write(text)
        self._buffer += text
        while True:
            for cut in ("\n", "\r"):    # \r: download progress redraws
                if cut in self._buffer:
                    line, self._buffer = self._buffer.split(cut, 1)
                    if line.strip():
                        self._events.put(("line", line))
                    break
            else:
                return len(text)

    def flush(self) -> None:
        self._echo.flush()
        if self._buffer.strip():
            self._events.put(("line", self._buffer))
        self._buffer = ""


class TaskRunner:
    """Run one command function at a time on a worker thread.

    Results arrive on ``events`` as ``("line", text)`` while running and
    ``("done", exit_code)`` at the end; the panel polls the queue from the
    Tk main loop, which keeps all widget work on the right thread.
    """

    def __init__(self) -> None:
        self.events: queue.Queue = queue.Queue()
        self.stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def launch(self, fn, *args, **kwargs) -> bool:
        if self.busy:
            return False
        self.stop.clear()

        def target():
            tee = _Tee(self.events, sys.stdout)
            original, sys.stdout = sys.stdout, tee
            try:
                code = fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 -- surfaced in the panel
                self.events.put(("line", f"error: {exc}"))
                code = 1
            finally:
                sys.stdout = original
                tee.flush()
            self.events.put(("done", code))

        self._thread = threading.Thread(target=target, name="task",
                                        daemon=True)
        self._thread.start()
        return True

    def request_stop(self) -> None:
        self.stop.set()

    def join(self, timeout: float) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

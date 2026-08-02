"""Keystroke delivery.

Spaces and punctuation are skipped -- the minigame accepts letters only, and
lowercase is fine. Two typists exist: one that prints what it would send, and
one that actually sends it.

:class:`TypingWorker` decouples typing from scanning: the scan loop used to
block for the whole keystroke burst (~400 ms for a long item name), during
which nothing was watching the screen -- a short-lived word could spawn and
die inside that gap. The worker types from its own thread, always picking
the most urgent queued word, and drops anything that has waited so long the
word is gone anyway.
"""

from __future__ import annotations

import random
import threading
import time
from typing import Callable

from .config import Behaviour


class DryRunTypist:
    """Records what would have been typed. The default everywhere."""

    live = False

    def __init__(self, behaviour: Behaviour | None = None) -> None:
        self.behaviour = behaviour or Behaviour()
        self.typed: list[str] = []
        self.clicked: list[tuple[int, int]] = []

    def type(self, text: str) -> None:
        self.typed.append(text)

    def click(self, x: int, y: int) -> None:
        self.clicked.append((x, y))


class DirectInputTypist:
    """Sends real keystrokes via pydirectinput (works with DirectX games)."""

    live = True

    def __init__(self, behaviour: Behaviour | None = None) -> None:
        try:
            import pydirectinput
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "pydirectinput is not installed. Run: uv sync --extra live"
            ) from exc
        pydirectinput.PAUSE = 0
        self._pydirectinput = pydirectinput
        self.behaviour = behaviour or Behaviour()
        self.typed: list[str] = []
        self.clicked: list[tuple[int, int]] = []
        self._min_seconds_per_char = self._pace()

    def _pace(self) -> float:
        """Seconds per character implied by the WPM cap (0 = uncapped).

        Valve patched a pause-typing exploit in this minigame in July 2026, so
        the leaderboard is watched. A cap keeps output within human range.
        """
        wpm = self.behaviour.max_wpm
        if wpm <= 0:
            return 0.0
        return 60.0 / (wpm * 5.0)   # 5 characters per "word", by convention

    def type(self, text: str) -> None:
        for char in text:
            self._pydirectinput.press(char)
            delay = random.uniform(*self.behaviour.key_delay)
            time.sleep(max(delay, self._min_seconds_per_char))
        self.typed.append(text)

    def click(self, x: int, y: int) -> None:
        self._pydirectinput.click(x, y)
        self.clicked.append((x, y))


def make_typist(live: bool, behaviour: Behaviour | None = None):
    return (DirectInputTypist if live else DryRunTypist)(behaviour)


class TypingWorker(threading.Thread):
    """Types queued words from its own thread, most urgent first.

    ``submit`` is called from the scan loop with a TypedWord; ``urgency``
    ranks queued words (smaller = closer to death) at pop time, so a word
    that arrives late but urgent overtakes everything already waiting.
    Words queued longer than ``stale_after`` are dropped: their automaton
    has almost certainly reached the platform, and the keystrokes would be
    pure waste. ``on_typed`` fires after the keys have actually been sent,
    with the time the word spent waiting.
    """

    def __init__(self, typist, urgency: Callable, on_typed: Callable | None = None,
                 stale_after: float = 4.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        super().__init__(daemon=True, name="typing-worker")
        self.typist = typist
        self.urgency = urgency
        self.on_typed = on_typed
        self.stale_after = stale_after
        self.clock = clock
        self.dropped_stale = 0
        self._pending: list[tuple[object, float]] = []
        self._condition = threading.Condition()
        self._stopped = False

    def submit(self, word) -> None:
        with self._condition:
            if self._stopped:
                return
            if any(w.keystrokes == word.keystrokes for w, _ in self._pending):
                return
            self._pending.append((word, self.clock()))
            self._condition.notify()

    def run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._stopped:
                    self._condition.wait(0.1)
                if self._stopped:
                    return
                index = min(
                    range(len(self._pending)),
                    key=lambda i: self.urgency(self._pending[i][0].detection))
                word, enqueued = self._pending.pop(index)
            waited = self.clock() - enqueued
            if waited > self.stale_after:
                self.dropped_stale += 1
                continue
            self.typist.type(word.keystrokes)
            if self.on_typed:
                self.on_typed(word, waited)

    def stop(self, timeout: float = 5.0) -> None:
        """Stop immediately; pending words belong to a round that is over."""
        with self._condition:
            self._stopped = True
            self._condition.notify()
        if self.is_alive():
            self.join(timeout)

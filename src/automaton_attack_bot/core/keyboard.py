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

    def __init__(self, behaviour: Behaviour | None = None,
                 origin: tuple[int, int] = (0, 0)) -> None:
        self.behaviour = behaviour or Behaviour()
        self.origin = origin
        self.typed: list[str] = []
        self.clicked: list[tuple[int, int]] = []
        self.last_key_times: list[float] = []

    def type(self, text: str) -> None:
        now = time.monotonic()
        self.last_key_times = [now] * len(text)
        self.typed.append(text)

    def click(self, x: int, y: int) -> None:
        """x, y are frame (captured-monitor) coordinates; the click goes
        to virtual-screen absolute. On a non-primary monitor the two
        differ by the monitor's origin -- clicks used to land on the
        wrong monitor entirely."""
        self.clicked.append((x + self.origin[0], y + self.origin[1]))


class DirectInputTypist:
    """Sends real keystrokes via pydirectinput (works with DirectX games)."""

    live = True

    def __init__(self, behaviour: Behaviour | None = None,
                 origin: tuple[int, int] = (0, 0)) -> None:
        try:
            import pydirectinput
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "pydirectinput is not installed. Run: uv sync"
            ) from exc
        pydirectinput.PAUSE = 0
        self._pydirectinput = pydirectinput
        self.behaviour = behaviour or Behaviour()
        self.origin = origin
        self.typed: list[str] = []
        self.clicked: list[tuple[int, int]] = []
        # Monotonic send time of every key of the last word, recorded at
        # keyDown -- the audit trail for keystroke-level forensics.
        self.last_key_times: list[float] = []
        self._min_seconds_per_char = self._pace()

    def _pace(self) -> float:
        """Seconds per character implied by the WPM cap (0 = uncapped).

        The default is uncapped, full machine speed. --max-wpm is the
        opt-in throttle for anyone who wants human-plausible pacing.
        """
        wpm = self.behaviour.max_wpm
        if wpm <= 0:
            return 0.0
        return 60.0 / (wpm * 5.0)   # 5 characters per "word", by convention

    def type(self, text: str) -> None:
        """Send the keystrokes, pacing to the WPM cap when one is set.

        Throttled keys are HELD for part of the interval: press() taps
        with zero hold, and a zero-length tap can fall between the
        game's input polls entirely -- at 150 WPM the tracker consumed
        ~1 of every 3 keys (DAZZLE: 6 sent in 0.5s, accepted over 1.5s;
        SHADOW BLADE: 11 sent, 1 accepted, word struck). Full speed
        keeps the plain tap: burst input lands, and the records were
        set with it.
        """
        key_times = []
        for char in text:
            delay = max(random.uniform(*self.behaviour.key_delay),
                        self._min_seconds_per_char)
            key_times.append(time.monotonic())
            if delay > 0:
                hold = min(0.05, delay * 0.4)
                self._pydirectinput.keyDown(char)
                time.sleep(hold)
                self._pydirectinput.keyUp(char)
                time.sleep(max(0.0, delay - hold))
            else:
                self._pydirectinput.press(char)
        self.last_key_times = key_times
        self.typed.append(text)

    def click(self, x: int, y: int) -> None:
        """Frame coordinates in, virtual-screen click out (see DryRunTypist
        .click)."""
        absolute = (x + self.origin[0], y + self.origin[1])
        self._pydirectinput.click(*absolute)
        self.clicked.append(absolute)


def make_typist(live: bool, behaviour: Behaviour | None = None,
                origin: tuple[int, int] = (0, 0)):
    return (DirectInputTypist if live else DryRunTypist)(behaviour, origin)


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

    # Under load, weak guesses lose their seat: keyboard time is the scarce
    # resource, and a 0.67 match from a fly-in partial usually gets retyped
    # correctly a scan later anyway. Verbatim/fallback words are exempt --
    # they are the only shot at out-of-corpus words.
    TRIAGE_DEPTH = 3
    TRIAGE_BELOW = 0.75

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
        self.dropped_triage = 0
        self._pending: list[tuple[object, float]] = []
        self._condition = threading.Condition()
        self._stopped = False
        self._typing_keys = 0   # keystrokes of the word being typed now

    def pending_keystrokes(self) -> int:
        """Keystrokes queued plus in-progress -- the keyboard's backlog.

        The engine stretches its dedup windows by this backlog's service
        time: with --max-wpm a word is still being typed long after it
        was submitted, and treating it as 'typed but still visible'
        double-queued nearly every word of a throttled round.
        """
        with self._condition:
            queued = sum(len(w.keystrokes) for w, _ in self._pending)
        return queued + self._typing_keys

    def submit(self, word) -> None:
        with self._condition:
            if self._stopped:
                return
            if any(w.keystrokes == word.keystrokes for w, _ in self._pending):
                return
            self._pending.append((word, self.clock()))
            self._condition.notify()

    @staticmethod
    def _boost_priority() -> None:
        """Keystrokes must never wait behind OCR compute (Windows).

        The typing thread's time slices were getting starved when the
        scan pipeline saturated the CPU; visible as inter-key stutter.
        """
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            THREAD_PRIORITY_HIGHEST = 2
            kernel32.SetThreadPriority(kernel32.GetCurrentThread(),
                                       THREAD_PRIORITY_HIGHEST)
        except Exception:   # noqa: BLE001 - best effort, non-Windows etc.
            pass

    def run(self) -> None:
        self._boost_priority()
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
            if self._triage(word):
                self.dropped_triage += 1
                continue
            self._typing_keys = len(word.keystrokes)
            try:
                self.typist.type(word.keystrokes)
            finally:
                self._typing_keys = 0
            if self.on_typed:
                self.on_typed(word, waited)

    def _triage(self, word) -> bool:
        with self._condition:
            backlog = len(self._pending)
        if backlog < self.TRIAGE_DEPTH:
            return False
        match = word.detection.match
        if match is None or match.source == "fallback":
            return False
        return match.score < self.TRIAGE_BELOW

    def stop(self, timeout: float = 5.0) -> None:
        """Stop immediately; pending words belong to a round that is over."""
        with self._condition:
            self._stopped = True
            self._condition.notify()
        if self.is_alive():
            self.join(timeout)

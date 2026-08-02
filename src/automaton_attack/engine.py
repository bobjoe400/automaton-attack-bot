"""The scan/decide/type loop.

The multiplier resets when a word escapes untyped -- wrong keystrokes are
free. So the engine is greedy: every corpus match is typed the moment it is
seen. If the next scan reads the same word better, the corrected guess is a
different name and types too; the earlier attempt was just stray keys.

Two guards remain, both about not WASTING keyboard time rather than about
being right:

* :class:`Confirmer` -- only fallback reads (raw OCR that matched nothing)
  are held until the identical text repeats on two consecutive scans. A
  fallback has no corpus anchor, and OCR errors vary frame to frame
  (METEORHAKIMER one scan, METEORHARIMER the next), so without the repeat
  gate every flicker would burn keystrokes on a fresh garbage variant.
  Exact repetition is also the one signal the read is actually right.
  Tesseract's own confidence score is not usable for this -- it returned 0
  on a correctly-read long phrase.
* :class:`Deduper` -- a word already typed near the same spot is not retyped
  for a few seconds. In live play a completed word disappears, so seeing it
  again later means keystrokes were dropped or it genuinely respawned; both
  are cases where retyping is the right move.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator

import numpy as np

from .config import Settings
from .detect import Detection, Detector
from .keyboard import DryRunTypist


@dataclass
class TypedWord:
    """A word the engine decided to type."""

    timestamp: float
    detection: Detection
    keystrokes: str

    @property
    def name(self) -> str:
        return self.detection.name

    @property
    def source(self) -> str:
        return self.detection.match.source if self.detection.match else "?"


@dataclass
class Stats:
    frames: int = 0
    detections: int = 0
    typed: int = 0
    by_source: dict[str, int] = field(default_factory=dict)
    suppressed_duplicate: int = 0
    awaiting_confirmation: int = 0

    def record(self, word: TypedWord) -> None:
        self.typed += 1
        self.by_source[word.source] = self.by_source.get(word.source, 0) + 1


class Deduper:
    """Suppresses a word recently typed at roughly the same position."""

    def __init__(self, radius: int = 120, ttl: float = 3.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.radius = radius
        self.ttl = ttl
        self.clock = clock
        self._entries: list[tuple[str, tuple[int, int], float]] = []

    def seen(self, name: str, pos: tuple[int, int]) -> bool:
        now = self.clock()
        self._entries = [e for e in self._entries if now - e[2] < self.ttl]
        for entry_name, entry_pos, _ in self._entries:
            if entry_name != name:
                continue
            if (abs(entry_pos[0] - pos[0]) < self.radius
                    and abs(entry_pos[1] - pos[1]) < self.radius):
                return True
        return False

    def mark(self, name: str, pos: tuple[int, int]) -> None:
        self._entries.append((name, pos, self.clock()))


class Confirmer:
    """Holds fallback reads back until a scan repeats them exactly.

    Corpus matches pass straight through: typing a wrong guess costs only
    keystrokes, while holding a right one can cost the word.
    """

    def __init__(self) -> None:
        self.pending: set[str] = set()

    @staticmethod
    def _is_fallback(detection: Detection) -> bool:
        return bool(detection.match) and detection.match.source == "fallback"

    def ready(self, detection: Detection) -> bool:
        if not self._is_fallback(detection):
            return True
        return detection.name in self.pending

    def update(self, detections: Iterable[Detection]) -> None:
        self.pending = {
            d.name for d in detections if self._is_fallback(d)
        }


class Engine:
    def __init__(self, detector: Detector, typist=None,
                 settings: Settings | None = None) -> None:
        self.detector = detector
        self.settings = settings or detector.settings
        self.typist = typist or DryRunTypist(self.settings.behaviour)
        behaviour = self.settings.behaviour
        self.deduper = Deduper(behaviour.dedup_radius, behaviour.dedup_ttl)
        self.confirmer = Confirmer()
        self.stats = Stats()
        self.last_detections: list[Detection] = []
        # How a decided word becomes keystrokes. The default types inline;
        # live mode replaces this with TypingWorker.submit so the scan loop
        # never blocks on the keyboard.
        self.dispatch: Callable[[TypedWord], None] = (
            lambda word: self.typist.type(word.keystrokes))

    # Where Hoodwink stands, in panel fractions: words die when their
    # automaton reaches this point, so distance to it is time-to-live.
    # Spawns from below start close and are urgent immediately.
    PLATFORM = (0.5, 0.66)

    def _urgency(self, detection: Detection) -> float:
        panel_w, panel_h = self.settings.geometry.panel_size
        bx, by, bw, bh = detection.box
        dx = (bx + bw / 2) / panel_w - self.PLATFORM[0]
        dy = (by + bh / 2) / panel_h - self.PLATFORM[1]
        return dx * dx + dy * dy

    def process(self, timestamp: float,
                frame: np.ndarray) -> list[TypedWord]:
        """Scan one frame and type whatever clears both guards.

        Words closest to the platform are typed first: everything queued
        behind a keystroke burst drops a little further while it waits, so
        the word about to die must not wait behind a fresh spawn.
        """
        detections = self.detector.detect(frame, include_unmatched=True,
                                          timestamp=timestamp)
        detections.sort(key=self._urgency)
        self.last_detections = detections
        self.stats.frames += 1
        self.stats.detections += sum(1 for d in detections if d.match)
        typed = []
        for detection in detections:
            if not detection.match:
                continue
            if self.deduper.seen(detection.name, detection.pos):
                self.stats.suppressed_duplicate += 1
                continue
            if not self.confirmer.ready(detection):
                self.stats.awaiting_confirmation += 1
                continue
            self.deduper.mark(detection.name, detection.pos)
            word = TypedWord(timestamp, detection, detection.match.keystrokes)
            self.dispatch(word)
            self.stats.record(word)
            typed.append(word)
        # Update after the loop: a word must survive a full scan-to-scan gap,
        # not be confirmed by its own detection.
        self.confirmer.update(detections)
        return typed

    def run(self, frames: Iterable[tuple[float, np.ndarray]]
            ) -> Iterator[TypedWord]:
        for timestamp, frame in frames:
            yield from self.process(timestamp, frame)

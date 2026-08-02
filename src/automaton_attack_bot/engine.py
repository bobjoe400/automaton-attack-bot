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
from .lexicon import Match, to_key


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
    """Holds fallback reads back until they stay stable over real TIME.

    Corpus matches pass straight through: typing a wrong guess costs only
    keystrokes, while holding a right one can cost the word.

    Stability is measured in seconds, not scans: with pipelined scanning,
    consecutive scans can be ~70ms apart and see essentially the same
    frame, so an OCR mangle 'repeats identically' without meaning
    anything. (A pipelining regression typed CRYSTLY, ABADDOMNIKNIGHT and
    friends this way.) A read must persist MIN_STABLE_AGE across whatever
    number of scans that spans.
    """

    MIN_STABLE_AGE = 0.35   # seconds a fallback read must persist

    def __init__(self) -> None:
        self._first_seen: dict[str, float] = {}

    @staticmethod
    def _is_fallback(detection: Detection) -> bool:
        return bool(detection.match) and detection.match.source == "fallback"

    def ready(self, detection: Detection, now: float) -> bool:
        if not self._is_fallback(detection):
            return True
        first = self._first_seen.get(detection.name)
        return first is not None and now - first >= self.MIN_STABLE_AGE

    def update(self, detections: Iterable[Detection], now: float) -> None:
        current = {d.name for d in detections if self._is_fallback(d)}
        self._first_seen = {
            name: self._first_seen.get(name, now) for name in current
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
        self._raw_first_seen: dict[str, float] = {}
        # How a decided word becomes keystrokes. The default types inline;
        # live mode replaces this with TypingWorker.submit so the scan loop
        # never blocks on the keyboard.
        self.dispatch: Callable[[TypedWord], None] = (
            lambda word: self.typist.type(word.keystrokes))

    # Where Hoodwink stands, in panel fractions: words die when their
    # automaton reaches this point, so distance to it is time-to-live.
    # Spawns from below start close and are urgent immediately.
    PLATFORM = (0.5, 0.66)
    # A label rendered below this is a word that ALREADY struck -- the
    # game shows the killer at the platform for a moment. Typing it is
    # pure waste, and because it is the closest thing to the platform it
    # used to hijack the top of the urgency queue right after every
    # strike, starving living words exactly when the combo was rebuilding.
    # (Matches analyze.PLATFORM_BAND.)
    STRIKE_BAND = 0.70
    # Rough conversion factors for the deadline estimate: automatons cross
    # about half the panel in a ~4s word lifetime, and a keystroke costs
    # ~22ms with jitter.
    APPROACH_SPEED = 0.12       # panel-fractions per second
    SECONDS_PER_KEY = 0.003     # SendInput overhead; no artificial delay

    def _urgency(self, detection: Detection) -> float:
        """Smaller = must start typing sooner.

        Deadline minus service time, not bare distance: a 30-key voice
        line needs ~0.7s of keyboard before it completes, so it must
        start earlier than a 4-key word at the same range. Two phrases
        died this exact way -- queued behind each other while both fell.
        """
        panel_w, panel_h = self.settings.geometry.panel_size
        bx, by, bw, bh = detection.box
        dx = (bx + bw / 2) / panel_w - self.PLATFORM[0]
        dy = (by + bh / 2) / panel_h - self.PLATFORM[1]
        deadline = (dx * dx + dy * dy) ** 0.5 / self.APPROACH_SPEED
        keys = len(detection.match.keystrokes) if detection.match else 0
        return deadline - keys * self.SECONDS_PER_KEY

    def process(self, timestamp: float,
                frame: np.ndarray) -> list[TypedWord]:
        """Scan one frame and type whatever clears both guards."""
        detections = self.detector.detect(frame, include_unmatched=True,
                                          timestamp=timestamp)
        return self.process_detections(timestamp, detections)

    def process_detections(self, timestamp: float,
                           detections: list[Detection]) -> list[TypedWord]:
        """Decide and dispatch for one scan's detections.

        Split from detection so scans can be OCRed in a pipeline (two in
        flight) while decisions stay strictly ordered -- the confirmer's
        consecutive-scan semantics depend on order.

        Words closest to the platform are typed first: everything queued
        behind a keystroke burst drops a little further while it waits, so
        the word about to die must not wait behind a fresh spawn.
        """
        panel_h = self.settings.geometry.panel_size[1]
        detections = [
            d for d in detections
            if (d.box[1] + d.box[3]) / panel_h < self.STRIKE_BAND
        ]
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
            if not self.confirmer.ready(detection, timestamp):
                self.stats.awaiting_confirmation += 1
                continue
            self.deduper.mark(detection.name, detection.pos)
            word = TypedWord(timestamp, detection, detection.match.keystrokes)
            self.dispatch(word)
            self.stats.record(word)
            typed.append(word)
        typed.extend(self._type_embedded(timestamp, detections))
        typed.extend(self._insure_weak_matches(timestamp, detections))
        # Update after the loop: a word must survive real time on screen,
        # not be confirmed by its own detection.
        self.confirmer.update(detections, timestamp)
        return typed

    def _type_embedded(self, timestamp: float,
                       detections: list[Detection]) -> list[TypedWord]:
        """Mine merged-cluster reads for words embedded letter-perfect.

        Interleaved pile-ups OCR as mush ('TEMPLAR ASSMANTA STYLE'), and
        the pile-up is exactly where words die unseen -- VISAGE lost a
        combo with one letter typed inside such a cluster. An exact vocab
        key inside a long weak read is near-certain to be a real word on
        screen, so it types immediately; keys inside the primary match's
        own key complete automatically when the match is typed.
        """
        if self.settings.safe_mode:
            return []
        lexicon = getattr(self.detector, "lexicon", None)
        if lexicon is None:
            return []
        matching = self.settings.matching
        out = []
        for detection in detections:
            match = detection.match
            if match and match.score >= matching.strong_match:
                continue
            exclude = to_key(match.name) if match else None
            for name in lexicon.embedded_words(detection.raw,
                                               exclude_key=exclude):
                if self.deduper.seen(name, detection.pos):
                    continue
                self.deduper.mark(name, detection.pos)
                embedded = Match(name, 1.0, "vocab")
                word = TypedWord(timestamp,
                                 Detection(box=detection.box,
                                           raw=detection.raw,
                                           match=embedded),
                                 embedded.keystrokes)
                self.dispatch(word)
                self.stats.record(word)
                out.append(word)
        return out

    def _insure_weak_matches(self, timestamp: float,
                             detections: list[Detection]) -> list[TypedWord]:
        """Also type the verbatim read when its match is only a guess.

        A word missing from the corpus gets fuzzy-stolen by whatever scores
        >=0.62, and that wrong guess used to suppress the verbatim fallback
        entirely -- HYPOTHERMIA died behind a weak steal. A read that
        repeats identically across scans is what is on screen, so when its
        best match is weak, the verbatim read is typed too: one of the two
        completes the word, and the loser is stray keys.
        """
        matching = self.settings.matching
        current: set[str] = set()
        insured = []
        for detection in detections:
            key = to_key(detection.raw)
            if len(key) < matching.fallback_min_length:
                continue
            current.add(key)
            match = detection.match
            if (match is None or match.source == "fallback"
                    or match.score >= matching.strong_match):
                continue    # pure fallbacks have their own gate (Confirmer)
            if self.settings.safe_mode:
                continue
            first = self._raw_first_seen.get(key)
            if first is None or timestamp - first < Confirmer.MIN_STABLE_AGE:
                # Stability is time, not scans: pipelined scans ~70ms apart
                # can read the same frame's mangle identically twice.
                continue
            verbatim = Match(key, 0.0, "fallback")
            if verbatim.keystrokes == match.keystrokes:
                continue
            if self.deduper.seen(key, detection.pos):
                continue
            self.deduper.mark(key, detection.pos)
            word = TypedWord(timestamp,
                             Detection(box=detection.box, raw=detection.raw,
                                       match=verbatim),
                             verbatim.keystrokes)
            self.dispatch(word)
            self.stats.record(word)
            insured.append(word)
        self._raw_first_seen = {
            key: self._raw_first_seen.get(key, timestamp) for key in current
        }
        return insured

    def run(self, frames: Iterable[tuple[float, np.ndarray]]
            ) -> Iterator[TypedWord]:
        for timestamp, frame in frames:
            yield from self.process(timestamp, frame)

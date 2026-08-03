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
from dataclasses import dataclass, field, replace
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

    # Entries must outlive the longest override window a caller may ask
    # for (service-stretched and lock-step windows exceed the base ttl);
    # pruning at base ttl silently disabled every longer window.
    RETENTION = 8.0

    def __init__(self, radius: int = 120, ttl: float = 3.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.radius = radius
        self.ttl = ttl
        self.clock = clock
        self._entries: list[tuple[str, tuple[int, int], float]] = []

    def seen(self, name: str, pos: tuple[int, int],
             ttl: float | None = None) -> bool:
        """``ttl`` overrides the configured window for this check only --
        the engine stretches or shortens it per word."""
        now = self.clock()
        retention = max(self.RETENTION, self.ttl)
        self._entries = [e for e in self._entries if now - e[2] < retention]
        window = self.ttl if ttl is None else ttl
        for entry_name, entry_pos, marked in self._entries:
            if entry_name != name:
                continue
            if now - marked >= window:
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
        # The keyboard's backlog in keystrokes; live mode wires this to
        # TypingWorker.pending_keystrokes so dedup windows can stretch by
        # the real service time (see _dedup_ttl).
        self.queue_keystrokes: Callable[[], int] = lambda: 0
        # Live age ledger: when each on-screen word was first read. Words
        # live ~3.5s from first readable label to the platform no matter
        # their path -- an arcing word looks geometrically safe at its
        # apex moments before it plummets.
        self._ages: list[dict] = []
        self._now = 0.0

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
    # The game only accepts the TOP word of a stacked pile: a lower word's
    # keystrokes do nothing until everything above it has cleared, so its
    # effective deadline is later by roughly one word-clear per rank.
    STACK_RANK_DELAY = 0.15     # seconds per position down the stack

    # Two labels are one bundle when they overlap horizontally by at least
    # this fraction of the narrower one and the vertical gap between them
    # is under this many label-heights. MAGIC WAND hung on screen for ~2s
    # with a phrase 16px above it: separate blobs, one bundle in-game.
    STACK_MIN_OVERLAP = 0.5
    STACK_MAX_GAP_LINES = 1.4
    # A typed word still visible this deep is a word whose keystrokes were
    # eaten (a bundle held the top slot) or dropped -- either way it is
    # about to strike, and the normal dedup window is far too patient:
    # WRAITH KING was typed at depth 0.55, blocked, and the 1.2s TTL only
    # let the retype land at 0.68 -- one scan from the platform.
    DANGER_BAND = 0.55
    DANGER_RETYPE_TTL = 0.4

    def _char_seconds(self) -> float:
        """Seconds one keystroke takes under the current speed settings."""
        wpm = self.settings.behaviour.max_wpm
        return 60.0 / (wpm * 5.0) if wpm > 0 else self.SECONDS_PER_KEY

    # Lock-step mode engages when keystrokes cost real time (<=600 WPM).
    # The game feeds ONE active word at a time and DISCARDS keys that
    # don't match it (run27 frame-by-frame: MJOLLNIR's entire first
    # typing ran during BUTTERFLY's lock -- every key wasted, audibly).
    LOCKSTEP_THRESHOLD = 0.02
    # After the selected word's keystrokes END, wait this long for the
    # kill to render and reach us through the scan pipeline before
    # concluding it survived. Run30: the refeed clock ran from EMIT, so
    # every word longer than it was instantly re-typed off a stale
    # pipeline frame the moment the keyboard went idle -- 17 of 49
    # words double-typed, ~190 ghost keys, 44% overhead.
    KILL_CONFIRM = 0.35
    # Detection can flicker for a scan or two (fast mover over a light
    # barrel); a word stays a believed-alive candidate this long after
    # its last sighting.
    FLICKER_WINDOW = 0.6

    def _process_lockstep(self, timestamp: float,
                          detections: list[Detection]) -> list[TypedWord]:
        """Throttled play: six rules, one ledger, nothing else.

        1. Never interrupt a started word (run32: detection flicker is
           OUR problem, never evidence about the game).
        2. Never idle while an eligible word exists.
        3. One word in flight, ever (run29).
        4. Eligibility comes from the LEDGER, not the scan: a word is a
           candidate if seen within the flicker window, and either never
           typed or seen again after its typing ended plus a
           kill-confirm beat -- a killed word vanishes instantly, so
           surviving your own keystrokes means it needs more of them
           (runs 25/30). Divers that blink out of the deciding scan stay
           candidates (SVEN, run32).
        5. Earliest deadline first: min(position, lifetime) minus
           service time, stacks top-first (runs 20/28).
        6. No speculation under a WPM cap: insurance and embedded mining
           assume wrong keys are free; here keys are time (run27).
        """
        if self.queue_keystrokes() > 0:
            return []                                       # rules 1 + 3
        candidates = []
        for entry in self._ages:
            det = entry.get("det")
            if det is None or not det.match:
                continue
            if timestamp - entry["last"] > self.FLICKER_WINDOW:
                continue                    # believed gone (or truly dead)
            typed_end = entry.get("typed_end")
            if typed_end is not None:
                if timestamp < typed_end + self.KILL_CONFIRM:
                    continue                # keys just landed; await kill
                if entry["last"] <= typed_end:
                    continue                # never seen again: presumed dead
            candidates.append((entry, det))
        if not candidates:
            return []
        entry, det = min(candidates, key=lambda c: self._urgency(c[1]))
        if not self.confirmer.ready(det, timestamp):
            self.stats.awaiting_confirmation += 1
            return []
        entry["typed_end"] = (timestamp
                             + len(det.match.keystrokes)
                             * self._char_seconds())
        return [self._emit(timestamp, det)]                 # rules 2 + 5

    def _dedup_ttl(self, detection: Detection) -> float | None:
        """Retype window for a typed word that is still visible (live
        only: on tape typed words never vanish, so replays -- which floor
        the TTL at 3.0 -- must not rapid-fire retypes).

        A typed word still visible is still ALIVE -- a killed word
        vanishes instantly, points and all. Deep words get the short
        danger window. Both windows stretch by the keyboard's service
        time (queue backlog plus this word's own keystrokes): with
        --max-wpm a word is still being TYPED long after submission, and
        the instant-typing assumption double-queued nearly every word of
        a 100-WPM round while real words waited 4s."""
        if self.settings.behaviour.dedup_ttl > 2.0:
            return None
        panel_h = self.settings.geometry.panel_size[1]
        bottom = (detection.box[1] + detection.box[3]) / panel_h
        base = (self.DANGER_RETYPE_TTL if bottom >= self.DANGER_BAND
                else self.settings.behaviour.dedup_ttl)
        own = len(detection.match.keystrokes) if detection.match else 0
        service = (self.queue_keystrokes() + own) * self._char_seconds()
        return base + service

    def _group_stacks(self, detections: list[Detection]) -> list[Detection]:
        """Tag vertically-adjacent, horizontally-overlapping labels as one
        stack (top-to-bottom ranks, shared bundle box).

        The line splitter tags stacks whose labels touch; this pass also
        catches bundles whose labels stay separate blobs. Grouping is
        transitive: three labels chained by pairwise adjacency are one
        stack of three.
        """
        if len(detections) < 2:
            return detections
        parent = list(range(len(detections)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(detections)):
            for j in range(len(detections)):
                if i >= j:
                    continue
                bi, bj = detections[i].box, detections[j].box
                upper, lower = (bi, bj) if bi[1] <= bj[1] else (bj, bi)
                # Stacked lines sit below each other; boxes at the same
                # height are side-by-side (or the same spot), not a stack.
                if lower[1] < upper[1] + int(upper[3] * 0.6):
                    continue
                gap = lower[1] - (upper[1] + upper[3])
                if gap > max(bi[3], bj[3]) * self.STACK_MAX_GAP_LINES:
                    continue
                overlap = (min(bi[0] + bi[2], bj[0] + bj[2])
                           - max(bi[0], bj[0]))
                if overlap >= self.STACK_MIN_OVERLAP * min(bi[2], bj[2]):
                    parent[find(i)] = find(j)

        groups: dict[int, list[int]] = {}
        for i in range(len(detections)):
            groups.setdefault(find(i), []).append(i)
        out = list(detections)
        for members in groups.values():
            if len(members) < 2:
                continue
            members.sort(key=lambda i: detections[i].box[1])
            x0 = min(detections[i].box[0] for i in members)
            y0 = min(detections[i].box[1] for i in members)
            x1 = max(detections[i].box[0] + detections[i].box[2]
                     for i in members)
            y1 = max(detections[i].box[1] + detections[i].box[3]
                     for i in members)
            bundle = (x0, y0, x1 - x0, y1 - y0)
            for rank, i in enumerate(members):
                out[i] = replace(detections[i], group_box=bundle,
                                 stack_rank=rank)
        return out

    # Observed live: strikes land 3.2-3.5s after the label first became
    # readable, regardless of trajectory.
    WORD_LIFETIME = 3.4
    AGE_FORGET = 1.0        # unseen this long = the word is gone; forget it
    AGE_MATCH_RADIUS = 300  # px a word can drift and still be itself

    def _update_ages(self, timestamp: float,
                     detections: list[Detection]) -> None:
        for d in detections:
            if not d.name:
                continue
            for entry in self._ages:
                if (entry["name"] == d.name
                        and abs(entry["pos"][0] - d.pos[0]) < self.AGE_MATCH_RADIUS
                        and abs(entry["pos"][1] - d.pos[1]) < self.AGE_MATCH_RADIUS):
                    entry["pos"] = d.pos
                    entry["last"] = timestamp
                    entry["det"] = d
                    break
            else:
                self._ages.append({"name": d.name, "pos": d.pos,
                                   "first": timestamp, "last": timestamp,
                                   "det": d})
        self._ages = [e for e in self._ages
                      if timestamp - e["last"] <= self.AGE_FORGET]
        self._now = timestamp

    def _time_left(self, detection: Detection) -> float:
        """Seconds until this word's lifetime runs out, if we know it."""
        for entry in self._ages:
            if (entry["name"] == detection.name
                    and abs(entry["pos"][0] - detection.pos[0]) < self.AGE_MATCH_RADIUS
                    and abs(entry["pos"][1] - detection.pos[1]) < self.AGE_MATCH_RADIUS):
                return self.WORD_LIFETIME - (self._now - entry["first"])
        return float("inf")

    def _urgency(self, detection: Detection) -> float:
        """Smaller = must start typing sooner.

        Deadline minus service time, not bare distance: a 30-key voice
        line needs ~0.7s of keyboard before it completes, so it must
        start earlier than a 4-key word at the same range. Two phrases
        died this exact way -- queued behind each other while both fell.

        The deadline is the SOONER of position and age: words live ~3.5s
        from first readable label no matter the path, and an arcing word
        reads as geometrically safe at its apex moments before it
        plummets -- age catches what position cannot.

        Stacked words use the whole pile's position (they fall together)
        plus a per-rank delay, so a stack always types top-first: bundles
        used to linger while we typed their ineligible lower words, then
        vanish all at once when the retype cycle finally hit the top one.
        """
        panel_w, panel_h = self.settings.geometry.panel_size
        bx, by, bw, bh = detection.group_box or detection.box
        dx = (bx + bw / 2) / panel_w - self.PLATFORM[0]
        dy = (by + bh / 2) / panel_h - self.PLATFORM[1]
        deadline = (dx * dx + dy * dy) ** 0.5 / self.APPROACH_SPEED
        deadline = min(deadline, self._time_left(detection))
        keys = len(detection.match.keystrokes) if detection.match else 0
        return (deadline - keys * self.SECONDS_PER_KEY
                + detection.stack_rank * self.STACK_RANK_DELAY)

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
        detections = self._group_stacks(detections)
        self._update_ages(timestamp, detections)
        detections.sort(key=self._urgency)
        self.last_detections = detections
        self.stats.frames += 1
        self.stats.detections += sum(1 for d in detections if d.match)
        if self._char_seconds() >= self.LOCKSTEP_THRESHOLD:
            typed = self._process_lockstep(timestamp, detections)
            self.confirmer.update(detections, timestamp)
            return typed
        typed = []
        for detection in detections:
            if not detection.match:
                continue
            if self.deduper.seen(detection.name, detection.pos,
                                 ttl=self._dedup_ttl(detection)):
                self.stats.suppressed_duplicate += 1
                continue
            if not self.confirmer.ready(detection, timestamp):
                self.stats.awaiting_confirmation += 1
                continue
            typed.append(self._emit(timestamp, detection))
        typed.extend(self._type_embedded(timestamp, detections))
        typed.extend(self._insure_weak_matches(timestamp, detections))
        # Update after the loop: a word must survive real time on screen,
        # not be confirmed by its own detection.
        self.confirmer.update(detections, timestamp)
        return typed

    def _emit(self, timestamp: float, detection: Detection,
              match: Match | None = None) -> TypedWord:
        """Mark the deduper, build the word, dispatch it, count it.

        With ``match``, the word types as that match instead of the
        detection's own (embedded mining and verbatim insurance ride on
        another detection's box and raw read).
        """
        if match is not None:
            detection = replace(detection, match=match)
        self.deduper.mark(detection.name, detection.pos)
        word = TypedWord(timestamp, detection, detection.match.keystrokes)
        self.dispatch(word)
        self.stats.record(word)
        return word

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
                if self.deduper.seen(name, detection.pos,
                                     ttl=self._dedup_ttl(detection)):
                    continue
                out.append(self._emit(timestamp, detection,
                                      Match(name, 1.0, "vocab")))
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
            insured.append(self._emit(timestamp, detection, verbatim))
        self._raw_first_seen = {
            key: self._raw_first_seen.get(key, timestamp) for key in current
        }
        return insured

    def run(self, frames: Iterable[tuple[float, np.ndarray]]
            ) -> Iterator[TypedWord]:
        for timestamp, frame in frames:
            yield from self.process(timestamp, frame)

"""The two guards between detecting a word and typing it."""

import numpy as np
import pytest

from automaton_attack.config import Settings
from automaton_attack.detect import Detection
from automaton_attack.engine import Confirmer, Deduper, Engine
from automaton_attack.keyboard import DryRunTypist
from automaton_attack.lexicon import Match


def detection(name, score, pos=(100, 100), raw=None, source="vocab"):
    return Detection(
        box=(pos[0], pos[1], 120, 20),
        raw=raw if raw is not None else name,
        match=Match(name, score, source),
    )


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


# -- Deduper -------------------------------------------------------------
def test_deduper_suppresses_same_word_at_same_place():
    clock = FakeClock()
    dedup = Deduper(radius=120, ttl=3.0, clock=clock)
    assert not dedup.seen("BANE", (100, 100))
    dedup.mark("BANE", (100, 100))
    assert dedup.seen("BANE", (110, 105))


def test_deduper_allows_same_word_far_away():
    """Two copies of a word on screen at once are two words to type."""
    dedup = Deduper(radius=120, ttl=3.0, clock=FakeClock())
    dedup.mark("BANE", (100, 100))
    assert not dedup.seen("BANE", (600, 100))


def test_deduper_forgets_after_ttl():
    """Re-seeing a word later means keystrokes were dropped or it respawned.
    Both call for retyping."""
    clock = FakeClock()
    dedup = Deduper(radius=120, ttl=3.0, clock=clock)
    dedup.mark("BANE", (100, 100))
    clock.advance(3.5)
    assert not dedup.seen("BANE", (100, 100))


# -- Confirmer -----------------------------------------------------------
def test_high_confidence_types_immediately():
    confirmer = Confirmer(high_confidence=0.85)
    assert confirmer.ready(detection("BANE", 0.95))


def test_low_confidence_needs_a_second_sighting():
    confirmer = Confirmer(high_confidence=0.85)
    low = detection("METEOR HAMMER", 0.70)
    assert not confirmer.ready(low)
    confirmer.update([low])
    assert confirmer.ready(low)


def test_fallback_reads_always_need_confirmation():
    """Fallback matches score 0.0 precisely so they cannot skip this."""
    confirmer = Confirmer(high_confidence=0.85)
    fallback = detection("A LONG UNKNOWN PHRASE", 0.0, source="fallback")
    assert not confirmer.ready(fallback)


def test_confirmer_forgets_words_that_vanish():
    confirmer = Confirmer(high_confidence=0.85)
    low = detection("METEOR HAMMER", 0.70)
    confirmer.update([low])
    confirmer.update([])
    assert not confirmer.ready(low)


# -- Engine --------------------------------------------------------------
class FakeDetector:
    """Replays canned detections, one list per frame."""

    def __init__(self, frames, settings=None):
        self._frames = list(frames)
        self.settings = settings or Settings()
        self._index = 0

    def detect(self, frame, include_unmatched=False):
        if self._index >= len(self._frames):
            return []
        result = self._frames[self._index]
        self._index += 1
        return result


def run_engine(frames, settings=None):
    settings = settings or Settings()
    detector = FakeDetector(frames, settings)
    typist = DryRunTypist(settings.behaviour)
    engine = Engine(detector, typist, settings)
    blank = np.zeros((4, 4, 3), np.uint8)
    typed = []
    for index in range(len(frames)):
        typed.extend(engine.process(float(index), blank))
    return typed, typist, engine


def test_engine_types_high_confidence_word_once():
    typed, typist, _ = run_engine([[detection("BANE", 1.0)],
                                   [detection("BANE", 1.0)]])
    assert [w.name for w in typed] == ["BANE"]
    assert typist.typed == ["bane"]


def test_engine_holds_then_types_a_repeated_low_confidence_read():
    low = detection("METEOR HAMMER", 0.70)
    typed, typist, engine = run_engine([[low], [low]])
    assert [w.name for w in typed] == ["METEOR HAMMER"]
    assert typist.typed == ["meteorhammer"]
    assert engine.stats.awaiting_confirmation == 1


def test_engine_never_types_a_one_off_low_confidence_read():
    """A word glimpsed in a single scan is exactly what the guard is for."""
    typed, typist, _ = run_engine([[detection("METEOR HAMMER", 0.70)],
                                   [detection("BANE", 1.0)]])
    assert [w.name for w in typed] == ["BANE"]
    assert "meteorhammer" not in typist.typed


def test_engine_strips_spaces_and_punctuation_before_typing():
    typed, typist, _ = run_engine([[detection("BANE OF YOUR EXISTENCE.", 1.0,
                                              source="phrase")]])
    assert typist.typed == ["baneofyourexistence"]
    assert typed[0].source == "phrase"


def test_engine_counts_what_it_did():
    _, _, engine = run_engine([[detection("BANE", 1.0)],
                               [detection("BANE", 1.0)],
                               [detection("PUDGE", 1.0, pos=(400, 400))]])
    assert engine.stats.typed == 2
    assert engine.stats.frames == 3
    assert engine.stats.by_source == {"vocab": 2}
    assert engine.stats.suppressed_duplicate == 1


def test_dry_run_typist_sends_nothing_live():
    typist = DryRunTypist()
    assert typist.live is False


@pytest.mark.parametrize("score", [0.0, 0.5, 0.84])
def test_everything_below_the_bar_is_held(score):
    confirmer = Confirmer(high_confidence=0.85)
    assert not confirmer.ready(detection("SOMETHING", score))

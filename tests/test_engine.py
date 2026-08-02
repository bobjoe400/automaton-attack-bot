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
def test_corpus_matches_type_immediately_at_any_score():
    """Wrong keystrokes are free; a held word can escape and cost the
    multiplier. Every corpus match goes through on first sight."""
    confirmer = Confirmer()
    assert confirmer.ready(detection("BANE", 0.95))
    assert confirmer.ready(detection("METEOR HAMMER", 0.70))
    assert confirmer.ready(detection("SAND KING", 0.63))
    assert confirmer.ready(detection("BANE OF YOUR EXISTENCE.", 0.80,
                                     source="phrase"))


def test_fallback_reads_need_an_exact_repeat():
    """Raw OCR with no corpus anchor: repetition is the only evidence the
    read is right, and unrepeated flickers would burn keystrokes."""
    confirmer = Confirmer()
    fallback = detection("A LONG UNKNOWN PHRASE", 0.0, source="fallback")
    assert not confirmer.ready(fallback)
    confirmer.update([fallback])
    assert confirmer.ready(fallback)


def test_confirmer_forgets_fallbacks_that_vanish():
    confirmer = Confirmer()
    fallback = detection("A LONG UNKNOWN PHRASE", 0.0, source="fallback")
    confirmer.update([fallback])
    confirmer.update([])
    assert not confirmer.ready(fallback)


# -- Engine --------------------------------------------------------------
class FakeDetector:
    """Replays canned detections, one list per frame."""

    def __init__(self, frames, settings=None):
        self._frames = list(frames)
        self.settings = settings or Settings()
        self._index = 0

    def detect(self, frame, include_unmatched=False, timestamp=None):
        if self._index >= len(self._frames):
            return []
        result = list(self._frames[self._index])
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


def test_engine_types_low_confidence_corpus_match_on_first_sight():
    typed, typist, engine = run_engine([[detection("METEOR HAMMER", 0.70)]])
    assert [w.name for w in typed] == ["METEOR HAMMER"]
    assert typist.typed == ["meteorhammer"]
    assert engine.stats.awaiting_confirmation == 0


def test_engine_holds_then_types_a_repeated_fallback():
    fallback = detection("SOMELONGPHRASE", 0.0, source="fallback")
    typed, typist, engine = run_engine([[fallback], [fallback]])
    assert [w.name for w in typed] == ["SOMELONGPHRASE"]
    assert typist.typed == ["somelongphrase"]
    assert engine.stats.awaiting_confirmation == 1


def test_engine_never_types_a_one_off_fallback():
    """An unanchored read glimpsed once is exactly what the guard is for."""
    typed, typist, _ = run_engine(
        [[detection("SOMELONGPHRASE", 0.0, source="fallback")],
         [detection("BANE", 1.0)]])
    assert [w.name for w in typed] == ["BANE"]
    assert "somelongphrase" not in typist.typed


def test_a_corrected_read_types_as_a_new_word():
    """First scan guesses WEAVE, second reads WEAVER: both type. The first
    attempt is stray keys; the second completes the word."""
    typed, typist, _ = run_engine([[detection("WEAVE", 0.91)],
                                   [detection("WEAVER", 1.0)]])
    assert typist.typed == ["weave", "weaver"]


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


# -- monitor picking -------------------------------------------------------
def test_pick_monitor_prefers_the_overlapping_screen():
    from automaton_attack.capture import pick_monitor

    monitors = [
        {"left": 0, "top": 0, "width": 3840, "height": 1080},      # virtual
        {"left": 0, "top": 0, "width": 1920, "height": 1080},      # primary
        {"left": 1920, "top": 0, "width": 1920, "height": 1080},   # second
    ]
    dota_on_second = (2000, 50, 3800, 1000)
    assert pick_monitor(dota_on_second, monitors) == 2
    dota_on_primary = (10, 10, 1900, 1000)
    assert pick_monitor(dota_on_primary, monitors) == 1
    straddling_mostly_second = (1500, 0, 3500, 1080)
    assert pick_monitor(straddling_mostly_second, monitors) == 2


def test_pick_monitor_with_no_overlap_returns_none():
    from automaton_attack.capture import pick_monitor

    monitors = [
        {"left": 0, "top": 0, "width": 1920, "height": 1080},
        {"left": 0, "top": 0, "width": 1920, "height": 1080},
    ]
    assert pick_monitor((-5000, -5000, -4000, -4000), monitors) is None


def test_words_nearest_the_platform_type_first():
    """Automatons converge on Hoodwink at panel-centre; the word about to
    reach her must not wait behind a fresh spawn. Spawns from the bottom
    start close to the platform, so this covers them too."""
    far_top = detection("BANE", 1.0, pos=(60, 40))
    near_platform = detection("PUDGE", 1.0, pos=(430, 580))
    bottom_spawn = detection("LINA", 1.0, pos=(450, 900))
    typed, typist, _ = run_engine([[far_top, near_platform, bottom_spawn]])
    assert typist.typed == ["pudge", "lina", "bane"]


# -- verbatim insurance ------------------------------------------------------
def test_stable_read_with_weak_match_types_both():
    """HYPOTHERMIA scenario: an out-of-corpus word gets fuzzy-stolen at a
    weak score, and the wrong guess used to block the verbatim fallback.
    Once the read proves stable, both are typed -- one of them wins."""
    weak = detection("HYPNOTIZE", 0.65, raw="HYPOTHERMIA")
    typed, typist, _ = run_engine([[weak], [weak]])
    assert "hypnotize" in typist.typed          # the guess, on first sight
    assert "hypothermia" in typist.typed        # the insurance, once stable


def test_one_off_weak_match_gets_no_insurance():
    """A single sighting can be a misread; insurance waits for stability."""
    typed, typist, _ = run_engine([[detection("HYPNOTIZE", 0.65,
                                              raw="HYPOTHERMIA")]])
    assert typist.typed == ["hypnotize"]


def test_strong_matches_get_no_insurance():
    strong = detection("SILENCER", 0.93, raw="SLENCER")
    typed, typist, _ = run_engine([[strong], [strong]])
    assert typist.typed == ["silencer"]


def test_short_weak_reads_get_no_insurance():
    """'CIOAK'->CLOAK at 0.80 is a misread of a real word, not a gap; the
    length floor keeps insurance off it."""
    weak = detection("CLOAK", 0.80, raw="CIOAK")
    typed, typist, _ = run_engine([[weak], [weak]])
    assert typist.typed == ["cloak"]


def test_long_phrases_win_the_tie_at_equal_range():
    """Urgency is deadline MINUS service time, so at equal range the
    longer keystroke burden starts first. (At full typing speed the
    service term is small, but the tie-break still matters: two phrases
    once died queued behind each other.)"""
    phrase = detection("TOLD YOU A STORM WAS COMING!", 1.0, pos=(300, 300),
                       source="phrase")
    word = detection("AXE", 1.0, pos=(300, 300))
    typed, typist, _ = run_engine([[word, phrase]])
    assert typist.typed[0] == "toldyouastormwascoming"

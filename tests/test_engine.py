"""The two guards between detecting a word and typing it."""

import numpy as np
import pytest

from automaton_attack_bot.core.config import Settings
from automaton_attack_bot.core.detect import Detection
from automaton_attack_bot.core.engine import Confirmer, Deduper, Engine
from automaton_attack_bot.core.keyboard import DryRunTypist
from automaton_attack_bot.core.lexicon import Match


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
    assert confirmer.ready(detection("BANE", 0.95), 0.0)
    assert confirmer.ready(detection("METEOR HAMMER", 0.70), 0.0)
    assert confirmer.ready(detection("SAND KING", 0.63), 0.0)
    assert confirmer.ready(detection("BANE OF YOUR EXISTENCE.", 0.80,
                                     source="phrase"), 0.0)


def test_fallback_reads_need_stability_over_time():
    """Raw OCR with no corpus anchor must persist for real TIME, not just
    consecutive scans: pipelined scans ~70ms apart can read the same
    frame's mangle identically twice (CRYSTLY, ABADDOMNIKNIGHT and other
    garbage got typed that way)."""
    confirmer = Confirmer()
    fallback = detection("A LONG UNKNOWN PHRASE", 0.0, source="fallback")
    assert not confirmer.ready(fallback, 0.0)
    confirmer.update([fallback], 0.0)
    assert not confirmer.ready(fallback, 0.1)   # same-frame repeat: no
    confirmer.update([fallback], 0.1)
    assert confirmer.ready(fallback, 0.5)       # persisted 0.5s: yes


def test_confirmer_forgets_fallbacks_that_vanish():
    confirmer = Confirmer()
    fallback = detection("A LONG UNKNOWN PHRASE", 0.0, source="fallback")
    confirmer.update([fallback], 0.0)
    confirmer.update([], 0.5)
    assert not confirmer.ready(fallback, 1.0)


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
    from automaton_attack_bot.core.capture import pick_monitor

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
    from automaton_attack_bot.core.capture import pick_monitor

    monitors = [
        {"left": 0, "top": 0, "width": 1920, "height": 1080},
        {"left": 0, "top": 0, "width": 1920, "height": 1080},
    ]
    assert pick_monitor((-5000, -5000, -4000, -4000), monitors) is None


def test_words_nearest_the_platform_type_first():
    """Automatons converge on Hoodwink at panel-centre; the word about to
    reach her must not wait behind a fresh spawn. Spawns from below start
    close to the platform (but above the strike band -- anything below
    that is already dead). Horizontally separate: overlapping labels
    would be a bundle and type top-first instead."""
    far_top = detection("BANE", 1.0, pos=(60, 40))
    near_platform = detection("PUDGE", 1.0, pos=(430, 500))
    below_spawn = detection("LINA", 1.0, pos=(500, 650))
    typed, typist, _ = run_engine([[far_top, near_platform, below_spawn]])
    assert typist.typed == ["lina", "pudge", "bane"]


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


def test_process_detections_matches_process():
    """The pipelined path (detect elsewhere, decide here) must behave
    identically to the inline path."""
    settings = Settings()
    typist = DryRunTypist(settings.behaviour)
    engine = Engine(FakeDetector([], settings), typist, settings)
    words = engine.process_detections(
        1.0, [detection("BANE", 1.0), detection("PUDGE", 1.0,
                                                pos=(430, 580))])
    assert [w.name for w in words] == ["PUDGE", "BANE"]   # urgency order
    assert typist.typed == ["pudge", "bane"]


def test_embedded_words_type_from_weak_cluster_reads():
    """VISAGE died with one letter typed inside an interleaved pile-up.
    A weak merged read containing a vocab key letter-perfect types that
    word immediately."""
    from automaton_attack_bot.core.lexicon import Lexicon

    settings = Settings()
    detector = FakeDetector([], settings)
    detector.lexicon = Lexicon(["Templar Assassin", "Manta Style"])
    typist = DryRunTypist(settings.behaviour)
    engine = Engine(detector, typist, settings)
    weak = detection("TEMPLAR ASSASSIN", 0.69,
                     raw="TEMPLAR ASSMANTA STYLE")
    engine.process_detections(0.0, [weak])
    assert "templarassassin" in typist.typed
    assert "mantastyle" in typist.typed


def test_strike_band_corpses_are_never_typed():
    """A label below 70% panel height is a word that already struck --
    the game's kill display. KAYA's corpse got typed at top queue
    priority and starved living words right after a strike."""
    corpse = detection("KAYA", 1.0, pos=(524, 819))
    living = detection("TANGO", 1.0, pos=(461, 600))
    typed, typist, _ = run_engine([[corpse, living]])
    assert typist.typed == ["tango"]


# -- stacked bundles ---------------------------------------------------------
def stacked(name, rank, group, pos):
    """A word that is one line of a taller pile."""
    return Detection(
        box=(pos[0], pos[1], 120, 20),
        raw=name,
        match=Match(name, 1.0, "vocab"),
        group_box=group,
        stack_rank=rank,
    )


def test_stacks_type_top_first_regardless_of_range():
    """The game only accepts the TOP word of a pile: bundles lingered
    while we typed ineligible lower words, then vanished all at once when
    the retype cycle finally hit the top one. Top-first, always -- even
    though the bottom word is nearest the platform."""
    group = (400, 500, 140, 80)
    bottom = stacked("PUDGE", 2, group, (400, 560))
    middle = stacked("BANE", 1, group, (400, 530))
    top = stacked("MARCI", 0, group, (400, 500))
    typed, typist, _ = run_engine([[bottom, middle, top]])
    assert typist.typed == ["marci", "bane", "pudge"]


def test_a_stack_uses_the_piles_deadline_not_each_words():
    """Stack members fall together; the pile's top word must not wait
    behind a lone word that is farther out."""
    group = (430, 560, 140, 60)      # pile right by the platform
    pile_top = stacked("LINA", 0, group, (430, 560))
    lone_far = detection("BANE", 1.0, pos=(60, 40))
    typed, typist, _ = run_engine([[lone_far, pile_top]])
    assert typist.typed[0] == "lina"


def test_separate_blobs_that_bundle_type_top_first():
    """The MAGIC WAND case: a phrase 16px above a word, separate blobs,
    one in-game bundle. The wand's first keystrokes were eaten while the
    phrase above held the top slot -- so the phrase must type first even
    though the word below is nearer the platform."""
    phrase = Detection(box=(300, 423, 520, 20), raw="YOU'RE IN OVER YOUR HEAD.",
                       match=Match("YOU'RE IN OVER YOUR HEAD.", 1.0, "phrase"))
    wand = Detection(box=(430, 459, 200, 20), raw="MAGIC WAND",
                     match=Match("MAGIC WAND", 1.0, "vocab"))
    typed, typist, _ = run_engine([[wand, phrase]])
    assert typist.typed == ["youreinoveryourhead", "magicwand"]


def test_side_by_side_words_do_not_bundle():
    """Horizontally separate words at similar heights are independent --
    normal urgency order applies (nearest the platform first)."""
    left = detection("BANE", 1.0, pos=(100, 420))
    right = detection("PUDGE", 1.0, pos=(700, 430))
    typed, typist, _ = run_engine([[left, right]])
    assert typist.typed == ["pudge", "bane"]


# -- danger-zone retype ------------------------------------------------------
def run_clocked(frames, settings=None, step=0.5):
    """Like run_engine, but with a controllable dedup clock."""
    settings = settings or Settings()
    detector = FakeDetector(frames, settings)
    typist = DryRunTypist(settings.behaviour)
    engine = Engine(detector, typist, settings)
    clock = FakeClock()
    engine.deduper = Deduper(radius=settings.behaviour.dedup_radius,
                             ttl=settings.behaviour.dedup_ttl, clock=clock)
    blank = np.zeros((4, 4, 3), np.uint8)
    for index in range(len(frames)):
        engine.process(index * step, blank)
        clock.advance(step)
    return typist


def test_a_deep_word_that_would_not_die_retypes_fast():
    """LEGION COMMANDER: typed once, keys eaten by the bundle above, and
    it stayed ALIVE for 2.5s -- a killed word vanishes instantly, so a
    typed word still visible needs retyping. Direction is irrelevant:
    bottom-spawned words RISE toward the platform (Legion rose the whole
    time); a sinking-only gate once cost it 1.6 extra seconds."""
    typed_at = detection("LEGION COMMANDER", 1.0, pos=(222, 643))
    risen = detection("LEGION COMMANDER", 1.0, pos=(236, 613))
    typist = run_clocked([[typed_at], [risen]], step=0.5)
    assert typist.typed == ["legioncommander", "legioncommander"]


def test_a_shallow_word_keeps_the_calm_dedup_window():
    shallow = detection("BANE", 1.0, pos=(100, 100))
    typist = run_clocked([[shallow], [shallow]], step=0.5)
    assert typist.typed == ["bane"]


def test_replay_ttls_disable_the_danger_retype():
    """On tape a typed word never disappears; replays floor dedup_ttl at
    3.0 and must not rapid-fire retypes at deep words."""
    data = Settings().to_dict()
    data["behaviour"]["dedup_ttl"] = 3.0
    settings = Settings.from_dict(data)
    deep = detection("WRAITH KING", 1.0, pos=(430, 545))
    typist = run_clocked([[deep], [deep]], settings=settings, step=0.5)
    assert typist.typed == ["wraithking"]


def test_throttled_typing_does_not_double_type():
    """Run25 at 100 WPM: nearly every word typed TWICE. A capped word is
    still being typed when the instant-typing dedup window expires, so
    the window stretches by the word's own service time. (With several
    words visible there is no held lock, so the sole-visible fast refeed
    does not apply.)"""
    data = Settings().to_dict()
    data["behaviour"]["max_wpm"] = 100.0
    settings = Settings.from_dict(data)
    deep = detection("SKULL BASHER", 1.0, pos=(430, 545))
    other = detection("AXE", 1.0, pos=(100, 100))
    typist = run_clocked([[deep, other], [deep, other]],
                         settings=settings, step=0.5)
    assert typist.typed.count("skullbasher") == 1


def test_keyboard_backlog_stretches_the_window_too():
    """A word behind a deep queue has not even STARTED typing when the
    base window expires."""
    data = Settings().to_dict()
    data["behaviour"]["max_wpm"] = 100.0
    settings = Settings.from_dict(data)
    detector = FakeDetector([], settings)
    typist = DryRunTypist(settings.behaviour)
    engine = Engine(detector, typist, settings)
    engine.queue_keystrokes = lambda: 40        # 40 keys * 0.12s = 4.8s
    deep = detection("AXE", 1.0, pos=(430, 545))
    assert engine._dedup_ttl(deep) > 4.8


# -- lock-step throttled play ------------------------------------------------
def wpm_settings(wpm=150.0):
    data = Settings().to_dict()
    data["behaviour"]["max_wpm"] = wpm
    return Settings.from_dict(data)


def test_lockstep_never_emits_while_keys_are_in_flight():
    """Run29: 'active word vanished' used to mean 'done' -- but during a
    lock every word EXCEPT the selection is invisible, so the engine
    blind-queued words into 3s waits, keys typed into the void. One word
    in flight, ever."""
    settings = wpm_settings()
    detector = FakeDetector([], settings)
    typist = DryRunTypist(settings.behaviour)
    engine = Engine(detector, typist, settings)
    engine.queue_keystrokes = lambda: 8
    a = detection("KAYA", 1.0, pos=(300, 300))
    b = detection("MJOLLNIR", 1.0, pos=(700, 300))
    assert engine.process_detections(0.0, [a, b]) == []
    assert typist.typed == []


def test_lockstep_feeds_next_word_when_several_are_visible():
    """Several visible words = no lock held; the most urgent becomes the
    selection with our first key."""
    a = detection("KAYA", 1.0, pos=(300, 300))
    b = detection("MJOLLNIR", 1.0, pos=(700, 300))
    typist = run_clocked([[a, b], [a, b]], settings=wpm_settings(), step=0.5)
    assert typist.typed == ["kaya", "mjollnir"]


def test_lockstep_moves_on_when_the_active_word_dies():
    a = detection("KAYA", 1.0, pos=(300, 300))
    b = detection("MJOLLNIR", 1.0, pos=(700, 300))
    typist = run_clocked([[a, b], [b]], settings=wpm_settings(), step=0.5)
    assert typist.typed == ["kaya", "mjollnir"]


def test_lockstep_refeeds_the_sole_visible_word_fast():
    """Exactly one visible word IS the game's selection: while it
    survives our keystrokes it needs more of them, on a short clock --
    a full retype's tail completes it wherever its progress stands."""
    a = detection("KAYA", 1.0, pos=(300, 300))
    typist = run_clocked([[a], [a], [a]], settings=wpm_settings(), step=0.5)
    assert typist.typed == ["kaya", "kaya", "kaya"]


def test_lockstep_skips_insurance_and_embedded():
    """Keys are time at capped WPM; speculative typing is off."""
    weak = detection("HYPNOTIZE", 0.65, raw="HYPOTHERMIA")
    typist = run_clocked([[weak], [weak]], settings=wpm_settings(), step=0.2)
    assert typist.typed == ["hypnotize"]


# -- word ages ---------------------------------------------------------------
def test_an_old_high_word_outranks_a_fresh_closer_one():
    """Words live ~3.5s from first readable label no matter the path; an
    arcing word looks geometrically safe at its apex right before it
    plummets. Age is a deadline position cannot see."""
    old_high = detection("LYCAN", 1.0, pos=(200, 120))
    fresh_mid = detection("KHANDA", 1.0, pos=(430, 500))
    settings = Settings()
    detector = FakeDetector([], settings)
    typist = DryRunTypist(settings.behaviour)
    engine = Engine(detector, typist, settings)
    engine.process_detections(0.0, [old_high])
    engine.process_detections(2.8, [old_high, fresh_mid])
    # LYCAN was typed at 0.0; at 2.8 it has ~0.6s left vs KHANDA's ~1.1s
    assert engine._urgency(old_high) < engine._urgency(fresh_mid)


def test_ages_forget_words_that_left_the_screen():
    settings = Settings()
    engine = Engine(FakeDetector([], settings), DryRunTypist(), settings)
    seen = detection("LYCAN", 1.0, pos=(200, 120))
    engine.process_detections(0.0, [seen])
    engine.process_detections(2.0, [])          # gone for 2s: killed
    engine.process_detections(2.1, [seen])      # a NEW spawn of the name
    assert engine._time_left(seen) > 3.0

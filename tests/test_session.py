"""Game-state tracking, panel location, and the full-game regression.

The clip-marked tests use ``clips/fullgame.mp4``: start screen, one full
round, game-over screen with a total score of 750.
"""

from pathlib import Path

import numpy as np
import pytest

from automaton_attack_bot.core.config import Settings
from automaton_attack_bot.core.session import (
    GameState,
    SessionTracker,
    _fuzzy_contains,
    locate_panel,
)

FULLGAME = Path("clips") / "fullgame.mp4"
FULLGAME_SCORE = 750


# -- unit ------------------------------------------------------------------
@pytest.mark.parametrize("read, key", [
    ("AUTOMATONATTACK", "AUTOMATONATTACK"),
    ("AUTOMATONATACK", "AUTOMATONATTACK"),    # observed rapidocr read
    ("CAMEOVER", "GAMEOVER"),                 # observed rapidocr read
    ("GAMEOVER", "GAMEOVER"),
])
def test_observed_title_misreads_still_match(read, key):
    assert _fuzzy_contains(read, key)


@pytest.mark.parametrize("read, key", [
    ("", "GAMEOVER"),
    ("RAZO", "GAMEOVER"),           # stray gameplay word in the title strip
    ("HIGHSCORES", "AUTOMATONATTACK"),
])
def test_unrelated_text_does_not_match(read, key):
    assert not _fuzzy_contains(read, key)


def test_button_positions_land_inside_the_panel():
    settings = Settings()
    tracker = SessionTracker(backend=None, settings=settings)
    x0, y0, x1, y1 = settings.geometry.panel
    for state in (GameState.START_SCREEN, GameState.GAME_OVER):
        bx, by = tracker.button_position(state)
        assert x0 < bx < x1
        assert y0 < by < y1
    # PLAY sits lower than PLAY AGAIN on their respective screens
    assert (tracker.button_position(GameState.START_SCREEN)[1]
            > tracker.button_position(GameState.GAME_OVER)[1])


def test_with_panel_rescales_geometry_and_blobs():
    settings = Settings()
    # same panel, doubled in size
    x0, y0, x1, y1 = settings.geometry.panel
    scaled = settings.with_panel((100, 50, 100 + 2 * (x1 - x0),
                                  50 + 2 * (y1 - y0)))
    assert scaled.geometry.panel[0] == 100
    assert scaled.blobs.min_width == settings.blobs.min_width * 2
    assert scaled.geometry.hud_boxes[0][2] == settings.geometry.hud_boxes[0][2] * 2


def test_locate_panel_rejects_a_blank_frame():
    assert locate_panel(np.zeros((1080, 1920, 3), np.uint8)) is None


# -- playing-state hardening -------------------------------------------------
def _paint_boxes(frame, hsv_color, boxes, rng):
    import cv2

    settings = Settings()
    x0, y0, _, _ = settings.geometry.panel
    h, s, v = hsv_color
    for bx0, by0, bx1, by1 in boxes:
        shape = (by1 - by0, bx1 - bx0)
        pixels = np.stack([
            np.clip(rng.normal(h, 1.5, shape), 0, 179),
            np.clip(rng.normal(s, 8, shape), 0, 255),
            np.clip(rng.uniform(v - 60, v + 45, shape), 0, 255),
        ], axis=-1).astype(np.uint8)
        frame[y0 + by0:y0 + by1, x0 + bx0:x0 + bx1] = \
            cv2.cvtColor(pixels, cv2.COLOR_HSV2BGR)


def _classify_synthetic(frame):
    tracker = SessionTracker(backend=None, settings=Settings())
    return tracker.classify(0.0, frame)


def test_gold_menu_text_is_not_playing():
    """The arcade pages leak saturated gold UI text into the HUD regions;
    it must fail the colour plausibility gate, not read as a running game.
    This is a real false positive observed on the start screen."""
    rng = np.random.default_rng(3)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    gold = (25, 190, 200)      # far more saturated than the HUD yellow-green
    _paint_boxes(frame, gold, Settings().geometry.hud_boxes, rng)
    assert _classify_synthetic(frame) is not GameState.PLAYING


def test_one_lit_box_is_not_playing():
    """Score, timer and high-score are all present in a real round; a single
    lit rectangle is page furniture, not gameplay."""
    rng = np.random.default_rng(3)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    hud = (39, 102, 165)       # genuine HUD colour, but only one box
    _paint_boxes(frame, hud, Settings().geometry.hud_boxes[:1], rng)
    assert _classify_synthetic(frame) is not GameState.PLAYING


def test_all_boxes_in_hud_colour_is_playing():
    rng = np.random.default_rng(3)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    hud = (39, 102, 165)
    _paint_boxes(frame, hud, Settings().geometry.hud_boxes, rng)
    assert _classify_synthetic(frame) is GameState.PLAYING


# -- full-game clip ----------------------------------------------------------
pytestmark_clips = pytest.mark.clips


@pytest.fixture(scope="module")
def fullgame_frames():
    if not FULLGAME.exists():
        pytest.skip(f"{FULLGAME} not present")
    import cv2

    capture = cv2.VideoCapture(str(FULLGAME))
    fps = capture.get(cv2.CAP_PROP_FPS) or 60

    def grab(seconds):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(seconds * fps))
        ok, frame = capture.read()
        assert ok, f"no frame at {seconds}s"
        return frame

    yield grab
    capture.release()


@pytest.fixture(scope="module")
def tracker():
    from automaton_attack_bot.core.ocr import OcrUnavailable, get_backend

    try:
        backend = get_backend("auto")
    except (OcrUnavailable, Exception) as exc:  # noqa: BLE001
        pytest.skip(f"no OCR backend: {exc}")
    return SessionTracker(backend, Settings())


@pytest.mark.clips
def test_locate_panel_matches_measured_geometry(fullgame_frames):
    located = locate_panel(fullgame_frames(1.0))
    assert located is not None
    for got, expected in zip(located, Settings().geometry.panel):
        assert abs(got - expected) <= 10


@pytest.mark.clips
def test_states_classify_across_the_round(fullgame_frames, tracker):
    assert tracker.classify(1.0, fullgame_frames(1.0)) is GameState.START_SCREEN
    assert tracker.classify(30.0, fullgame_frames(30.0)) is GameState.PLAYING
    assert tracker.classify(70.4, fullgame_frames(70.4)) is GameState.GAME_OVER


@pytest.mark.clips
def test_final_score_is_read(fullgame_frames, tracker):
    assert tracker.read_final_score(fullgame_frames(70.4)) == FULLGAME_SCORE


@pytest.mark.clips
def test_full_game_replay_walks_the_whole_lifecycle():
    from automaton_attack_bot.core.capture import VideoSource
    from automaton_attack_bot.core.detect import Detector
    from automaton_attack_bot.core.engine import Engine
    from automaton_attack_bot.core.lexicon import Lexicon
    from automaton_attack_bot.core.ocr import get_backend

    settings = Settings()
    # replay semantics: typed words never vanish on tape (see test_clips)
    data = settings.to_dict()
    data["behaviour"]["dedup_ttl"] = max(3.0, data["behaviour"]["dedup_ttl"])
    settings = Settings.from_dict(data)
    backend = get_backend("auto")
    lexicon = Lexicon.load()
    source = VideoSource(FULLGAME, settings.behaviour.replay_stride)
    settings = settings.for_resolution(source.width, source.height)
    detector = Detector(lexicon, backend, settings)
    engine = Engine(detector, settings=settings)
    tracker = SessionTracker(backend, settings)

    for timestamp, frame in source.frames():
        if tracker.classify(timestamp, frame) is GameState.PLAYING:
            engine.process(timestamp, frame)

    states = [t.state for t in tracker.transitions]
    assert states == [GameState.START_SCREEN, GameState.PLAYING,
                      GameState.GAME_OVER]
    assert tracker.final_score == FULLGAME_SCORE
    assert engine.stats.typed >= 40      # a full round of words came through


# -- score parsing -----------------------------------------------------------
@pytest.mark.parametrize("row, expected", [
    ("TOTAL SCORE 750", 750),
    ("TOTAL SCORE 1,150", 1150),
    ("TOTALSCORE1150", 1150),
    ("TOTALSCOREL150", 1150),      # "1," read as L -- observed live
    ("TOTAL SCORC750", 750),       # label mangled: trailing digits only
    ("TOTAL SCORE I,I50", 1150),   # ones as I
    ("", None),
    ("PLAY AGAIN", None),
])
def test_parse_score_survives_ocr_noise(row, expected):
    from automaton_attack_bot.core.session import parse_score

    assert parse_score(row) == expected


def _score_tracker(reads):
    tracker = SessionTracker(backend=None, settings=Settings())
    read_iter = iter(reads)
    tracker.read_final_score = lambda frame: next(read_iter)
    tracker._read_bright_text = lambda region: "GAMEOVER"
    return tracker


def _feed(tracker, reads, start=0.0, step=0.3):
    frame = np.zeros((1080, 1920, 3), np.uint8)
    timestamps = [start + i * step for i in range(len(reads))]
    for timestamp in timestamps:
        tracker._last_ocr = -1e9        # bypass the OCR throttle
        tracker.classify(timestamp, frame)


def test_score_settles_on_the_stable_maximum():
    """The count-up animation can repeat a value long enough to fool a
    short window: 478 was reported for a final 17,770. Settling requires
    a 3-read identical tail on the LARGEST twice-confirmed value, 2 s
    after the modal appeared."""
    reads = [150, 478, 478, 3000, 9000, 17770, 17770, 17770]
    tracker = _score_tracker(reads)
    _feed(tracker, reads)
    assert tracker.score_settled
    assert tracker.final_score == 17770


def test_mid_animation_repeat_does_not_settle_early():
    reads = [478, 478, 478]             # all inside the first 0.9 s
    tracker = _score_tracker(reads)
    _feed(tracker, reads)
    assert not tracker.score_settled    # too soon: animation may be running
    assert tracker.final_score == 478   # still the best provisional value


def test_single_wild_misread_cannot_become_the_score():
    """One glitched huge read (77770) must not outrank a value the OCR
    confirmed repeatedly."""
    reads = [1150, 77770, 1150, 1150, 1150, 1150, 1150, 1150]
    tracker = _score_tracker(reads)
    _feed(tracker, reads)
    assert tracker.score_settled
    assert tracker.final_score == 1150


@pytest.mark.parametrize("text, expected", [
    ("0:24", 24),
    ("1:00", 60),
    ("0.07", 7),           # OCR reads the colon as a dot sometimes
    ("1:37", None),        # the round clock never exceeds 1:00 -- misread
    ("TIME", None),
    ("", None),
])
def test_parse_timer(text, expected):
    from automaton_attack_bot.core.session import parse_timer

    assert parse_timer(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("X3.0", 3.0),
    ("x1.5", 1.5),
    ("*2,0", 2.0),      # OCR reads the dot as a comma sometimes
    ("3.0", 3.0),
    ("X3O", 3.0),       # dot lost, zero read as O -- still unambiguous
    ("X15", 1.5),       # decimals are only ever 0 or 5
    ("XO", None),       # a single digit is not enough
    ("X17", None),      # not a half-step value
    ("X8.0", None),     # out of the game's x1.0-x3.0 range ('3' misread)
    ("SCORE", None),
    ("", None),
])
def test_parse_multiplier(text, expected):
    from automaton_attack_bot.core.session import parse_multiplier

    assert parse_multiplier(text) == expected


# -- click gating -------------------------------------------------------
@pytest.mark.clips
def test_the_real_play_again_button_passes_verification(fullgame_frames,
                                                        tracker):
    assert tracker.verify_button(fullgame_frames(70.4),
                                 GameState.GAME_OVER)


def test_a_buttonless_game_over_fails_verification():
    """Clicks once landed on bare background for a whole session; the
    game-over click is text-gated (the start screen's ornate plate defeats
    OCR, so its guard is the drive loop's futile-click breaker)."""
    tracker = SessionTracker(backend=None, settings=Settings())
    frame = np.zeros((1080, 1920, 3), np.uint8)
    assert not tracker.verify_button(frame, GameState.GAME_OVER)


def test_start_screen_clicks_are_not_text_gated():
    tracker = SessionTracker(backend=None, settings=Settings())
    frame = np.zeros((1080, 1920, 3), np.uint8)
    assert tracker.verify_button(frame, GameState.START_SCREEN)

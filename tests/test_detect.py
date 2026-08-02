"""Blob filtering and static-text suppression."""

import numpy as np
import pytest

from automaton_attack.config import Settings
from automaton_attack.detect import Detector
from automaton_attack.lexicon import Lexicon


class NoOcr:
    name = "none"

    def read(self, image):
        return ""


@pytest.fixture()
def detector():
    return Detector(Lexicon(["Placeholder"]), NoOcr(), Settings())


def text_like_roi(h=22, w=120):
    """Letter-ish stripes: ~18% fill, like real rendered text."""
    roi = np.zeros((h, w), np.uint8)
    for x in range(0, w, 11):
        roi[3:h-3, x:x+2] = 255
    return roi


def test_text_density_blob_survives(detector):
    mask = np.zeros((965, 963), np.uint8)
    mask[400:422, 300:420] = text_like_roi()
    assert detector.blobs(mask)


def test_sparse_speckle_blob_is_dropped(detector):
    """Automaton bodies leak sparse speckle through the colour mask; each
    one that reaches OCR costs ~50 ms. Fill separates them cleanly
    (text >= 0.13, speckle <= 0.08 measured on live footage)."""
    rng = np.random.default_rng(5)
    mask = np.zeros((965, 963), np.uint8)
    region = rng.random((22, 120)) < 0.05     # 5% fill
    mask[400:422, 300:420][region] = 255
    assert detector.blobs(mask) == []


def _hud_coloured_block():
    import cv2

    return cv2.cvtColor(
        np.dstack([np.full((22, 120), 41, np.uint8),
                   np.full((22, 120), 95, np.uint8),
                   text_like_roi() // 255 * 180]).astype(np.uint8),
        cv2.COLOR_HSV2BGR)


def test_static_text_in_the_top_band_is_suppressed(detector):
    """HUD labels live in the top band; text sitting still there for the
    whole window is furniture (this is where mis-anchored geometry once
    leaked 'HIGH SCORE' as a word)."""
    frame = np.zeros((1080, 1920, 3), np.uint8)
    x0, y0, _, _ = Settings().geometry.panel
    frame[y0 + 130:y0 + 152, x0 + 300:x0 + 420] = _hud_coloured_block()

    hits = [bool(detector.detect(frame, include_unmatched=True, timestamp=t))
            for t in (0.0, 0.5, 1.0, 1.5, 2.5)]
    assert hits[0] is True          # fresh text is a candidate word
    assert hits[-1] is False        # still there 2.5s later: furniture


def test_slow_words_in_the_play_field_are_never_suppressed(detector):
    """VOID SPIRIT descended slowly enough to sit pixel-still for over two
    seconds, got classified as furniture, and died invisible. The play
    field must never be suppressible, however still the word."""
    frame = np.zeros((1080, 1920, 3), np.uint8)
    x0, y0, _, _ = Settings().geometry.panel
    frame[y0 + 500:y0 + 522, x0 + 300:x0 + 420] = _hud_coloured_block()

    hits = [bool(detector.detect(frame, include_unmatched=True, timestamp=t))
            for t in (0.0, 0.5, 1.0, 1.5, 2.5, 4.0)]
    assert all(hits)


def test_moving_text_is_not_suppressed(detector):
    import cv2

    stripes = text_like_roi()
    block = cv2.cvtColor(
        np.dstack([np.full((22, 120), 41, np.uint8),
                   np.full((22, 120), 95, np.uint8),
                   stripes // 255 * 180]).astype(np.uint8),
        cv2.COLOR_HSV2BGR)
    x0, y0, _, _ = Settings().geometry.panel
    for i, t in enumerate((0.0, 0.5, 1.0, 1.5, 2.5)):
        frame = np.zeros((1080, 1920, 3), np.uint8)
        px = 300 + i * 30          # drifts like a real word
        frame[y0 + 500:y0 + 522, x0 + px:x0 + px + 120] = block
        result = detector.detect(frame, include_unmatched=True, timestamp=t)
    assert result                   # last scan still sees it


def test_stacked_words_are_split_into_lines(detector):
    """Automatons converge and their labels pile up; a two-line stack must
    yield two word boxes, not be rejected as 'too tall'."""
    mask = np.zeros((965, 963), np.uint8)
    mask[400:422, 300:420] = text_like_roi()
    mask[430:452, 320:460] = text_like_roi(w=140)
    boxes = detector.blobs(mask)
    assert len(boxes) == 2
    heights = sorted(box[3] for box in boxes)
    assert all(12 < h < 45 for h in heights)
    tops = sorted(box[1] for box in boxes)
    assert abs(tops[0] - 400) <= 4
    assert abs(tops[1] - 430) <= 4


def test_three_line_stack_splits_into_three(detector):
    mask = np.zeros((965, 963), np.uint8)
    for i in range(3):
        mask[400 + i * 30:422 + i * 30, 300:420] = text_like_roi()
    assert len(detector.blobs(mask)) == 3


def test_single_words_are_unaffected_by_the_splitter(detector):
    mask = np.zeros((965, 963), np.uint8)
    mask[400:422, 300:420] = text_like_roi()
    assert len(detector.blobs(mask)) == 1


def test_wrapped_phrase_lines_are_stitched_in_reading_order():
    """A long voice line wraps to two on-screen lines; typed separately
    (bottom first, by urgency) it never completes. Stitched, it types top
    line first as one phrase."""
    from automaton_attack.detect import Detection
    from automaton_attack.lexicon import Lexicon

    lexicon = Lexicon(
        ["Placeholder"],
        phrases=["There's a fine line between bravery and stupidity."])
    detector = Detector(lexicon, NoOcr(), Settings())
    top = Detection(box=(300, 400, 400, 22),
                    raw="THERE'S A FINE LINE BETWEEN BRAVERY AND",
                    match=None)
    bottom = Detection(box=(380, 428, 120, 22), raw="STUPIDITY.",
                       match=None)
    stitched = detector._stitch_wrapped_lines([bottom, top])
    assert len(stitched) == 1
    assert stitched[0].match is not None
    assert stitched[0].match.source == "phrase"
    assert stitched[0].match.keystrokes.startswith("theresafineline")
    assert stitched[0].match.keystrokes.endswith("stupidity")


def test_stacked_independent_words_are_not_stitched():
    from automaton_attack.detect import Detection
    from automaton_attack.lexicon import Lexicon, Match

    lexicon = Lexicon(["Earth Spirit", "Phantom Assassin"])
    detector = Detector(lexicon, NoOcr(), Settings())
    a = Detection(box=(300, 400, 200, 22), raw="EARTH SPIRIT",
                  match=Match("EARTH SPIRIT", 1.0, "vocab"))
    b = Detection(box=(310, 428, 260, 22), raw="PHANTOM ASSASSIN",
                  match=Match("PHANTOM ASSASSIN", 1.0, "vocab"))
    assert len(detector._stitch_wrapped_lines([a, b])) == 2


def test_distant_lines_are_not_stitched():
    from automaton_attack.detect import Detection
    from automaton_attack.lexicon import Lexicon

    lexicon = Lexicon(
        ["Placeholder"],
        phrases=["There's a fine line between bravery and stupidity."])
    detector = Detector(lexicon, NoOcr(), Settings())
    top = Detection(box=(300, 400, 400, 22),
                    raw="THERE'S A FINE LINE BETWEEN BRAVERY AND",
                    match=None)
    far_below = Detection(box=(380, 700, 120, 22), raw="STUPIDITY.",
                          match=None)
    assert len(detector._stitch_wrapped_lines([top, far_below])) == 2


def test_three_line_wrapped_phrase_is_stitched():
    """Narrow phrases wrap to THREE lines ('YOU'LL LOOK GOOD / WITH AN
    APPLE IN / YER MOUTH'); the chain stitcher must absorb all of them,
    not just a pair. A DIVINE RAPIER hidden behind this exact cluster
    escaped while the phrase resolved late."""
    from automaton_attack.detect import Detection
    from automaton_attack.lexicon import Lexicon

    lexicon = Lexicon(
        ["Placeholder"],
        phrases=["You'll look good with an apple in yer mouth!"])
    detector = Detector(lexicon, NoOcr(), Settings())
    lines = [
        Detection(box=(320, 400, 300, 22), raw="YOU'LL LOOK GOOD",
                  match=None),
        Detection(box=(310, 428, 320, 22), raw="WITH AN APPLE IN",
                  match=None),
        Detection(box=(350, 456, 200, 22), raw="YER MOUTH", match=None),
    ]
    stitched = detector._stitch_wrapped_lines(lines)
    assert len(stitched) == 1
    assert stitched[0].match is not None
    assert stitched[0].match.keystrokes == "youlllookgoodwithanappleinyermouth"

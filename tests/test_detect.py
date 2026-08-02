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


def test_static_text_is_suppressed(detector):
    """Words never stop moving; pixels lit continuously for the whole
    window are HUD furniture, wherever the geometry put them."""
    frame = np.zeros((1080, 1920, 3), np.uint8)
    x0, y0, _, _ = Settings().geometry.panel
    # paint a text-like block at a fixed position, in the word colour
    import cv2

    block = cv2.cvtColor(
        np.dstack([np.full((22, 120), 41, np.uint8),
                   np.full((22, 120), 95, np.uint8),
                   text_like_roi() // 255 * 180]).astype(np.uint8),
        cv2.COLOR_HSV2BGR)
    frame[y0 + 500:y0 + 522, x0 + 300:x0 + 420] = block

    hits = [bool(detector.detect(frame, include_unmatched=True, timestamp=t))
            for t in (0.0, 0.5, 1.0, 1.5, 2.5)]
    assert hits[0] is True          # fresh text is a candidate word
    assert hits[-1] is False        # still there 2.5s later: furniture


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

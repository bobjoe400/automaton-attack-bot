"""HUD-anchored colour calibration.

The HUD boxes share the target words' colour, so whatever a display
transform (HDR, night light, LUTs) does to one, it does to the other. These
tests paint synthetic frames where both are shifted together and check that
detection follows.
"""

import cv2
import numpy as np
import pytest

from automaton_attack_bot.core import autocolor
from automaton_attack_bot.core.config import Settings
from automaton_attack_bot.core.detect import Detector
from automaton_attack_bot.core.lexicon import Lexicon


SETTINGS = Settings()
RNG_SEED = 7

# Word-pixel colour under reference conditions: inside the tuned range,
# matching the measured footage (H 39-44, S 68-119, V 165-196).
REFERENCE_WORD_HSV = (41, 95, 180)


def paint_hud(frame, hsv, rng):
    """Fill the HUD boxes with text-like pixels of the given colour.

    Real HUD text has antialiasing and glow, so its pixels spread around the
    nominal colour; the sampler's anchors (medians, V extremes) assume that
    spread exists. Uniform noise approximates it well enough.
    """
    x0, y0, _, _ = SETTINGS.geometry.panel
    h, s, v = hsv
    for hx0, hy0, hx1, hy1 in SETTINGS.geometry.hud_boxes:
        shape = (hy1 - hy0, hx1 - hx0)
        pixels = np.stack([
            np.clip(rng.normal(h, 1.5, shape), 0, 179),
            np.clip(rng.normal(s, 8, shape), 0, 255),
            np.clip(rng.uniform(v - 60, v + 45, shape), 0, 255),
        ], axis=-1).astype(np.uint8)
        frame[y0 + hy0:y0 + hy1, x0 + hx0:x0 + hx1] = \
            cv2.cvtColor(pixels, cv2.COLOR_HSV2BGR)


def paint_word(frame, hsv, pos=(700, 500), size=(150, 20)):
    x0, y0, _, _ = SETTINGS.geometry.panel
    px, py = pos
    w, h = size
    block = np.full((h, w, 3), hsv, np.uint8)
    frame[y0 + py:y0 + py + h, x0 + px:x0 + px + w] = \
        cv2.cvtColor(block, cv2.COLOR_HSV2BGR)


def synthetic_frame(shift=(0, 0, 0)):
    """A black 1080p frame with HUD text and one word, both colour-shifted."""
    rng = np.random.default_rng(RNG_SEED)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    dh, ds, dv = shift
    ref_hud = (autocolor.REFERENCE.hue_median,
               autocolor.REFERENCE.sat_median,
               (autocolor.REFERENCE.val_p5 + autocolor.REFERENCE.val_p95) / 2)
    hud = (ref_hud[0] + dh, ref_hud[1] + ds, ref_hud[2] + dv)
    word = tuple(np.array(REFERENCE_WORD_HSV) + shift)
    paint_hud(frame, hud, rng)
    paint_word(frame, word)
    return frame


@pytest.fixture(scope="module")
def detector_factory():
    lexicon = Lexicon(["Placeholder"])   # blob tests never reach matching

    class NoOcr:
        name = "none"

        def read(self, image):
            return ""

    def make(auto):
        data = Settings().to_dict()
        data["color"]["auto"] = auto
        return Detector(lexicon, NoOcr(), Settings.from_dict(data))

    return make


# -- measure -------------------------------------------------------------
def test_measure_finds_the_hud_colour():
    frame = synthetic_frame()
    x0, y0, x1, y1 = SETTINGS.geometry.panel
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    anchor = autocolor.measure(hsv, SETTINGS.geometry.hud_boxes)
    assert anchor is not None
    assert anchor.hue_median == pytest.approx(
        autocolor.REFERENCE.hue_median, abs=3)
    assert anchor.sat_median == pytest.approx(
        autocolor.REFERENCE.sat_median, abs=10)


def test_measure_rejects_a_frame_without_hud():
    hsv = np.zeros((965, 963, 3), np.uint8)
    assert autocolor.measure(hsv, SETTINGS.geometry.hud_boxes) is None


# -- shifted_range ---------------------------------------------------------
def test_zero_shift_keeps_the_tuned_range():
    got = autocolor.shifted_range(
        autocolor.REFERENCE, SETTINGS.color.hsv_lo, SETTINGS.color.hsv_hi)
    assert got == (SETTINGS.color.hsv_lo, SETTINGS.color.hsv_hi)


def test_range_follows_a_hue_shift():
    anchor = autocolor.ColorAnchor(
        hue_median=autocolor.REFERENCE.hue_median + 10,
        sat_median=autocolor.REFERENCE.sat_median,
        val_p5=autocolor.REFERENCE.val_p5,
        val_p95=autocolor.REFERENCE.val_p95,
    )
    lo, hi = autocolor.shifted_range(
        anchor, SETTINGS.color.hsv_lo, SETTINGS.color.hsv_hi)
    assert lo[0] == SETTINGS.color.hsv_lo[0] + 10
    assert hi[0] == SETTINGS.color.hsv_hi[0] + 10
    assert (lo[1], lo[2]) == SETTINGS.color.hsv_lo[1:]


def test_an_implausible_shift_is_rejected():
    anchor = autocolor.ColorAnchor(
        hue_median=autocolor.REFERENCE.hue_median + 40,   # not a display
        sat_median=autocolor.REFERENCE.sat_median,
        val_p5=autocolor.REFERENCE.val_p5,
        val_p95=autocolor.REFERENCE.val_p95,
    )
    assert autocolor.shifted_range(
        anchor, SETTINGS.color.hsv_lo, SETTINGS.color.hsv_hi) is None


# -- end to end ------------------------------------------------------------
def find_word_blobs(detector, frame):
    return detector.blobs(detector.word_mask(frame))


def test_reference_conditions_detect_with_and_without_auto(detector_factory):
    frame = synthetic_frame()
    assert find_word_blobs(detector_factory(auto=False), frame)
    assert find_word_blobs(detector_factory(auto=True), frame)


def test_night_light_style_shift_needs_auto_color(detector_factory):
    """Hue -8, value -35: warm-shifted and dimmed, like a night-light filter.

    The word colour leaves the static range entirely; the HUD sample brings
    it back. This is the scenario the feature exists for.
    """
    frame = synthetic_frame(shift=(-8, 0, -35))
    assert not find_word_blobs(detector_factory(auto=False), frame)
    assert find_word_blobs(detector_factory(auto=True), frame)


def test_auto_color_keeps_last_good_calibration(detector_factory):
    """A frame without a HUD (scene transition) must not lose calibration."""
    detector = detector_factory(auto=True)
    shifted = synthetic_frame(shift=(-8, 0, -35))
    assert find_word_blobs(detector, shifted)
    calibrated = detector.active_range

    blank = np.zeros((1080, 1920, 3), np.uint8)
    detector.word_mask(blank)
    assert detector.active_range == calibrated

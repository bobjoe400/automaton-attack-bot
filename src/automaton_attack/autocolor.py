"""HUD-anchored colour calibration.

The score, timer and high-score boxes are rendered in the same yellow-green
as the target words and sit at fixed positions, which makes them a free
colour reference: whatever HDR tone-mapping, night-light filter or driver
LUT sits between the game and the framebuffer is applied to the HUD text and
the words alike. Sampling the HUD tells us what the words look like *right
now*, so the detection range can follow the display instead of going silently
blind when it is not the display the range was tuned on.

The calibration is a *shift*, not a re-derivation: HUD glyphs carry glow and
antialiasing that target words do not, so their raw pixel spread is wider
than the word range should be. Instead, stable anchors of the HUD sample --
hue median, saturation median, and the 5th/95th percentiles of value (the
dim and bright edges of the text; measured spread across the reference
footage is <=4 units, while the middle percentiles swing by ~40 as the HUD
pulses) -- are compared against the same anchors recorded from the validated
recordings, and the tuned word range is translated by the difference.

On a display matching the reference conditions the deltas are ~0 and the
mask is identical to the hand-tuned one.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

Box = tuple[int, int, int, int]

# Candidate window for "this pixel is HUD text": generous enough to survive a
# real display transform, tight enough to exclude the dark panel behind the
# glyphs. The anchors below were measured through this same window.
BROAD_LO = (15, 35, 95)
BROAD_HI = (75, 255, 255)

# Below this many candidate pixels the HUD is not on screen (menu, loading
# screen, wrong scene) and the sample is meaningless.
MIN_PIXELS = 300

# A shift larger than this is not a display transform, it is a sampling
# accident -- e.g. gold Dota UI text occupying the boxes outside the
# minigame. Reject rather than clamp: a wrong calibration is worse than the
# tuned default, and a transform this wild needs `automaton calibrate`
# anyway.
MAX_SHIFT = (20, 60, 60)


@dataclass(frozen=True)
class ColorAnchor:
    """Stable summary of the HUD text colour in one frame."""

    hue_median: float
    sat_median: float
    val_p5: float
    val_p95: float


# Anchors of the validated reference footage (16 frames across both clips,
# 1920x1080 SDR). Regenerate with tests/measure_reference_anchor.py if the
# HSV range in config.py is ever retuned.
REFERENCE = ColorAnchor(
    hue_median=35.0,
    sat_median=104.5,
    val_p5=97.5,
    val_p95=227.0,
)


def measure(hsv_panel: np.ndarray, hud_boxes: tuple[Box, ...]) -> ColorAnchor | None:
    """Sample the HUD boxes of an HSV panel crop. None if there is no HUD."""
    samples = []
    for x0, y0, x1, y1 in hud_boxes:
        region = hsv_panel[y0:y1, x0:x1]
        if region.size == 0:
            continue
        mask = cv2.inRange(region, BROAD_LO, BROAD_HI)
        samples.append(region[mask > 0])
    if not samples:
        return None
    pixels = np.concatenate(samples)
    if len(pixels) < MIN_PIXELS:
        return None
    return ColorAnchor(
        hue_median=float(np.median(pixels[:, 0])),
        sat_median=float(np.median(pixels[:, 1])),
        val_p5=float(np.percentile(pixels[:, 2], 5)),
        val_p95=float(np.percentile(pixels[:, 2], 95)),
    )


def shifted_range(
    anchor: ColorAnchor,
    hsv_lo: tuple[int, int, int],
    hsv_hi: tuple[int, int, int],
    reference: ColorAnchor = REFERENCE,
) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    """Translate the tuned word range by the anchor's offset from reference.

    Returns None when the offset is implausibly large (see MAX_SHIFT).
    """
    dh = anchor.hue_median - reference.hue_median
    ds = anchor.sat_median - reference.sat_median
    dv_lo = anchor.val_p5 - reference.val_p5
    dv_hi = anchor.val_p95 - reference.val_p95

    if (abs(dh) > MAX_SHIFT[0] or abs(ds) > MAX_SHIFT[1]
            or abs(dv_lo) > MAX_SHIFT[2] or abs(dv_hi) > MAX_SHIFT[2]):
        return None

    def clamp(value: float, hi: int) -> int:
        return max(0, min(hi, round(value)))

    lo = (clamp(hsv_lo[0] + dh, 179),
          clamp(hsv_lo[1] + ds, 255),
          clamp(hsv_lo[2] + dv_lo, 255))
    hi = (clamp(hsv_hi[0] + dh, 179),
          clamp(hsv_hi[1] + ds, 255),
          clamp(hsv_hi[2] + dv_hi, 255))
    return lo, hi

"""Turning a frame into candidate words.

Isolate the target words by colour, merge their letters into per-word blobs,
OCR each blob, then resolve the read against the lexicon.

Only the small yellow-green target text is detected. Each word also renders a
second time as a larger desaturated-white "input tracker" that highlights the
next letter you owe -- useful for a human, redundant for us, and deliberately
outside the HSV range.

Two things share the words' colour and must not reach OCR:

* the automatons themselves -- gold bodies leaking sparse speckle through the
  mask; filtered by fill density (see Blobs.min_fill);
* HUD text -- masked by the configured boxes, and additionally by *static
  suppression*: any pixel lit continuously for a couple of seconds is
  interface furniture, because words never stop moving. This catches HUD
  text wherever it is, even when the geometry is imperfect (a mis-located
  panel once had the bot typing 'HIGH SCORE' every three seconds).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

# Static suppression: a pixel lit in every sample across this span is HUD.
STATIC_WINDOW = 2.0        # seconds
STATIC_MIN_SAMPLES = 4
STATIC_SAMPLE_GAP = 0.2    # don't hoard near-duplicate masks at high fps

from . import autocolor
from .config import Settings
from .lexicon import Lexicon, Match
from .ocr import OcrBackend, prepare


@dataclass(frozen=True)
class Detection:
    """One word found on screen."""

    box: tuple[int, int, int, int]   # x, y, w, h in panel-local coords
    raw: str                         # what OCR read
    match: Match | None              # what we resolved it to

    @property
    def pos(self) -> tuple[int, int]:
        return self.box[0], self.box[1]

    @property
    def name(self) -> str:
        return self.match.name if self.match else ""

    @property
    def score(self) -> float:
        return self.match.score if self.match else 0.0


class Detector:
    def __init__(
        self,
        lexicon: Lexicon,
        backend: OcrBackend,
        settings: Settings | None = None,
    ) -> None:
        self.lexicon = lexicon
        self.backend = backend
        self.settings = settings or Settings()
        self._kernel = np.ones(self.settings.blobs.dilate_kernel, np.uint8)
        self._calibrated: tuple | None = None   # last good HUD-derived range
        self.last_anchor: autocolor.ColorAnchor | None = None
        self._mask_history: deque[tuple[float, np.ndarray]] = deque()

    @property
    def active_range(self) -> tuple:
        """The HSV range currently in use (calibrated if available)."""
        if self._calibrated:
            return self._calibrated
        return self.settings.color.hsv_lo, self.settings.color.hsv_hi

    # -- stages ----------------------------------------------------------
    def crop_panel(self, frame_bgr: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.settings.geometry.panel
        return frame_bgr[y0:y1, x0:x1]

    def word_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Binary mask of target-word pixels, HUD regions blanked."""
        crop = self.crop_panel(frame_bgr)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        color = self.settings.color
        if color.auto:
            # The HUD boxes share the words' colour; sample them to follow
            # whatever the display (HDR, night light, LUTs) is doing to it.
            anchor = autocolor.measure(hsv, self.settings.geometry.hud_boxes)
            if anchor:
                self.last_anchor = anchor
                calibrated = autocolor.shifted_range(
                    anchor, color.hsv_lo, color.hsv_hi)
                if calibrated:
                    self._calibrated = calibrated
        lo, hi = self.active_range
        mask = cv2.inRange(hsv, lo, hi)
        # The score, timer and high-score boxes use the same yellow-green.
        for hx0, hy0, hx1, hy1 in self.settings.geometry.hud_boxes:
            mask[hy0:hy1, hx0:hx1] = 0
        return mask

    def blobs(self, mask: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Merge letters into word-sized boxes and filter out noise."""
        merged = cv2.dilate(mask, self._kernel)
        count, _, stats, _ = cv2.connectedComponentsWithStats(merged)
        blobs = self.settings.blobs
        boxes = []
        for i in range(1, count):
            bx, by, bw, bh, area = stats[i]
            if area < blobs.min_area or bw < blobs.min_width:
                continue
            if not blobs.min_height < bh < blobs.max_height:
                continue
            # Sparse speckle (automaton bodies) is not text; every one of
            # these that reaches OCR costs ~50 ms of scan latency.
            roi = mask[by:by + bh, bx:bx + bw]
            if roi.mean() / 255.0 < blobs.min_fill:
                continue
            boxes.append((int(bx), int(by), int(bw), int(bh)))
        return boxes

    def _static_mask(self, mask: np.ndarray,
                     timestamp: float | None) -> np.ndarray | None:
        """Pixels lit continuously across the history window: HUD, not words."""
        if timestamp is None:
            return None
        history = self._mask_history
        if history and history[-1][1].shape != mask.shape:
            history.clear()             # geometry changed mid-run
        if not history or timestamp - history[-1][0] >= STATIC_SAMPLE_GAP:
            history.append((timestamp, mask.copy()))
        while len(history) > 2 and timestamp - history[1][0] >= STATIC_WINDOW:
            history.popleft()
        if (len(history) < STATIC_MIN_SAMPLES
                or timestamp - history[0][0] < STATIC_WINDOW):
            return None
        static = history[0][1].copy()
        for _, past in list(history)[1:]:
            np.bitwise_and(static, past, out=static)
        return static

    # -- full pass -------------------------------------------------------
    def detect(self, frame_bgr: np.ndarray,
               include_unmatched: bool = False,
               timestamp: float | None = None) -> list[Detection]:
        mask = self.word_mask(frame_bgr)
        static = self._static_mask(mask, timestamp)
        if static is not None:
            mask[static > 0] = 0
        results = []
        for box in self.blobs(mask):
            raw = self.backend.read(prepare(mask, box))
            match = self.lexicon.match(
                raw, allow_fallback=not self.settings.safe_mode
            ) if raw else None
            if match or include_unmatched:
                results.append(Detection(box=box, raw=raw, match=match))
        return results

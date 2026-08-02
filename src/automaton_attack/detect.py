"""Turning a frame into candidate words.

Isolate the target words by colour, merge their letters into per-word blobs,
OCR each blob, then resolve the read against the lexicon.

Only the small yellow-green target text is detected. Each word also renders a
second time as a larger desaturated-white "input tracker" that highlights the
next letter you owe -- useful for a human, redundant for us, and deliberately
outside the HSV range.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

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

    # -- stages ----------------------------------------------------------
    def crop_panel(self, frame_bgr: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.settings.geometry.panel
        return frame_bgr[y0:y1, x0:x1]

    def word_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Binary mask of target-word pixels, HUD regions blanked."""
        crop = self.crop_panel(frame_bgr)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.settings.color.hsv_lo,
                           self.settings.color.hsv_hi)
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
            boxes.append((int(bx), int(by), int(bw), int(bh)))
        return boxes

    # -- full pass -------------------------------------------------------
    def detect(self, frame_bgr: np.ndarray,
               include_unmatched: bool = False) -> list[Detection]:
        mask = self.word_mask(frame_bgr)
        results = []
        for box in self.blobs(mask):
            raw = self.backend.read(prepare(mask, box))
            match = self.lexicon.match(
                raw, allow_fallback=not self.settings.safe_mode
            ) if raw else None
            if match or include_unmatched:
                results.append(Detection(box=box, raw=raw, match=match))
        return results

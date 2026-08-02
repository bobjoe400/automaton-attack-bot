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

import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

# Static suppression: a pixel lit in every sample across this span is HUD.
STATIC_WINDOW = 2.0        # seconds
STATIC_MIN_SAMPLES = 4
STATIC_SAMPLE_GAP = 0.2    # don't hoard near-duplicate masks at high fps
# Suppression applies ONLY to the top band, where HUD labels live (and
# where mis-anchored geometry once leaked them). A slow word in the play
# field can sit near-still for seconds -- VOID SPIRIT descended slowly
# enough to be classified as furniture and died invisible.
STATIC_BAND_FRACTION = 0.20

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
    # Stack context: when this word is one line of a taller pile-up, the
    # whole pile's box and this word's top-to-bottom position in it. The
    # game only accepts the TOP word of a stack, so typing order within a
    # stack must be top-first regardless of who is nearest the platform.
    group_box: tuple[int, int, int, int] | None = None
    stack_rank: int = 0

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
        # Two scans may be detected concurrently (pipelined); calibration
        # and static-history state is shared and must be mutated under a
        # lock. OCR itself runs unlocked.
        self._state_lock = threading.Lock()
        # Blobs are OCRed concurrently when the backend can take it (an
        # OcrPool); a busy screen has 6-10 blobs at ~50 ms each.
        workers = getattr(backend, "size", 1)
        self._ocr_executor = (ThreadPoolExecutor(max_workers=workers,
                                                 thread_name_prefix="ocr")
                              if workers > 1 else None)

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
                calibrated = autocolor.shifted_range(
                    anchor, color.hsv_lo, color.hsv_hi)
                with self._state_lock:
                    self.last_anchor = anchor
                    if calibrated:
                        self._calibrated = calibrated
        lo, hi = self.active_range
        mask = cv2.inRange(hsv, lo, hi)
        # The score, timer and high-score boxes use the same yellow-green.
        for hx0, hy0, hx1, hy1 in self.settings.geometry.hud_boxes:
            mask[hy0:hy1, hx0:hx1] = 0
        return mask

    def blobs(self, mask: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Word-sized boxes only; see _blobs_with_stacks for the details."""
        return [box for box, _, _ in self._blobs_with_stacks(mask)]

    def _blobs_with_stacks(
            self, mask: np.ndarray
    ) -> list[tuple[tuple[int, int, int, int],
                    tuple[int, int, int, int] | None, int]]:
        """Merge letters into word-sized boxes and filter out noise.

        A component TALLER than one text line is not noise -- it is words
        stacked on top of each other (automatons converge, their labels
        pile up). Rejecting tall components made whole clusters invisible
        until the words drifted apart, which is how a five-word pile-up at
        0:04 on the clock cost a combo. Tall components are split into
        text lines instead.

        Returns (box, stack_box, rank) per word: stack_box is the whole
        pile's box (None for a lone word) and rank is the word's
        top-to-bottom position in its pile.
        """
        merged = cv2.dilate(mask, self._kernel)
        count, _, stats, _ = cv2.connectedComponentsWithStats(merged)
        blobs = self.settings.blobs
        boxes = []
        for i in range(1, count):
            bx, by, bw, bh, area = stats[i]
            if area < blobs.min_area or bw < blobs.min_width:
                continue
            if bh <= blobs.min_height:
                continue
            split = bh >= blobs.max_height
            candidates = ([(int(bx), int(by), int(bw), int(bh))]
                          if not split
                          else self._split_lines(mask, bx, by, bw, bh))
            kept = []
            for cx, cy, cw, ch in candidates:
                if cw < blobs.min_width:
                    continue
                if not blobs.min_height < ch < blobs.max_height:
                    continue
                # Sparse speckle (automaton bodies) is not text; every one
                # that reaches OCR costs ~50 ms of scan latency.
                roi = mask[cy:cy + ch, cx:cx + cw]
                if roi.size == 0 or roi.mean() / 255.0 < blobs.min_fill:
                    continue
                kept.append((cx, cy, cw, ch))
            # Two or more lines out of one component is a stack: keep the
            # pile's box and each line's top-to-bottom rank with it.
            group = ((int(bx), int(by), int(bw), int(bh))
                     if split and len(kept) >= 2 else None)
            for rank, box in enumerate(sorted(kept, key=lambda b: b[1])):
                boxes.append((box, group, rank if group else 0))
        return boxes

    def _split_lines(self, mask: np.ndarray, bx: int, by: int,
                     bw: int, bh: int) -> list[tuple[int, int, int, int]]:
        """Split a tall component into its text lines.

        Bands of lit rows in the pre-dilation mask, separated by empty
        rows, are individual words; each band's box is re-fit to its lit
        columns. Words whose rows genuinely interleave cannot be split
        and stay lost until they separate -- but the common case is a
        clean few-pixel gap between stacked labels.
        """
        region = mask[by:by + bh, bx:bx + bw]
        lit_rows = region.max(axis=1) > 0
        bands = []
        start = None
        for row, lit in enumerate(lit_rows):
            if lit and start is None:
                start = row
            elif not lit and start is not None:
                bands.append((start, row))
                start = None
        if start is not None:
            bands.append((start, len(lit_rows)))

        out = []
        for row0, row1 in bands:
            band = region[row0:row1]
            columns = np.flatnonzero(band.max(axis=0) > 0)
            if columns.size == 0:
                continue
            out.append((bx + int(columns[0]), by + row0,
                        int(columns[-1] - columns[0] + 1), row1 - row0))
        return out

    def _static_mask(self, mask: np.ndarray,
                     timestamp: float | None) -> np.ndarray | None:
        """Pixels lit continuously across the history window: HUD, not words."""
        if timestamp is None:
            return None
        with self._state_lock:
            history = self._mask_history
            if history and history[-1][1].shape != mask.shape:
                history.clear()             # geometry changed mid-run
            if not history or timestamp - history[-1][0] >= STATIC_SAMPLE_GAP:
                history.append((timestamp, mask.copy()))
            while (len(history) > 2
                   and timestamp - history[1][0] >= STATIC_WINDOW):
                history.popleft()
            if (len(history) < STATIC_MIN_SAMPLES
                    or timestamp - history[0][0] < STATIC_WINDOW):
                return None
            snapshot = [past for _, past in history]
        static = snapshot[0].copy()
        for past in snapshot[1:]:
            np.bitwise_and(static, past, out=static)
        return static

    # -- full pass -------------------------------------------------------
    def detect(self, frame_bgr: np.ndarray,
               include_unmatched: bool = False,
               timestamp: float | None = None) -> list[Detection]:
        mask = self.word_mask(frame_bgr)
        static = self._static_mask(mask, timestamp)
        if static is not None:
            band_end = int(mask.shape[0] * STATIC_BAND_FRACTION)
            static[band_end:, :] = 0    # the play field is never furniture
            mask[static > 0] = 0
        blob_meta = self._blobs_with_stacks(mask)
        boxes = [meta[0] for meta in blob_meta]
        if self._ocr_executor is not None and len(boxes) > 1:
            raws = list(self._ocr_executor.map(
                lambda box: self.backend.read(prepare(mask, box)), boxes))
        else:
            raws = [self.backend.read(prepare(mask, box)) for box in boxes]
        detections = []
        retry_indices = []
        for index, (box, raw) in enumerate(zip(boxes, raws)):
            match = self.lexicon.match(
                raw, allow_fallback=not self.settings.safe_mode
            ) if raw else None
            detections.append(Detection(
                box=box, raw=raw, match=match,
                group_box=blob_meta[index][1],
                stack_rank=blob_meta[index][2]))
            if match is None or match.source == "fallback":
                retry_indices.append(index)
        # Second chance for glare fragments: explosion glow and spawn
        # flashes degrade the threshold mask far more than the raw pixels
        # ('FNU!' where BLITZ KNUCKLES was legible to the eye). Re-OCR the
        # colour crop for the few unmatched blobs.
        for index in retry_indices[:3]:
            box = boxes[index]
            second = self._read_colour_crop(frame_bgr, box)
            if not second:
                continue
            match = self.lexicon.match(
                second, allow_fallback=not self.settings.safe_mode)
            old = detections[index]
            if match is not None and (old.match is None
                                      or match.score > old.match.score):
                detections[index] = Detection(
                    box=box, raw=second, match=match,
                    group_box=old.group_box, stack_rank=old.stack_rank)
        detections = self._stitch_wrapped_lines(detections)
        return [d for d in detections if d.match or include_unmatched]

    def _read_colour_crop(self, frame_bgr: np.ndarray,
                          box: tuple[int, int, int, int]) -> str:
        bx, by, bw, bh = box
        x0, y0, _, _ = self.settings.geometry.panel
        pad = 4
        crop = frame_bgr[max(0, y0 + by - pad):y0 + by + bh + pad,
                         max(0, x0 + bx - pad):x0 + bx + bw + pad]
        if crop.size == 0:
            return ""
        upscaled = cv2.resize(crop, None, fx=2, fy=2,
                              interpolation=cv2.INTER_CUBIC)
        return self.backend.read(upscaled)

    # A wrapped phrase's second line starts within a line-height below the
    # first; a merge is accepted only on a confident corpus match.
    STITCH_MIN_SCORE = 0.85

    def _stitch_wrapped_lines(
            self, detections: list[Detection]) -> list[Detection]:
        """Rejoin phrases that wrap onto two on-screen lines.

        The line splitter (rightly) separates stacked words, but a LONG
        voice line wraps into exactly the same shape. If two vertically
        adjacent, horizontally overlapping lines jointly match the corpus
        convincingly -- and they weren't both confident matches on their
        own -- they are one phrase, typed in reading order: top line
        first. 'There's a fine line between bravery and stupidity.' died
        as two separately-typed fallback lines to make the point.
        """
        if len(detections) < 2:
            return detections
        detections = sorted(detections, key=lambda d: (d.box[1], d.box[0]))
        max_gap = self.settings.blobs.max_height
        consumed = [False] * len(detections)
        out = []
        for i, top in enumerate(detections):
            if consumed[i]:
                continue
            # Grow a chain of vertically adjacent, overlapping lines and
            # keep the best-matching prefix. A wide phrase wraps to TWO
            # lines; a narrow one wraps to THREE ("YOU'LL LOOK GOOD /
            # WITH AN APPLE IN / YER MOUTH").
            chain = [i]
            best: tuple[list[int], object] | None = None
            current = top
            for j in range(i + 1, len(detections)):
                if consumed[j]:
                    continue
                below = detections[j]
                gap = below.box[1] - (current.box[1] + current.box[3])
                if gap > max_gap:
                    break
                if gap < -5:
                    continue
                overlap = (min(current.box[0] + current.box[2],
                               below.box[0] + below.box[2])
                           - max(current.box[0], below.box[0]))
                if overlap < 0.5 * min(current.box[2], below.box[2]):
                    continue
                if (below.match and below.match.score >= 0.9
                        and all(detections[k].match
                                and detections[k].match.score >= 0.9
                                for k in chain)):
                    continue        # confident words stay independent
                chain.append(j)
                current = below
                joined = " ".join(detections[k].raw.strip() for k in chain)
                match = self.lexicon.match(joined, allow_fallback=False)
                if match is not None and match.score >= self.STITCH_MIN_SCORE:
                    best = (list(chain), match)
                if len(chain) >= 4:
                    break
            if best is not None:
                indices, match = best
                parts = [detections[k] for k in indices]
                x0 = min(p.box[0] for p in parts)
                y0 = parts[0].box[1]
                x1 = max(p.box[0] + p.box[2] for p in parts)
                y1 = parts[-1].box[1] + parts[-1].box[3]
                joined = " ".join(p.raw.strip() for p in parts)
                out.append(Detection(box=(x0, y0, x1 - x0, y1 - y0),
                                     raw=joined, match=match))
                for k in indices:
                    consumed[k] = True
            else:
                out.append(top)
                consumed[i] = True
        return out

"""Session awareness: where the minigame is and what state it is in.

Three screens matter:

* the **start screen** -- "AUTOMATON ATTACK" modal with a PLAY button;
* **gameplay** -- HUD (score/timer/high-score) visible in yellow-green;
* the **game-over screen** -- "GAME OVER", the total score, PLAY AGAIN.

State detection is cheap and layered. Gameplay is recognised by the HUD
colour sample (:func:`autocolor.measure`), which the detector computes
anyway; the modal screens cover the HUD area with dark navy, so a missing
sample plus a readable title distinguishes the other two. Titles are OCRed
from a fixed strip and fuzzy-matched, since the backends misread stylised
caps ("CAMEOVER", "AUTOMATONATACK").

The panel border itself is auto-located from the frame: the minigame sits in
a bright rectangular frame on a dark page, so the strongest vertical and
horizontal edge projections pin its four sides. That replaces hard-coded
geometry when the window is not at the reference position/resolution.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

import cv2
import numpy as np

from . import autocolor
from .config import Settings
from .ocr import OcrBackend, make_digit_reader

# Regions in panel fractions (x0, y0, x1, y1), measured off the reference
# footage. Fractions survive any panel size the auto-locator finds.
TITLE_REGION = (0.26, 0.135, 0.74, 0.26)     # covers both modal titles
# "Total Score      17,770" -- padded vertically: a few pixels of geometry
# drift once clipped the digits into a phantom '86'.
SCORE_ROW_REGION = (0.227, 0.263, 0.767, 0.322)
TIMER_REGION = (0.40, 0.02, 0.585, 0.085)    # the "0:24" above TIME
MULTIPLIER_REGION = (0.04, 0.102, 0.23, 0.143)  # the "x1.5" under SCORE
PLAY_BUTTON = (0.4965, 0.857)                # start screen
PLAY_AGAIN_BUTTON = (0.4965, 0.790)          # game-over screen

TITLE_OCR_THRESHOLD = 120    # modal titles are bright on dark navy
TITLE_MATCH_CUTOFF = 0.7

# "A round is running" requires every HUD box to be individually lit with
# this many candidate pixels. The arcade pages leak gold-ish UI text into
# parts of the HUD area, but only real gameplay lights score, timer AND
# high-score at once.
PER_BOX_MIN_PIXELS = 150

START_TITLE_KEY = "AUTOMATONATTACK"
GAME_OVER_KEY = "GAMEOVER"


class GameState(enum.Enum):
    UNKNOWN = "unknown"
    START_SCREEN = "start-screen"
    PLAYING = "playing"
    GAME_OVER = "game-over"


def _letters(text: str) -> str:
    return "".join(c for c in text.upper() if c.isalpha())


# OCR confuses these with digits in the score line ("1," reads as "L").
# Applied only to the text after the SCORE label, never to words.
_DIGIT_LOOKALIKES = str.maketrans({"O": "0", "I": "1", "L": "1", "l": "1",
                                   "B": "8", "S": "5", "Z": "2"})


def parse_multiplier(text: str) -> float | None:
    """Combo multiplier from an "x1.5"-style read.

    Multipliers are always digit-dot-digit with the decimal in {0, 5}, so
    a read that lost its separator ('X3O' for x3.0 -- note the 0 read as
    O) is still unambiguous once lookalikes are translated.
    """
    text = text.upper().translate(_DIGIT_LOOKALIKES)
    value = None
    match = re.search(r"(\d)\s?[.,]\s?(\d)", text)
    if match:
        value = float(f"{match.group(1)}.{match.group(2)}")
    else:
        digits = [c for c in text if c.isdigit()]
        if len(digits) == 2:
            value = float(f"{digits[0]}.{digits[1]}")
    # The game's multipliers run x1.0-x3.0 in half steps; anything else is
    # a misread ('3' has come back as '8', 'SCORE' as '5C0RE').
    if value is None or value * 2 != int(value * 2) or not 1.0 <= value <= 3.0:
        return None
    return value


def parse_timer(text: str) -> int | None:
    """Seconds remaining from a "0:24"-style read."""
    match = re.search(r"(\d{1,2})[:.](\d{2})", text)
    if not match:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


def parse_score(row: str) -> int | None:
    """Extract the number from a "Total Score 1,150" OCR read.

    The read is messy: separators vanish, and digits come back as
    lookalike letters ("1,150" has been observed as "L150"). Everything
    after the SCORE label is treated as digits-in-disguise; without a
    label, only a clean trailing number is trusted.
    """
    text = row.upper().replace(",", "").replace(".", "").replace(" ", "")
    label = text.rfind("SCOR")
    if label >= 0:
        tail = text[label + 4:].lstrip("E").translate(_DIGIT_LOOKALIKES)
        digits = "".join(c for c in tail if c.isdigit())
        return int(digits) if digits else None
    match = re.search(r"(\d+)$", text)
    return int(match.group(1)) if match else None


def _fuzzy_contains(text: str, key: str) -> bool:
    if key in text:
        return True
    if not text:
        return False
    return SequenceMatcher(None, text, key).ratio() >= TITLE_MATCH_CUTOFF


# ---------------------------------------------------------------------------
# Panel auto-location
# ---------------------------------------------------------------------------
EDGE_GRADIENT = 40      # per-pixel gradient that counts as "an edge here"
MIN_SIDE_COVER = 0.45   # each side must be an edge along >=45% of ITS length
CANDIDATES_PER_SIDE = 6


def _edge_line_candidates(profile: np.ndarray, start: int, stop: int,
                          min_cover: float = 0.2) -> list[int]:
    """Strongest sustained-edge positions in a range, deduplicated.

    The border line is a few pixels wide, so both of its flanks peak;
    positions within 10 px keep only the strongest.
    """
    order = np.argsort(profile[start:stop])[::-1][:CANDIDATES_PER_SIDE * 3]
    picked: list[int] = []
    for offset in order:
        index = int(offset) + start
        if profile[index] < min_cover:
            break
        if all(abs(index - p) > 10 for p in picked):
            picked.append(index)
        if len(picked) >= CANDIDATES_PER_SIDE:
            break
    return picked


def locate_panel(frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Find the minigame frame by its border edges. None if implausible.

    Candidate edge lines are combined into rectangles and each rectangle is
    scored by how continuous its four sides are *within its own bounds* --
    a real frame has four mutually-consistent sides, while a bright modal
    edge, HUD text or an FPS counter is strong in one direction only.
    (Full-screen argmax once mis-anchored the geometry badly enough that
    the HUD's own labels were typed as words.)
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    edges_v = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)) > EDGE_GRADIENT
    edges_h = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)) > EDGE_GRADIENT

    col_cover = edges_v.mean(axis=0)
    row_cover = edges_h.mean(axis=1)
    lefts = _edge_line_candidates(col_cover, 0, width // 2)
    rights = _edge_line_candidates(col_cover, width // 2, width)
    tops = _edge_line_candidates(row_cover, 0, height // 2)
    bottoms = _edge_line_candidates(row_cover, height // 2, height)

    def side_cover(edge_map, fixed, lo, hi, vertical):
        window = slice(max(0, fixed - 2), fixed + 3)
        if vertical:
            return edge_map[lo:hi, window].max(axis=1).mean()
        return edge_map[window, lo:hi].max(axis=0).mean()

    best_score, best_rect = 0.0, None
    for x0 in lefts:
        for x1 in rights:
            panel_w = x1 - x0
            if panel_w < 0.3 * width:
                continue
            for y0 in tops:
                for y1 in bottoms:
                    panel_h = y1 - y0
                    if panel_h < 0.5 * height:
                        continue
                    # Every real observation is ~square: 0.998-1.005 across
                    # 1080p and 1440p captures. A bad lock at 1.107 once
                    # blinded a whole session, so the tolerance is tight.
                    if not 0.93 <= panel_w / panel_h <= 1.07:
                        continue
                    sides = (
                        side_cover(edges_v, x0, y0, y1, vertical=True),
                        side_cover(edges_v, x1, y0, y1, vertical=True),
                        side_cover(edges_h, y0, x0, x1, vertical=False),
                        side_cover(edges_h, y1, x0, x1, vertical=False),
                    )
                    if min(sides) < MIN_SIDE_COVER:
                        continue
                    score = sum(sides)
                    if score > best_score:
                        best_score, best_rect = score, (x0, y0, x1, y1)
    return best_rect


# ---------------------------------------------------------------------------
# State tracking
# ---------------------------------------------------------------------------
@dataclass
class Transition:
    timestamp: float
    state: GameState


class SessionTracker:
    """Classifies frames into game states and reads the final score."""

    def __init__(self, backend: OcrBackend, settings: Settings,
                 ocr_interval: float = 0.25) -> None:
        self.backend = backend
        self.settings = settings
        self.ocr_interval = ocr_interval
        # Digits (score, timer) read best through tesseract's whitelist on
        # the raw grayscale; None when tesseract isn't installed.
        self.digit_reader = make_digit_reader()
        self.state = GameState.UNKNOWN
        self.transitions: list[Transition] = []
        self.final_score: int | None = None
        # The game-over score counts up when the modal appears, and the
        # animation can repeat a value long enough to fool a short
        # agreement window (478 was reported for a final 17,770). Settling
        # now needs a 3-read identical tail, at least 2 s after the modal
        # appeared, on the LARGEST value seen at least twice -- the score
        # only counts upward, so a mid-animation value can never be the
        # maximum once the animation passes it.
        self.score_settled = False
        self._score_reads: list[int] = []
        self._game_over_at: float | None = None
        self._last_ocr = -1e9

    # -- geometry helpers -------------------------------------------------
    def _panel(self, frame: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.settings.geometry.panel
        return frame[y0:y1, x0:x1]

    def _region(self, panel: np.ndarray,
                fractions: tuple[float, float, float, float]) -> np.ndarray:
        ph, pw = panel.shape[:2]
        fx0, fy0, fx1, fy1 = fractions
        return panel[int(fy0 * ph):int(fy1 * ph), int(fx0 * pw):int(fx1 * pw)]

    def button_position(self, state: GameState) -> tuple[int, int]:
        """Screen coordinates of the button that starts a round."""
        x0, y0, x1, y1 = self.settings.geometry.panel
        fx, fy = (PLAY_BUTTON if state is GameState.START_SCREEN
                  else PLAY_AGAIN_BUTTON)
        return round(x0 + fx * (x1 - x0)), round(y0 + fy * (y1 - y0))

    # -- OCR helpers --------------------------------------------------------
    def _read_bright_text(self, region: np.ndarray) -> str:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, TITLE_OCR_THRESHOLD, 255,
                                cv2.THRESH_BINARY)
        if int(mask.sum()) // 255 < 50:     # nothing bright: no modal text
            return ""
        upscaled = cv2.resize(255 - mask, None, fx=2, fy=2,
                              interpolation=cv2.INTER_CUBIC)
        return self.backend.read(upscaled)

    def _read_digits(self, region: np.ndarray) -> str:
        """Digits off a raw crop via the digit reader (None-safe)."""
        if self.digit_reader is None or region.size == 0:
            return ""
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        upscaled = cv2.resize(255 - gray, None, fx=3, fy=3,
                              interpolation=cv2.INTER_CUBIC)
        return self.digit_reader.read(upscaled)

    def read_final_score(self, frame: np.ndarray) -> int | None:
        """Read "Total Score NNN" off the game-over screen.

        The digit reader (tesseract, digits whitelist, raw grayscale) reads
        comma-grouped totals exactly where the general backend garbles them
        ('17,770' came back '17,7%'); fall back to the thresholded general
        read when tesseract isn't installed.
        """
        region = self._region(self._panel(frame), SCORE_ROW_REGION)
        digits = "".join(c for c in self._read_digits(region) if c.isdigit())
        if digits:
            return int(digits)
        row = self._read_bright_text(region)
        return parse_score(row)

    def read_multiplier(self, frame: np.ndarray) -> float | None:
        """The combo multiplier under the score, or None when unreadable.

        Logged during play so a combo loss is findable in the log (and the
        footage) without OCRing the whole recording after the fact. The
        glyphs are small, so this read uses a lower threshold and a larger
        upscale than the modal-title path.
        """
        region = self._region(self._panel(frame), MULTIPLIER_REGION)
        if region.size == 0 or self.backend is None:
            return None
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
        if int(mask.sum()) // 255 < 20:
            return None
        upscaled = cv2.resize(255 - mask, None, fx=3, fy=3,
                              interpolation=cv2.INTER_CUBIC)
        # A white margin helps the recogniser on tiny glyph strips.
        upscaled = cv2.copyMakeBorder(upscaled, 12, 12, 12, 12,
                                      cv2.BORDER_CONSTANT, value=255)
        return parse_multiplier(self.backend.read(upscaled))

    def read_timer(self, frame: np.ndarray) -> int | None:
        """Seconds left on the round clock, or None when unreadable.

        The clock is the game's own timestamp: logging words against it is
        exact regardless of capture latency or where a recording starts.
        """
        region = self._region(self._panel(frame), TIMER_REGION)
        text = self._read_digits(region)
        if not text:
            text = self._read_bright_text(region)
        return parse_timer(text)

    def _hud_visible(self, hsv: np.ndarray) -> bool:
        """True only when the gameplay HUD -- not lookalike UI -- is up.

        Two conditions, both aimed at the arcade's gold UI text, which can
        drift into the HUD regions on menu screens:

        * score, timer and high-score must each be lit individually -- gold
          leakage rarely covers all three fixed rectangles at once;
        * the combined sample must pass the same plausibility gate as the
          colour calibration. Gold is far more saturated than the HUD's
          yellow-green, so it fails the shift bound.
        """
        boxes = self.settings.geometry.hud_boxes
        if any(autocolor.box_candidates(hsv, box) < PER_BOX_MIN_PIXELS
               for box in boxes):
            return False
        anchor = autocolor.measure(hsv, boxes)
        if anchor is None:
            return False
        color = self.settings.color
        return autocolor.shifted_range(anchor, color.hsv_lo,
                                       color.hsv_hi) is not None

    # -- classification -----------------------------------------------------
    def classify(self, timestamp: float, frame: np.ndarray) -> GameState:
        panel = self._panel(frame)
        hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
        if self._hud_visible(hsv):
            return self._advance(timestamp, GameState.PLAYING)

        # No HUD: a modal screen, a transition, or not the minigame at all.
        # Title OCR is throttled; between reads the last state stands.
        if timestamp - self._last_ocr < self.ocr_interval:
            return self.state
        self._last_ocr = timestamp
        title = _letters(self._read_bright_text(
            self._region(panel, TITLE_REGION)))
        if _fuzzy_contains(title, GAME_OVER_KEY):
            if self.state is not GameState.GAME_OVER:
                self._game_over_at = timestamp
            # Read until settled, then FREEZE: later frames can misread (a
            # lost leading digit turned 5685 into 685 after the fact) and
            # must not overwrite a settled value.
            if not self.score_settled:
                score = self.read_final_score(frame)
                if score is not None:
                    self._score_reads.append(score)
                self._update_score(timestamp)
            return self._advance(timestamp, GameState.GAME_OVER)
        if _fuzzy_contains(title, START_TITLE_KEY):
            return self._advance(timestamp, GameState.START_SCREEN)
        return self._advance(timestamp, GameState.UNKNOWN)

    def _update_score(self, timestamp: float) -> None:
        """Best current estimate, and whether it can be trusted as final.

        The best estimate at any moment is the largest value that has been
        read at least twice (a single wild misread cannot become final).
        It is *settled* once the last three reads all agree on it and the
        modal has been up for 2 s -- long enough for the count-up to pass
        any value it briefly repeated.
        """
        from collections import Counter

        counts = Counter(self._score_reads)
        confirmed = [value for value, count in counts.items() if count >= 2]
        if not confirmed:
            return
        best = max(confirmed)
        self.final_score = best
        tail = self._score_reads[-3:]
        since = (self._game_over_at if self._game_over_at is not None
                 else timestamp)
        age = timestamp - since
        self.score_settled = (len(tail) == 3 and set(tail) == {best}
                              and age >= 2.0)

    def _advance(self, timestamp: float, state: GameState) -> GameState:
        if state is not self.state:
            self.transitions.append(Transition(timestamp, state))
            self.state = state
            if state is GameState.PLAYING:
                self.final_score = None
                self.score_settled = False
                self._score_reads = []
                self._game_over_at = None
        return state

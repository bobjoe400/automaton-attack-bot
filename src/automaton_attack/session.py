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
from .ocr import OcrBackend

# Regions in panel fractions (x0, y0, x1, y1), measured off the reference
# footage. Fractions survive any panel size the auto-locator finds.
TITLE_REGION = (0.26, 0.135, 0.74, 0.26)     # covers both modal titles
SCORE_ROW_REGION = (0.227, 0.272, 0.767, 0.313)   # "Total Score      750"
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
def locate_panel(frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Find the minigame frame by its border edges. None if implausible.

    The panel is a near-square bright frame filling most of the screen
    height on an otherwise dark page, so the strongest vertical edge in each
    horizontal half and the strongest horizontal edge in each vertical half
    are its four sides.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    col_edges = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)).sum(axis=0)
    row_edges = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)).sum(axis=1)

    x0 = int(np.argmax(col_edges[: width // 2]))
    x1 = int(np.argmax(col_edges[width // 2:])) + width // 2
    y0 = int(np.argmax(row_edges[: height // 2]))
    y1 = int(np.argmax(row_edges[height // 2:])) + height // 2

    panel_w, panel_h = x1 - x0, y1 - y0
    if panel_h < 0.5 * height:            # too small to be the minigame
        return None
    if not 0.85 <= panel_w / panel_h <= 1.15:   # the panel is near-square
        return None
    return x0, y0, x1, y1


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
        self.state = GameState.UNKNOWN
        self.transitions: list[Transition] = []
        self.final_score: int | None = None
        # The game-over score counts up when the modal appears; a single
        # read mid-animation is wrong (150 observed for a final 1,150).
        # settled = the same non-None value on two consecutive reads.
        self.score_settled = False
        self._last_score_read: int | None = None
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

    def read_final_score(self, frame: np.ndarray) -> int | None:
        """Read "Total Score NNN" off the game-over screen."""
        row = self._read_bright_text(
            self._region(self._panel(frame), SCORE_ROW_REGION))
        return parse_score(row)

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
            score = self.read_final_score(frame)
            self.score_settled = (score is not None
                                  and score == self._last_score_read)
            self._last_score_read = score
            if score is not None:
                self.final_score = score
            return self._advance(timestamp, GameState.GAME_OVER)
        if _fuzzy_contains(title, START_TITLE_KEY):
            return self._advance(timestamp, GameState.START_SCREEN)
        return self._advance(timestamp, GameState.UNKNOWN)

    def _advance(self, timestamp: float, state: GameState) -> GameState:
        if state is not self.state:
            self.transitions.append(Transition(timestamp, state))
            self.state = state
            if state is GameState.PLAYING:
                self.final_score = None
                self.score_settled = False
                self._last_score_read = None
        return state

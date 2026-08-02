"""Keystroke delivery.

Spaces and punctuation are skipped -- the minigame accepts letters only, and
lowercase is fine. Two typists exist: one that prints what it would send, and
one that actually sends it.
"""

from __future__ import annotations

import random
import time

from .config import Behaviour


class DryRunTypist:
    """Records what would have been typed. The default everywhere."""

    live = False

    def __init__(self, behaviour: Behaviour | None = None) -> None:
        self.behaviour = behaviour or Behaviour()
        self.typed: list[str] = []

    def type(self, text: str) -> None:
        self.typed.append(text)


class DirectInputTypist:
    """Sends real keystrokes via pydirectinput (works with DirectX games)."""

    live = True

    def __init__(self, behaviour: Behaviour | None = None) -> None:
        try:
            import pydirectinput
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "pydirectinput is not installed. Run: uv sync --extra live"
            ) from exc
        pydirectinput.PAUSE = 0
        self._pydirectinput = pydirectinput
        self.behaviour = behaviour or Behaviour()
        self.typed: list[str] = []
        self._min_seconds_per_char = self._pace()

    def _pace(self) -> float:
        """Seconds per character implied by the WPM cap (0 = uncapped).

        Valve patched a pause-typing exploit in this minigame in July 2026, so
        the leaderboard is watched. A cap keeps output within human range.
        """
        wpm = self.behaviour.max_wpm
        if wpm <= 0:
            return 0.0
        return 60.0 / (wpm * 5.0)   # 5 characters per "word", by convention

    def type(self, text: str) -> None:
        for char in text:
            self._pydirectinput.press(char)
            delay = random.uniform(*self.behaviour.key_delay)
            time.sleep(max(delay, self._min_seconds_per_char))
        self.typed.append(text)


def make_typist(live: bool, behaviour: Behaviour | None = None):
    return (DirectInputTypist if live else DryRunTypist)(behaviour)

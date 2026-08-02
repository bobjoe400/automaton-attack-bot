"""Turn session log lines into display state. Pure functions, tested.

The drive loop's ``say`` channel already announces every state change,
combo event and score; parsing those lines means the panel needs no hooks
inside the engine, and anything the bot learns to report shows up here
for free.
"""

from __future__ import annotations

import re

_STATE = re.compile(r"--- (.+?) ---")
_COMBO = re.compile(r"combo x([\d.]+)")
_COMBO_LOST = re.compile(r"!!! COMBO LOST x[\d.]+ -> x([\d.]+)")
_SCORE = re.compile(r"total score: (\d+)")
_TYPED = re.compile(r"^(\d+) words typed over")
_TIMESTAMP = re.compile(r"^\[\s*[\d.]+s\] ")

# state-line value -> (banner label, banner colour)
STATES = {
    "playing": ("ACTIVE", "#1f8a3b"),
    "start-screen": ("READY", "#1565c0"),
    "game-over": ("GAME OVER", "#7b1fa2"),
    "unknown": ("SEARCHING", "#8a6d3b"),
    "idle": ("IDLE", "#546e7a"),
    "working": ("WORKING", "#546e7a"),
}


def interpret(line: str) -> dict:
    """Map one log line to display updates (subset of: state, combo,
    combo_lost, score, typed)."""
    updates: dict = {}
    state = _STATE.search(line)
    if state and state.group(1) in STATES:
        updates["state"] = state.group(1)
    lost = _COMBO_LOST.search(line)
    combo = lost or _COMBO.search(line)
    if combo:
        updates["combo"] = combo.group(1)
        updates["combo_lost"] = bool(lost)
    score = _SCORE.search(line)
    if score:
        updates["score"] = score.group(1)
    typed = _TYPED.search(line.strip())
    if typed:
        updates["typed"] = typed.group(1)
    return updates


def display_text(line: str) -> str:
    """The activity feed shows the message, not the raw log record."""
    return _TIMESTAMP.sub("", line).rstrip()

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


_LOG_PATH = re.compile(r"Session log \(full detail\): (.+)$")
_CAPTURE = re.compile(
    r"Live capture on monitor (\S+) \((\d+x\d+)\), OCR=(\w+)")
_SCORE_LINE = re.compile(r"total score: (\S+)")

# Ordered (pattern, friendly text) -- first hit wins. Lines that match
# nothing produce no status update: the status bar shows curated moments,
# not the log.
_STATUS = [
    (re.compile(r"--- unknown ---"), "Searching for the minigame..."),
    (re.compile(r"--- start-screen ---"), "Start screen found"),
    (re.compile(r"--- playing ---"), "Round in progress"),
    (re.compile(r"--- game-over ---"), "Round over"),
    (re.compile(r"Located minigame panel"), "Minigame panel located"),
    (re.compile(r"\[dry-run\] would click"), "Dry run: would click PLAY"),
    (re.compile(r"clicking PLAY AGAIN"), "Clicking PLAY AGAIN"),
    (re.compile(r"clicking PLAY"), "Clicking PLAY"),
    (re.compile(r"no button text at the click target"),
     "State looks wrong -- holding the click"),
    (re.compile(r"clicks are not landing"),
     "Clicks are not landing -- re-locating the panel"),
    (re.compile(r"Nothing recognised after 5s"),
     "Can't see the minigame -- is it on the captured monitor?"),
    (re.compile(r"capture stalled"), "Capture stalled -- recovering"),
    (re.compile(r"!!! COMBO LOST"), "Combo lost!"),
    (re.compile(r"Loaded \d+ vocabulary"), "Word corpus loaded"),
    (re.compile(r"Falling back to a dry run"),
     "Input libraries missing -- dry run only"),
]


def status_message(line: str) -> str | None:
    """One friendly sentence for the status bar, or None to leave it be."""
    score = _SCORE_LINE.search(line)
    if score:
        return f"Round over -- score {score.group(1)}"
    for pattern, text in _STATUS:
        if pattern.search(line):
            return text
    return None


def session_info(line: str) -> str | None:
    """The capture-setup chip: monitor, resolution, OCR backend."""
    m = _CAPTURE.search(line)
    if m:
        return f"monitor {m.group(1)} · {m.group(2)} · {m.group(3)}"
    return None


def log_path(line: str) -> str | None:
    """The session log file the current run is writing."""
    m = _LOG_PATH.search(line)
    return m.group(1).strip() if m else None

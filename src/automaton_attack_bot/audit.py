"""Keystroke-to-frame audit: which keys actually landed.

``automaton audit CLIP --log RUN.log`` replays a recording against the
session's keystroke ledger (every key's real send time is in the log) and
judges, key by key, whether the word's input tracker advanced on screen.
The output is a per-word mark string -- ``+`` landed, ``.`` ghost, ``?``
untrackable -- plus a list of non-progressive bouts with log and video
timestamps to jump to.

Method (validated on run31 frame-by-frame): the tracker text under a
word's label renders consumed letters gray and remaining letters white,
so progress = gray / (gray + white) climbs 0 -> 1 as keys land. A key
"landed" when that fraction steps up within 0.2s of the send time.

Ground truth learned this way so far: a word accepts keys only while its
label is actually rendered -- fly-in, scenery occlusion (BUTTERFLY behind
a barrel) and the gray lock-out all suppress input identically.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

TYPED_RE = re.compile(
    r"^\[\s*([\d.]+)s\] (\S.*?)\s{2,}(?:vocab|phrase|fallback)"
    r"\s+score=\S+ ocr=.*\[queued [\d.]+s\]"
    r"\s*\[keys mono=([\d.]+)((?: \+[\d.]+)*)\]")
TRACE_RE = re.compile(r"^\s+\(\s*(\d+),\s*(\d+)\) ocr=.* -> (.+) \(\d\.\d\d\)$")
STAMP_RE = re.compile(r"^\[\s*([\d.]+)s\]")
EPOCH_RE = re.compile(r"capture epoch: monotonic ([\d.]+)")
PANEL_RE = re.compile(r"Located minigame panel at \((\d+), (\d+), (\d+), (\d+)\)")
PLAYING_RE = re.compile(r"^\[\s*([\d.]+)s\] --- playing ---")
OVER_RE = re.compile(r"^\[\s*([\d.]+)s\] --- (?:game-over|unknown) ---")

# Judgement windows around each key's send time (seconds).
BEFORE = (-0.20, 0.02)
AFTER = (0.03, 0.20)
MIN_TEXT_PIXELS = 120


@dataclass
class KeyedWord:
    name: str
    keys: list[float]                    # log-time of every key actually sent
    progress: list = field(default_factory=list)   # (log_t, fraction)
    marks: str = ""


@dataclass
class SessionLedger:
    epoch: float
    panel: tuple[int, int, int, int]
    rounds: list[tuple[float, float]]
    words: list[KeyedWord]
    traces: dict[str, list[tuple[float, int, int]]]


def parse_log(path: str) -> SessionLedger:
    epoch = None
    panel = None
    rounds: list[list[float]] = []
    words: list[KeyedWord] = []
    traces: dict[str, list[tuple[float, int, int]]] = defaultdict(list)
    last_ts = 0.0
    for line in open(path, encoding="utf-8"):
        s = STAMP_RE.match(line)
        if s:
            last_ts = float(s.group(1))
        m = EPOCH_RE.search(line)
        if m and epoch is None:
            epoch = float(m.group(1))
        m = PANEL_RE.search(line)
        if m:
            panel = tuple(int(g) for g in m.groups())
        m = PLAYING_RE.match(line)
        if m:
            rounds.append([float(m.group(1)), None])
        m = OVER_RE.match(line)
        if m and rounds and rounds[-1][1] is None:
            rounds[-1][1] = float(m.group(1))
        m = TYPED_RE.match(line)
        if m and epoch is not None:
            keys = [float(m.group(3)) - epoch]
            for d in m.group(4).split():
                keys.append(keys[-1] + float(d))
            words.append(KeyedWord(m.group(2).strip(), keys))
            continue
        m = TRACE_RE.match(line)
        if m and m.group(3).strip() != "-":
            traces[m.group(3).strip()].append(
                (last_ts, int(m.group(1)), int(m.group(2))))
    if epoch is None:
        raise ValueError(
            "log has no 'capture epoch' line -- recorded before keystroke "
            "logging existed?")
    if panel is None:
        raise ValueError("log has no located-panel line")
    done = [(a, b) for a, b in rounds if b is not None]
    if not done:
        raise ValueError("log contains no completed round")
    return SessionLedger(epoch, panel, done, words, dict(traces))


def judge(word: KeyedWord) -> str:
    """Mark string for one word from its progress samples."""
    marks = []
    for kt in word.keys:
        before = [p for t, p in word.progress
                  if kt + BEFORE[0] <= t <= kt + BEFORE[1]]
        after = [p for t, p in word.progress
                 if kt + AFTER[0] <= t <= kt + AFTER[1]]
        if not before or not after:
            marks.append("?")
            continue
        step = max(after) - min(before)
        marks.append("+" if step >= 0.4 / len(word.keys) else ".")
    word.marks = "".join(marks)
    return word.marks


def run_audit(clip: str, log_path: str, round_index: int | None = None,
              progress=print) -> str:
    import cv2

    from .core.config import Settings
    from .core.ocr import get_backend
    from .core.session import (GameState, SessionTracker, locate_panel)

    ledger = parse_log(log_path)

    cap = cv2.VideoCapture(clip)
    if not cap.isOpened():
        raise FileNotFoundError(clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_seconds = frame_count / fps

    # -- pick the round this clip shows ----------------------------------
    if round_index is not None:
        r0, r1 = ledger.rounds[round_index - 1]
    else:
        r0, r1 = min(ledger.rounds,
                     key=lambda r: abs((r[1] - r[0]) - clip_seconds))
    progress(f"round: log {r0:.1f}-{r1:.1f}s "
             f"({ledger.rounds.index((r0, r1)) + 1} of {len(ledger.rounds)})")

    # -- anchor: find the clip time where the round clock reads a value --
    settings = Settings()
    backend = get_backend("auto")
    ok, first = cap.read()
    if not ok:
        raise RuntimeError("empty clip")
    settings = settings.for_resolution(first.shape[1], first.shape[0])
    clip_panel = None
    tracker = None
    offset = None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    probes = []
    for i in range(0, frame_count, int(fps / 2)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        if clip_panel is None:
            clip_panel = locate_panel(frame)
            if clip_panel:
                settings = settings.with_panel(clip_panel)
                tracker = SessionTracker(backend, settings)
            continue
        if tracker.classify(i / fps, frame) is not GameState.PLAYING:
            continue
        clock = tracker.read_timer(frame)
        if clock is not None and clock < 60:
            # log time at this clock = round start + (60 - clock)
            probes.append(i / fps - (r0 + 60 - clock))
        if len(probes) >= 5:
            break
    if not probes or clip_panel is None:
        raise RuntimeError("could not anchor the clip to the round clock")
    probes.sort()
    offset = probes[len(probes) // 2]       # video = log + offset
    progress(f"anchor: video = log + {offset:.2f}s   "
             f"clip panel {clip_panel}")

    sx = (clip_panel[2] - clip_panel[0]) / (ledger.panel[2] - ledger.panel[0])
    sy = (clip_panel[3] - clip_panel[1]) / (ledger.panel[3] - ledger.panel[1])

    words = [w for w in ledger.words if r0 <= w.keys[0] <= r1]

    def pos_at(name, t):
        cands = [c for c in ledger.traces.get(name, ())
                 if abs(c[0] - t) < 2.5]
        if not cands:
            return None
        return min(cands, key=lambda c: abs(c[0] - t))[1:]

    windows = [(w.keys[0] - 0.45, w.keys[-1] + 0.65) for w in words]
    t_lo = min(w[0] for w in windows) + offset
    t_hi = max(w[1] for w in windows) + offset
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(t_lo * fps)))
    fi = max(0, int(t_lo * fps))
    next_report = 0.0
    while True:
        ok, frame = cap.read()
        if not ok or fi / fps > t_hi:
            break
        log_t = fi / fps - offset
        fi += 1
        span = (log_t - r0) / max(1.0, r1 - r0)
        if span >= next_report:
            progress(f"  auditing {100 * min(1.0, max(0.0, span)):3.0f}%")
            next_report += 0.25
        hsv = None
        for w, (w0, w1) in zip(words, windows):
            if not (w0 <= log_t <= w1):
                continue
            p = pos_at(w.name, log_t)
            if not p:
                continue
            if hsv is None:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lx = int(p[0] * sx) + clip_panel[0]
            ly = int(p[1] * sy) + clip_panel[1]
            n = len(w.keys)
            x0, x1 = max(0, lx - 40), min(frame.shape[1], lx + n * 24 + 90)
            y0, y1 = ly + 12, min(frame.shape[0], ly + 75)
            crop = hsv[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            s, v = crop[..., 1], crop[..., 2]
            white = int(((s < 80) & (v > 155)).sum())
            gray = int(((s < 80) & (v > 92) & (v <= 155)).sum())
            if white + gray > MIN_TEXT_PIXELS:
                w.progress.append((log_t, gray / (white + gray)))
    cap.release()

    lines = []
    judged = landed = 0
    bouts = []
    for w in words:
        marks = judge(w) if len(w.progress) >= 4 else "?" * len(w.keys)
        w.marks = marks
        judged += sum(1 for m in marks if m in "+.")
        landed += marks.count("+")
        lines.append(f"  {w.keys[0]:7.2f}  {w.name:38} [{marks}]")
        if marks.count(".") >= max(2, len(w.keys) // 2):
            bouts.append(w)
    out = ["Per-word key audit (+ landed, . ghost, ? untrackable):"]
    out += lines
    out.append(f"\nkeys judged: {judged}  landed: {landed} "
               f"({100 * landed / max(1, judged):.0f}%)")
    if bouts:
        out.append("\nNon-progressive bouts (jump here in the clip):")
        for w in bouts:
            out.append(f"  {w.name:30} log {w.keys[0]:6.2f}s = "
                       f"video {w.keys[0] + offset:6.2f}s   [{w.marks}]")
    else:
        out.append("\nNo non-progressive bouts found.")
    return "\n".join(out)

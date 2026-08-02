"""The typing worker: urgency order, staleness, and events."""

import time

from automaton_attack.config import Settings
from automaton_attack.detect import Detection
from automaton_attack.engine import Engine, TypedWord
from automaton_attack.keyboard import DryRunTypist, TypingWorker
from automaton_attack.lexicon import Match


def word(name, pos, timestamp=0.0):
    detection = Detection(box=(pos[0], pos[1], 120, 20), raw=name,
                          match=Match(name, 1.0, "vocab"))
    return TypedWord(timestamp, detection, detection.match.keystrokes)


def urgency(detection):
    settings = Settings()
    return Engine._urgency(
        type("E", (), {"settings": settings, "PLATFORM": Engine.PLATFORM})(),
        detection)


def drain(worker, expected, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(worker.typist.typed) >= expected:
            return
        time.sleep(0.02)


def test_most_urgent_word_types_first():
    typist = DryRunTypist()
    worker = TypingWorker(typist, urgency)
    # submit before starting so ordering is deterministic
    worker.submit(word("BANE", (60, 40)))            # far corner
    worker.submit(word("PUDGE", (430, 580)))         # at the platform
    worker.submit(word("LINA", (450, 900)))          # bottom spawn
    worker.start()
    drain(worker, 3)
    worker.stop()
    assert typist.typed == ["pudge", "lina", "bane"]


def test_duplicate_keystrokes_are_not_queued_twice():
    typist = DryRunTypist()
    worker = TypingWorker(typist, urgency)
    worker.submit(word("BANE", (100, 100)))
    worker.submit(word("BANE", (105, 102)))
    worker.start()
    drain(worker, 1)
    worker.stop()
    assert typist.typed == ["bane"]


def test_stale_words_are_dropped_not_typed():
    """A word that waited past its lifetime is gone from the screen;
    typing it would be pure keystroke waste."""
    clock = [0.0]
    typist = DryRunTypist()
    worker = TypingWorker(typist, urgency, stale_after=4.0,
                          clock=lambda: clock[0])
    worker.submit(word("BANE", (100, 100)))
    clock[0] = 10.0                     # five seconds past stale
    worker.start()
    deadline = time.monotonic() + 2.0
    while worker.dropped_stale == 0 and time.monotonic() < deadline:
        time.sleep(0.02)
    worker.stop()
    assert typist.typed == []
    assert worker.dropped_stale == 1


def test_on_typed_fires_after_the_keys_are_sent():
    events = []
    typist = DryRunTypist()
    worker = TypingWorker(typist, urgency,
                          on_typed=lambda w, waited: events.append(
                              (w.keystrokes, waited)))
    worker.submit(word("BANE", (100, 100)))
    worker.start()
    drain(worker, 1)
    worker.stop()
    assert [e[0] for e in events] == ["bane"]
    assert events[0][1] >= 0.0

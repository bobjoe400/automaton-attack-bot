"""The typing worker: urgency order, staleness, and events."""

import time

from automaton_attack_bot.core.config import Settings
from automaton_attack_bot.core.detect import Detection
from automaton_attack_bot.core.engine import Engine, TypedWord
from automaton_attack_bot.core.keyboard import DryRunTypist, TypingWorker
from automaton_attack_bot.core.lexicon import Match


def word(name, pos, timestamp=0.0):
    detection = Detection(box=(pos[0], pos[1], 120, 20), raw=name,
                          match=Match(name, 1.0, "vocab"))
    return TypedWord(timestamp, detection, detection.match.keystrokes)


def urgency(detection):
    class _StubDetector:
        settings = Settings()

        def detect(self, *args, **kwargs):
            return []

    return Engine(_StubDetector(), settings=Settings())._urgency(detection)


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


def test_weak_guesses_are_triaged_when_the_queue_is_deep():
    """Keyboard time is the scarce resource under load: a 0.67 fly-in
    partial gets retyped correctly a scan later anyway, so it loses its
    seat when 3+ words are waiting."""
    from automaton_attack_bot.core.detect import Detection
    from automaton_attack_bot.core.lexicon import Match

    typist = DryRunTypist()
    worker = TypingWorker(typist, urgency)

    def scored(name, score, pos, source="vocab"):
        d = Detection(box=(pos[0], pos[1], 120, 20), raw=name,
                      match=Match(name, score, source))
        return TypedWord(0.0, d, d.match.keystrokes)

    # ROT sits closest to the platform, so it pops FIRST -- while four
    # other words are still waiting. That is exactly when a weak guess
    # must lose its seat. (A weak guess popping after the queue drains
    # types normally; see the shallow-queue test.)
    worker.submit(scored("ROT", 0.67, (430, 590)))          # weak, urgent
    worker.submit(scored("PUDGE", 1.0, (430, 500)))
    worker.submit(scored("LINA", 1.0, (450, 900)))
    worker.submit(scored("BANE", 1.0, (60, 40)))
    worker.submit(scored("UNKNOWNWORD", 0.0, (200, 200), source="fallback"))
    worker.start()
    drain(worker, 4)
    worker.stop()
    assert "rot" not in typist.typed            # triaged under load
    assert "unknownword" in typist.typed        # fallbacks are exempt
    assert worker.dropped_triage == 1


def test_weak_guesses_type_when_the_queue_is_shallow():
    from automaton_attack_bot.core.detect import Detection
    from automaton_attack_bot.core.lexicon import Match

    typist = DryRunTypist()
    worker = TypingWorker(typist, urgency)
    d = Detection(box=(100, 100, 120, 20), raw="ROT",
                  match=Match("ROT", 0.67, "vocab"))
    worker.submit(TypedWord(0.0, d, d.match.keystrokes))
    worker.start()
    drain(worker, 1)
    worker.stop()
    assert typist.typed == ["rot"]


def test_clicks_translate_frame_coords_to_the_capture_monitor():
    """Frames are monitor-relative, clicks are virtual-screen absolute; on
    a non-primary monitor the difference sent PLAY clicks to the wrong
    screen entirely."""
    from automaton_attack_bot.core.keyboard import DryRunTypist

    typist = DryRunTypist(origin=(2560, -180))
    typist.click(965, 979)
    assert typist.clicked == [(3525, 799)]


def test_default_origin_leaves_clicks_untouched():
    from automaton_attack_bot.core.keyboard import DryRunTypist

    typist = DryRunTypist()
    typist.click(965, 979)
    assert typist.clicked == [(965, 979)]

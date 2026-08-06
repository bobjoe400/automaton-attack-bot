"""The drive loop's function-local imports only execute when a session
actually starts -- which is how a broken relative import survived the whole
suite and detonated on the first live Start. These tests force that path."""

from automaton_attack_bot.core.config import Settings
from automaton_attack_bot.core.drive import drive, summarise
from automaton_attack_bot.core.keyboard import DryRunTypist
from automaton_attack_bot.core.lexicon import Lexicon


def _run(threaded):
    settings = Settings()
    engine, tracker = drive(
        iter([]), Lexicon(["Pudge"]), None, settings,
        DryRunTypist(settings.behaviour), threaded=threaded)
    return engine, tracker


def test_drive_constructs_and_returns_on_no_frames():
    engine, tracker = _run(threaded=False)
    assert engine.stats.frames == 0
    assert tracker.transitions == []
    summarise(engine, tracker)      # must not raise either


def test_threaded_drive_spins_up_and_down_cleanly():
    """The live path: typing worker thread + scan pipeline."""
    engine, tracker = _run(threaded=True)
    assert engine.stats.frames == 0

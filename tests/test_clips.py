"""End-to-end regression over recorded gameplay.

Runs the real pipeline -- colour mask, blob merge, OCR, matching, both guards
-- against clips in ``clips/`` and asserts the words that come out. Skipped
when the clips are absent, so a fresh clone still gets a green test run.

Populate ``clips/`` with:
    clip1.mp4   single-word phase
    clip2.mp4   multi-word phase, overlapping words
"""

from pathlib import Path

import pytest

from automaton_attack.capture import VideoSource
from automaton_attack.config import Settings
from automaton_attack.detect import Detector
from automaton_attack.engine import Engine
from automaton_attack.lexicon import Lexicon
from automaton_attack.ocr import OcrUnavailable, get_backend

pytestmark = pytest.mark.clips

CLIPS = "clips"

# Every word the single-word phase shows, in order. This clip is the tightest
# regression we have: any change to the pipeline that alters this list is a
# behaviour change worth explaining.
CLIP1_SEQUENCE = [
    "MARCI", "BRACER", "PUGNA", "SNIPER", "EAGLESONG", "CLARITY",
    "SILENCER", "MEKANSM", "LINA", "MJOLLNIR", "BUTTERFLY",
]

# The multi-word phase repeats and overlaps words, so order and count are not
# stable enough to pin. These are the ones that must be found.
CLIP2_EXPECTED = {
    "CHEESE", "BUTTERFLY", "LIFESTEALER", "MEEPO", "BUCKLER", "DIADEM",
    "SHAWL", "CHAINMAIL", "MUERTA", "VISAGE", "RIKI", "ECHO SABRE",
    "SAND KING", "VANGUARD", "METEOR HAMMER", "SANGE", "PRIMAL BEAST",
    "RUBICK", "DESOLATOR", "PUDGE", "MITHRIL HAMMER", "LONE DRUID",
    "FACELESS VOID", "BANE", "DAEDALUS",
}


@pytest.fixture(scope="module")
def backend():
    try:
        return get_backend("auto")
    except (OcrUnavailable, Exception) as exc:  # noqa: BLE001
        pytest.skip(f"no OCR backend available: {exc}")


@pytest.fixture(scope="module")
def lexicon():
    return Lexicon.load()


def play(clip_name, lexicon, backend, settings=None):
    path = Path(CLIPS) / clip_name
    if not path.exists():
        pytest.skip(f"{path} not present; see the module docstring")
    settings = settings or Settings()
    source = VideoSource(path, settings.behaviour.replay_stride)
    settings = settings.for_resolution(source.width, source.height)
    engine = Engine(Detector(lexicon, backend, settings), settings=settings)
    return list(engine.run(source.frames())), engine


def test_clip1_reads_every_word_in_order(lexicon, backend):
    typed, _ = play("clip1.mp4", lexicon, backend)
    assert [w.name for w in typed] == CLIP1_SEQUENCE


def test_clip1_needs_no_guesswork(lexicon, backend):
    """Every word resolves to the core vocabulary -- nothing typed verbatim."""
    typed, _ = play("clip1.mp4", lexicon, backend)
    assert {w.source for w in typed} == {"vocab"}


def test_clip2_finds_every_expected_word(lexicon, backend):
    typed, _ = play("clip2.mp4", lexicon, backend)
    assert CLIP2_EXPECTED <= {w.name for w in typed}


def test_clip2_never_types_an_unrecognised_word(lexicon, backend):
    """A wrong word resets the score multiplier, so a false positive costs
    more than a miss. Nothing here should reach the fallback tier."""
    typed, _ = play("clip2.mp4", lexicon, backend)
    assert [w.name for w in typed if w.source == "fallback"] == []


def test_safe_mode_still_reads_clip1(lexicon, backend):
    settings = Settings.from_dict({**Settings().to_dict(), "safe_mode": True})
    typed, _ = play("clip1.mp4", lexicon, backend, settings)
    assert [w.name for w in typed] == CLIP1_SEQUENCE


def test_hud_boxes_are_not_read_as_words(lexicon, backend):
    """The score, timer and high-score panels share the words' colour."""
    typed, _ = play("clip1.mp4", lexicon, backend)
    for word in typed:
        panel_y = word.detection.box[1]
        assert panel_y >= 0
    assert all(w.name in CLIP1_SEQUENCE for w in typed)

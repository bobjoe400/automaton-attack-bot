"""Matching behaviour, including the OCR misreads observed in the clips."""

import pytest

from automaton_attack.config import Matching
from automaton_attack.lexicon import Lexicon, to_key


@pytest.fixture(scope="module")
def lexicon():
    return Lexicon.load()


@pytest.fixture(scope="module")
def vocab_only():
    return Lexicon.load(with_phrases=False)


def test_to_key_strips_spaces_and_punctuation():
    assert to_key("Bane of your existence.") == "BANEOFYOUREXISTENCE"
    assert to_key("Sange & Yasha") == "SANGEYASHA"


def test_keystrokes_are_letters_only_lowercase(vocab_only):
    match = vocab_only.match("ECHO SABRE")
    assert match is not None
    assert match.keystrokes == "echosabre"


@pytest.mark.parametrize("raw, expected", [
    ("MARCI", "MARCI"),
    ("BUTTERFLY", "BUTTERFLY"),
    ("EAGLESONG", "EAGLESONG"),
    ("ECHO SABRE", "ECHO SABRE"),
])
def test_clean_reads_match_exactly(vocab_only, raw, expected):
    match = vocab_only.match(raw)
    assert match is not None
    assert match.name == expected
    assert match.score == pytest.approx(1.0)
    assert match.source == "vocab"


@pytest.mark.parametrize("raw, expected", [
    # Real misreads captured from the clips. Both OCR backends mangle these,
    # differently -- the fuzzy tier is what makes them survivable.
    ("METEORHARIMER", "METEOR HAMMER"),
    ("METEOR HANIMER", "METEOR HAMMER"),
    ("MITHRILHAMPEIER", "MITHRIL HAMMER"),
    ("MITHRIL HAM:IER", "MITHRIL HAMMER"),
    ("FRITAALBEAST", "PRIMAL BEAST"),
    ("M]OLLNI!!", "MJOLLNIR"),
    ("SAND KINT.", "SAND KING"),
])
def test_ocr_misreads_still_resolve(vocab_only, raw, expected):
    match = vocab_only.match(raw)
    assert match is not None
    assert match.name == expected
    assert match.score >= Matching().vocab_cutoff


def test_ambiguous_read_prefers_the_extension(vocab_only):
    """'WEAVEP' scores WEAVE 0.91 and WEAVER 0.83 -- too close to call.
    Typing WEAVER completes WEAVE on the way through, so the extension is
    the safe guess; typing WEAVE against WEAVER leaves it a letter short."""
    match = vocab_only.match("WEAVEP")
    assert match is not None
    assert match.name == "WEAVER"


def test_single_letter_extension_is_cheap_insurance(vocab_only):
    """Even a clean WEAVE read types WEAVER: ratio('WEAVE','WEAVER')=0.91
    is within epsilon, and the maths is asymmetric -- the extension costs
    one stray keystroke if the word really was WEAVE, but typing WEAVE
    against a WEAVER leaves it unfinished and resets the multiplier."""
    match = vocab_only.match("WEAVE")
    assert match is not None
    assert match.name == "WEAVER"
    assert match.score == pytest.approx(1.0)   # confidence is the best match's


def test_distant_extensions_are_not_preferred(vocab_only):
    """SANGE must not balloon into SANGE AND YASHA (0.55, far outside
    epsilon) -- extensions only win a near-tie, not every prefix."""
    match = vocab_only.match("SANGE")
    assert match is not None
    assert match.name == "SANGE"


def test_voice_line_matches_via_phrase_corpus(lexicon):
    match = lexicon.match("BANEOFYOUREXISTENCE")
    assert match is not None
    assert match.source == "phrase"
    assert match.name == "BANE OF YOUR EXISTENCE."
    assert match.score >= Matching().phrase_cutoff


def test_short_reads_are_rejected(vocab_only):
    assert vocab_only.match("AB") is None


def test_unmatched_long_read_falls_back_to_itself(vocab_only):
    raw = "SOMETHINGNOTINTHEVOCABULARY"
    match = vocab_only.match(raw)
    assert match is not None
    assert match.source == "fallback"
    # score 0.0 forces the engine's two-scan confirmation before typing
    assert match.score == 0.0


def test_safe_mode_disables_the_fallback_tier(vocab_only):
    assert vocab_only.match("SOMETHINGNOTINTHEVOCABULARY",
                            allow_fallback=False) is None


def test_removed_items_stay_out_of_the_vocabulary(vocab_only):
    """Cornucopia and Eternal Shroud left the game; they must not steal
    fuzzy matches from live items."""
    keys = {key for _, key in vocab_only.vocab}
    assert "CORNUCOPIA" not in keys
    assert "ETERNALSHROUD" not in keys


def test_recent_patch_items_are_present(vocab_only):
    keys = {key for _, key in vocab_only.vocab}
    for item in ("SHAWL", "CHASMSTONE", "SPLINTMAIL", "WIZARDHAT",
                 "HYDRASBREATH"):
        assert item in keys, f"{item} missing from vocab.txt"

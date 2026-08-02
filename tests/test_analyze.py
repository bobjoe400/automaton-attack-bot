"""Post-round read analysis."""

from automaton_attack_bot.analyze import ReadLog
from automaton_attack_bot.core.detect import Detection
from automaton_attack_bot.core.lexicon import Match


def det(raw, name=None, score=0.0):
    match = Match(name, score, "vocab") if name else None
    return Detection(box=(100, 100, 120, 20), raw=raw, match=match)


def test_stable_weak_match_is_flagged():
    log = ReadLog()
    for t in (10.0, 10.5, 11.0):
        log.add(t, det(" ETERNAL SHROUD", "INFERNAL SHRED", 0.69))
    weak = log.weak_matches()
    assert len(weak) == 1
    assert weak[0].best_name == "INFERNAL SHRED"
    assert "ETERNAL SHROUD" in weak[0].sample_raw


def test_stable_unmatched_read_is_a_gap():
    log = ReadLog()
    log.add(5.0, det("IMPERIA"))
    log.add(5.5, det("IMPERIA"))
    gaps = log.unmatched()
    assert len(gaps) == 1
    assert gaps[0].key == "IMPERIA"


def test_one_off_reads_are_noise_not_findings():
    """Partial fly-in reads never repeat identically; they must not clutter
    the report."""
    log = ReadLog()
    log.add(5.0, det("TUS"))          # too short
    log.add(5.5, det("BUTTERFL"))     # seen once only
    log.add(6.0, det("BUTTERFLY", "BUTTERFLY", 1.0))
    log.add(6.5, det("BUTTERFLY", "BUTTERFLY", 1.0))
    assert log.unmatched() == []
    assert log.weak_matches() == []


def test_confident_matches_stay_out_of_the_report():
    log = ReadLog()
    for t in (1.0, 1.5, 2.0):
        log.add(t, det("SHAWL", "SHAWL", 1.0))
    assert log.weak_matches() == []
    assert "every stable read matched" in log.report()

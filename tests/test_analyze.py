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


# -- screen-time tracking ----------------------------------------------------
def test_lifetimes_merge_sightings_and_split_on_gaps():
    from automaton_attack_bot.analyze import ReadLog

    log = ReadLog()
    for t in (10.0, 10.2, 10.4, 10.6):
        log.add_lifetime(t, "LEGION COMMANDER", 0.5, clock=30)
    log.add_lifetime(15.0, "LEGION COMMANDER", 0.4, clock=25)  # respawn
    episodes = [e for e in log.episodes if e.name == "LEGION COMMANDER"]
    assert len(episodes) == 2
    assert abs(episodes[0].duration - 0.6) < 1e-9
    assert episodes[0].sightings == 4


def test_lingerers_flags_the_outlier():
    from automaton_attack_bot.analyze import ReadLog

    log = ReadLog()
    for start, name in ((0, "BANE"), (2, "PUDGE"), (4, "LINA")):
        for t in (start, start + 0.3, start + 0.6):
            log.add_lifetime(t, name, 0.4)
    for t in [10.0 + 0.2 * i for i in range(30)]:          # 5.8s lingerer
        log.add_lifetime(t, "LEGION COMMANDER", 0.55)
    average, slow = log.lingerers()
    assert [e.name for e in slow] == ["LEGION COMMANDER"]
    assert average < 2.0 < slow[0].duration


# -- keystroke audit ---------------------------------------------------------
def test_audit_judge_marks_landed_and_ghost_keys():
    from automaton_attack_bot.audit import KeyedWord, judge

    word = KeyedWord("KAYA", keys=[10.0, 10.1, 10.2, 10.3])
    # progress rises through the first keys then flattens; the windows
    # overlap adjacent keys by design (empirically calibrated on run31),
    # so the last flat key is the unambiguous ghost
    word.progress = [(9.85, 0.0), (10.05, 0.3), (10.15, 0.6),
                     (10.28, 0.6), (10.42, 0.6)]
    assert judge(word) == "+++."


def test_audit_parse_log_reads_ledger_epoch_and_rounds(tmp_path):
    from automaton_attack_bot.audit import parse_log

    log = tmp_path / "run.log"
    log.write_text(
        "capture epoch: monotonic 1000.000 = t 0.000\n"
        "Located minigame panel at (473, 52, 1447, 1024) "
        "(configured geometry re-anchored).\n"
        "[   1.00s] --- playing ---\n"
        "    ( 100, 200) ocr='KAYA' -> KAYA (1.00)\n"
        "[   2.00s] KAYA         vocab    score=1.00 ocr='KAYA'  "
        "[queued 0.00s]  [keys mono=1002.000 +0.100 +0.100 +0.100]\n"
        "[  60.00s] --- game-over ---\n", encoding="utf-8")
    ledger = parse_log(str(log))
    assert ledger.rounds == [(1.0, 60.0)]
    assert len(ledger.words) == 1
    assert ledger.words[0].keys[0] == 2.0        # mono 1002 - epoch 1000
    assert abs(ledger.words[0].keys[-1] - 2.3) < 1e-9
    assert ledger.traces["KAYA"] == [(1.0, 100, 200)]

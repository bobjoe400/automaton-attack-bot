"""The panel's log-line interpreter -- the GUI's only logic worth testing
headless. Lines here are verbatim from real session logs."""

from automaton_attack_bot.gui.feed import display_text, interpret


def test_state_transitions_are_recognised():
    assert interpret("[  12.34s] --- playing ---") == {"state": "playing"}
    assert interpret("[   1.00s] --- start-screen ---") == {
        "state": "start-screen"}
    assert interpret("[  70.40s] --- game-over ---")["state"] == "game-over"


def test_combo_lines_carry_value_and_loss_flag():
    up = interpret("[  20.10s] combo x2.5, clock 0:41")
    assert up == {"combo": "2.5", "combo_lost": False}
    lost = interpret("[  33.00s] !!! COMBO LOST x3.0 -> x1.0, clock 0:28")
    assert lost == {"combo": "1.0", "combo_lost": True}


def test_score_and_summary_lines():
    over = interpret("[  70.40s] GAME OVER -- total score: 37205")
    assert over["score"] == "37205"
    assert interpret("55 words typed over 812 scans")["typed"] == "55"


def test_unremarkable_lines_update_nothing():
    assert interpret("Loaded 1526 vocabulary entries and 47292 voice lines.") \
        == {}
    assert interpret("") == {}


def test_display_text_strips_the_log_timestamp():
    assert display_text("[  12.34s] --- playing ---") == "--- playing ---"
    assert display_text("no timestamp here") == "no timestamp here"

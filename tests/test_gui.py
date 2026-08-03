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


# -- play option parity ------------------------------------------------------
def test_default_panel_settings_produce_a_plain_run():
    from automaton_attack_bot.gui.app import play_argv

    assert play_argv() == ["--rounds", "1"]


def test_every_panel_option_maps_to_its_flag():
    from automaton_attack_bot.gui.app import play_argv

    argv = play_argv(dry_run=True, rounds=3, max_wpm=300, auto_start=False,
                     keep_running=True, safe_mode=True, auto_color=False,
                     locate_panel=False, phrases=False, debug=True,
                     monitor="2", ocr="tesseract")
    assert argv == ["--rounds", "3", "--dry-run", "--max-wpm", "300",
                    "--no-auto-start", "--keep-running", "--safe-mode",
                    "--no-auto-color", "--no-locate-panel", "--no-phrases",
                    "--debug", "--monitor", "2", "--ocr", "tesseract"]


def test_panel_argv_always_parses_with_the_real_cli():
    """The parity guarantee: whatever the panel builds, the terminal
    parser accepts -- the two entrances cannot drift apart."""
    from automaton_attack_bot.cli import build_parser
    from automaton_attack_bot.gui.app import play_argv

    parser = build_parser()
    for argv in (play_argv(),
                 play_argv(dry_run=True, max_wpm=450.0, monitor="3",
                           ocr="rapidocr", safe_mode=True, debug=True,
                           auto_start=False, keep_running=True,
                           auto_color=False, locate_panel=False,
                           phrases=False, rounds=9)):
        args = parser.parse_args(argv)
        assert args.command is None
    assert args.rounds == 9
    assert args.max_wpm == 450.0
    assert args.monitor == "3"


# -- status translation ------------------------------------------------------
def test_meaningful_lines_translate_to_friendly_status():
    from automaton_attack_bot.gui.feed import status_message

    assert status_message("[   2.76s] --- start-screen ---") \
        == "Start screen found"
    assert status_message("[   2.76s] clicking PLAY at (1275, 1184)") \
        == "Clicking PLAY"
    assert status_message("[  36.55s] !!! COMBO LOST x3.0 -> x2.0") \
        == "Combo lost!"
    assert status_message("[  67.43s] GAME OVER -- total score: 32140") \
        == "Round over -- score 32140"
    assert status_message("!!! capture stalled (frozen frames) -- x") \
        == "Capture stalled -- recovering"


def test_noise_lines_leave_the_status_alone():
    from automaton_attack_bot.gui.feed import status_message

    assert status_message("[  20.10s] combo x2.5, clock 0:41") is None
    assert status_message("55 words typed over 812 scans") is None
    assert status_message("random chatter") is None


def test_capture_line_becomes_the_info_chip():
    from automaton_attack_bot.gui.feed import log_path, session_info

    line = ("Live capture on monitor 3 (2560x1440), OCR=rapidocr x8. "
            "TYPING ENABLED.")
    assert session_info(line) == "monitor 3 · 2560x1440 · rapidocr"
    assert session_info("combo x2.0") is None
    assert log_path("Session log (full detail): logs\run-1.log") \
        == "logs\run-1.log"

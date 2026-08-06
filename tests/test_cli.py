"""CLI argument plumbing."""

from automaton_attack_bot.cli import build_parser


def test_bare_invocation_opens_the_control_panel():
    """No subcommand means GUI; --cli plays in the terminal instead."""
    args = build_parser().parse_args([])
    assert args.command is None
    assert not args.cli               # dispatched to the control panel
    assert args.monitor == "auto"
    assert not args.no_auto_start     # auto-start is on by default
    assert not args.dry_run           # typing is the default
    assert args.rounds == 1


def test_cli_flag_selects_the_terminal_play_loop():
    args = build_parser().parse_args(["--cli", "--rounds", "2"])
    assert args.cli
    assert args.rounds == 2


def test_play_flags_parse_at_top_level():
    args = build_parser().parse_args(["--dry-run", "--rounds", "3",
                                      "--monitor", "2"])
    assert args.dry_run
    assert args.rounds == 3
    assert args.monitor == "2"


def test_subcommands_still_dispatch():
    args = build_parser().parse_args(["replay", "clip.mp4"])
    assert args.command == "replay"
    assert args.clip == "clip.mp4"


def test_python_dash_m_runs_the_cli():
    from automaton_attack_bot import __main__
    from automaton_attack_bot.cli import main

    assert __main__.main is main

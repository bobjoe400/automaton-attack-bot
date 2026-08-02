"""CLI argument plumbing."""

from automaton_attack_bot.cli import build_parser


def test_bare_invocation_is_the_play_command():
    """Everything run-like lives directly on `automaton` -- there is no
    'run' subcommand."""
    args = build_parser().parse_args([])
    assert args.command is None       # dispatched to the play loop
    assert args.monitor == "auto"
    assert not args.no_auto_start     # auto-start is on by default
    assert not args.dry_run           # typing is the default
    assert args.rounds == 1


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

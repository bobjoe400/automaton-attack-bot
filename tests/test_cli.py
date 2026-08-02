"""CLI argument plumbing."""

from automaton_attack.cli import build_parser, resolve_argv


def test_bare_invocation_means_run():
    assert resolve_argv([]) == ["run"]


def test_explicit_arguments_pass_through():
    assert resolve_argv(["replay", "clip.mp4"]) == ["replay", "clip.mp4"]


def test_run_defaults_are_hands_off():
    args = build_parser().parse_args(["run"])
    assert args.monitor == "auto"
    assert not args.no_auto_start     # auto-start is on by default
    assert not args.live              # but typing still needs --live
    assert args.rounds == 1


def test_python_dash_m_automaton_alias():
    import automaton

    from automaton_attack.cli import main

    assert automaton.main is main

"""Alias package so ``python -m automaton`` works.

The real code lives in :mod:`automaton_attack`; this exists purely to make
the obvious module name run the CLI.
"""

from automaton_attack.cli import main

__all__ = ["main"]

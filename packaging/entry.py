"""PyInstaller entry point.

Two jobs. First: the package's own ``__main__.py`` uses a relative import,
which only works with package context -- something a frozen entry script
does not have, so this absolute-import shim is what the exe boots instead.

Second: the exe is built windowed (no console), because double-clicking it
should open the control panel and nothing else. When it IS launched from a
terminal, attach to that terminal's console so CLI usage still prints; when
there is no parent console, wire the streams to devnull so ``print`` works
everywhere without a console ever being created.
"""

import multiprocessing
import os
import sys


def _wire_console() -> None:
    import ctypes

    if ctypes.windll.kernel32.AttachConsole(-1):    # parent process console
        try:
            sys.stdout = open("CONOUT$", "w", buffering=1)
            sys.stderr = open("CONOUT$", "w", buffering=1)
            sys.stdin = open("CONIN$", "r")
            return
        except OSError:
            pass
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")


if sys.stdout is None or sys.stderr is None:        # windowed build
    _wire_console()

from automaton_attack_bot.cli import main           # noqa: E402

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())

"""PyInstaller entry point.

The package's own ``__main__.py`` uses a relative import, which only works
with package context -- something a frozen entry script does not have. This
absolute-import shim is what the exe boots instead.
"""

import multiprocessing
import sys

from automaton_attack_bot.cli import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())

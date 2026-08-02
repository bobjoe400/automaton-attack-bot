"""Control panel: run the bot and its tools from one window.

The GUI owns no bot logic. Every button dispatches the same command
functions the terminal uses, on a worker thread with stdout teed into the
panel (:mod:`tasks`), and the status display is driven by interpreting the
session's own log lines (:mod:`feed`).
"""

from .app import run_gui

__all__ = ["run_gui"]

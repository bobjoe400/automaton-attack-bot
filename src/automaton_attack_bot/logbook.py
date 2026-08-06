"""Two-level session logging: everything to file, signal to console."""

from pathlib import Path


class SessionLog:
    """Two-level session logging: everything to file, signal to console.

    ``say`` is the console channel (state changes, combo telemetry,
    scores); ``trace`` is the detail channel (every typed word, every OCR
    read) and goes to the file only -- unless no file is attached (replay,
    analyze) or --debug echoes it. The file gets BOTH levels, always, so
    a round can be analysed after the fact without any console spam.
    """

    def __init__(self) -> None:
        self._file = None
        self.echo_detail = False

    def attach(self, path: Path, echo_detail: bool) -> None:
        self._file = path.open("w", encoding="utf-8")
        self.echo_detail = echo_detail

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def _to_file(self, text: str, flush: bool) -> None:
        if self._file is not None:
            self._file.write(text + "\n")
            if flush:
                self._file.flush()

    def say(self, text: str = "") -> None:
        print(text)
        self._to_file(text, flush=True)

    def trace(self, text: str) -> None:
        # No flush: a syscall per detail line would tax the scan-consume
        # path for nothing. The OS buffers ~8KB blocks; say() events and
        # close() flush, so the signal always reaches disk immediately.
        self._to_file(text, flush=False)
        if self._file is None or self.echo_detail:
            print(text)


LOG = SessionLog()

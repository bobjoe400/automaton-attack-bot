"""OCR backends.

Two are available:

``rapidocr``
    Default. Pure pip install (ONNX Runtime + bundled PP-OCR models), so a
    clone-and-run setup needs no system packages. We already segment words
    ourselves, so only the recognition model runs -- detection and angle
    classification are skipped.

``tesseract``
    The backend the prototype was validated with. Needs the tesseract binary
    installed separately; used automatically only if it is already present.

Both are fed the same thing: the binary word mask, upscaled and inverted to
black-on-white.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

UPSCALE = 3
PAD = 6

# Tesseract: one text line, capitals only -- the game renders targets in caps.
TESSERACT_CONFIG = (
    "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
)

WINDOWS_TESSERACT_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
)


class OcrBackend(Protocol):
    name: str

    def read(self, image: np.ndarray) -> str:
        """Return the text in a single-line image, uppercase."""


def prepare(mask: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a blob out of the mask and render it as black text on white."""
    bx, by, bw, bh = box
    roi = mask[max(0, by - PAD):by + bh + PAD,
               max(0, bx - PAD):bx + bw + PAD]
    if roi.size == 0:
        return roi
    roi = cv2.resize(roi, None, fx=UPSCALE, fy=UPSCALE,
                     interpolation=cv2.INTER_CUBIC)
    return 255 - roi


def clean_text(text: str) -> str:
    """Printable-ASCII-only uppercase.

    The game's target words are ASCII, but PP-OCR's recognition model is
    multilingual and happily emits CJK punctuation for glyph fragments.
    Anything non-ASCII is noise for us -- and it crashes printing on
    Windows consoles (cp1252) if allowed through.
    """
    return "".join(c for c in text if c.isascii() and c.isprintable()).upper()


class RapidOcrBackend:
    """PP-OCR recognition via ONNX Runtime. No system dependencies."""

    name = "rapidocr"

    def __init__(self) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:  # pragma: no cover - env dependent
            raise OcrUnavailable(
                "rapidocr-onnxruntime is not installed. Run: uv sync"
            ) from exc
        # One compute thread per engine: parallelism comes from the POOL.
        # The default (-1) spawns a thread per core PER SESSION -- eight
        # engines once amounted to ~128 compute threads on a 16-thread CPU
        # that was also running the game, and the typing thread starved.
        self._engine = RapidOCR(intra_op_num_threads=1,
                                inter_op_num_threads=1)

    def read(self, image: np.ndarray) -> str:
        if image.size == 0:
            return ""
        # The recogniser expects 3 channels; our mask is single-channel.
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        result, _ = self._engine(image, use_det=False, use_cls=False,
                                 use_rec=True)
        if not result:
            return ""
        return clean_text(" ".join(str(line[0]) for line in result))


class TesseractBackend:
    """Tesseract via pytesseract. Requires the binary on PATH."""

    name = "tesseract"

    def __init__(self, binary: str | Path | None = None,
                 config: str = TESSERACT_CONFIG) -> None:
        try:
            import pytesseract
        except ImportError as exc:  # pragma: no cover - env dependent
            raise OcrUnavailable(
                "pytesseract is not installed. Run: uv sync --extra tesseract"
            ) from exc
        path = str(binary) if binary else find_tesseract()
        if not path:
            raise OcrUnavailable(
                "tesseract binary not found. Install it, or use the default "
                "--ocr rapidocr backend, which needs no system packages."
            )
        pytesseract.pytesseract.tesseract_cmd = path
        self._pytesseract = pytesseract
        self.binary = path
        self.config = config

    def read(self, image: np.ndarray) -> str:
        if image.size == 0:
            return ""
        text = self._pytesseract.image_to_string(image, config=self.config)
        return clean_text(" ".join(text.split()))


# HUD numbers (final score, timer). PP-OCR's recognition model garbles
# digit groups at HUD size ('17,770' came back as '17,7%'), while tesseract
# with a digit whitelist on the raw grayscale reads them exactly. Optional:
# None when tesseract isn't installed, and callers fall back to the
# general backend.
DIGIT_CONFIG = "--psm 7 -c tessedit_char_whitelist=0123456789,:"


def make_digit_reader() -> TesseractBackend | None:
    try:
        return TesseractBackend(config=DIGIT_CONFIG)
    except OcrUnavailable:
        return None


class OcrUnavailable(RuntimeError):
    """Raised when a requested backend cannot be constructed."""


def find_tesseract() -> str | None:
    found = shutil.which("tesseract")
    if found:
        return found
    for candidate in WINDOWS_TESSERACT_PATHS:
        if candidate and Path(candidate).exists():
            return candidate
    return None


class OcrPool:
    """N engines behind one ``read``, so blobs can be OCRed in parallel.

    onnxruntime releases the GIL during inference, so distinct engine
    instances on distinct threads genuinely run concurrently -- but one
    RapidOCR instance is not guaranteed re-entrant, hence a borrow queue
    rather than a shared engine.
    """

    def __init__(self, factory, size: int) -> None:
        import queue

        self._queue: queue.Queue = queue.Queue()
        engines = [factory() for _ in range(max(1, size))]
        for engine in engines:
            self._queue.put(engine)
        self.name = f"{engines[0].name} x{len(engines)}"
        self.size = len(engines)

    def read(self, image: np.ndarray) -> str:
        engine = self._queue.get()
        try:
            return engine.read(image)
        finally:
            self._queue.put(engine)


BACKENDS = ("rapidocr", "tesseract")
# Sized for TWO pipelined scans of a busy screen (4-8 blobs each).
POOL_SIZE = min(8, max(2, (os.cpu_count() or 4) - 2))


def get_backend(name: str = "auto", pool: bool = True) -> OcrBackend:
    """Build an OCR backend by name. ``auto`` prefers rapidocr.

    With ``pool`` (the default), rapidocr is wrapped in an OcrPool so the
    detector can read several word blobs at once; a busy screen used to
    pay ~50 ms per blob serially. Tesseract shells out per call and is
    already parallel-safe as a single instance.
    """
    if name == "auto":
        try:
            return get_backend("rapidocr", pool=pool)
        except OcrUnavailable:
            return TesseractBackend()
    if name == "rapidocr":
        if pool and POOL_SIZE > 1:
            return OcrPool(RapidOcrBackend, POOL_SIZE)
        return RapidOcrBackend()
    if name == "tesseract":
        return TesseractBackend()
    raise ValueError(f"unknown OCR backend {name!r}; choose from {BACKENDS}")

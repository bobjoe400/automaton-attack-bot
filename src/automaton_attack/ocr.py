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
        self._engine = RapidOCR()

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
        return " ".join(str(line[0]) for line in result).upper()


class TesseractBackend:
    """Tesseract via pytesseract. Requires the binary on PATH."""

    name = "tesseract"

    def __init__(self, binary: str | Path | None = None) -> None:
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

    def read(self, image: np.ndarray) -> str:
        if image.size == 0:
            return ""
        text = self._pytesseract.image_to_string(image,
                                                 config=TESSERACT_CONFIG)
        return " ".join(text.split()).upper()


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


BACKENDS = ("rapidocr", "tesseract")


def get_backend(name: str = "auto") -> OcrBackend:
    """Build an OCR backend by name. ``auto`` prefers rapidocr."""
    if name == "auto":
        try:
            return RapidOcrBackend()
        except OcrUnavailable:
            return TesseractBackend()
    if name == "rapidocr":
        return RapidOcrBackend()
    if name == "tesseract":
        return TesseractBackend()
    raise ValueError(f"unknown OCR backend {name!r}; choose from {BACKENDS}")

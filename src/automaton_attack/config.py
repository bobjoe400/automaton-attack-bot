"""Tunable settings for detection, matching and typing.

Every constant the prototype hard-coded lives here, grouped by pipeline stage
and serialisable to JSON so a machine-specific calibration can be saved and
reloaded (``automaton calibrate --save``).

Geometry is expressed for a 1920x1080 screen; :meth:`Settings.for_resolution`
rescales it for other resolutions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

REFERENCE_RESOLUTION = (1920, 1080)

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class Geometry:
    """Where the minigame lives on screen.

    ``panel`` is in screen coordinates; ``hud_boxes`` are panel-local regions
    that share the target words' yellow-green and must be masked out. Do not
    replace them with a full-width top band: words legitimately fly through
    y~114.
    """

    panel: Box = (481, 57, 1444, 1022)
    hud_boxes: tuple[Box, ...] = (
        (40, 0, 220, 110),    # score + multiplier
        (390, 0, 560, 110),   # timer
        (720, 0, 940, 110),   # high score
    )

    @property
    def panel_size(self) -> tuple[int, int]:
        x0, y0, x1, y1 = self.panel
        return x1 - x0, y1 - y0


@dataclass(frozen=True)
class Color:
    """HSV range (OpenCV scale) isolating the yellow target words.

    Measured off recorded footage at H 39-44, S 68-119, V 165-196; the range
    below is padded around that. The white input tracker (S~4, V~230) falls
    outside it and is ignored, which is what we want -- it is redundant.

    With ``auto`` on (the default), the range is treated as relative to the
    reference display conditions: each scan samples the HUD text -- rendered
    in the same yellow-green -- and translates the range by however far the
    display has shifted it (HDR, night light, driver LUTs). See autocolor.py.
    """

    hsv_lo: tuple[int, int, int] = (35, 50, 140)
    hsv_hi: tuple[int, int, int] = (50, 150, 230)
    auto: bool = True


@dataclass(frozen=True)
class Blobs:
    """Letter-merging and blob-filtering parameters.

    The dilate kernel is deliberately wide and short so letters merge into one
    word but stacked words stay separate.
    """

    dilate_kernel: tuple[int, int] = (5, 25)
    min_area: int = 800
    min_width: int = 50
    min_height: int = 12
    max_height: int = 45


@dataclass(frozen=True)
class Matching:
    """Fuzzy-match acceptance thresholds.

    The multiplier resets when a word ESCAPES untyped, not when wrong
    letters are sent -- stray keystrokes are free. So corpus matches are
    typed on first sight at any accepted score: a wrong guess costs only
    keyboard time, hesitating can cost the word. Only fallback reads (raw
    OCR with no corpus anchor) wait for an exact repeat across two scans;
    OCR errors vary frame to frame (METEORHAKIMER vs METEORHARIMER), so
    repetition is the one signal that a fallback read is worth the keys.
    Tesseract's own confidence is NOT usable here -- it reported 0 on a
    correctly-read long phrase -- so nothing gates on it.
    """

    vocab_cutoff: float = 0.62
    min_ocr_length: int = 3
    # When a longer vocab entry's key merely EXTENDS the best match's key
    # (WEAVE -> WEAVER) and scores within this margin, type the longer one:
    # its keystrokes complete the shorter word on the way through, so it
    # covers both readings.
    extension_epsilon: float = 0.10
    # The phrase corpus is ~40k lines, so a wrong steal types an entire wrong
    # sentence -- pure wasted time. Hence a much higher bar than the vocab.
    phrase_cutoff: float = 0.80
    phrase_min_length: int = 10
    # Clean OCR that matched nothing is typed verbatim above this length.
    fallback_min_length: int = 8


@dataclass(frozen=True)
class Behaviour:
    """Loop pacing, dedup and keystroke delivery."""

    scan_interval: float = 0.05      # live: seconds between screen grabs
    replay_stride: int = 15          # replay: scan every Nth frame (~4/s @60fps)
    dedup_radius: int = 120          # px; same word near same spot = same word
    dedup_ttl: float = 3.0           # seconds
    key_delay: tuple[float, float] = (0.010, 0.030)  # per-key jitter, seconds
    # Valve patched a pause-typing exploit in this minigame, so they do watch
    # it. Cap effective speed to something a fast human could produce.
    max_wpm: float = 0.0             # 0 disables the cap


@dataclass(frozen=True)
class Settings:
    geometry: Geometry = field(default_factory=Geometry)
    color: Color = field(default_factory=Color)
    blobs: Blobs = field(default_factory=Blobs)
    matching: Matching = field(default_factory=Matching)
    behaviour: Behaviour = field(default_factory=Behaviour)
    # Skip the verbatim-typing fallback tier. Unmatched reads are usually
    # OCR-mangled, and typing them burns keyboard time that confident words
    # need -- but skipping guarantees missing any word absent from the
    # corpus, and a missed word is what resets the multiplier. Off by
    # default for that reason.
    safe_mode: bool = False

    # -- resolution ------------------------------------------------------
    def for_resolution(self, width: int, height: int) -> Settings:
        """Rescale geometry and blob filters from the 1920x1080 reference."""
        ref_w, ref_h = REFERENCE_RESOLUTION
        if (width, height) == (ref_w, ref_h):
            return self
        sx, sy = width / ref_w, height / ref_h

        def scale_box(b: Box) -> Box:
            x0, y0, x1, y1 = b
            return (round(x0 * sx), round(y0 * sy),
                    round(x1 * sx), round(y1 * sy))

        geometry = Geometry(
            panel=scale_box(self.geometry.panel),
            hud_boxes=tuple(scale_box(b) for b in self.geometry.hud_boxes),
        )
        blobs = Blobs(
            dilate_kernel=(max(1, round(self.blobs.dilate_kernel[0] * sy)),
                           max(1, round(self.blobs.dilate_kernel[1] * sx))),
            min_area=max(1, round(self.blobs.min_area * sx * sy)),
            min_width=max(1, round(self.blobs.min_width * sx)),
            min_height=max(1, round(self.blobs.min_height * sy)),
            max_height=max(2, round(self.blobs.max_height * sy)),
        )
        return replace(self, geometry=geometry, blobs=blobs)

    def with_panel(self, panel: Box) -> Settings:
        """Re-anchor geometry on an auto-located panel.

        HUD boxes and blob filters scale with the panel itself rather than
        the full frame: the panel's render size is what sets glyph size.
        """
        ref = Geometry()
        ref_w, ref_h = ref.panel_size
        x0, y0, x1, y1 = panel
        sx, sy = (x1 - x0) / ref_w, (y1 - y0) / ref_h
        geometry = Geometry(
            panel=panel,
            hud_boxes=tuple(
                (round(bx0 * sx), round(by0 * sy),
                 round(bx1 * sx), round(by1 * sy))
                for bx0, by0, bx1, by1 in ref.hud_boxes),
        )
        ref_blobs = Blobs()
        blobs = Blobs(
            dilate_kernel=(max(1, round(ref_blobs.dilate_kernel[0] * sy)),
                           max(1, round(ref_blobs.dilate_kernel[1] * sx))),
            min_area=max(1, round(ref_blobs.min_area * sx * sy)),
            min_width=max(1, round(ref_blobs.min_width * sx)),
            min_height=max(1, round(ref_blobs.min_height * sy)),
            max_height=max(2, round(ref_blobs.max_height * sy)),
        )
        return replace(self, geometry=geometry, blobs=blobs)

    # -- persistence -----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n",
                        encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Settings:
        def build(kind, key):
            payload = data.get(key)
            if not payload:
                return kind()
            fields = {f for f in kind.__dataclass_fields__}
            clean = {k: v for k, v in payload.items() if k in fields}
            # JSON has no tuples; dataclass fields that are tuples come back
            # as lists and would break equality/typing downstream.
            for k, v in clean.items():
                if isinstance(v, list):
                    clean[k] = tuple(tuple(i) if isinstance(i, list) else i
                                     for i in v)
            return kind(**clean)

        return cls(
            geometry=build(Geometry, "geometry"),
            color=build(Color, "color"),
            blobs=build(Blobs, "blobs"),
            matching=build(Matching, "matching"),
            behaviour=build(Behaviour, "behaviour"),
            safe_mode=bool(data.get("safe_mode", False)),
        )

    @classmethod
    def load(cls, path: str | Path) -> Settings:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


DEFAULT_CONFIG_NAME = "automaton.json"


def find_config(explicit: str | Path | None = None) -> Path | None:
    """Resolve a config file: explicit path, else ./automaton.json."""
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"config file not found: {path}")
        return path
    local = Path.cwd() / DEFAULT_CONFIG_NAME
    return local if local.exists() else None


def load_settings(explicit: str | Path | None = None) -> tuple[Settings, Path | None]:
    path = find_config(explicit)
    return (Settings.load(path) if path else Settings()), path

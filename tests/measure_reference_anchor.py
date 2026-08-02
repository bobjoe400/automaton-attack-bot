"""Recompute autocolor.REFERENCE from the reference clips.

Run when the HSV range in config.py is retuned or the reference footage
changes, and paste the printed anchor into autocolor.py:

    uv run python tests/measure_reference_anchor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from automaton_attack import autocolor
from automaton_attack.config import Settings

CLIPS = (Path("clips") / "clip1.mp4", Path("clips") / "clip2.mp4")
FRAMES_PER_CLIP = 8


def main() -> int:
    settings = Settings()
    x0, y0, x1, y1 = settings.geometry.panel
    anchors = []
    for clip in CLIPS:
        if not clip.exists():
            print(f"error: {clip} missing", file=sys.stderr)
            return 1
        capture = cv2.VideoCapture(str(clip))
        fps = capture.get(cv2.CAP_PROP_FPS) or 60
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for seconds in np.linspace(2, total / fps - 2, FRAMES_PER_CLIP):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(seconds * fps))
            ok, frame = capture.read()
            if not ok:
                continue
            hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
            anchor = autocolor.measure(hsv, settings.geometry.hud_boxes)
            if anchor:
                anchors.append(anchor)
        capture.release()

    if not anchors:
        print("error: no usable frames", file=sys.stderr)
        return 1

    def med(field: str) -> float:
        return round(float(np.median([getattr(a, field) for a in anchors])), 1)

    print(f"{len(anchors)} frames sampled. Paste into autocolor.py:\n")
    print("REFERENCE = ColorAnchor(")
    print(f"    hue_median={med('hue_median')},")
    print(f"    sat_median={med('sat_median')},")
    print(f"    val_p5={med('val_p5')},")
    print(f"    val_p95={med('val_p95')},")
    print(")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

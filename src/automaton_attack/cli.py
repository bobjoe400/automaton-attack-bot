"""Command-line interface.

    automaton doctor              check the environment
    automaton replay CLIP         run detection over a recording (never types)
    automaton run                 live capture; add --live to send keystrokes
    automaton match TEXT          ask the lexicon what a read resolves to
    automaton calibrate           dump mask/blob diagnostics from a frame
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import DEFAULT_CONFIG_NAME, Settings, load_settings
from .lexicon import Lexicon


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", metavar="PATH",
                        help=f"settings JSON (default: ./{DEFAULT_CONFIG_NAME}"
                             " if present)")
    parser.add_argument("--ocr", default="auto",
                        choices=("auto", "rapidocr", "tesseract"),
                        help="OCR backend (default: auto, prefers rapidocr)")
    parser.add_argument("--no-phrases", action="store_true",
                        help="skip the ~39k voice-line corpus")
    parser.add_argument("--safe-mode", action="store_true",
                        help="never type unmatched OCR verbatim; a wrong word "
                             "resets the score multiplier")
    parser.add_argument("--debug", action="store_true",
                        help="print every blob, matched or not")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="automaton",
        description="Bot for Dota 2's Automaton Attack typing minigame.",
    )
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    subs = parser.add_subparsers(dest="command", required=True)

    doctor = subs.add_parser("doctor", help="check dependencies and data files")
    doctor.add_argument("--config", metavar="PATH", help="settings JSON")

    update = subs.add_parser(
        "update-data",
        help="rebuild the word corpora from odota/dotaconstants and "
             "mdiller/dotabase (run after a Dota patch)")
    update.add_argument("--vocab-only", action="store_true",
                        help="skip the ~70 MB voice-line download")

    replay = subs.add_parser(
        "replay", help="run detection over a recorded clip (never types)")
    replay.add_argument("clip", help="path to a video file")
    replay.add_argument("--stride", type=int,
                        help="scan every Nth frame (default: 15)")
    _add_common(replay)

    run = subs.add_parser("run", help="live screen capture")
    run.add_argument("--live", action="store_true",
                     help="actually send keystrokes (default: dry run)")
    run.add_argument("--monitor", type=int, default=1,
                     help="mss monitor index (default: 1)")
    run.add_argument("--max-wpm", type=float,
                     help="cap typing speed, in words per minute")
    _add_common(run)

    match = subs.add_parser("match", help="resolve text against the lexicon")
    match.add_argument("text", nargs="+", help="OCR read(s) to look up")
    match.add_argument("--no-phrases", action="store_true")
    match.add_argument("--config", metavar="PATH")

    calibrate = subs.add_parser(
        "calibrate", help="inspect detection on one frame and tune geometry")
    src = calibrate.add_mutually_exclusive_group(required=True)
    src.add_argument("--clip", help="take the frame from a recording")
    src.add_argument("--screen", action="store_true",
                     help="take the frame from the live screen")
    calibrate.add_argument("--at", type=float, default=5.0,
                           help="clip timestamp in seconds (default: 5)")
    calibrate.add_argument("--out", default="calibration",
                           help="directory for diagnostic images")
    calibrate.add_argument("--save-config", metavar="PATH", nargs="?",
                           const=DEFAULT_CONFIG_NAME,
                           help="write the resolved settings to JSON")
    _add_common(calibrate)
    return parser


def _settings_from_args(args) -> Settings:
    settings, path = load_settings(getattr(args, "config", None))
    if path:
        print(f"Using config {path}")
    if getattr(args, "safe_mode", False):
        settings = Settings.from_dict({**settings.to_dict(), "safe_mode": True})
    if getattr(args, "stride", None):
        data = settings.to_dict()
        data["behaviour"]["replay_stride"] = args.stride
        settings = Settings.from_dict(data)
    if getattr(args, "max_wpm", None):
        data = settings.to_dict()
        data["behaviour"]["max_wpm"] = args.max_wpm
        settings = Settings.from_dict(data)
    return settings


def _load_lexicon(args, settings: Settings) -> Lexicon:
    lexicon = Lexicon.load(
        matching=settings.matching,
        with_phrases=not getattr(args, "no_phrases", False),
    )
    print(f"Loaded {len(lexicon)} vocabulary entries"
          + (f" and {len(lexicon.phrase_names)} voice lines."
             if lexicon.phrase_names else " (no phrase corpus)."))
    return lexicon


def _describe(word) -> str:
    d = word.detection
    return (f"[{word.timestamp:7.2f}s] {word.name:<40} "
            f"{word.source:<8} score={d.score:.2f} ocr={d.raw!r}")


# -- commands ------------------------------------------------------------
def cmd_doctor(args) -> int:
    from . import corpus
    from .ocr import BACKENDS, OcrUnavailable, find_tesseract, get_backend

    print(f"automaton {__version__}  (python {sys.version.split()[0]})")
    ok = True

    for module in ("cv2", "numpy", "rapidfuzz"):
        try:
            __import__(module)
            print(f"  [ok]   {module}")
        except ImportError:
            print(f"  [FAIL] {module} missing -- run: uv sync")
            ok = False

    for name in BACKENDS:
        try:
            get_backend(name)
            extra = (f" ({find_tesseract()})" if name == "tesseract" else "")
            print(f"  [ok]   OCR backend: {name}{extra}")
        except (OcrUnavailable, Exception) as exc:  # noqa: BLE001
            detail = str(exc).split("\n")[0]
            print(f"  [--]   OCR backend: {name} unavailable -- {detail}")

    for module, purpose in (("mss", "screen capture"),
                            ("pydirectinput", "keystrokes")):
        try:
            __import__(module)
            print(f"  [ok]   {module} ({purpose})")
        except ImportError:
            print(f"  [--]   {module} missing -- live mode needs "
                  f"'uv sync --extra live'")

    for label, path in (("vocabulary", corpus.vocab_path()),
                        ("voice lines", corpus.phrases_path())):
        if path.exists():
            count = sum(1 for line in path.read_text(encoding="utf-8").splitlines()
                        if line.strip() and not line.startswith("#"))
            print(f"  [ok]   {label}: {count} entries ({path})")
        else:
            print(f"  [--]   {label}: not built yet -- fetched automatically "
                  f"on first run, or run: automaton update-data")

    settings, path = load_settings(getattr(args, "config", None))
    print(f"  [ok]   settings: {path or 'built-in defaults'}; "
          f"panel {settings.geometry.panel}")
    return 0 if ok else 1


def cmd_update_data(args) -> int:
    from . import corpus

    print(corpus.ATTRIBUTION + "\n")
    corpus.write_vocab()
    if not args.vocab_only:
        corpus.write_phrases()
    print("\nDone. The bot uses the new corpora on its next run.")
    return 0


def cmd_replay(args) -> int:
    from .capture import VideoSource
    from .detect import Detector
    from .engine import Engine
    from .ocr import get_backend

    settings = _settings_from_args(args)
    lexicon = _load_lexicon(args, settings)
    backend = get_backend(args.ocr)

    source = VideoSource(args.clip, settings.behaviour.replay_stride)
    settings = settings.for_resolution(source.width, source.height)
    print(f"{Path(args.clip).name}: {source.width}x{source.height} "
          f"@{source.fps:.0f}fps, {source.frame_count} frames; "
          f"OCR={backend.name}, scanning every "
          f"{settings.behaviour.replay_stride} frames\n")

    detector = Detector(lexicon, backend, settings)
    engine = Engine(detector, settings=settings)

    if args.debug:
        _replay_debug(source, detector, engine)
    else:
        for word in engine.run(source.frames()):
            print(_describe(word))

    stats = engine.stats
    print(f"\n{stats.typed} words typed over {stats.frames} scans "
          f"({dict(stats.by_source)}); "
          f"{stats.suppressed_duplicate} duplicates suppressed, "
          f"{stats.awaiting_confirmation} held for confirmation.")
    return 0


def _replay_debug(source, detector, engine) -> None:
    for timestamp, frame in source.frames():
        blobs = detector.blobs(detector.word_mask(frame))
        if blobs:
            print(f"[{timestamp:7.2f}s] {len(blobs)} blob(s)")
        for detection in detector.detect(frame, include_unmatched=True):
            bx, by, bw, bh = detection.box
            target = detection.match.name if detection.match else "-"
            print(f"    ({bx:4},{by:4}) {bw:3}x{bh:<3} ocr={detection.raw!r}"
                  f" -> {target} ({detection.score:.2f})")
        for word in engine.process(timestamp, frame):
            print(f"    >>> TYPE {word.keystrokes!r}")


def cmd_run(args) -> int:
    from .capture import ScreenSource
    from .detect import Detector
    from .engine import Engine
    from .keyboard import make_typist
    from .ocr import get_backend

    settings = _settings_from_args(args)
    lexicon = _load_lexicon(args, settings)
    backend = get_backend(args.ocr)

    source = ScreenSource(args.monitor, settings.behaviour.scan_interval)
    settings = settings.for_resolution(source.width, source.height)
    detector = Detector(lexicon, backend, settings)
    engine = Engine(detector, make_typist(args.live, settings.behaviour),
                    settings)

    mode = "TYPING ENABLED" if args.live else "DRY RUN (pass --live to type)"
    print(f"Live capture on monitor {args.monitor} "
          f"({source.width}x{source.height}), OCR={backend.name}. {mode}. "
          f"Ctrl+C to stop.\n")
    try:
        for word in engine.run(source.frames()):
            print(_describe(word))
    except KeyboardInterrupt:
        print("\nStopped.")
    stats = engine.stats
    print(f"{stats.typed} words typed over {stats.frames} scans "
          f"({dict(stats.by_source)}).")
    return 0


def cmd_match(args) -> int:
    settings = _settings_from_args(args)
    lexicon = _load_lexicon(args, settings)
    for text in args.text:
        match = lexicon.match(text)
        if match:
            print(f"{text!r} -> {match.name} "
                  f"[{match.source} {match.score:.2f}] "
                  f"types {match.keystrokes!r}")
        else:
            print(f"{text!r} -> no match")
    return 0


def cmd_calibrate(args) -> int:
    import cv2

    from .detect import Detector
    from .ocr import get_backend

    settings = _settings_from_args(args)
    lexicon = _load_lexicon(args, settings)
    backend = get_backend(args.ocr)

    if args.clip:
        from .capture import VideoSource

        source = VideoSource(args.clip)
        frame = source.frame_at(args.at)
        source.close()
        if frame is None:
            print(f"No frame at {args.at}s in {args.clip}", file=sys.stderr)
            return 1
    else:
        from .capture import ScreenSource

        frame = ScreenSource(1).grab()

    height, width = frame.shape[:2]
    settings = settings.for_resolution(width, height)
    detector = Detector(lexicon, backend, settings)

    mask = detector.word_mask(frame)
    blobs = detector.blobs(mask)
    print(f"\nFrame {width}x{height}; panel {settings.geometry.panel}")
    print(f"Mask: {int(mask.sum() // 255)} lit pixels, {len(blobs)} blob(s) "
          f"passing filters.")

    if not blobs:
        print("\nNo blobs. If the game was on screen, the HSV range is the "
              "usual culprit -- recorded clips are compressed, a raw "
              "framebuffer is not. Measure the letter pixels in mask.png and "
              "widen color.hsv_lo/hsv_hi in the config.")

    for detection in detector.detect(frame, include_unmatched=True):
        bx, by, bw, bh = detection.box
        target = detection.match.name if detection.match else "-"
        print(f"  ({bx:4},{by:4}) {bw:3}x{bh:<3} ocr={detection.raw!r} "
              f"-> {target} ({detection.score:.2f})")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    overlay = detector.crop_panel(frame).copy()
    for bx, by, bw, bh in blobs:
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
    for hx0, hy0, hx1, hy1 in settings.geometry.hud_boxes:
        cv2.rectangle(overlay, (hx0, hy0), (hx1, hy1), (255, 0, 0), 2)
    cv2.imwrite(str(out / "frame.png"), frame)
    cv2.imwrite(str(out / "mask.png"), mask)
    cv2.imwrite(str(out / "blobs.png"), overlay)
    print(f"\nWrote frame.png, mask.png, blobs.png to {out}/ "
          f"(red = detected words, blue = masked HUD).")

    if args.save_config:
        path = settings.save(args.save_config)
        print(f"Wrote settings to {path}")
    return 0


COMMANDS = {
    "doctor": cmd_doctor,
    "update-data": cmd_update_data,
    "replay": cmd_replay,
    "run": cmd_run,
    "match": cmd_match,
    "calibrate": cmd_calibrate,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except KeyboardInterrupt:
        return 130
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

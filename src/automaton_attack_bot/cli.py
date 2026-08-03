"""Command-line interface.

    automaton                     open the control panel (--cli plays in the
                                  terminal; --dry-run rehearses)
    automaton doctor              check the environment
    automaton replay CLIP         run detection over a recording (never types)
    automaton analyze CLIP        report misses and per-word screen time
    automaton match TEXT          ask the lexicon what a read resolves to
    automaton calibrate           dump mask/blob diagnostics from a frame
    automaton update-data         rebuild the word corpora from upstream
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .core.config import DEFAULT_CONFIG_NAME, Settings, load_settings
from .core.drive import drive, summarise
from .core.lexicon import Lexicon
from .logbook import LOG


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
                        help="never type unmatched OCR verbatim; saves "
                             "keyboard time for confident words, but any "
                             "word missing from the corpus will escape")
    parser.add_argument("--no-auto-color", action="store_true",
                        help="use the configured HSV range as-is instead of "
                             "calibrating it against the HUD text each scan")
    parser.add_argument("--no-locate-panel", action="store_true",
                        help="trust the configured panel geometry instead of "
                             "finding the minigame frame on screen")
    parser.add_argument("--debug", action="store_true",
                        help="print every blob, matched or not")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="automaton",
        description="Bot for Dota 2's Automaton Attack typing minigame. "
                    "With no subcommand it opens the control panel; with "
                    "--cli it plays in the terminal: finds the Dota window "
                    "and the minigame panel, clicks PLAY, types the round, "
                    "reports the score and exits.",
    )
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    parser.add_argument("--cli", action="store_true",
                        help="play in the terminal instead of opening the "
                             "control panel")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be typed/clicked instead of "
                             "doing it")
    parser.add_argument("--monitor", default="auto",
                        help="monitor to capture: a number, or 'auto' to "
                             "find the Dota 2 window (default: auto)")
    parser.add_argument("--max-wpm", type=float,
                        help="cap typing speed, in words per minute")
    parser.add_argument("--no-auto-start", action="store_true",
                        help="don't click PLAY / PLAY AGAIN when a start or "
                             "game-over screen is showing")
    parser.add_argument("--rounds", type=int, default=1,
                        help="rounds to play before exiting (default: 1)")
    parser.add_argument("--keep-running", action="store_true",
                        help="don't exit when the game-over screen appears")
    _add_common(parser)
    subs = parser.add_subparsers(dest="command", required=False)

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

    analyze = subs.add_parser(
        "analyze",
        help="replay a recording and report reads that look like misses")
    analyze.add_argument("clip", help="path to a video file")
    analyze.add_argument("--stride", type=int,
                         help="scan every Nth frame (default: 15)")
    _add_common(analyze)

    gui = subs.add_parser(
        "gui", help="control panel: start/stop the bot, watch its status")
    gui.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)

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
    data = settings.to_dict()
    changed = False
    if getattr(args, "safe_mode", False):
        data["safe_mode"] = True
        changed = True
    if getattr(args, "no_auto_color", False):
        data["color"]["auto"] = False
        changed = True
    if getattr(args, "stride", None):
        data["behaviour"]["replay_stride"] = args.stride
        changed = True
    if getattr(args, "max_wpm", None):
        data["behaviour"]["max_wpm"] = args.max_wpm
        changed = True
    return Settings.from_dict(data) if changed else settings


def _attach_session_log(prefix: str, echo_detail: bool) -> Path:
    """Open logs/<prefix>-<stamp>.log and route the session log to it."""
    import datetime

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"{prefix}-{stamp}.log"
    LOG.attach(log_path, echo_detail=echo_detail)
    LOG.say(f"Session log (full detail): {log_path}")
    return log_path


def _load_lexicon(args, settings: Settings) -> Lexicon:
    lexicon = Lexicon.load(
        matching=settings.matching,
        with_phrases=not getattr(args, "no_phrases", False),
    )
    print(f"Loaded {len(lexicon)} vocabulary entries"
          + (f" and {len(lexicon.phrase_names)} voice lines."
             if lexicon.phrase_names else " (no phrase corpus)."))
    return lexicon


# -- commands ------------------------------------------------------------
def cmd_doctor(args) -> int:
    from . import corpus
    from .core.ocr import BACKENDS, OcrUnavailable, find_tesseract, get_backend

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
                  f"'uv sync'")

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
    from .core.capture import VideoSource
    from .core.keyboard import DryRunTypist
    from .core.ocr import get_backend

    _attach_session_log("replay", echo_detail=args.debug)

    settings = _settings_from_args(args)
    lexicon = _load_lexicon(args, settings)
    backend = get_backend(args.ocr)

    source = VideoSource(args.clip, settings.behaviour.replay_stride)
    settings = settings.for_resolution(source.width, source.height)
    # On tape nothing disappears when typed, so the live short-TTL
    # retype-if-still-visible rule would spam duplicates; floor it.
    if settings.behaviour.dedup_ttl < 3.0:
        data = settings.to_dict()
        data["behaviour"]["dedup_ttl"] = 3.0
        settings = Settings.from_dict(data)
    print(f"{Path(args.clip).name}: {source.width}x{source.height} "
          f"@{source.fps:.0f}fps, {source.frame_count} frames; "
          f"OCR={backend.name}, scanning every "
          f"{settings.behaviour.replay_stride} frames\n")

    try:
        engine, tracker = drive(
            source.frames(), lexicon, backend, settings,
            DryRunTypist(settings.behaviour),
            locate=not args.no_locate_panel,
            stop_on_game_over=False,

        )
        summarise(engine, tracker)
    finally:
        LOG.close()
    return 0


def _until_stopped(frames, stop_event):
    for item in frames:
        if stop_event.is_set():
            return
        yield item


def cmd_run(args, stop_event=None) -> int:
    """Play. ``stop_event`` is the GUI's stop button: the screen-capture
    generator is endless, so a cooperative cut is the clean way out of
    the drive loop from another thread."""
    from .core.capture import ScreenSource, find_game_monitor
    from .core.keyboard import make_typist
    from .core.ocr import get_backend

    _attach_session_log("run", echo_detail=args.debug)

    settings = _settings_from_args(args)
    lexicon = _load_lexicon(args, settings)
    backend = get_backend(args.ocr)

    monitor = args.monitor
    if monitor == "auto":
        found = find_game_monitor()
        if found:
            print(f"Dota 2 window found on monitor {found}.")
        else:
            print("No Dota 2 window found; capturing the primary monitor "
                  "(--monitor N to override).")
        monitor = found or 1
    source = ScreenSource(int(monitor), settings.behaviour.scan_interval,
                          on_stall=LOG.say)
    settings = settings.for_resolution(source.width, source.height)
    live = not args.dry_run
    if live:
        try:
            typist = make_typist(True, settings.behaviour,
                                 origin=source.origin)
        except RuntimeError as exc:
            # A clone missing the input libs still gets a useful run.
            print(f"note: {exc}")
            print("Falling back to a dry run.")
            live = False
    if not live:
        typist = make_typist(False, settings.behaviour, origin=source.origin)

    mode = ("TYPING ENABLED (--dry-run to rehearse)" if live
            else "DRY RUN (no keys or clicks will be sent)")
    print(f"Live capture on monitor {monitor} "
          f"({source.width}x{source.height}), OCR={backend.name}. {mode}. "
          f"Ctrl+C to stop.\n")
    frames = source.frames()
    if stop_event is not None:
        frames = _until_stopped(frames, stop_event)
    try:
        engine, tracker = drive(
            frames, lexicon, backend, settings, typist,
            locate=not args.no_locate_panel,
            auto_start=not args.no_auto_start,
            rounds=args.rounds,
            stop_on_game_over=not args.keep_running,

            threaded=True,
        )
        summarise(engine, tracker)
        return 0
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0
    finally:
        LOG.close()


def cmd_analyze(args) -> int:
    from .analyze import PLATFORM_BAND, ReadLog
    from .core.capture import VideoSource
    from .core.detect import Detector
    from .core.ocr import get_backend
    from .core.session import GameState, SessionTracker, locate_panel

    settings = _settings_from_args(args)
    lexicon = _load_lexicon(args, settings)
    backend = get_backend(args.ocr)

    source = VideoSource(args.clip, settings.behaviour.replay_stride)
    settings = settings.for_resolution(source.width, source.height)
    log = ReadLog()
    detector = None
    tracker = None

    for timestamp, frame in source.frames():
        if detector is None:
            panel = None if args.no_locate_panel else locate_panel(frame)
            if panel:
                drift = max(abs(a - b) for a, b in
                            zip(panel, settings.geometry.panel))
                # Rebuilds are cheap, and even a few pixels of offset can
                # clip the tight digit crops (score, timer).
                if drift > 3:
                    settings = settings.with_panel(panel)
            elif not args.no_locate_panel:
                continue    # wait for a frame the panel can be found on
            detector = Detector(lexicon, backend, settings)
            tracker = SessionTracker(backend, settings)
        if tracker.classify(timestamp, frame) is not GameState.PLAYING:
            continue
        clock = tracker.read_timer(frame)
        panel_height = settings.geometry.panel_size[1]
        for detection in detector.detect(frame, include_unmatched=True,
                                         timestamp=timestamp):
            log.add(timestamp, detection, clock=clock)
            bottom = detection.box[1] + detection.box[3]
            if (detection.name
                    and bottom / panel_height < PLATFORM_BAND):
                log.add_lifetime(timestamp, detection.name,
                                 bottom / panel_height, clock=clock)
            if bottom / panel_height >= PLATFORM_BAND and (detection.name
                                                  or detection.raw.strip()):
                log.add_strike(timestamp, clock,
                               detection.name or detection.raw.strip())

    print()
    print(log.report())
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

    from .core.detect import Detector
    from .core.ocr import get_backend

    settings = _settings_from_args(args)
    lexicon = _load_lexicon(args, settings)
    backend = get_backend(args.ocr)

    if args.clip:
        from .core.capture import VideoSource

        source = VideoSource(args.clip)
        frame = source.frame_at(args.at)
        source.close()
        if frame is None:
            print(f"No frame at {args.at}s in {args.clip}", file=sys.stderr)
            return 1
    else:
        from .core.capture import ScreenSource

        frame = ScreenSource(1).grab()

    height, width = frame.shape[:2]
    settings = settings.for_resolution(width, height)
    detector = Detector(lexicon, backend, settings)

    import cv2 as _cv2

    from .core import autocolor
    from .core.session import PER_BOX_MIN_PIXELS

    mask = detector.word_mask(frame)
    blobs = detector.blobs(mask)
    print(f"\nFrame {width}x{height}; panel {settings.geometry.panel}")
    hsv_panel = _cv2.cvtColor(detector.crop_panel(frame), _cv2.COLOR_BGR2HSV)
    counts = [autocolor.box_candidates(hsv_panel, box)
              for box in settings.geometry.hud_boxes]
    print(f"HUD boxes (score/timer/high-score) candidate pixels: {counts} "
          f"(each needs >= {PER_BOX_MIN_PIXELS} to count as 'playing')")
    configured = (settings.color.hsv_lo, settings.color.hsv_hi)
    active = detector.active_range
    if detector.last_anchor:
        anchor = detector.last_anchor
        print(f"HUD colour sample: hue {anchor.hue_median:.0f}, "
              f"sat {anchor.sat_median:.0f}, "
              f"val {anchor.val_p5:.0f}-{anchor.val_p95:.0f}")
        if active != configured:
            print(f"Calibrated HSV range {active[0]}-{active[1]} "
                  f"(configured: {configured[0]}-{configured[1]})")
        else:
            print("Display matches reference conditions; configured HSV "
                  "range used unchanged.")
    elif settings.color.auto:
        print("No usable HUD colour sample in this frame (minigame not on "
              "screen?); configured HSV range used.")
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


def cmd_gui(args) -> int:
    from .gui import run_gui

    return run_gui(args)


COMMANDS = {
    "analyze": cmd_analyze,
    "doctor": cmd_doctor,
    "update-data": cmd_update_data,
    "replay": cmd_replay,
    "run": cmd_run,
    "gui": cmd_gui,
    "match": cmd_match,
    "calibrate": cmd_calibrate,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        # No subcommand opens the control panel; --cli plays in the
        # terminal the way bare `automaton` used to.
        default = "run" if getattr(args, "cli", False) else "gui"
        return COMMANDS[args.command or default](args)
    except KeyboardInterrupt:
        return 130
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

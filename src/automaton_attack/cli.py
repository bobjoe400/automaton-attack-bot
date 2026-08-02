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
    run.add_argument("--monitor", default="auto",
                     help="monitor to capture: a number, or 'auto' to find "
                          "the Dota 2 window (default: auto)")
    run.add_argument("--max-wpm", type=float,
                     help="cap typing speed, in words per minute")
    run.add_argument("--no-auto-start", action="store_true",
                     help="don't click PLAY / PLAY AGAIN when a start or "
                          "game-over screen is showing")
    run.add_argument("--rounds", type=int, default=1,
                     help="rounds to play before exiting (default: 1)")
    run.add_argument("--keep-running", action="store_true",
                     help="don't exit when the game-over screen appears")
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
    if getattr(args, "no_auto_color", False):
        data = settings.to_dict()
        data["color"]["auto"] = False
        settings = Settings.from_dict(data)
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


def _drive(frames, lexicon, backend, settings, typist, *,
           locate=True, auto_start=False, rounds=1,
           stop_on_game_over=True, debug=False):
    """Session-aware scan loop shared by replay and run.

    Tracks game state alongside word detection: types only while the game is
    playing, optionally clicks PLAY / PLAY AGAIN, reports the final score,
    and re-anchors geometry when the panel is found away from its configured
    position.
    """
    from .detect import Detector
    from .engine import Engine
    from .session import GameState, SessionTracker, locate_panel

    detector = Detector(lexicon, backend, settings)
    engine = Engine(detector, typist, settings)
    tracker = SessionTracker(backend, settings)
    located = False
    rounds_done = 0
    last_click = -1e9
    first_timestamp = None
    hinted = False
    seen_playing = False
    game_over_at = None
    score_reported = False

    for timestamp, frame in frames:
        if first_timestamp is None:
            first_timestamp = timestamp
        if locate and not located:
            panel = locate_panel(frame)
            if panel:
                drift = max(abs(a - b) for a, b in
                            zip(panel, settings.geometry.panel))
                if drift > 8:
                    settings = settings.with_panel(panel)
                    detector = Detector(lexicon, backend, settings)
                    engine.detector = detector
                    tracker = SessionTracker(backend, settings)
                    print(f"Located minigame panel at {panel} "
                          f"(configured geometry re-anchored).")
                located = True

        previous = tracker.state
        state = tracker.classify(timestamp, frame)
        if state is not previous:
            print(f"[{timestamp:7.2f}s] --- {state.value} ---")
        if (not tracker.transitions and not hinted
                and timestamp - first_timestamp > 5.0):
            hinted = True
            print("Nothing recognised after 5s -- is the minigame visible "
                  "on the captured monitor? (--debug shows every OCR read)")

        if state is GameState.PLAYING:
            seen_playing = True
            for word in engine.process(timestamp, frame):
                print(_describe(word))
            if debug:
                for d in engine.last_detections:
                    print(f"    ({d.box[0]:4},{d.box[1]:4}) ocr={d.raw!r} "
                          f"-> {d.name or '-'} ({d.score:.2f})")
            continue

        if state is GameState.GAME_OVER:
            if previous is not GameState.GAME_OVER:
                game_over_at = timestamp
                score_reported = False
            # The displayed score counts up as the modal appears; report
            # only once the same value has been read twice (or it has had
            # ample time to finish animating).
            if not score_reported and (tracker.score_settled
                                       or timestamp - game_over_at > 4.0):
                score_reported = True
                score = tracker.final_score
                print(f"[{timestamp:7.2f}s] GAME OVER -- total score: "
                      f"{score if score is not None else 'unreadable'}"
                      + ("" if seen_playing
                         else " (stale: no round played yet)"))
                if seen_playing:
                    rounds_done += 1
                    if rounds_done >= rounds and stop_on_game_over:
                        break

        if (auto_start and timestamp - last_click > 2.0
                and state in (GameState.START_SCREEN, GameState.GAME_OVER)):
            if state is GameState.GAME_OVER and (
                    not score_reported or rounds_done >= rounds):
                continue
            x, y = tracker.button_position(state)
            label = ("PLAY" if state is GameState.START_SCREEN
                     else "PLAY AGAIN")
            if typist.live:
                print(f"[{timestamp:7.2f}s] clicking {label} at ({x}, {y})")
            else:
                print(f"[{timestamp:7.2f}s] [dry-run] would click {label} "
                      f"at ({x}, {y})")
            typist.click(x, y)
            last_click = timestamp

    return engine, tracker


def _summarise(engine, tracker) -> None:
    stats = engine.stats
    print(f"\n{stats.typed} words typed over {stats.frames} scans "
          f"({dict(stats.by_source)}); "
          f"{stats.suppressed_duplicate} duplicates suppressed, "
          f"{stats.awaiting_confirmation} held for confirmation.")
    if tracker.transitions:
        path = " -> ".join(t.state.value for t in tracker.transitions)
        print(f"Session: {path}"
              + (f"; final score {tracker.final_score}"
                 if tracker.final_score is not None else ""))


def cmd_replay(args) -> int:
    from .capture import VideoSource
    from .keyboard import DryRunTypist
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

    engine, tracker = _drive(
        source.frames(), lexicon, backend, settings,
        DryRunTypist(settings.behaviour),
        locate=not args.no_locate_panel,
        stop_on_game_over=False,
        debug=args.debug,
    )
    _summarise(engine, tracker)
    return 0


def cmd_run(args) -> int:
    from .capture import ScreenSource, find_game_monitor
    from .keyboard import make_typist
    from .ocr import get_backend

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
    source = ScreenSource(int(monitor), settings.behaviour.scan_interval)
    settings = settings.for_resolution(source.width, source.height)
    typist = make_typist(args.live, settings.behaviour)

    mode = "TYPING ENABLED" if args.live else "DRY RUN (pass --live to type)"
    print(f"Live capture on monitor {monitor} "
          f"({source.width}x{source.height}), OCR={backend.name}. {mode}. "
          f"Ctrl+C to stop.\n")
    try:
        engine, tracker = _drive(
            source.frames(), lexicon, backend, settings, typist,
            locate=not args.no_locate_panel,
            auto_start=not args.no_auto_start,
            rounds=args.rounds,
            stop_on_game_over=not args.keep_running,
            debug=args.debug,
        )
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0
    _summarise(engine, tracker)
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

    import cv2 as _cv2

    from . import autocolor
    from .session import PER_BOX_MIN_PIXELS

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


COMMANDS = {
    "doctor": cmd_doctor,
    "update-data": cmd_update_data,
    "replay": cmd_replay,
    "run": cmd_run,
    "match": cmd_match,
    "calibrate": cmd_calibrate,
}


def resolve_argv(argv: list[str] | None) -> list[str]:
    """Bare invocation means an auto-configured run.

    ``uv run automaton`` / ``python -m automaton`` with no arguments behaves
    like ``automaton run``: find the Dota window, find the panel, watch the
    session, click PLAY when it shows up -- dry run unless --live is given.
    """
    if argv is None:
        argv = sys.argv[1:]
    return argv if argv else ["run"]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(resolve_argv(argv))
    try:
        return COMMANDS[args.command](args)
    except KeyboardInterrupt:
        return 130
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

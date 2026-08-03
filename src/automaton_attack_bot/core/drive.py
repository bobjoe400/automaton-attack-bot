"""The session driver: scan, decide, type, track -- shared by run and replay."""

from ..logbook import LOG


def describe(word) -> str:
    d = word.detection
    return (f"[{word.timestamp:7.2f}s] {word.name:<40} "
            f"{word.source:<8} score={d.score:.2f} ocr={d.raw!r}")


def drive(frames, lexicon, backend, settings, typist, *,
           locate=True, auto_start=False, rounds=1,
           stop_on_game_over=True, threaded=False):
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
    worker = None
    scan_pipeline = None
    if threaded:
        from concurrent.futures import ThreadPoolExecutor

        from .keyboard import TypingWorker

        def emit(word, waited):
            LOG.trace(describe(word) + f"  [queued {waited:.2f}s]")

        worker = TypingWorker(typist, engine._urgency, on_typed=emit)
        engine.dispatch = worker.submit
        worker.start()
        # Detection is pipelined: while one scan's OCR runs, the next
        # frame's starts. Decisions stay strictly ordered (the confirmer's
        # consecutive-scan semantics require it); only the OCR overlaps.
        scan_pipeline = ThreadPoolExecutor(max_workers=2,
                                           thread_name_prefix="scan")
    try:
        return _drive_loop(
            frames, lexicon, backend, settings, typist, engine, tracker,
            worker, scan_pipeline,
            locate=locate, auto_start=auto_start, rounds=rounds,
            stop_on_game_over=stop_on_game_over)
    finally:
        if scan_pipeline is not None:
            scan_pipeline.shutdown(wait=False, cancel_futures=True)
        if worker is not None:
            worker.stop()
            if worker.dropped_stale:
                LOG.say(f"{worker.dropped_stale} queued word(s) dropped as "
                        f"stale.")
            if worker.dropped_triage:
                LOG.say(f"{worker.dropped_triage} weak guess(es) skipped "
                        f"while the keyboard queue was deep.")


def _drive_loop(frames, lexicon, backend, settings, typist, engine, tracker,
                worker, scan_pipeline, *, locate, auto_start, rounds,
                stop_on_game_over):
    from collections import deque

    from .detect import Detector
    from .session import GameState, SessionTracker, locate_panel

    # Geometry is "proven" once the tracker recognises any game state with
    # it. Until then, keep re-locating: locking on the first plausible
    # rectangle once blinded a whole session when a transitional frame
    # produced a wrong-but-plausible panel.
    proven = False
    rounds_done = 0
    last_click = -1e9
    first_timestamp = None
    hinted = False
    seen_playing = False
    game_over_at = None
    score_reported = False
    last_multiplier = None
    last_multiplier_read = -1e9
    futile_clicks = 0       # clicks (or refusals) with no state change since
    in_flight = deque()     # (timestamp, future) of pipelined detections
    last_scan_submit = -1e9
    FUTILE_CLICK_LIMIT = 3
    # Below this spacing, two scans see essentially the same frame and
    # 'stability' stops meaning anything (see Confirmer.MIN_STABLE_AGE).
    MIN_SCAN_SPACING = 0.08

    def drain(block: bool = False) -> None:
        while in_flight and (block or in_flight[0][1].done()
                             or len(in_flight) >= 2):
            scan_ts, future = in_flight.popleft()
            for word in engine.process_detections(scan_ts, future.result()):
                if worker is None:
                    LOG.trace(describe(word))

    for timestamp, frame in frames:
        if first_timestamp is None:
            first_timestamp = timestamp
        if locate and not proven:
            panel = locate_panel(frame)
            if panel:
                drift = max(abs(a - b) for a, b in
                            zip(panel, settings.geometry.panel))
                # Rebuilds are cheap, and even a few pixels of offset can
                # clip the tight digit crops (score, timer).
                if drift > 3:
                    settings = settings.with_panel(panel)
                    detector = Detector(lexicon, backend, settings)
                    engine.detector = detector
                    tracker = SessionTracker(backend, settings)
                    LOG.say(f"Located minigame panel at {panel} "
                            f"(configured geometry re-anchored).")

        previous = tracker.state
        state = tracker.classify(timestamp, frame)
        if state is not GameState.UNKNOWN:
            proven = True
        if state is not previous:
            LOG.say(f"[{timestamp:7.2f}s] --- {state.value} ---")
            futile_clicks = 0       # the screen responded; clicks work
        if (not tracker.transitions and not hinted
                and timestamp - first_timestamp > 5.0):
            hinted = True
            LOG.say("Nothing recognised after 5s -- is the minigame visible "
                    "on the captured monitor? (--debug shows every OCR read)")

        if state is GameState.PLAYING:
            if not seen_playing or previous is not GameState.PLAYING:
                # New round: the multiplier legitimately restarts at x1.0;
                # comparing across rounds printed phantom COMBO LOST lines.
                last_multiplier = None
            seen_playing = True
            # Words first -- telemetry OCR must never delay a keystroke.
            if scan_pipeline is not None:
                if timestamp - last_scan_submit >= MIN_SCAN_SPACING:
                    last_scan_submit = timestamp
                    in_flight.append((timestamp, scan_pipeline.submit(
                        engine.detector.detect, frame,
                        include_unmatched=True, timestamp=timestamp)))
                drain()
            else:
                for word in engine.process(timestamp, frame):
                    if worker is None:  # threaded mode prints at type time
                        LOG.trace(describe(word))
            # Log the combo so a loss is findable in the log (and footage)
            # without a post-hoc OCR scrub of the whole recording.
            if timestamp - last_multiplier_read >= 0.5:
                last_multiplier_read = timestamp
                multiplier = tracker.read_multiplier(frame)
                if multiplier is not None and multiplier != last_multiplier:
                    clock = tracker.read_timer(frame)
                    clock_note = (f", clock {clock // 60}:{clock % 60:02d}"
                                  if clock is not None else "")
                    if (last_multiplier is not None
                            and multiplier < last_multiplier):
                        LOG.say(f"[{timestamp:7.2f}s] !!! COMBO LOST "
                                f"x{last_multiplier} -> x{multiplier}"
                                f"{clock_note}")
                    else:
                        LOG.say(f"[{timestamp:7.2f}s] combo x{multiplier}"
                                f"{clock_note}")
                    last_multiplier = multiplier
            for d in engine.last_detections:
                LOG.trace(f"    ({d.box[0]:4},{d.box[1]:4}) ocr={d.raw!r} "
                          f"-> {d.name or '-'} ({d.score:.2f})")
            continue

        if state is not GameState.PLAYING and in_flight:
            # The round is over; scans still in the pipeline belong to it.
            in_flight.clear()

        if state is GameState.GAME_OVER:
            if previous is not GameState.GAME_OVER:
                game_over_at = timestamp
                score_reported = False
            # The displayed score counts up as the modal appears; report
            # only once the settled criteria hold (or the animation has had
            # ample time and the best confirmed value stands).
            if not score_reported and (tracker.score_settled
                                       or timestamp - game_over_at > 5.0):
                score_reported = True
                score = tracker.final_score
                LOG.say(f"[{timestamp:7.2f}s] GAME OVER -- total score: "
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
            # Three strikes on either guard and the geometry lock is
            # considered wrong: a hub page once classified as start-screen
            # and got clicked, and a mislocated panel means the click spot
            # is bare background either way.
            if futile_clicks >= FUTILE_CLICK_LIMIT:
                LOG.say(f"[{timestamp:7.2f}s] clicks are not landing -- "
                        f"dropping the geometry lock to re-locate.")
                proven = False
                futile_clicks = 0
                last_click = timestamp
                continue
            if not tracker.verify_button(frame, state):
                LOG.say(f"[{timestamp:7.2f}s] {state.value} but no button "
                        f"text at the click target -- not clicking.")
                futile_clicks += 1
                last_click = timestamp
                continue
            x, y = tracker.button_position(state)
            label = ("PLAY" if state is GameState.START_SCREEN
                     else "PLAY AGAIN")
            if typist.live:
                LOG.say(f"[{timestamp:7.2f}s] clicking {label} at ({x}, {y})")
            else:
                LOG.say(f"[{timestamp:7.2f}s] [dry-run] would click {label} "
                        f"at ({x}, {y})")
            typist.click(x, y)
            futile_clicks += 1
            last_click = timestamp

    return engine, tracker


def summarise(engine, tracker) -> None:
    stats = engine.stats
    LOG.say(f"\n{stats.typed} words typed over {stats.frames} scans "
            f"({dict(stats.by_source)}); "
            f"{stats.suppressed_duplicate} duplicates suppressed, "
            f"{stats.awaiting_confirmation} held for confirmation.")
    if tracker.transitions:
        path = " -> ".join(t.state.value for t in tracker.transitions)
        LOG.say(f"Session: {path}"
                + (f"; final score {tracker.final_score}"
                   if tracker.final_score is not None else ""))

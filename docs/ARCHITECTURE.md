# Architecture

How the bot works, end to end.

## Code layout

```
src/automaton_attack_bot/
  cli.py       argument parsing and the command handlers
  logbook.py   two-level session log (full detail to file, signal to console)
  analyze.py   offline miss forensics for recordings
  corpus.py    word-corpus download/build/cache, with data/ holding
               custom_vocab.txt
  core/        the real-time bot: config, capture, ocr, autocolor, detect,
               lexicon, engine, keyboard, session, and drive (the shared
               scan-decide-type loop)
  gui/         the control panel: app (window), tasks (worker thread with
               stdout teed into the panel), feed (log-line -> status)
```

The GUI owns no bot logic: its buttons dispatch the same command functions
the terminal uses, and its status display is parsed from the session's own
log lines.

## The pipeline

Perception, decision and action run as a real-time pipeline of cooperating
threads, none of which waits on another except where correctness demands it:

1. **Capture thread** grabs the screen continuously and serves the freshest
   frame; each grab is fingerprinted, and a frozen framebuffer (~0.3s of
   identical content — it happens at round restarts) re-initialises the
   grabber and confesses to the log.
2. **Detection pipeline** runs two scans in flight: HSV-isolate the yellow
   target words (auto-calibrated each scan against the HUD text, which shares
   their colour), split too-tall blobs into stacked lines, stitch wrapped
   phrases back together (up to four lines), and OCR every blob concurrently
   on a pool of single-threaded PP-OCR engines. Decisions consume scans
   strictly in order.
3. **Matching** tries, in order: core vocabulary (fuzzy, accept ≥0.62, prefer
   the longer prefix-extension on near-ties — typing WEAVER completes WEAVE on
   the way through); unique containment (a fly-in fragment inside exactly one
   vocab key IS that word); the voice-line corpus (accept ≥0.80); embedded
   mining (interleaved pile-ups OCR as mush that contains component words
   letter-perfect — type them all); verbatim fallback for clean unmatched
   reads, gated on 0.35s of real-time stability; and verbatim insurance
   whenever the best match is weak, because a wrong guess costs nothing and a
   blocked word costs the combo.
4. **Scheduling**: words type nearest-the-platform-first (deadline minus
   typing time); labels below 70% of panel height are the game's kill display
   and are never typed; a word still visible ~1.2s after its keystrokes did
   not complete and is retyped — typing is self-correcting.
5. **Typing worker** sends keystrokes at full speed from a HIGHEST-priority
   thread, dropping stale or hopeless entries under load.
6. **Session tracking** reads the round state, combo multiplier, clock and
   final score from the HUD (digits via a tesseract whitelist where
   available — PP-OCR garbles `17,770` into `17,7%`), settles the score
   against its count-up animation, and clicks PLAY / PLAY AGAIN itself.

## Configuration

Defaults are tuned for 1920x1080 and rescale automatically for other
resolutions; the panel is located live by its border, so the window position
doesn't matter. To override anything, write a config and edit it:

```bash
uv run automaton calibrate --clip clips/clip1.mp4 --at 8 --save-config
```

That writes `automaton.json` (picked up automatically from the working
directory) plus annotated diagnostic images.

## Game notes

- 60-second round, no fail state. Words fly in from the sides and below and
  converge on Hoodwink at the platform; a word that reaches her untyped
  resets the score multiplier to 1.
- **Wrong keystrokes cost nothing** — input isn't targeted at a word, stray
  letters just don't advance anything. So the bot is greedy: attempt
  everything the moment it's seen, and let a better read on the next scan
  correct it. Hesitation loses multipliers, guessing doesn't.
- Voice lines score far more than single words (100–250 vs 10–50 observed).
- Set Dota to English or the words are localised and nothing matches.
- Typing runs at full machine speed by default. `--max-wpm` is the opt-in
  throttle if you want human-plausible pacing (Valve has patched exploits in
  this minigame before, so the leaderboard is at least occasionally watched —
  your call).


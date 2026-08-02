# Automaton Attack Bot

A computer-vision bot that plays **Automaton Attack**, the typing minigame in
Dota 2's Dark Carnival: Midnight Run event (Chapter 3, Engine Car).

It watches the screen, finds the yellow target words, reads them, matches them
against a corpus of hero/item/ability names and ~47k in-game voice lines, and
types them at machine speed.

Best validated live result: **37,205**, set during a streak of **six
consecutive zero-combo-loss rounds** — about 26× the leaderboard high score
this project first encountered. The road there is documented in the commit
history: every combo loss across nineteen live rounds was diagnosed from the
bot's own logs and footage, root-caused, fixed, and pinned with a regression
test.

## Install

Requires [uv](https://docs.astral.sh/uv/). Nothing else — the OCR model ships
as a Python wheel, so there are no system packages to install.

```bash
git clone git@github.com:bobjoe400/automaton-attack-bot.git && cd automaton-attack-bot
uv sync
uv run automaton doctor
```

Add `--extra live` to install screen capture and keystroke delivery (Windows):

```bash
uv sync --extra live
```

On first run the bot downloads and caches its word data (~70 MB, one time)
from the community projects that maintain it — see
[Word data](#word-data) below.

## Use

The bare command plays a round, start to finish:

```bash
uv run automaton
```

(`python -m automaton` works too.) It finds the monitor the Dota 2 window is
on, finds the minigame panel by its frame, clicks PLAY, types the round,
reports the final score and exits at the game-over screen. `--rounds N`
plays several games; `--no-auto-start` leaves the buttons alone;
`--keep-running` ignores the game-over screen.

To rehearse without touching the keyboard or mouse — everything is printed
instead of sent:

```bash
uv run automaton --dry-run
```

(A clone without the live extras installed falls back to a dry run on its
own.)

Every run writes a full-detail session log to `logs/run-<stamp>.log` — every
typed word with its OCR read and queue latency, plus every scan's per-blob
detail — while the console shows only the signal: state transitions, combo
telemetry with the round clock (`!!! COMBO LOST x3.0 -> x1.5, clock 0:19`),
scores, and summaries. A round's forensics live in one file.

Other commands:

| Command | What it does |
| --- | --- |
| *(none)* | Play: find the game, click PLAY, type the round; `--dry-run` to rehearse |
| `analyze CLIP` | Replays a recording and reports likely misses: platform strikes (words that reached Hoodwink), stable reads with weak or no match — vocabulary gaps surface as a list |
| `replay CLIP` | Runs detection over a recording; never types |
| `doctor` | Checks dependencies, OCR backends and data files |
| `match TEXT` | Asks the lexicon what an OCR read resolves to |
| `calibrate` | Dumps mask/blob diagnostics from one frame |
| `update-data` | Rebuilds the word corpora from upstream (run after a patch) |

Useful flags: `--debug` echoes the per-scan detail to the console,
`--safe-mode` refuses to type anything that did not match the corpus,
`--no-phrases` drops the voice-line corpus, `--ocr tesseract` switches
backends, `--max-wpm` caps typing speed.

## How it works

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

## The iteration loop

The workflow that produced every fix in this repo: play a round, then feed
the artifacts back —

1. `logs/run-<stamp>.log` — the combo telemetry pins any loss to a round
   clock; the per-scan detail shows exactly what was read at that moment.
2. `automaton analyze recording.mp4` — names words that reached the platform
   and stable reads with weak or missing matches.
3. Real words the corpus lacks go one-per-line into
   [custom_vocab.txt](src/automaton_attack/data/custom_vocab.txt) — applied
   at load, no rebuild. Trust observation over documentation: several
   "retired" words turned out to be live targets.

## Development

```bash
uv run pytest                    # unit tests
uv run pytest -m clips           # clip regressions (needs clips/ populated)
```

Drop recordings in `clips/` — see [clips/README.md](clips/README.md) for the
expected files. Every regression test in the suite was derived from a live
failure; the clip tests replay real rounds and pin the expected outcomes.

See [docs/HANDOFF.md](docs/HANDOFF.md) for the original prototype notes —
historical context; several of its claims were disproven by live play, and
the code documents the corrections where they matter.

## Word data

The bot ships **no word lists of its own**. On first run (or via
`automaton update-data`) it builds its corpora from the community projects
that maintain this data, and caches them locally
(`%LOCALAPPDATA%\automaton-attack` on Windows, `~/.cache/automaton-attack`
elsewhere; override with `AUTOMATON_DATA_DIR`):

- **[odota/dotaconstants](https://github.com/odota/dotaconstants)** — hero,
  item and ability names, the same constants the OpenDota API is built on.
- **[mdiller/dotabase](https://github.com/mdiller/dotabase)** — a SQLite
  export of Dota's game files; its `responses` table provides ~47k voice
  lines.

Both track Valve's patches, so `automaton update-data` after a patch is the
whole maintenance story. The one hand-maintained file is
[custom_vocab.txt](src/automaton_attack/data/custom_vocab.txt): words seen in
the minigame that upstream doesn't carry (event words like `Banana` and
`Screeauk`, renamed heroes, voice lines the game renders differently), and
retired items to exclude so they can't steal fuzzy matches from live ones.

# Automaton Attack Bot

A computer-vision bot that plays **Automaton Attack**, the typing minigame in
Dota 2's Dark Carnival: Midnight Run event (Chapter 3, Engine Car).

It watches the screen, finds the yellow target words, reads them, matches them
against a corpus of hero/item/ability names and ~39k in-game voice lines, and
types them.

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

Everything is auto-configured and nothing touches your keyboard or mouse
until you pass `--live`. The bare command is a full dry run:

```bash
uv run automaton
```

(`python -m automaton` works too.) That finds the monitor the Dota 2 window
is on, finds the minigame panel by its frame, watches the session, and prints
everything it *would* do — the words it reads, the buttons it would click,
the final score. When it looks right, let it play:

```bash
uv run automaton run --live
```

That clicks PLAY on the start screen, plays a round, reports the final score
and exits when the game-over screen appears. `--rounds N` plays several
games; `--no-auto-start` leaves the buttons alone; `--keep-running` ignores
the game-over screen.

Replay a recording instead of the live screen — the safest way to check a
config change:

```bash
uv run automaton replay clips/clip1.mp4
```

Other commands:

| Command | What it does |
| --- | --- |
| `doctor` | Checks dependencies, OCR backends and data files |
| `replay CLIP` | Runs detection over a recording; never types |
| `run` | Live capture; dry run unless `--live` |
| `match TEXT` | Asks the lexicon what an OCR read resolves to |
| `calibrate` | Dumps mask/blob diagnostics from one frame |
| `update-data` | Rebuilds the word corpora from upstream (run after a patch) |

Useful flags: `--debug` prints every blob including unmatched ones,
`--safe-mode` refuses to type anything that did not match the corpus,
`--no-phrases` drops the voice-line corpus, `--ocr tesseract` switches
backends, `--max-wpm` caps typing speed.

## How it works

1. **Capture** a frame (`mss` live, OpenCV for recordings).
2. **Isolate** target words by HSV colour, then blank the three HUD boxes that
   share the same yellow-green.
3. **Merge** letters into word blobs with a wide, short dilate kernel, then
   filter by area and height.
4. **Read** each blob with PP-OCR (recognition only — we do our own
   segmentation).
5. **Match** in three tiers: the core vocabulary (difflib, accept ≥0.62), then
   the voice-line corpus (rapidfuzz, accept ≥0.80 because a wrong steal types a
   whole wrong sentence), then the read itself if it is long and clean. When a
   longer word merely extends the best match (WEAVE → WEAVER) at a near-tied
   score, the longer one wins: its keystrokes complete the shorter word on the
   way through, so it covers both readings.
6. **Type immediately, letters only, lowercase** — the game lets you skip
   spaces and punctuation, and wrong keystrokes cost nothing (see below), so
   every corpus match is typed on first sight. Only fallback reads — raw OCR
   that matched nothing — wait for the identical text on two consecutive
   scans, because OCR errors vary frame to frame and unrepeated flickers
   would burn keyboard time on garbage.

## Configuration

Defaults are tuned for 1920x1080 and rescale automatically for other
resolutions. To override anything, write a config and edit it:

```bash
uv run automaton calibrate --clip clips/clip1.mp4 --at 8 --save-config
```

That writes `automaton.json` (picked up automatically from the working
directory) plus annotated diagnostic images. The HSV range is the setting most
likely to need retuning: recorded clips are compression-softened, a live
framebuffer is not.

## Game notes

- 60-second round, no fail state. Words fly in from the sides and vanish at the
  centre if unfinished.
- **A word that escapes untyped resets the score multiplier to 1** — but wrong
  keystrokes cost nothing; input isn't targeted at a word, stray letters just
  don't advance anything. So the bot is greedy: attempt everything the moment
  it's seen, and let a better read on the next scan correct it. Hesitation
  loses multipliers, guessing doesn't.
- Voice lines score far more than single words (100 vs 10 observed).
- Set Dota to English or the words are localised and nothing matches.
- Valve patched a pause-typing exploit in this minigame in July 2026, so the
  leaderboard is watched. `--max-wpm` keeps output in human range.

## Development

```bash
uv run pytest                    # unit tests
uv run pytest -m clips           # clip regressions (needs clips/ populated)
```

Drop recordings in `clips/` as `clip1.mp4` (single-word phase) and `clip2.mp4`
(multi-word phase) to enable the regression tests.

See [docs/HANDOFF.md](docs/HANDOFF.md) for the measurements and reasoning
behind every constant, and the open questions that remain.

## Word data

The bot ships **no word lists of its own**. On first run (or via
`automaton update-data`) it builds its corpora from the community projects
that maintain this data, and caches them locally
(`%LOCALAPPDATA%\automaton-attack` on Windows, `~/.cache/automaton-attack`
elsewhere; override with `AUTOMATON_DATA_DIR`):

- **[odota/dotaconstants](https://github.com/odota/dotaconstants)** — hero,
  item and ability names, the same constants the OpenDota API is built on.
- **[mdiller/dotabase](https://github.com/mdiller/dotabase)** — a SQLite
  export of Dota's game files; its `responses` table provides ~40k voice
  lines.

Both track Valve's patches, so `automaton update-data` after a patch is the
whole maintenance story. The one hand-maintained file is
[custom_vocab.txt](src/automaton_attack/data/custom_vocab.txt): words seen in
the minigame that upstream doesn't carry, and retired items to exclude so
they can't steal fuzzy matches from live ones.

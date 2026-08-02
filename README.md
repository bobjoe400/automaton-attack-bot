# Automaton Attack Bot

A computer-vision bot that plays **Automaton Attack**, the typing minigame in
Dota 2's Dark Carnival: Midnight Run event. It watches the screen, reads the
target words, matches them against Dota's names and ~47k voice lines, and
types them at machine speed.

Best live result: **37,205**, during six consecutive zero-combo-loss rounds
(the leaderboard high score it first met was 1,430).

## Quick start

Requires [uv](https://docs.astral.sh/uv/) — no system packages.

```bash
git clone git@github.com:bobjoe400/automaton-attack-bot.git && cd automaton-attack-bot
uv sync --extra live
uv run automaton doctor
```

Open the minigame in Dota, then:

```bash
uv run automaton
```

That finds the Dota window, clicks PLAY, types the round, reports the score
and exits. First run downloads ~70 MB of word data (once) from
[odota/dotaconstants](https://github.com/odota/dotaconstants) and
[mdiller/dotabase](https://github.com/mdiller/dotabase) — the community
projects that did the hard part.

Useful variations:

```bash
uv run automaton --dry-run     # rehearse: print everything, touch nothing
uv run automaton --rounds 3    # play several games back to back
```

Every run writes a full-detail log to `logs/run-<stamp>.log`; the console
shows only state changes, combo telemetry and scores.

## Commands

| Command | What it does |
| --- | --- |
| *(none)* | Play a round (`--dry-run` to rehearse, `--rounds N` for several) |
| `analyze CLIP` | Report likely misses from a recording: platform strikes, weak/unmatched reads |
| `replay CLIP` | Run detection over a recording; never types |
| `doctor` | Check dependencies, OCR backends and data files |
| `match TEXT` | Ask the lexicon what an OCR read resolves to |
| `calibrate` | Dump mask/blob diagnostics from one frame |
| `update-data` | Rebuild the word corpora from upstream (run after a patch) |

## Learn more

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — the perception/decision/typing
  pipeline, configuration, and game mechanics notes.
- [docs/WORKFLOW.md](docs/WORKFLOW.md) — the iteration loop (logs → analyze →
  vocab), testing, and how the word data is built and maintained.
- [docs/HANDOFF.md](docs/HANDOFF.md) — the original prototype notes, kept as
  historical context.
- [clips/README.md](clips/README.md) — recordings the regression tests expect.

If a word the game shows isn't in the corpus, add it to
[custom_vocab.txt](src/automaton_attack_bot/data/custom_vocab.txt) — one line,
applied on the next run.

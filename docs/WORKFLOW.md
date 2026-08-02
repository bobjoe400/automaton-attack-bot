# Workflow

How this bot gets better, and how its data stays current.

## The iteration loop

The workflow that produced every fix in this repo: play a round, then feed
the artifacts back —

1. `logs/run-<stamp>.log` — the combo telemetry pins any loss to a round
   clock; the per-scan detail shows exactly what was read at that moment.
2. `automaton analyze recording.mp4` — names words that reached the platform
   and stable reads with weak or missing matches.
3. Real words the corpus lacks go one-per-line into
   [custom_vocab.txt](../src/automaton_attack_bot/data/custom_vocab.txt) — applied
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
[custom_vocab.txt](../src/automaton_attack_bot/data/custom_vocab.txt): words seen in
the minigame that upstream doesn't carry (event words like `Banana` and
`Screeauk`, renamed heroes, voice lines the game renders differently), and
retired items to exclude so they can't steal fuzzy matches from live ones.

# Clips

Recorded gameplay used by the offline regression tests. The files themselves
are gitignored — they are large, and they are Valve's pixels, not ours.

Drop recordings here as:

| File | What it should contain |
| --- | --- |
| `clip1.mp4` | The single-word phase: one target on screen at a time |
| `clip2.mp4` | The multi-word phase: overlapping simultaneous targets |
| `fullgame.mp4` | A whole session: start screen, one full round, game-over screen (its expected total score is pinned in `tests/test_session.py`) |

Then:

```bash
uv run pytest -m clips
```

Without them, `pytest -m clips` skips and the rest of the suite still passes.

Record at 1920x1080 if you can. Other resolutions work — geometry rescales
automatically — but the defaults were measured at 1080p, so that is the
configuration the assertions describe.

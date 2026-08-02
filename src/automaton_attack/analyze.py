"""Post-round analysis: find the words the bot handled badly.

``automaton analyze CLIP`` replays a recording, aggregates every OCR read
by its letters-only key, and reports the ones that look like misses:

* **unmatched** -- a stable read that resolved to nothing: a vocabulary gap
  (IMPERIA before it was added), or OCR garbage if it never repeats.
* **weak match** -- a read that repeated identically while its best match
  stayed fuzzy. A stable read IS what is on screen (OCR errors vary frame
  to frame), so a stable read with a sub-0.9 match usually means the match
  is wrong: 'ETERNAL SHROUD' stealing INFERNAL SHRED at 0.69, 'OUTWORLD
  DESTROYER' typing the hero's old name at 0.79.

Real words repeat across scans; one-off partial reads from words flying in
do not. Requiring the same key twice filters nearly all of the noise.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .lexicon import to_key

STABLE_MIN_SIGHTINGS = 2
STRONG_MATCH = 0.90
MIN_KEY_LENGTH = 4


@dataclass
class ReadRecord:
    key: str
    sightings: int = 0
    raws: Counter = field(default_factory=Counter)
    matches: Counter = field(default_factory=Counter)
    best_score: float = 0.0
    best_name: str | None = None
    first_seen: float = 0.0
    last_seen: float = 0.0
    # Round clock (seconds remaining) at first sighting, when readable.
    # The game's own timestamp: exact regardless of capture latency or
    # where a recording was trimmed.
    clock: int | None = None

    @property
    def stable(self) -> bool:
        return self.sightings >= STABLE_MIN_SIGHTINGS

    @property
    def sample_raw(self) -> str:
        return self.raws.most_common(1)[0][0]


# A word label rendered this far down the panel is a word whose automaton
# just reached Hoodwink -- words never render there otherwise. Checked
# offline only: the live loop spends nothing on it.
PLATFORM_BAND = 0.70


class ReadLog:
    """Aggregates detections across a whole recording."""

    def __init__(self) -> None:
        self.records: dict[str, ReadRecord] = {}
        self.strikes: list[tuple[float, int | None, str]] = []

    def add_strike(self, timestamp: float, clock: int | None,
                   label: str) -> None:
        # collapse repeats of the same label within a couple of seconds
        if self.strikes and self.strikes[-1][2] == label \
                and timestamp - self.strikes[-1][0] < 2.0:
            return
        self.strikes.append((timestamp, clock, label))

    def add(self, timestamp: float, detection,
            clock: int | None = None) -> None:
        key = to_key(detection.raw)
        if len(key) < MIN_KEY_LENGTH:
            return
        record = self.records.setdefault(key, ReadRecord(key))
        if record.sightings == 0:
            record.first_seen = timestamp
            record.clock = clock
        record.sightings += 1
        record.last_seen = timestamp
        record.raws[detection.raw.strip()] += 1
        if detection.match:
            record.matches[detection.match.name] += 1
            if (detection.match.score > record.best_score
                    or record.best_name is None):
                record.best_score = detection.match.score
                record.best_name = detection.match.name

    # -- classification ----------------------------------------------------
    def unmatched(self) -> list[ReadRecord]:
        """Stable reads that resolved to nothing at all: vocabulary gaps."""
        return sorted(
            (r for r in self.records.values()
             if r.stable and not r.matches),
            key=lambda r: -r.sightings)

    def weak_matches(self) -> list[ReadRecord]:
        """Stable reads whose best match never got convincing.

        The read repeating identically means OCR is seeing it right, so a
        fuzzy score this low points at the match, not the read.
        """
        out = []
        for record in self.records.values():
            if not record.stable or not record.matches:
                continue
            if record.best_score >= STRONG_MATCH:
                continue
            # Exclude reads that are themselves the match's key -- those
            # are exact hits reported at 1.0 by definition.
            out.append(record)
        return sorted(out, key=lambda r: -r.sightings)

    def report(self) -> str:
        lines = []
        if self.strikes:
            lines.append("Words that reached the platform (combo losses):")
            for timestamp, clock, label in self.strikes:
                note = (f" (clock {clock // 60}:{clock % 60:02d})"
                        if clock is not None else "")
                lines.append(f"  {label!r:36} at {timestamp:5.1f}s{note}")
        weak = self.weak_matches()
        gaps = self.unmatched()
        def when(r: ReadRecord) -> str:
            base = f"{r.first_seen:5.1f}-{r.last_seen:5.1f}s"
            if r.clock is not None:
                base += f" (clock {r.clock // 60}:{r.clock % 60:02d})"
            return base

        if weak:
            lines.append("Stable reads with only a weak match -- the match "
                         "is probably the wrong word:")
            for r in weak:
                lines.append(
                    f"  {r.sample_raw!r:36} seen {r.sightings}x at "
                    f"{when(r)} -> best match "
                    f"{r.best_name} ({r.best_score:.2f})")
        if gaps:
            lines.append("Stable reads that matched nothing:")
            for r in gaps:
                lines.append(
                    f"  {r.sample_raw!r:36} seen {r.sightings}x at "
                    f"{when(r)}")
        if not lines:
            return ("No suspicious reads and no platform strikes: every "
                    "stable read matched convincingly.")
        lines.append(
            "\nIf one of these is a real word, add it to "
            "src/automaton_attack/data/custom_vocab.txt (it applies on the "
            "next run, no rebuild needed).")
        return "\n".join(lines)

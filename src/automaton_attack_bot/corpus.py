"""Corpus management: download, build, cache.

The word lists are not ours and are not shipped with the bot. They are built
on first run from the projects that actually maintain this data:

* **odota/dotaconstants** (https://github.com/odota/dotaconstants) --
  hero, item and ability names, as used by the OpenDota API.
* **mdiller/dotabase** (https://github.com/mdiller/dotabase) -- a SQLite
  export of the game's VPK files; its ``responses`` table holds every hero
  voice line.

Both are refreshed from Valve's data by their maintainers, so
``automaton update-data`` after a patch is the whole maintenance story --
there is no list of ours to curate. The one exception is
``data/custom_vocab.txt``: hand-collected words seen in the minigame that
upstream does not carry, plus a short list of retired items to *exclude*
(they linger upstream but would steal fuzzy matches from live words).

Everything is cached under :func:`data_dir`, overridable with the
``AUTOMATON_DATA_DIR`` environment variable.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import tempfile
import urllib.request
from importlib import resources
from pathlib import Path

ATTRIBUTION = """\
Word data comes from two community projects; if this bot is useful, they did
the hard part:
  odota/dotaconstants  https://github.com/odota/dotaconstants  (names)
  mdiller/dotabase     https://github.com/mdiller/dotabase     (voice lines)"""

DOTACONSTANTS_BASE = (
    "https://raw.githubusercontent.com/odota/dotaconstants/master/build/"
)
DOTACONSTANTS_FILES = ("heroes.json", "items.json", "abilities.json")

DOTABASE_SQL_URL = (
    "https://raw.githubusercontent.com/mdiller/dotabase/master/"
    "dotabase/dotabase.db.sql"
)

VOCAB_FILE = "vocab.txt"
PHRASES_FILE = "phrases.txt"

# Bump when the builder's filters change meaning (a word un-retired, a new
# reject rule): caches built by older code silently miss words otherwise --
# a perfectly-read 'ETERNAL SHROUD' once fuzzy-stole INFERNAL SHRED because
# the on-disk cache predated its un-retirement.
CORPUS_VERSION = 3
VERSION_PREFIX = "# corpus-version:"

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

# Name-field rows that never appear as target words.
REJECT_SUBSTRINGS = (
    "recipe", "Recipe", "tooltip", "Tooltip", "DUMMY", "dummy",
    "empty", "Empty", "seasonal", "Seasonal",
)

# Rows whose *key* marks them as non-words: talent descriptions
# ("+0.5s Stun Duration") and Ability Draft variants ("Cold Snap (AD)").
REJECT_KEY_PREFIXES = ("special_bonus_",)
REJECT_KEY_SUFFIXES = ("_ad",)

# Real names are plain text: letters first, then letters/digits/space and the
# few marks that actually occur ("Blast Off!", "Detonate M.A.D.",
# "Sange & Yasha"). Talent fragments fail this on their leading +/-/digit,
# scepter-note rows on colons, parentheses or embedded markup.
NAME_SHAPE = re.compile(r"^[A-Za-z][A-Za-z0-9 '\-.!&]*$")

# In the game's target words, but retired from the game itself. They linger
# in dotaconstants and would steal fuzzy matches from live words. Only list
# a word here on OBSERVATION, not on patch notes: Eternal Shroud was listed
# as retired by the prototype's notes, then showed up as a live target word.
RETIRED = {
    "Cornucopia",
}

# Voice lines: long enough to be worth a fuzzy match, plain enough to be
# typeable. Long lines WRAP on screen rather than being excluded from the
# game: "There's a fine line between bravery and stupidity." (51 chars)
# appeared as a two-line target and died unmatched under the old 48 cap.
# The detector stitches wrapped lines back together.
PHRASE_MIN_LENGTH = 8
PHRASE_MAX_LENGTH = 72
PHRASE_ALLOWED = re.compile(r"^[A-Za-z ',.!?-]+$")

# A line known to be in both dotabase and the minigame; if the filter drops
# it, the filter is wrong.
PHRASE_CANARY = "Bane of your existence."


def data_dir() -> Path:
    override = os.environ.get("AUTOMATON_DATA_DIR")
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA",
                                   Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME",
                                   Path.home() / ".cache"))
    return base / "automaton-attack"


def vocab_path() -> Path:
    return data_dir() / VOCAB_FILE


def phrases_path() -> Path:
    return data_dir() / PHRASES_FILE


def custom_vocab_path() -> Path:
    return Path(str(resources.files("automaton_attack_bot.data")
                    .joinpath("custom_vocab.txt")))


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------
def _fetch(url: str, label: str) -> bytes:
    print(f"  downloading {label} ...", flush=True)
    request = urllib.request.Request(
        url, headers={"User-Agent": "automaton-attack-bot corpus fetch"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
def _usable_name(name: str, key: str = "") -> bool:
    if not name or len(name) < 3:
        return False
    if key.startswith(REJECT_KEY_PREFIXES) or key.endswith(REJECT_KEY_SUFFIXES):
        return False
    if name in RETIRED:
        return False
    if any(bad in name for bad in REJECT_SUBSTRINGS):
        return False
    return NAME_SHAPE.match(name) is not None


def custom_entries() -> tuple[set[str], set[str]]:
    """Hand-maintained additions and exclusions from custom_vocab.txt.

    Lines starting with ``-`` are exclusions; everything else is an addition.
    """
    path = custom_vocab_path()
    additions, exclusions = set(), set()
    if not path.exists():
        return additions, exclusions
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            exclusions.add(line[1:].strip())
        else:
            additions.add(line)
    return additions, exclusions


def build_vocab(verbose: bool = True) -> list[str]:
    names: set[str] = set()
    for filename in DOTACONSTANTS_FILES:
        data = json.loads(_fetch(DOTACONSTANTS_BASE + filename,
                                 f"dotaconstants {filename}"))
        rows = (data.items() if isinstance(data, dict)
                else ((str(i), row) for i, row in enumerate(data)))
        count = 0
        for key, row in rows:
            if not isinstance(row, dict):
                continue
            name = row.get("localized_name") or row.get("dname")
            if isinstance(name, str) and _usable_name(name.strip(), key):
                names.add(name.strip())
                count += 1
        if verbose:
            print(f"    {count} usable names from {filename}")

    additions, exclusions = custom_entries()
    names |= additions
    names -= exclusions
    names -= RETIRED
    return sorted(names, key=str.lower)


def write_vocab() -> Path:
    entries = build_vocab()
    path = vocab_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (f"{VERSION_PREFIX} {CORPUS_VERSION}\n"
              "# Generated by `automaton update-data` from "
              "odota/dotaconstants -- do not edit.\n"
              "# Add words to src/automaton_attack_bot/data/custom_vocab.txt "
              "instead.\n")
    path.write_text(header + "\n".join(entries) + "\n", encoding="utf-8")
    print(f"  {len(entries)} vocabulary entries -> {path}")
    return path


# ---------------------------------------------------------------------------
# Phrases
# ---------------------------------------------------------------------------
def _acceptable_phrase(text: str) -> bool:
    if not PHRASE_MIN_LENGTH <= len(text) <= PHRASE_MAX_LENGTH:
        return False
    if not PHRASE_ALLOWED.match(text):
        return False
    return any(ch.isalpha() for ch in text)


def build_phrases() -> list[str]:
    sql = _fetch(DOTABASE_SQL_URL, "dotabase voice lines (~70 MB, one-time)")
    print("  building phrase corpus (a minute or so) ...", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "dotabase.db"
        connection = sqlite3.connect(db)
        connection.executescript(sql.decode("utf-8", errors="ignore"))
        rows = connection.execute("SELECT text FROM responses").fetchall()
        connection.close()

    seen: dict[str, str] = {}
    for (text,) in rows:
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not _acceptable_phrase(text):
            continue
        key = "".join(c for c in text.upper() if c.isalnum())
        seen.setdefault(key, text)
    phrases = sorted(seen.values(), key=str.lower)

    if PHRASE_CANARY not in phrases:
        raise RuntimeError(
            f"phrase filter dropped the canary line {PHRASE_CANARY!r}; "
            "refusing to write a corpus that is provably missing lines")
    return phrases


def write_phrases() -> Path:
    phrases = build_phrases()
    path = phrases_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (f"{VERSION_PREFIX} {CORPUS_VERSION}\n"
              "# Generated by `automaton update-data` from mdiller/dotabase "
              "-- do not edit.\n")
    path.write_text(header + "\n".join(phrases) + "\n", encoding="utf-8")
    print(f"  {len(phrases)} voice lines -> {path}")
    return path


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def _cache_current(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open(encoding="utf-8") as handle:
            first = handle.readline().strip()
    except OSError:
        return False
    return first == f"{VERSION_PREFIX} {CORPUS_VERSION}"


def ensure(with_phrases: bool = True, force: bool = False) -> tuple[Path, Path | None]:
    """Return corpus paths, building anything missing or stale.

    First run downloads and builds; after that it is a pair of header
    checks. A cache written by an older builder is rebuilt, not trusted.
    """
    need_vocab = force or not _cache_current(vocab_path())
    need_phrases = with_phrases and (force or not _cache_current(phrases_path()))
    if need_vocab or need_phrases:
        print("Corpus data missing -- fetching. " + ATTRIBUTION)
        try:
            if need_vocab:
                write_vocab()
            if need_phrases:
                write_phrases()
        except OSError as exc:
            raise RuntimeError(
                f"could not build the word corpus ({exc}). The bot needs one "
                "successful `automaton update-data` run with network access; "
                "after that everything is cached locally."
            ) from exc
    return vocab_path(), phrases_path() if with_phrases else None

"""Word lists and fuzzy matching.

Two corpora, consulted in order:

* the vocabulary -- ~1.5k hero/item/ability names and Dota terms, built from
  odota/dotaconstants (plus ``data/custom_vocab.txt`` adjustments);
* the phrase corpus -- ~39k in-game voice lines, built from mdiller/dotabase.

Both are downloaded and cached by :mod:`automaton_attack.corpus` on first use.

Everything is matched on a *key*: the uppercase, letters-and-digits-only form
of the string. The minigame lets you skip spaces and punctuation while typing,
so they carry no information for us and only add noise to the fuzzy score.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Literal

from . import corpus
from .config import Matching

Source = Literal["vocab", "phrase", "fallback"]


def to_key(text: str) -> str:
    """Uppercase, letters/digits only -- the form everything is matched on."""
    return "".join(c for c in text.upper() if c.isalnum())


@dataclass(frozen=True)
class Match:
    name: str
    score: float
    source: Source

    @property
    def keystrokes(self) -> str:
        """What actually gets typed: lowercase letters and digits only."""
        return "".join(c for c in self.name.lower() if c.isalnum())


def _read_entries(path: Path) -> list[str]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            entries.append(line)
    return entries


class Lexicon:
    """Fuzzy-matches OCR output against the vocab, then the phrase corpus."""

    def __init__(
        self,
        vocab: Iterable[str],
        phrases: Iterable[str] = (),
        matching: Matching | None = None,
    ) -> None:
        self.matching = matching or Matching()
        self.vocab: list[tuple[str, str]] = [
            (name.upper(), to_key(name)) for name in vocab
        ]
        phrase_names = [p.upper() for p in phrases]
        self.phrase_names: list[str] = phrase_names
        self.phrase_keys: list[str] = [to_key(p) for p in phrase_names]

    # -- construction ----------------------------------------------------
    @classmethod
    def load(
        cls,
        vocab_path: str | Path | None = None,
        phrases_path: str | Path | None = None,
        matching: Matching | None = None,
        with_phrases: bool = True,
    ) -> Lexicon:
        """Load the cached corpora (fetching on first use), or explicit files."""
        if vocab_path is None and phrases_path is None:
            vocab_file, phrase_file = corpus.ensure(with_phrases=with_phrases)
        else:
            vocab_file = Path(vocab_path) if vocab_path else corpus.ensure(False)[0]
            phrase_file = Path(phrases_path) if phrases_path else None
        if not vocab_file.exists():
            raise FileNotFoundError(f"vocabulary not found: {vocab_file}")
        phrases: list[str] = []
        if with_phrases and phrase_file and phrase_file.exists():
            phrases = _read_entries(phrase_file)
        return cls(_read_entries(vocab_file), phrases, matching)

    def __len__(self) -> int:
        return len(self.vocab)

    # -- matching --------------------------------------------------------
    def match_vocab(self, raw: str) -> tuple[str | None, float]:
        """Best core-vocabulary match for an OCR read.

        difflib rather than rapidfuzz on purpose: the accept threshold of 0.62
        was tuned against SequenceMatcher's ratio, and the two scorers do not
        agree closely enough to reuse the number. The corpus is small and the
        length prefilter skips most of it, so this is not the hot path -- OCR
        is.
        """
        key = to_key(raw)
        if len(key) < self.matching.min_ocr_length:
            return None, 0.0
        best, best_score = None, 0.0
        sm = SequenceMatcher(None, key)
        tolerance = max(3, len(key) // 3)
        for name, entry_key in self.vocab:
            if abs(len(entry_key) - len(key)) > tolerance:
                continue
            sm.set_seq2(entry_key)
            # Both are cheap upper bounds on ratio(); skip anything that
            # cannot beat the incumbent.
            if (sm.real_quick_ratio() <= best_score
                    or sm.quick_ratio() <= best_score):
                continue
            score = sm.ratio()
            if score > best_score:
                best, best_score = name, score
        if best is not None:
            best = self._prefer_extension(key, best, best_score)
        return best, best_score

    def _prefer_extension(self, read_key: str, best: str,
                          best_score: float) -> str:
        """Swap a near-tied match for the longest entry whose key extends it.

        Typing WEAVER when the word is WEAVE still completes WEAVE at the
        fifth letter -- the trailing key is a harmless stray -- while typing
        WEAVE when the word is WEAVER leaves it one letter short, and an
        unfinished word is a lost multiplier. So when the OCR read is
        ambiguous between a word and its extension (scores within
        extension_epsilon), the extension is the safe choice: its keystrokes
        cover both. The reported score stays the best match's, since that is
        the evidence something matched at all.
        """
        best_key = to_key(best)
        chosen, chosen_len = best, len(best_key)
        floor = best_score - self.matching.extension_epsilon
        sm = SequenceMatcher(None, read_key)
        for name, entry_key in self.vocab:
            if len(entry_key) <= chosen_len:
                continue
            if not entry_key.startswith(best_key):
                continue
            sm.set_seq2(entry_key)
            if sm.ratio() >= floor:
                chosen, chosen_len = name, len(entry_key)
        return chosen

    def match_phrase(self, raw: str) -> tuple[str | None, float]:
        """Best voice-line match. Deliberately strict -- see phrase_cutoff."""
        if not self.phrase_keys:
            return None, 0.0
        key = to_key(raw)
        if len(key) < self.matching.phrase_min_length:
            return None, 0.0
        from rapidfuzz import fuzz, process

        hit = process.extractOne(
            key, self.phrase_keys, scorer=fuzz.ratio,
            score_cutoff=self.matching.phrase_cutoff * 100,
        )
        if hit is None:
            return None, 0.0
        return self.phrase_names[hit[2]], hit[1] / 100.0

    def match(self, raw: str, allow_fallback: bool = True) -> Match | None:
        """Resolve an OCR read to something typeable, or None.

        Tiers: core vocab, then the phrase corpus, then -- if the read is long
        and clean enough -- the read itself. Corpus matches are typed on
        sight; fallback matches (score 0.0, source "fallback") are held by
        the engine until the identical read repeats on consecutive scans.
        """
        name, score = self.match_vocab(raw)
        if not name or score < self.matching.vocab_cutoff:
            phrase_name, phrase_score = self.match_phrase(raw)
            if phrase_name and phrase_score > score:
                return Match(phrase_name, phrase_score, "phrase")
        if name and score >= self.matching.vocab_cutoff:
            return Match(name, score, "vocab")
        if allow_fallback and len(to_key(raw)) >= self.matching.fallback_min_length:
            return Match(raw.upper(), 0.0, "fallback")
        return None

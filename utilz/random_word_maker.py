from __future__ import annotations

import functools
import itertools
from pathlib import Path
import secrets
import string

VOWELS = "aeiou"
CONSONANTS = "".join(ch for ch in string.ascii_lowercase if ch not in VOWELS)
SYSTEM_WORDNET_NOUN_PATHS: tuple[Path, ...] = (
    Path("/usr/share/wordnet/index.noun"),
    Path("/usr/share/wordnet/dict/index.noun"),
)
SYSTEM_DICTIONARY_PATHS: tuple[Path, ...] = (
    Path("/usr/share/dict/american-english"),
    Path("/usr/share/dict/british-english"),
    Path("/usr/share/dict/words"),
)
WORD_PATTERNS: tuple[tuple[str, bool], ...] = (
    ("CVC", False),
    ("CVCV", False),
    ("CVC", True),
    ("CVCV", True),
)


def pattern_capacity(pattern: str, with_digit: bool = False) -> int:
    capacity = 1
    for symbol in pattern:
        if symbol == "C":
            capacity *= len(CONSONANTS)
        elif symbol == "V":
            capacity *= len(VOWELS)
        else:
            raise ValueError(f"Unsupported pattern symbol: {symbol}")
    return capacity * (10 if with_digit else 1)


def suffix_from_index(pattern: str, index: int, with_digit: bool = False) -> str:
    if with_digit:
        digit = str(index % 10)
        index //= 10
    else:
        digit = ""

    chars = []
    for symbol in reversed(pattern):
        alphabet = CONSONANTS if symbol == "C" else VOWELS
        index, remainder = divmod(index, len(alphabet))
        chars.append(alphabet[remainder])
    return "".join(reversed(chars)) + digit


def _normalize_word(raw_word: str) -> str:
    word = raw_word.strip().lower().replace("_", "")
    if len(word) < 3 or not word.isalpha():
        return ""
    return word


def _iter_system_wordnet_nouns():
    for path in SYSTEM_WORDNET_NOUN_PATHS:
        if not path.exists():
            continue
        with path.open(encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line or line.startswith("  "):
                    continue
                lemma = _normalize_word(line.split(" ")[0])
                if lemma:
                    yield lemma
        return


def _iter_nltk_wordnet_nouns():
    try:
        from nltk.corpus import wordnet as nltk_wordnet
    except ImportError:
        return

    try:
        for synset in nltk_wordnet.all_synsets("n"):
            for lemma in synset.lemma_names():
                word = _normalize_word(lemma)
                if word:
                    yield word
    except LookupError:
        return


def _iter_system_dictionary_words():
    for path in SYSTEM_DICTIONARY_PATHS:
        if not path.exists():
            continue
        with path.open(encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                word = _normalize_word(line)
                if word:
                    yield word


@functools.lru_cache(maxsize=1)
def _real_word_inventory() -> tuple[str, ...]:
    ordered_words: list[str] = []
    seen: set[str] = set()

    for word in itertools.chain(_iter_system_wordnet_nouns(), _iter_nltk_wordnet_nouns()):
        if word in seen:
            continue
        seen.add(word)
        ordered_words.append(word)

    for word in _iter_system_dictionary_words():
        if word in seen:
            continue
        seen.add(word)
        ordered_words.append(word)

    return tuple(ordered_words)


def ordered_real_word_suffixes():
    yield from _real_word_inventory()


def ordered_word_suffixes():
    yield from ordered_real_word_suffixes()
    for pattern, with_digit in WORD_PATTERNS:
        for index in range(pattern_capacity(pattern, with_digit)):
            yield suffix_from_index(pattern, index, with_digit)


def logical_word_capacity() -> int:
    return len(_real_word_inventory()) + sum(
        pattern_capacity(pattern, with_digit) for pattern, with_digit in WORD_PATTERNS
    )


def random_pronounceable_suffix(min_len: int = 5) -> str:
    length = max(5, int(min_len))
    pattern = "".join("CV"[i % 2] for i in range(length))
    return suffix_from_index(pattern, secrets.randbelow(pattern_capacity(pattern)))

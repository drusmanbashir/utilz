from __future__ import annotations

import secrets
from pathlib import Path
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


def _normalize_word(raw_word: str) -> str:
    word = raw_word.strip().lower().replace("_", "")
    if len(word) < 3 or not word.isalpha():
        return ""
    return word


def _iter_real_words():
    seen = set()
    for path in SYSTEM_WORDNET_NOUN_PATHS + SYSTEM_DICTIONARY_PATHS:
        try:
            handle = path.open(encoding="utf-8", errors="ignore")
        except FileNotFoundError:
            continue
        with handle:
            for line in handle:
                raw = line.split(" ")[0] if "wordnet" in str(path) else line
                word = _normalize_word(raw)
                if word and word not in seen:
                    seen.add(word)
                    yield word


def random_fake_word(min_length: int = 3, max_length: int = 5) -> str:
    length = secrets.randbelow(max_length - min_length + 1) + min_length
    return "".join(
        (CONSONANTS, VOWELS)[i % 2][
            secrets.randbelow(len((CONSONANTS, VOWELS)[i % 2]))
        ]
        for i in range(length)
    )


def random_real_word(min_length: int = 3, max_length: int = 5) -> str:
    words = [word for word in _iter_real_words() if min_length <= len(word) <= max_length]
    if words:
        return secrets.choice(words)
    return random_fake_word(min_length, max_length)

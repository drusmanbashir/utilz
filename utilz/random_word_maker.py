import secrets
from pathlib import Path
import string
from typing import Collection

import yaml

VOWELS = "aeiou"
CONSONANTS = "".join(ch for ch in string.ascii_lowercase if ch not in VOWELS)

_DATA_DIR = Path(__file__).parent.parent
SHORT_NAMES_YAML = _DATA_DIR / "names_short.yaml"
LONG_NAMES_YAML = _DATA_DIR / "names_long.yaml"
# Legacy txt paths kept for backward-compatibility fallback
SHORT_NAMES_PATH = _DATA_DIR / "names_short.txt"
LONG_NAMES_PATH = _DATA_DIR / "names_long.txt"

# Module-level cache so YAML files are only parsed once per process.
_yaml_cache: dict[str, dict] = {}


def _load_yaml(path: Path) -> dict:
    key = str(path)
    if key not in _yaml_cache:
        with path.open(encoding="utf-8") as fh:
            _yaml_cache[key] = yaml.safe_load(fh)
    return _yaml_cache[key]


def _words_from_yaml(yaml_path: Path, bucket: str, min_length: int, max_length: int) -> list[str]:
    data = _load_yaml(yaml_path)
    return [
        w for w in data.get(bucket, [])
        if min_length <= len(w) <= max_length
    ]


def _words_from_txt(txt_path: Path, min_length: int, max_length: int) -> list[str]:
    with txt_path.open(encoding="utf-8", errors="ignore") as fh:
        return [
            line.strip() for line in fh
            if min_length <= len(line.strip()) <= max_length
        ]


def _collect_words(min_length: int, max_length: int, bucket: str) -> list[str]:
    """Return all words in *bucket* ('common' or 'uncommon') within the length range.

    Falls back to reading the legacy ``.txt`` files when the YAML files are absent.
    """
    words: list[str] = []
    if SHORT_NAMES_YAML.exists():
        if min_length <= 5:
            words.extend(_words_from_yaml(SHORT_NAMES_YAML, bucket, min_length, max_length))
        if max_length > 5:
            words.extend(_words_from_yaml(LONG_NAMES_YAML, bucket, min_length, max_length))
    else:
        # Legacy fallback
        if min_length <= 5:
            words.extend(_words_from_txt(SHORT_NAMES_PATH, min_length, max_length))
        if max_length > 5:
            words.extend(_words_from_txt(LONG_NAMES_PATH, min_length, max_length))
    return words


def random_fake_word(min_length: int = 3, max_length: int = 5) -> str:
    length = secrets.randbelow(max_length - min_length + 1) + min_length
    return "".join(
        (CONSONANTS, VOWELS)[i % 2][
            secrets.randbelow(len((CONSONANTS, VOWELS)[i % 2]))
        ]
        for i in range(length)
    )


def random_real_word(
    min_length: int = 3,
    max_length: int = 5,
    caller_context: str = "",
    used_names: Collection[str] | None = None,
) -> str:
    """Return a random real word within [min_length, max_length].

    When *caller_context* and *used_names* are supplied the function first
    tries to pick an unused word from the ``common`` bucket.  Once every
    common word of the requested length has been used it falls through to
    the ``uncommon`` bucket.  If no real word is available at all, a fake
    word is generated as a final fallback.

    Args:
        min_length: Minimum word length (inclusive).
        max_length: Maximum word length (inclusive).
        caller_context: An identifier for the calling project / namespace
            (e.g. a wandb project name).  Used only for clarity; the actual
            exclusion is driven by *used_names*.
        used_names: A collection of names already used within the caller
            context that should not be returned again.
    """
    used: frozenset[str] = frozenset(used_names) if used_names else frozenset()

    for bucket in ("common", "uncommon"):
        candidates = _collect_words(min_length, max_length, bucket)
        eligible = [w for w in candidates if w not in used]
        if eligible:
            return secrets.choice(eligible)

    return random_fake_word(min_length, max_length)

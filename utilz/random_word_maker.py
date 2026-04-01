import secrets
from pathlib import Path
import string

VOWELS = "aeiou"
CONSONANTS = "".join(ch for ch in string.ascii_lowercase if ch not in VOWELS)
SHORT_NAMES_PATH = Path(__file__).parent.parent / "names_short.txt"
LONG_NAMES_PATH = Path(__file__).parent.parent / "names_long.txt"


def random_fake_word(min_length: int = 3, max_length: int = 5) -> str:
    length = secrets.randbelow(max_length - min_length + 1) + min_length
    return "".join(
        (CONSONANTS, VOWELS)[i % 2][
            secrets.randbelow(len((CONSONANTS, VOWELS)[i % 2]))
        ]
        for i in range(length)
    )


def random_real_word(min_length: int = 3, max_length: int = 5) -> str:
    words = []
    if min_length <= 5:
        with SHORT_NAMES_PATH.open(encoding="utf-8", errors="ignore") as handle:
            words.extend(line.strip() for line in handle if min_length <= len(line.strip()) <= max_length)
    if max_length > 5:
        with LONG_NAMES_PATH.open(encoding="utf-8", errors="ignore") as handle:
            words.extend(line.strip() for line in handle if min_length <= len(line.strip()) <= max_length)
    if words:
        return secrets.choice(words)
    return random_fake_word(min_length, max_length)

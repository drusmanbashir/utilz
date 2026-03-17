import secrets
import string

VOWELS = "aeiou"
CONSONANTS = "".join(ch for ch in string.ascii_lowercase if ch not in VOWELS)
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


def ordered_word_suffixes():
    for pattern, with_digit in WORD_PATTERNS:
        for index in range(pattern_capacity(pattern, with_digit)):
            yield suffix_from_index(pattern, index, with_digit)


def logical_word_capacity() -> int:
    return sum(pattern_capacity(pattern, with_digit) for pattern, with_digit in WORD_PATTERNS)


def random_pronounceable_suffix(min_len: int = 5) -> str:
    length = max(5, int(min_len))
    pattern = "".join("CV"[i % 2] for i in range(length))
    return suffix_from_index(pattern, secrets.randbelow(pattern_capacity(pattern)))

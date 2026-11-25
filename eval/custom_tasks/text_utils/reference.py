import re
from collections import Counter


def word_frequency(text: str) -> dict:
    return dict(Counter(re.findall(r"[a-z0-9']+", text.lower())))


def most_common(text: str, n: int) -> list:
    if n < 0:
        raise ValueError("n must be >= 0")
    items = sorted(word_frequency(text).items(), key=lambda kv: (-kv[1], kv[0]))
    return items[:n]


def is_anagram(a: str, b: str) -> bool:
    norm = lambda s: sorted(c for c in s.lower() if not c.isspace())
    return norm(a) == norm(b)

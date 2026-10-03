"""Time-sortable, URL-safe identifiers."""

from __future__ import annotations

import secrets
import time

_ALPHABET = "0123456789abcdefghjkmnpqrstvwxyz"  # Crockford base32, lowercase


def _b32(value: int, length: int) -> str:
    chars = []
    for _ in range(length):
        chars.append(_ALPHABET[value & 31])
        value >>= 5
    return "".join(reversed(chars))


def new_id(prefix: str = "") -> str:
    """Return an id like ``task_01hx3k...`` that sorts by creation time."""
    millis = int(time.time() * 1000)
    rand = secrets.randbits(40)
    body = _b32(millis, 9) + _b32(rand, 8)
    return f"{prefix}_{body}" if prefix else body


def short_id(n: int = 8) -> str:
    return "".join(secrets.choice(_ALPHABET) for _ in range(n))

"""Shared scoring normalization; keep digits, diacritics and word order."""
from __future__ import annotations

import re
import unicodedata
from typing import Any


def normalize_vi(text: str) -> str:
    """Conservative ASR normalization: Unicode, case, punctuation, whitespace.

    Digits and Vietnamese diacritics are deliberately preserved. This makes
    errors in dosages, dates and identifiers visible instead of scoring them
    away.
    """
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = "".join(" " if unicodedata.category(ch)[0] in {"P", "S"} else ch for ch in text)
    return re.sub(r"\s+", " ", text).strip()


def text_from_hypothesis(value: Any) -> str:
    if isinstance(value, str):
        return value
    if hasattr(value, "text"):
        return str(value.text)
    if isinstance(value, (tuple, list)) and value:
        return text_from_hypothesis(value[0])
    return str(value)

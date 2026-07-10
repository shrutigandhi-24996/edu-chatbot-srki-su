"""Lightweight text preprocessing shared across the pipeline."""
from __future__ import annotations

import re

_WHITESPACE = re.compile(r"\s+")


def preprocess(text: str) -> str:
    """Normalize whitespace and strip control characters for model input."""
    if not text:
        return ""
    text = text.replace("\u00a0", " ")
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def normalize(text: str) -> str:
    """Lowercased, punctuation-light form used for keyword matching."""
    text = preprocess(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return _WHITESPACE.sub(" ", text).strip()

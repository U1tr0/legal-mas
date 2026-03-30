from __future__ import annotations

import re
from typing import Any

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s\.-]+", flags=re.UNICODE)
_TOKEN_RE = re.compile(r"[a-zа-я0-9]+(?:[.-][a-zа-я0-9]+)*", flags=re.IGNORECASE)
_ARTICLE_CONTEXT_RE = re.compile(
    r"(?:ч\.\s*\d+\s*)?(?:п\.\s*\d+\s*)?(?:ст(?:атья|\.)?\s*)?(\d+(?:\.\d+)+)",
    flags=re.IGNORECASE,
)


def normalize_text(text: Any) -> str:
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    text = text.lower().replace("ё", "е")
    text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def tokenize_for_bm25(text: Any) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return _TOKEN_RE.findall(normalized)


def normalize_article_number(value: Any) -> str | None:
    if value is None:
        return None

    if not isinstance(value, str):
        value = str(value)

    match = re.search(r"\d+(?:\.\d+)+", value.strip())
    if not match:
        return None

    parts = [str(int(part)) if part.isdigit() else part for part in match.group(0).split(".")]
    return ".".join(parts)


def extract_explicit_article_numbers(text: Any) -> list[str]:
    normalized_text = normalize_text(text)
    if not normalized_text:
        return []

    seen: set[str] = set()
    article_numbers: list[str] = []

    for raw_match in _ARTICLE_CONTEXT_RE.findall(normalized_text):
        normalized = normalize_article_number(raw_match)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        article_numbers.append(normalized)

    return article_numbers

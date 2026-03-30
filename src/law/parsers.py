from __future__ import annotations

import re
from typing import Iterable, List, Optional, Union

from src.utils.schemas import ArticleRef


_CODE_CANON = {
    "коап": "КоАП РФ",
    "ук": "УК РФ",
    "гк": "ГК РФ",
    "тк": "ТК РФ",
    "гпк": "ГПК РФ",
    "апк": "АПК РФ",
}

_SHORT_FORM = re.compile(
    r"""
    (?:
        (?:ч\.?\s*(?P<part>\d+)\s*)?      # ч.4
        (?:ст\.?|статья)\s*              # ст. / статья
        (?P<article>\d+(?:\.\d+)?)        # 12.15
        (?:\s*ч\.?\s*(?P<part2>\d+))?     # ... ч.4 (если часть указана после статьи)
    )
    \s*
    (?P<code>
        КоАП|УК|ГК|ТК|ГПК|АПК
    )
    \s*РФ
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

_LONG_KOAP = re.compile(
    r"""
    (?:част[ьи]\s*(?P<part>\d+)\s*)?
    стать[ьи]\s*(?P<article>\d+(?:\.\d+)?)
    \s*
    (?:Кодекса\s+Российской\s+Федерации\s+об\s+административных\s+правонарушениях)
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

_POINT_FORM = re.compile(
    r"""
    п\.?\s*(?P<point>\d+)\s*
    (?:ч\.?\s*(?P<part>\d+)\s*)?
    (?:ст\.?|статья)\s*(?P<article>\d+(?:\.\d+)?)
    \s*
    (?P<code>КоАП|УК|ГК|ТК|ГПК|АПК)\s*РФ
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


def _canon_code(code_token: str) -> str:
    key = code_token.strip().lower()
    return _CODE_CANON.get(key, code_token.strip() + " РФ")


def _dedupe_keep_order(items: List[ArticleRef]) -> List[ArticleRef]:
    seen = set()
    out: List[ArticleRef] = []
    for a in items:
        k = (a.code, a.article, a.part, a.point)
        if k in seen:
            continue
        seen.add(k)
        out.append(a)
    return out


def parse_articles(raw: Union[str, Iterable[str]]) -> List[ArticleRef]:
    """Извлекает ссылки на статьи из строки или набора текстовых фрагментов."""
    if raw is None:
        return []

    if isinstance(raw, str):
        texts = [raw]
    else:
        texts = [str(x) for x in raw if x]

    found: List[ArticleRef] = []

    for text in texts:
        t = " ".join(text.split())  # Упрощаем пробелы перед поиском по шаблонам.

        for m in _POINT_FORM.finditer(t):
            code = _canon_code(m.group("code"))
            article = m.group("article")
            part = m.group("part")
            point = m.group("point")
            found.append(ArticleRef(code=code, article=article, part=part, point=point, raw=m.group(0)))

        for m in _SHORT_FORM.finditer(t):
            code = _canon_code(m.group("code"))
            article = m.group("article")
            part = m.group("part") or m.group("part2")
            found.append(ArticleRef(code=code, article=article, part=part, point=None, raw=m.group(0)))

        for m in _LONG_KOAP.finditer(t):
            code = "КоАП РФ"
            article = m.group("article")
            part = m.group("part")
            found.append(ArticleRef(code=code, article=article, part=part, point=None, raw=m.group(0)))

    return _dedupe_keep_order(found)

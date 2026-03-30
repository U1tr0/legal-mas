from __future__ import annotations

import argparse
import json
import os
import re
from datetime import date
from typing import Any, Iterable, Optional

import pandas as pd

from src.law.parsers import parse_articles


MONTH_MAP = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
}


def load_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise RuntimeError(f"JSONL parse error on line {line_no}: {e}") from e


def normalize_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_decision_date_from_text(full_text: Optional[str]) -> Optional[date]:
    if not full_text:
        return None

    m = re.search(
        r"\b(\d{1,2})\s+"
        r"(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)"
        r"\s+(\d{4})\s+года\b",
        full_text,
        flags=re.IGNORECASE,
    )
    if not m:
        return None

    day = int(m.group(1))
    month = MONTH_MAP.get(m.group(2).lower())
    year = int(m.group(3))

    if not month:
        return None

    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_court_name_from_text(full_text: Optional[str]) -> Optional[str]:
    if not full_text:
        return None

    head = full_text[:2500].upper()
    if "ВЕРХОВНЫЙ СУД" in head and "РОССИЙСКОЙ ФЕДЕРАЦИИ" in head:
        return "Верховный Суд РФ"

    return None


def _clean_spaces(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clip_by_stoppers(text: str, max_len: int = 1800) -> str:
    stoppers = [
        r"\bСуд\b.*\bсчитает\b",
        r"\bСуд\b.*\bприходит к выводу\b",
        r"\bПроверив материалы\b",
        r"\bОценив доводы\b",
        r"\bРуководствуясь\b",
        r"\bНа основании изложенного\b",
        r"\bПри таких обстоятельствах\b",
        r"\bДоводы\b.*\bжалоб[ыа]\b",
        r"\bСогласно статье\b",
        r"\bВ силу\b",
        r"\bПОСТАНОВИЛ\b",
        r"\bОПРЕДЕЛИЛ\b",
        r"\bРЕШИЛ\b",
        r"\bоставить\s+без\s+изменения\b",
        r"\bбез\s+удовлетворения\b",
    ]

    cut: Optional[int] = None
    for pat in stoppers:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if not m:
            continue

        pos = m.start()
        if pos >= 200:
            cut = pos if cut is None else min(cut, pos)

    if cut is not None:
        text = text[:cut]

    return text[:max_len].strip()


def extract_summary_v2(full_text: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Вытаскивает фактический блок и обрывает текст до рассуждений суда."""
    if not full_text:
        return None, None

    text = _clean_spaces(full_text)

    patterns = [
        (
            r"(Как\s+усматривается\s+из\s+материалов\s+дела(?:\s+об\s+административном\s+правонарушении)?[\s\S]{100,5000}?)"
            r"(?=\n\s*(УСТАНОВИЛ|УСТАНОВИЛА|ПОСТАНОВИЛ|ОПРЕДЕЛИЛ|РЕШИЛ|Проверив|Оценив|Руководствуясь|$))",
            "materials_block",
            200,
            False,
        ),
        (
            r"(Как\s+следует\s+из\s+материалов\s+дела(?:\s+об\s+административном\s+правонарушении)?[\s\S]{100,5000}?)"
            r"(?=\n\s*(УСТАНОВИЛ|УСТАНОВИЛА|ПОСТАНОВИЛ|ОПРЕДЕЛИЛ|РЕШИЛ|Проверив|Оценив|Руководствуясь|$))",
            "materials_follows_as_block",
            200,
            False,
        ),
        (
            r"(Из\s+материалов\s+дела\s+следует[\s\S]{150,3500}?)"
            r"(?=\n\s*(Проверив|Оценив|Суд\s+считает|Руководствуясь|ПОСТАНОВИЛ|ОПРЕДЕЛИЛ|РЕШИЛ|$))",
            "materials_follows_block",
            200,
            False,
        ),
        (
            r"((?:Согласно|В\s+соответствии\s+с)\s+материалам\s+дела[\s\S]{120,5000}?)"
            r"(?=\n\s*(Проверив|Оценив|Суд\s+считает|Руководствуясь|ПОСТАНОВИЛ|ОПРЕДЕЛИЛ|РЕШИЛ|$))",
            "materials_according_block",
            200,
            False,
        ),
        (
            r"(Установлено,\s+что[\s\S]{150,4000}?)"
            r"(?=\n\s*(Проверив|Оценив|Суд\s+считает|Руководствуясь|ПОСТАНОВИЛ|ОПРЕДЕЛИЛ|РЕШИЛ|$))",
            "ustanovleno_block",
            200,
            False,
        ),
        (
            r"((?:Основанием|Основаниями)\s+для\s+"
            r"(?:привлечения|возбуждения|наступления)"
            r"[\s\S]{0,500}?"
            r"(?:послужил[ао]?|явил(?:ось|ся)|стало)"
            r"[\s\S]{0,80}?"
            r"то,\s+что[\s\S]{120,4500}?)"
            r"(?=(?:\n\s*)?(Фактические\s+обстоятельства|Проверив|Оценив|Суд\s+считает|Руководствуясь|ПОСТАНОВИЛ|ОПРЕДЕЛИЛ|РЕШИЛ|$))",
            "basis_liability_block",
            200,
            False,
        ),
        (
            r"(УСТАНОВИЛ[А]?\s*:?\s*[\s\S]{200,6000}?)"
            r"(?=\n\s*(ПОСТАНОВИЛ|ОПРЕДЕЛИЛ|РЕШИЛ|Руководствуясь|На\s+основании|$))",
            "ustanovil_block",
            250,
            True,
        ),
    ]

    for pattern, source, min_len, strip_header in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if not m:
            continue

        summary = m.group(1).strip()

        if strip_header:
            summary = re.sub(r"^УСТАНОВИЛ[А]?\s*:?\s*", "", summary, flags=re.IGNORECASE)

        summary = " ".join(summary.split())
        summary = _clip_by_stoppers(summary, max_len=1800)

        if len(summary) >= min_len:
            return summary, source

    return None, None


def raw_to_record(raw: dict[str, Any]) -> Optional[dict[str, Any]]:
    case_id = raw.get("case_id") or raw.get("id")
    if not case_id:
        return None

    full_text = normalize_text(raw.get("full_text"))

    site_meta = raw.get("site_meta")
    site_articles = site_meta.get("articles") if isinstance(site_meta, dict) else None

    articles_raw = []
    if isinstance(site_articles, list):
        seen = set()
        for article in site_articles:
            article = " ".join(str(article).split())
            if article and article not in seen:
                articles_raw.append(article)
                seen.add(article)

    articles_norm = parse_articles(articles_raw)
    decision_date = parse_decision_date_from_text(full_text)
    court_name = parse_court_name_from_text(full_text)
    summary, summary_source = extract_summary_v2(full_text)

    return {
        "case_id": str(case_id),
        "source": raw.get("source") or "vsrf",
        "source_url": raw.get("source_url"),
        "pdf_url": raw.get("pdf_url"),
        "pdf_path": raw.get("pdf_path"),
        "document_type": raw.get("document_type"),
        "list_date_raw": raw.get("list_date_raw"),
        "decision_date": decision_date.isoformat() if decision_date else None,
        "court_name": court_name,
        "articles_raw": articles_raw,
        "articles_norm": [a.model_dump() for a in articles_norm],
        "full_text": full_text,
        "summary": summary,
        "summary_source": summary_source,
        "created_at": raw.get("created_at"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/cases_raw.jsonl")
    ap.add_argument("--output", default="data/processed/cases.parquet")
    ap.add_argument("--drop-no-text", action="store_true", help="Drop records where full_text is empty")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    records: list[dict[str, Any]] = []
    stats = {
        "total": 0,
        "kept": 0,
        "dropped_no_id": 0,
        "dropped_no_text": 0,
        "empty_articles_norm": 0,
        "summary_is_null": 0,
    }

    for raw in load_jsonl(args.input):
        stats["total"] += 1
        rec = raw_to_record(raw)

        if rec is None:
            stats["dropped_no_id"] += 1
            continue

        if args.drop_no_text and not rec.get("full_text"):
            stats["dropped_no_text"] += 1
            continue

        if not rec.get("articles_norm"):
            stats["empty_articles_norm"] += 1

        if rec.get("summary") is None:
            stats["summary_is_null"] += 1

        records.append(rec)
        stats["kept"] += 1

    df = pd.DataFrame(records)

    print("=== PREPARE DATASET STATS ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

    if not df.empty:
        for col in ["full_text", "summary"]:
            if col in df.columns:
                lens = df[col].fillna("").map(lambda x: len(x) if isinstance(x, str) else 0)
                print(
                    f"{col}: mean={lens.mean():.1f} "
                    f"p50={lens.quantile(0.5):.0f} "
                    f"p95={lens.quantile(0.95):.0f} "
                    f"max={lens.max():.0f}"
                )

        if "summary_source" in df.columns:
            print("\nsummary_source distribution (incl NULL):")
            print(df["summary_source"].fillna("NULL").value_counts())

    df.to_parquet(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

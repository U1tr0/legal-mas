from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd


def safe_text(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, str):
        x = x.strip()
        return x if x else None
    x = str(x).strip()
    return x if x else None


def parse_articles_norm(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []

    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]

    if hasattr(value, "tolist"):
        try:
            converted = value.tolist()
            if isinstance(converted, list):
                return [x for x in converted if isinstance(x, dict)]
        except Exception:
            pass

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict)]
        except Exception:
            return []

    return []


def format_articles_for_text(articles_norm: list[dict[str, Any]]) -> list[str]:
    parts: list[str] = []

    for article in articles_norm:
        code = safe_text(article.get("code"))
        article_num = safe_text(article.get("article"))
        part = safe_text(article.get("part"))
        point = safe_text(article.get("point"))

        if not code or not article_num:
            continue

        chunk = []
        if point:
            chunk.append(f"п. {point}")
        if part:
            chunk.append(f"ч. {part}")
        chunk.append(f"ст. {article_num}")
        chunk.append(code)

        parts.append(" ".join(chunk))

    seen = set()
    uniq = []
    for x in parts:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def build_case_title(row: pd.Series) -> str:
    case_id = safe_text(row.get("case_id")) or "без номера"
    court_name = safe_text(row.get("court_name"))
    decision_date = safe_text(row.get("decision_date"))

    parts = [f"Дело {case_id}"]
    if court_name:
        parts.append(court_name)
    if decision_date:
        parts.append(decision_date)

    return " | ".join(parts)


def build_case_retrieval_text(row: pd.Series) -> str:
    chunks: list[str] = []

    case_id = safe_text(row.get("case_id"))
    court_name = safe_text(row.get("court_name"))
    decision_date = safe_text(row.get("decision_date"))
    summary = safe_text(row.get("summary"))
    full_text = safe_text(row.get("full_text"))

    articles_norm = parse_articles_norm(row.get("articles_norm"))
    article_strings = format_articles_for_text(articles_norm)

    if case_id:
        chunks.append(f"Дело {case_id}")
    if court_name:
        chunks.append(court_name)
    if decision_date:
        chunks.append(f"Дата решения: {decision_date}")
    if article_strings:
        chunks.append("Статьи: " + "; ".join(article_strings))
    if summary:
        chunks.append("Фабула дела:")
        chunks.append(summary)
    elif full_text:
        chunks.append("Текст документа:")
        chunks.append(full_text[:3000])

    return "\n".join(chunks).strip()


def build_case_record(row: pd.Series) -> dict[str, Any]:
    case_id = safe_text(row.get("case_id"))
    source = safe_text(row.get("source")) or "vsrf"
    summary = safe_text(row.get("summary"))
    full_text = safe_text(row.get("full_text"))

    articles_norm = parse_articles_norm(row.get("articles_norm"))
    article_strings = format_articles_for_text(articles_norm)

    return {
        "doc_id": f"case_{case_id}" if case_id else None,
        "doc_type": "case",
        "title": build_case_title(row),
        "retrieval_text": build_case_retrieval_text(row),
        "case_id": case_id,
        "source": source,
        "source_url": safe_text(row.get("source_url")),
        "pdf_url": safe_text(row.get("pdf_url")),
        "pdf_path": safe_text(row.get("pdf_path")),
        "document_type": safe_text(row.get("document_type")),
        "decision_date": safe_text(row.get("decision_date")),
        "court_name": safe_text(row.get("court_name")),
        "articles_norm": articles_norm,
        "articles_text": article_strings,
        "summary": summary,
        "full_text": full_text,
        "summary_status": safe_text(row.get("summary_status")),
        "summary_confidence": row.get("summary_confidence"),
    }


def save_jsonl(df: pd.DataFrame, path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    if hasattr(value, "tolist"):
        try:
            converted = value.tolist()
            if isinstance(converted, list):
                return [str(x).strip() for x in converted if str(x).strip()]
        except Exception:
            pass

    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []

    return []

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="data/processed/cases_with_summaries.parquet",
        help="Input parquet with processed cases and summaries",
    )
    ap.add_argument(
        "--output-parquet",
        default="data/processed/case_index.parquet",
        help="Output parquet path",
    )
    ap.add_argument(
        "--output-jsonl",
        default="data/processed/case_index.jsonl",
        help="Output jsonl path",
    )
    ap.add_argument(
        "--drop-no-retrieval-text",
        action="store_true",
        help="Drop rows where retrieval_text is empty",
    )
    args = ap.parse_args()

    df = pd.read_parquet(args.input)

    records = [build_case_record(row) for _, row in df.iterrows()]
    out_df = pd.DataFrame(records)

    if "articles_text" in out_df.columns:
        out_df["articles_text"] = out_df["articles_text"].apply(normalize_string_list)

    if args.drop_no_retrieval_text:
        out_df = out_df[out_df["retrieval_text"].fillna("").str.strip() != ""].copy()

    Path(args.output_parquet).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output_parquet, index=False)
    save_jsonl(out_df, args.output_jsonl)

    print("=== CASE INDEX PREPARED ===")
    print(f"Input: {args.input}")
    print(f"Rows: {len(out_df)}")
    print(f"Output parquet: {args.output_parquet}")
    print(f"Output jsonl: {args.output_jsonl}")
    print("\nColumns:")
    print(out_df.columns.tolist())

    empty_retrieval = int((out_df["retrieval_text"].fillna("").str.strip() == "").sum())
    empty_summary = int((out_df["summary"].fillna("").str.strip() == "").sum())

    print("\nSanity checks:")
    print(f"Empty retrieval_text: {empty_retrieval}")
    print(f"Empty summary: {empty_summary}")


if __name__ == "__main__":
    main()

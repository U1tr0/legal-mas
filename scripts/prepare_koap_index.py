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


def build_law_title(row: pd.Series) -> str:
    article_label = safe_text(row.get("article_label"))
    code_name = safe_text(row.get("code_name")) or "КоАП РФ"

    if article_label:
        return f"{code_name} | {article_label}"
    return code_name


def build_law_retrieval_text(row: pd.Series) -> str:
    chunks: list[str] = []

    code_name = safe_text(row.get("code_name"))
    section_title = safe_text(row.get("section_title"))
    chapter_title = safe_text(row.get("chapter_title"))
    article_label = safe_text(row.get("article_label"))
    article_text = safe_text(row.get("article_text"))

    if code_name:
        chunks.append(code_name)
    if section_title:
        chunks.append(section_title)
    if chapter_title:
        chunks.append(chapter_title)
    if article_label:
        chunks.append(article_label)
    if article_text:
        chunks.append(article_text)

    return "\n".join(chunks).strip()


def build_law_record(row: pd.Series) -> dict[str, Any]:
    article_id = safe_text(row.get("article_id"))
    article_number = safe_text(row.get("article_number"))
    article_label = safe_text(row.get("article_label"))
    article_title = safe_text(row.get("article_title"))
    article_text = safe_text(row.get("article_text"))

    return {
        "doc_id": article_id,
        "doc_type": "law_article",
        "title": build_law_title(row),
        "retrieval_text": build_law_retrieval_text(row),
        "code_name": safe_text(row.get("code_name")) or "КоАП РФ",
        "article_id": article_id,
        "article_number": article_number,
        "article_label": article_label,
        "article_title": article_title,
        "article_text": article_text,
        "chapter_title": safe_text(row.get("chapter_title")),
        "section_title": safe_text(row.get("section_title")),
        "source_file": safe_text(row.get("source_file")),
        "is_text_missing": not bool(article_text),
    }


def save_jsonl(df: pd.DataFrame, path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="data/processed/koap_articles.parquet",
        help="Input parquet with parsed КоАП articles",
    )
    ap.add_argument(
        "--output-parquet",
        default="data/processed/koap_index.parquet",
        help="Output parquet path",
    )
    ap.add_argument(
        "--output-jsonl",
        default="data/processed/koap_index.jsonl",
        help="Output jsonl path",
    )
    ap.add_argument(
        "--drop-empty-article-text",
        action="store_true",
        help="Drop rows where article_text is empty",
    )
    args = ap.parse_args()

    df = pd.read_parquet(args.input)

    records = [build_law_record(row) for _, row in df.iterrows()]
    out_df = pd.DataFrame(records)

    if args.drop_empty_article_text:
        out_df = out_df[out_df["article_text"].fillna("").str.strip() != ""].copy()

    Path(args.output_parquet).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output_parquet, index=False)
    save_jsonl(out_df, args.output_jsonl)

    print("=== KOAP INDEX PREPARED ===")
    print(f"Input: {args.input}")
    print(f"Rows: {len(out_df)}")
    print(f"Output parquet: {args.output_parquet}")
    print(f"Output jsonl: {args.output_jsonl}")
    print("\nColumns:")
    print(out_df.columns.tolist())

    empty_retrieval = int((out_df["retrieval_text"].fillna("").str.strip() == "").sum())
    empty_article_text = int((out_df["article_text"].fillna("").str.strip() == "").sum())

    print("\nSanity checks:")
    print(f"Empty retrieval_text: {empty_retrieval}")
    print(f"Empty article_text: {empty_article_text}")


if __name__ == "__main__":
    main()
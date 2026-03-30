from __future__ import annotations

import argparse
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import pandas as pd


ARTICLE_RE = re.compile(r"^Статья\s+(\d+(?:\.\d+)*)\.\s+(.+)$")
CHAPTER_RE = re.compile(r"^Глава\s+.+$")
SECTION_RE = re.compile(r"^Раздел\s+.+$")


def read_odt_text(path: str) -> str:
    with zipfile.ZipFile(path, "r") as z:
        xml_data = z.read("content.xml")

    root = ET.fromstring(xml_data)
    parts: list[str] = []

    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag in {"h", "p"}:
            text = "".join(elem.itertext()).strip()
            if text:
                parts.append(text)

    return "\n".join(parts)


def is_garant_line(line: str) -> bool:
    return line == "ГАРАНТ:" or line.startswith("См. ")


def is_service_article_title(article_title: str) -> bool:
    text = article_title.strip().lower()

    service_starts = (
        "изменена ",
        "изменен ",
        "утратила силу",
        "утратил силу",
        "введена ",
        "введен ",
        "дополнена ",
        "дополнен ",
        "исключена ",
        "исключен ",
    )

    return text.startswith(service_starts)


def clean_article_lines(lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    skip_info_block = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if is_garant_line(line):
            continue

        if line.startswith("Информация об изменениях:"):
            skip_info_block = True
            continue

        if skip_info_block:
            if re.match(r"^(Федеральным законом|Часть \d+|Пункт \d+|Абзац|Подпункт)", line):
                continue

            if re.match(r"^(\d+[\.\)]|[1-9]\)|Примечание\.|[А-ЯЁ])", line):
                skip_info_block = False
            else:
                continue

        cleaned.append(line)

    return cleaned


def finalize_article_text(lines: list[str]) -> str:
    lines = clean_article_lines(lines)
    return "\n".join(lines).strip()


def parse_koap_articles(text: str) -> list[dict[str, Optional[str]]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    records: list[dict[str, Optional[str]]] = []
    current_section: Optional[str] = None
    current_chapter: Optional[str] = None
    current_article: Optional[dict[str, Optional[str]]] = None
    current_article_lines: list[str] = []

    for line in lines:
        if SECTION_RE.match(line):
            current_section = line
            continue

        if CHAPTER_RE.match(line):
            current_chapter = line
            continue

        article_match = ARTICLE_RE.match(line)
        if article_match:
            article_number = article_match.group(1).strip()
            article_title = article_match.group(2).strip()

            if is_service_article_title(article_title):
                continue

            if current_article is not None:
                current_article["article_text"] = finalize_article_text(current_article_lines)
                records.append(current_article)

            current_article = {
                "article_id": f"koap_{article_number}",
                "article_number": article_number,
                "article_label": f"Статья {article_number}. {article_title}",
                "article_title": article_title,
                "article_text": "",
                "chapter_title": current_chapter,
                "section_title": current_section,
                "code_name": "КоАП РФ",
                "source_file": None,
            }
            current_article_lines = []
            continue

        if current_article is None:
            continue

        current_article_lines.append(line)

    if current_article is not None:
        current_article["article_text"] = finalize_article_text(current_article_lines)
        records.append(current_article)

    return records


def build_articles_dataframe(records: list[dict[str, Optional[str]]], source_file: str) -> pd.DataFrame:
    df = pd.DataFrame(records)

    if df.empty:
        return df

    df["source_file"] = source_file
    df["article_text"] = df["article_text"].fillna("").map(lambda x: x.strip())

    df = df[
        [
            "article_id",
            "code_name",
            "article_number",
            "article_label",
            "article_title",
            "article_text",
            "chapter_title",
            "section_title",
            "source_file",
        ]
    ].copy()

    return df


def save_jsonl(df: pd.DataFrame, path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to .odt file with КоАП РФ")
    ap.add_argument(
        "--output-parquet",
        default="data/processed/koap_articles.parquet",
        help="Output parquet path",
    )
    ap.add_argument(
        "--output-jsonl",
        default="data/processed/koap_articles.jsonl",
        help="Output jsonl path",
    )
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    text = read_odt_text(str(input_path))
    records = parse_koap_articles(text)
    df = build_articles_dataframe(records, source_file=input_path.name)

    if df.empty:
        raise RuntimeError("No articles were parsed from the input file.")

    Path(args.output_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output_parquet, index=False)
    save_jsonl(df, args.output_jsonl)

    print("=== KOAP ARTICLES PREPARED ===")
    print(f"Input: {input_path}")
    print(f"Articles parsed: {len(df)}")
    print(f"Output parquet: {args.output_parquet}")
    print(f"Output jsonl: {args.output_jsonl}")
    print("\nColumns:")
    print(df.columns.tolist())

    empty_text_count = int((df["article_text"].fillna("").str.strip() == "").sum())
    dup_count = int(df["article_number"].duplicated().sum())

    print("\nSanity checks:")
    print(f"Empty article_text: {empty_text_count}")
    print(f"Duplicate article_number: {dup_count}")

    print("\nFirst 3 articles:")
    preview_cols = ["article_number", "article_title", "chapter_title", "section_title"]
    print(df[preview_cols].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any, Dict, Optional

import pandas as pd
from tqdm import tqdm

from src.llm.deepseek_client import DeepSeekClient
from src.tasks.summary_generator import SummaryGenerator

from dotenv import load_dotenv
load_dotenv()


def safe_text(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, str):
        x = x.strip()
        return x if x else None
    return str(x).strip() or None


def normalize_summary_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Нормализует ответ модели и отсекает непригодные краткие фабулы."""
    summary = result.get("summary")
    evidence = result.get("evidence")
    status = result.get("status")
    confidence = result.get("confidence")
    model_name = result.get("model_name")
    raw_response = result.get("raw_response")

    bad_summaries = {
        "лицо совершило деяние, которое послужило основанием для привлечения к ответственности.",
        "лицо совершило правонарушение.",
        "было совершено деяние.",
        "совершено административное правонарушение.",
    }

    if isinstance(summary, str):
        summary_clean = " ".join(summary.split()).strip()
    else:
        summary_clean = None

    if isinstance(evidence, str):
        evidence_clean = " ".join(evidence.split()).strip()
    else:
        evidence_clean = None

    if summary_clean:
        if summary_clean.lower() in bad_summaries:
            summary_clean = None
            status = "null"

        if len(summary_clean) < 30:
            summary_clean = None
            status = "null"

    if status != "ok":
        summary_clean = None

    return {
        "summary": summary_clean,
        "summary_evidence": evidence_clean,
        "summary_status": status,
        "summary_confidence": confidence,
        "summary_model_name": model_name,
        "summary_raw_response": raw_response,
    }


def build_error_result(model_name: str, error: str) -> Dict[str, Any]:
    return {
        "summary": None,
        "summary_evidence": None,
        "summary_status": "error",
        "summary_confidence": 0.0,
        "summary_model_name": model_name,
        "summary_raw_response": error,
    }


def process_row(
    idx: int,
    full_text: Optional[str],
    api_key: str,
    model_name: str,
    sleep_sec: float,
) -> tuple[int, Dict[str, Any]]:
    client = DeepSeekClient(
        api_key=api_key,
        model_name=model_name,
    )
    generator = SummaryGenerator(client)

    if not full_text:
        result = {
            "summary": None,
            "summary_evidence": None,
            "summary_status": "no_text",
            "summary_confidence": 0.0,
            "summary_model_name": client.model_name,
            "summary_raw_response": None,
        }
        return idx, result

    try:
        raw_result = generator.generate_summary(full_text)
        result = normalize_summary_result(raw_result)
    except Exception as e:
        result = build_error_result(client.model_name, str(e))

    if sleep_sec > 0:
        time.sleep(sleep_sec)

    return idx, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/cases.parquet", help="Input parquet path")
    ap.add_argument("--output", default="data/processed/cases_with_summaries.parquet", help="Output parquet path")
    ap.add_argument("--text-col", default="full_text", help="Column with document text")
    ap.add_argument("--start-idx", type=int, default=0, help="Start row index")
    ap.add_argument("--limit", type=int, default=None, help="Max number of rows to process")
    ap.add_argument("--save-every", type=int, default=20, help="Save parquet every N processed rows")
    ap.add_argument("--sleep-sec", type=float, default=0.0, help="Sleep between requests")
    ap.add_argument("--api-key", default=None, help="DeepSeek API key. If omitted, reads from DEEPSEEK_API_KEY")
    ap.add_argument("--model-name", default="deepseek-chat", help="DeepSeek model name")
    ap.add_argument("--max-workers", type=int, default=1, help="Number of parallel workers for LLM calls")
    args = ap.parse_args()

    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DeepSeek API key not provided. Use --api-key or set DEEPSEEK_API_KEY")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    df = pd.read_parquet(args.input)

    max_workers = max(1, args.max_workers)
    client = DeepSeekClient(api_key=api_key, model_name=args.model_name)

    for col in [
        "summary",
        "summary_evidence",
        "summary_status",
        "summary_confidence",
        "summary_model_name",
        "summary_raw_response",
    ]:
        if col not in df.columns:
            df[col] = None

    end_idx = len(df) if args.limit is None else min(len(df), args.start_idx + args.limit)

    processed_since_save = 0
    pending: dict[Future[tuple[int, Dict[str, Any]]], int] = {}

    def apply_result(row_idx: int, norm: Dict[str, Any]) -> None:
        nonlocal processed_since_save
        for col, value in norm.items():
            df.at[row_idx, col] = value
        processed_since_save += 1
        if processed_since_save >= args.save_every:
            df.to_parquet(args.output, index=False)
            processed_since_save = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        progress = tqdm(total=end_idx - args.start_idx, desc="Generating summaries")

        for idx in range(args.start_idx, end_idx):
            full_text = safe_text(df.at[idx, args.text_col]) if args.text_col in df.columns else None

            if not full_text:
                norm = {
                    "summary": None,
                    "summary_evidence": None,
                    "summary_status": "no_text",
                    "summary_confidence": 0.0,
                    "summary_model_name": client.model_name,
                    "summary_raw_response": None,
                }
                apply_result(idx, norm)
                progress.update(1)
                continue

            existing_status = df.at[idx, "summary_status"]
            existing_summary = df.at[idx, "summary"]

            if existing_status == "ok" and safe_text(existing_summary):
                processed_since_save += 1
                if processed_since_save >= args.save_every:
                    df.to_parquet(args.output, index=False)
                    processed_since_save = 0
                progress.update(1)
                continue

            future = executor.submit(
                process_row,
                idx,
                full_text,
                api_key,
                args.model_name,
                args.sleep_sec,
            )
            pending[future] = idx

            # Не накапливаем слишком много незавершенных LLM-запросов.
            if len(pending) >= max_workers * 2:
                done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                for finished in done:
                    row_idx, norm = finished.result()
                    pending.pop(finished, None)
                    apply_result(row_idx, norm)
                    progress.update(1)

        while pending:
            done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
            for finished in done:
                row_idx, norm = finished.result()
                pending.pop(finished, None)
                apply_result(row_idx, norm)
                progress.update(1)

        progress.close()

    df.to_parquet(args.output, index=False)

    print(f"Saved: {args.output}")
    if "summary_status" in df.columns:
        print("\nsummary_status distribution:")
        print(df["summary_status"].fillna("NULL").value_counts(dropna=False))


if __name__ == "__main__":
    main()

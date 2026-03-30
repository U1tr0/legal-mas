from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.llm.deepseek_client import DeepSeekClient
from src.planner import LegalPlanner
from src.retrieval import LawGuidedRetriever

load_dotenv()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="User query to test")
    ap.add_argument("--law-top-k", type=int, default=5)
    ap.add_argument("--case-top-k", type=int, default=5)
    ap.add_argument("--model-name", default="deepseek-chat")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--koap-index", default="data/processed/koap_index.parquet")
    ap.add_argument("--case-index", default="data/processed/case_index.parquet")
    ap.add_argument("--use-planner-candidate-articles", action="store_true")
    args = ap.parse_args()

    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DeepSeek API key not provided. Use --api-key or set DEEPSEEK_API_KEY")

    planner = LegalPlanner(
        DeepSeekClient(
            api_key=api_key,
            model_name=args.model_name,
        )
    )
    retriever = LawGuidedRetriever(
        koap_index_path=Path(args.koap_index),
        case_index_path=Path(args.case_index),
        use_planner_candidate_articles=args.use_planner_candidate_articles,
    )

    plan = planner.plan(args.query)
    result = retriever.retrieve_from_plan(
        plan,
        law_top_k=args.law_top_k,
        case_top_k=args.case_top_k,
    )

    print("=== PLANNER OUTPUT ===")
    print(json.dumps(plan.model_dump(), ensure_ascii=False, indent=2))

    print("\n=== LAW RESULTS ===")
    print(
        json.dumps(
            result["law_results"],
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\n=== CASE RESULTS ===")
    print(
        json.dumps(
            result["case_results"],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

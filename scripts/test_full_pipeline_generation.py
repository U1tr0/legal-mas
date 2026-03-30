from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.generator import LegalAnswerGenerator
from src.llm.deepseek_client import DeepSeekClient
from src.llm.qwen_local_client import QwenLocalClient
from src.planner import LegalPlanner
from src.retrieval import LawGuidedRetriever

load_dotenv()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="User query to test end-to-end pipeline")
    ap.add_argument("--provider", choices=["qwen_local", "deepseek"], default="qwen_local")
    ap.add_argument("--model-name", default="deepseek-chat")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--model-path", default=None, help="Path to local Qwen GGUF model")
    ap.add_argument("--n-ctx", type=int, default=8192)
    ap.add_argument("--n-gpu-layers", type=int, default=35)
    ap.add_argument("--n-threads", type=int, default=8)
    ap.add_argument("--law-top-k", type=int, default=5)
    ap.add_argument("--case-top-k", type=int, default=5)
    ap.add_argument("--koap-index", default="data/processed/koap_index.parquet")
    ap.add_argument("--case-index", default="data/processed/case_index.parquet")
    ap.add_argument("--use-planner-candidate-articles", action="store_true")
    args = ap.parse_args()

    if args.provider == "qwen_local":
        model_path = args.model_path or os.getenv("QWEN_MODEL_PATH")
        if not model_path:
            raise ValueError("Qwen model path not provided. Use --model-path or set QWEN_MODEL_PATH")
        llm_client = QwenLocalClient(
            model_path=model_path,
            model_name=args.model_name if args.model_name != "deepseek-chat" else "Qwen2.5-7B-Instruct-GGUF",
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            n_threads=args.n_threads,
        )
    else:
        api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DeepSeek API key not provided. Use --api-key or set DEEPSEEK_API_KEY")
        llm_client = DeepSeekClient(api_key=api_key, model_name=args.model_name)

    planner = LegalPlanner(llm_client)
    retriever = LawGuidedRetriever(
        koap_index_path=Path(args.koap_index),
        case_index_path=Path(args.case_index),
        use_planner_candidate_articles=args.use_planner_candidate_articles,
    )
    generator = LegalAnswerGenerator(llm_client)

    plan = planner.plan(args.query)
    retrieval_result = retriever.retrieve_from_plan(plan, law_top_k=args.law_top_k, case_top_k=args.case_top_k)
    generated_answer = generator.generate(args.query, plan, retrieval_result)

    print("=== PLANNER OUTPUT ===")
    print(json.dumps(plan.model_dump(), ensure_ascii=False, indent=2))

    print("\n=== LAW EVIDENCE ===")
    print(json.dumps(retrieval_result["evidence_pack"]["law_evidence"], ensure_ascii=False, indent=2))

    print("\n=== CASE EVIDENCE ===")
    print(json.dumps(retrieval_result["evidence_pack"]["case_evidence"], ensure_ascii=False, indent=2))

    print("\n=== GENERATED ANSWER ===")
    print(json.dumps(generated_answer.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

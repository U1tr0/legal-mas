from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.generator import LegalAnswerGenerator
from src.llm.deepseek_client import DeepSeekClient
from src.llm.qwen_local_client import QwenLocalClient
from src.orchestrator import LegalDecisionPipeline
from src.planner import LegalPlanner
from src.retrieval import LawGuidedRetriever
from src.verifier import LegalAnswerVerifier

load_dotenv()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="User query to run through the full pipeline")
    ap.add_argument("--provider", choices=["qwen_local", "deepseek"], default="qwen_local")
    ap.add_argument("--model-name", default="deepseek-chat")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--n-ctx", type=int, default=8192)
    ap.add_argument("--n-gpu-layers", type=int, default=35)
    ap.add_argument("--n-threads", type=int, default=8)
    ap.add_argument("--koap-index", default="data/processed/koap_index.parquet")
    ap.add_argument("--case-index", default="data/processed/case_index.parquet")
    ap.add_argument("--law-top-k", type=int, default=5)
    ap.add_argument("--case-top-k", type=int, default=5)
    ap.add_argument("--expanded-law-top-k", type=int, default=8)
    ap.add_argument("--expanded-case-top-k", type=int, default=12)
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
    verifier = LegalAnswerVerifier(llm_client)
    pipeline = LegalDecisionPipeline(
        planner=planner,
        retriever=retriever,
        generator=generator,
        verifier=verifier,
        initial_law_top_k=args.law_top_k,
        initial_case_top_k=args.case_top_k,
        expanded_law_top_k=args.expanded_law_top_k,
        expanded_case_top_k=args.expanded_case_top_k,
    )

    result = pipeline.run(args.query)

    print("=== PLANNER OUTPUT ===")
    print(json.dumps(result.planner_output.model_dump() if result.planner_output else None, ensure_ascii=False, indent=2))

    print("\n=== RETRIEVAL SUMMARY ===")
    retrieval_summary = None
    if result.retrieval_result:
        retrieval_summary = {
            "candidate_article_numbers": result.retrieval_result.get("candidate_article_numbers"),
            "law_count": len((result.retrieval_result.get("evidence_pack") or {}).get("law_evidence") or []),
            "case_count": len((result.retrieval_result.get("evidence_pack") or {}).get("case_evidence") or []),
            "planner_candidate_articles_used": result.retrieval_result.get("planner_candidate_articles_used"),
            "search_queries": result.retrieval_result.get("search_queries"),
        }
    print(json.dumps(retrieval_summary, ensure_ascii=False, indent=2))

    print("\n=== GENERATED ANSWER ===")
    print(json.dumps(result.generated_answer.model_dump() if result.generated_answer else None, ensure_ascii=False, indent=2))

    print("\n=== VERIFICATION RESULT ===")
    print(json.dumps(result.verification_result.model_dump() if result.verification_result else None, ensure_ascii=False, indent=2))

    print("\n=== PIPELINE RESULT ===")
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))

    print("\n=== FINAL STATUS ===")
    print(f"status: {result.status}")
    print(f"stop_reason: {result.stop_reason}")
    print(f"final_answer: {result.final_answer}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import PipelineResult


def _to_plain_data(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [_to_plain_data(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_plain_data(item) for key, item in value.items()}
    return value


def _format_json(value: Any) -> str:
    return json.dumps(_to_plain_data(value), ensure_ascii=False, indent=2)


def build_retrieval_summary(retrieval_result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not retrieval_result:
        return None

    evidence_pack = retrieval_result.get("evidence_pack") or {}
    law_evidence = evidence_pack.get("law_evidence") or []
    case_evidence = evidence_pack.get("case_evidence") or []

    return {
        "query": retrieval_result.get("query"),
        "normalized_query": retrieval_result.get("normalized_query"),
        "fact_description": retrieval_result.get("fact_description"),
        "query_type": retrieval_result.get("query_type"),
        "domain": retrieval_result.get("domain"),
        "search_queries": retrieval_result.get("search_queries"),
        "candidate_article_numbers": retrieval_result.get("candidate_article_numbers"),
        "planner_candidate_articles_used": retrieval_result.get("planner_candidate_articles_used"),
        "law_count": len(law_evidence),
        "case_count": len(case_evidence),
    }


def build_pipeline_report_text(query: str, pipeline_result: PipelineResult) -> str:
    sections = [
        ("ИСХОДНЫЙ ЗАПРОС", query),
        (
            "КРАТКИЙ ИТОГ",
            "\n".join(
                [
                    f"status: {pipeline_result.status}",
                    f"stop_reason: {pipeline_result.stop_reason}",
                    f"final_answer: {pipeline_result.final_answer}",
                ]
            ),
        ),
        ("PLANNER OUTPUT", _format_json(pipeline_result.planner_output)),
        ("RETRIEVAL SUMMARY", _format_json(build_retrieval_summary(pipeline_result.retrieval_result))),
        ("RETRIEVAL RESULT", _format_json(pipeline_result.retrieval_result)),
        ("GENERATED ANSWER", _format_json(pipeline_result.generated_answer)),
        ("VERIFICATION RESULT", _format_json(pipeline_result.verification_result)),
        ("ITERATIONS", _format_json(pipeline_result.iterations)),
        ("TRACE", _format_json(pipeline_result.trace)),
        ("PIPELINE RESULT", _format_json(pipeline_result)),
    ]

    parts: list[str] = []
    for title, content in sections:
        parts.append("=" * 120)
        parts.append(title)
        parts.append("=" * 120)
        parts.append(content if isinstance(content, str) else str(content))
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def save_pipeline_report(path: str | Path, query: str, pipeline_result: PipelineResult) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_pipeline_report_text(query, pipeline_result), encoding="utf-8")
    return report_path

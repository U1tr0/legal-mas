from __future__ import annotations

import json
from typing import Any

from src.llm.base import BaseLLMClient
from src.retrieval.query_processing import extract_explicit_article_numbers, normalize_text

from .prompts import PLANNER_SYSTEM_PROMPT, PLANNER_USER_TEMPLATE
from .schemas import PlannerOutput


class LegalPlanner:
    _NOISE_MARKERS = {
        "жена",
        "жены",
        "муж",
        "мужа",
        "друг",
        "друга",
        "подруга",
        "роддом",
        "роды",
        "ребенок",
        "ребенка",
        "гости",
        "гостях",
        "позвонила",
        "позвонил",
        "срочно",
        "торопился",
        "торопилась",
        "испугался",
        "испугалась",
        "праздник",
        "застолье",
    }

    def __init__(self, llm_client: BaseLLMClient) -> None:
        self.llm_client = llm_client

    @classmethod
    def _is_noisy_search_query(cls, value: str) -> bool:
        lowered = normalize_text(value)
        markers_found = sum(1 for marker in cls._NOISE_MARKERS if marker in lowered.split())
        return markers_found >= 1

    def _fallback_plan(self, query: str, reason: str | None = None) -> PlannerOutput:
        normalized_query = normalize_text(query)
        explicit_articles = extract_explicit_article_numbers(query)

        return PlannerOutput(
            original_query=query,
            normalized_query=normalized_query or query,
            fact_description=normalized_query or query,
            query_type="other",
            domain="unknown",
            needs_clarification=False,
            clarification_reason=reason,
            retrieval_targets=["law", "cases"],
            candidate_articles=explicit_articles,
            search_queries=[normalized_query or query],
            extracted_facts={
                "short_summary": normalized_query or query,
                "legal_keywords": [],
                "event_markers": [],
            },
        )

    def _normalize_raw_payload(self, payload: dict[str, Any], query: str) -> dict[str, Any]:
        payload = dict(payload)
        payload["original_query"] = payload.get("original_query") or query
        payload["normalized_query"] = payload.get("normalized_query") or normalize_text(query) or query
        payload["fact_description"] = payload.get("fact_description") or payload["normalized_query"]

        explicit_articles = extract_explicit_article_numbers(query)
        candidate_articles = payload.get("candidate_articles") or []
        if isinstance(candidate_articles, list):
            payload["candidate_articles"] = [*candidate_articles, *explicit_articles]
        else:
            payload["candidate_articles"] = explicit_articles

        search_queries = payload.get("search_queries") or []
        if not isinstance(search_queries, list):
            search_queries = []

        normalized_search_queries: list[str] = []
        seen_queries: set[str] = set()
        for item in search_queries:
            cleaned = " ".join(str(item).split()).strip()
            if not cleaned:
                continue
            if self._is_noisy_search_query(cleaned):
                continue
            if cleaned in seen_queries:
                continue
            normalized_search_queries.append(cleaned)
            seen_queries.add(cleaned)

        if not normalized_search_queries:
            normalized_search_queries = [payload["normalized_query"]]

        payload["search_queries"] = normalized_search_queries[:3]

        domain = payload.get("domain")
        retrieval_targets = payload.get("retrieval_targets")
        if domain == "administrative_law":
            if not isinstance(retrieval_targets, list):
                retrieval_targets = []
            normalized_targets = [str(item).strip() for item in retrieval_targets]
            if "law" not in normalized_targets:
                normalized_targets.append("law")
            if "cases" not in normalized_targets:
                normalized_targets.append("cases")
            payload["retrieval_targets"] = normalized_targets

        return payload

    def plan(self, query: str) -> PlannerOutput:
        user_prompt = PLANNER_USER_TEMPLATE.format(query=query)

        try:
            raw_response = self.llm_client.generate(
                system_prompt=PLANNER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.0,
                max_tokens=1200,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            return self._fallback_plan(query, reason=f"planner_llm_error: {e}")

        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            return self._fallback_plan(query, reason="planner_invalid_json")

        try:
            normalized_payload = self._normalize_raw_payload(parsed, query)
            return PlannerOutput.model_validate(normalized_payload)
        except Exception as e:
            return self._fallback_plan(query, reason=f"planner_validation_error: {e}")

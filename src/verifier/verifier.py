from __future__ import annotations

import json
from typing import Any

from src.generator.schemas import GeneratedAnswer
from src.llm.base import BaseLLMClient
from src.planner.schemas import PlannerOutput

from .prompts import VERIFIER_SYSTEM_PROMPT, VERIFIER_USER_TEMPLATE
from .schemas import VerificationResult


class LegalAnswerVerifier:
    def __init__(self, llm_client: BaseLLMClient) -> None:
        self.llm_client = llm_client

    @staticmethod
    def _safe_text(value: Any) -> str | None:
        if value is None:
            return None
        value = " ".join(str(value).split()).strip()
        return value or None

    def _build_planner_context(self, planner_output: PlannerOutput) -> dict[str, Any]:
        return {
            "normalized_query": planner_output.normalized_query,
            "query_type": planner_output.query_type,
            "domain": planner_output.domain,
            "needs_clarification": planner_output.needs_clarification,
            "clarification_reason": planner_output.clarification_reason,
            "candidate_articles": planner_output.candidate_articles,
            "search_queries": planner_output.search_queries,
        }

    def _build_evidence_context(self, retrieval_result: dict[str, Any]) -> dict[str, Any]:
        evidence_pack = retrieval_result.get("evidence_pack") or {}
        law_evidence = evidence_pack.get("law_evidence") or []
        case_evidence = evidence_pack.get("case_evidence") or []

        compact_law_evidence = []
        for item in law_evidence:
            compact_law_evidence.append(
                {
                    "article_label": self._safe_text(item.get("article_label")),
                    "article_text": self._safe_text(item.get("article_text"))[:1600] if self._safe_text(item.get("article_text")) else None,
                }
            )

        compact_case_evidence = []
        for idx, item in enumerate(case_evidence):
            compact_case_evidence.append(
                {
                    "case_id": self._safe_text(item.get("case_id")),
                    "summary": self._safe_text(item.get("summary"))[:700] if self._safe_text(item.get("summary")) else None,
                    "full_text_excerpt": (
                        self._safe_text(item.get("full_text_excerpt"))[:700]
                        if idx < 2 and self._safe_text(item.get("full_text_excerpt"))
                        else None
                    ),
                    "articles_text": item.get("articles_text") or [],
                }
            )

        return {
            "candidate_article_numbers": retrieval_result.get("candidate_article_numbers") or [],
            "law_evidence": compact_law_evidence,
            "case_evidence": compact_case_evidence,
        }

    def _fallback_result(self, raw_response: str | None = None) -> VerificationResult:
        return VerificationResult(
            verdict="REWRITE",
            is_grounded=False,
            is_sufficient=False,
            problems=["verifier_parse_error"],
            explanation="Verifier could not produce a reliable structured assessment.",
            recommended_action="Review and rewrite the answer conservatively.",
            rewrite_focus="Check groundedness against retrieved evidence and remove unsupported claims.",
            confidence=0.0,
            raw_response=raw_response,
        )

    @staticmethod
    def _has_blocking_problem(problems: list[str]) -> bool:
        blocking_markers = (
            "unsupported",
            "hallucin",
            "invented",
            "wrong_article",
            "wrong_case",
            "not_grounded",
            "out_of_scope",
            "insufficient_evidence",
            "misaligned",
            "categorical",
        )
        lowered = " ".join(problems).lower()
        return any(marker in lowered for marker in blocking_markers)

    def _normalize_verification_result(
        self,
        result: VerificationResult,
        generated_answer: GeneratedAnswer,
        retrieval_result: dict[str, Any],
    ) -> VerificationResult:
        has_final_answer = bool(self._safe_text(generated_answer.final_answer))
        law_evidence = ((retrieval_result.get("evidence_pack") or {}).get("law_evidence") or [])
        has_law_evidence = bool(law_evidence)

        if (
            result.verdict == "REWRITE"
            and result.is_grounded
            and has_final_answer
            and has_law_evidence
            and result.confidence >= 0.65
            and not self._has_blocking_problem(result.problems)
        ):
            result.verdict = "ACCEPT"
            if not result.recommended_action:
                result.recommended_action = "Answer can be returned as sufficiently grounded."
            if not result.explanation:
                result.explanation = "The answer is sufficiently grounded and usable without mandatory rewrite."

        return result

    def verify(
        self,
        query: str,
        planner_output: PlannerOutput,
        retrieval_result: dict[str, Any],
        generated_answer: GeneratedAnswer,
    ) -> VerificationResult:
        planner_context = self._build_planner_context(planner_output)
        evidence_context = self._build_evidence_context(retrieval_result)

        user_prompt = VERIFIER_USER_TEMPLATE.format(
            query=query,
            planner_context_json=json.dumps(planner_context, ensure_ascii=False, separators=(",", ":")),
            evidence_context_json=json.dumps(evidence_context, ensure_ascii=False, separators=(",", ":")),
            generated_answer_json=json.dumps(generated_answer.model_dump(), ensure_ascii=False, separators=(",", ":")),
        )

        try:
            raw_response = self.llm_client.generate(
                system_prompt=VERIFIER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.0,
                max_tokens=1200,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            return self._fallback_result(raw_response=f"verifier_llm_error: {e}")

        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            return self._fallback_result(raw_response=raw_response)

        try:
            parsed["raw_response"] = raw_response
            result = VerificationResult.model_validate(parsed)
            return self._normalize_verification_result(result, generated_answer, retrieval_result)
        except Exception:
            return self._fallback_result(raw_response=raw_response)

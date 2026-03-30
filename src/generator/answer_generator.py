from __future__ import annotations

import json
from typing import Any

from src.llm.base import BaseLLMClient
from src.planner.schemas import PlannerOutput

from .prompts import GENERATOR_SYSTEM_PROMPT, GENERATOR_USER_TEMPLATE
from .schemas import GeneratedAnswer


class LegalAnswerGenerator:
    def __init__(self, llm_client: BaseLLMClient) -> None:
        self.llm_client = llm_client

    @staticmethod
    def _safe_text(value: Any) -> str | None:
        if value is None:
            return None
        value = " ".join(str(value).split()).strip()
        return value or None

    def _build_evidence_context(self, retrieval_result: dict[str, Any]) -> dict[str, Any]:
        evidence_pack = retrieval_result.get("evidence_pack") or {}
        law_evidence = evidence_pack.get("law_evidence") or []
        case_evidence = evidence_pack.get("case_evidence") or []

        compact_law_evidence = []
        for item in law_evidence:
            compact_law_evidence.append(
                {
                    "article_number": self._safe_text(item.get("article_number")),
                    "article_label": self._safe_text(item.get("article_label")),
                    "article_text": self._safe_text(item.get("article_text"))[:1800] if self._safe_text(item.get("article_text")) else None,
                    "score": item.get("score"),
                }
            )

        compact_case_evidence = []
        for idx, item in enumerate(case_evidence):
            compact_case_evidence.append(
                {
                    "case_id": self._safe_text(item.get("case_id")),
                    "summary": self._safe_text(item.get("summary"))[:700] if self._safe_text(item.get("summary")) else None,
                    "full_text_excerpt": (
                        self._safe_text(item.get("full_text_excerpt"))[:900]
                        if idx < 2 and self._safe_text(item.get("full_text_excerpt"))
                        else None
                    ),
                    "articles_text": item.get("articles_text") or [],
                    "source_url": self._safe_text(item.get("source_url")),
                    "score": item.get("score"),
                }
            )

        return {
            "query": retrieval_result.get("query"),
            "normalized_query": retrieval_result.get("normalized_query"),
            "query_type": retrieval_result.get("query_type"),
            "domain": retrieval_result.get("domain"),
            "candidate_article_numbers": retrieval_result.get("candidate_article_numbers") or [],
            "law_evidence": compact_law_evidence,
            "case_evidence": compact_case_evidence,
        }

    def _fallback_answer(self, raw_response: str | None = None) -> GeneratedAnswer:
        return GeneratedAnswer(
            question_summary="",
            applicable_laws=[],
            relevant_cases=[],
            legal_analysis=None,
            risk_factors=[],
            final_answer="По найденным материалам нельзя сделать надежный вывод без дополнительных подтвержденных данных. Нужна дополнительная проверка найденных норм и практики.",
            confidence=0.0,
            raw_response=raw_response,
        )

    def _build_non_empty_final_answer(
        self,
        answer: GeneratedAnswer,
        retrieval_result: dict[str, Any],
    ) -> str:
        if answer.final_answer:
            return answer.final_answer

        law_evidence = (retrieval_result.get("evidence_pack") or {}).get("law_evidence") or []
        law_numbers = [
            self._safe_text(item.get("article_number"))
            for item in law_evidence
            if self._safe_text(item.get("article_number"))
        ]
        unique_law_numbers: list[str] = []
        seen_numbers: set[str] = set()
        for article_number in law_numbers:
            if article_number in seen_numbers:
                continue
            seen_numbers.add(article_number)
            unique_law_numbers.append(article_number)

        if unique_law_numbers:
            law_fragment = f"По найденным материалам ситуация предварительно соотносится со статьями КоАП: {', '.join(unique_law_numbers[:3])}."
        else:
            law_fragment = "По найденным материалам можно дать только предварительный и ограниченный вывод."

        analysis_fragment = answer.legal_analysis
        if analysis_fragment:
            analysis_fragment = analysis_fragment.rstrip(". ")
            return f"{law_fragment} {analysis_fragment}. Итог требует осторожной оценки по найденным материалам."

        return f"{law_fragment} Точных оснований для более уверенного вывода в текущем evidence недостаточно."

    def generate(
        self,
        query: str,
        planner_output: PlannerOutput,
        retrieval_result: dict[str, Any],
        rewrite_focus: str | None = None,
    ) -> GeneratedAnswer:
        evidence_context = self._build_evidence_context(retrieval_result)
        generation_instruction = self._safe_text(rewrite_focus) or "Нет дополнительной инструкции. Сгенерируй лучший grounded-ответ по evidence."
        user_prompt = GENERATOR_USER_TEMPLATE.format(
            query=query,
            planner_output_json=json.dumps(planner_output.model_dump(), ensure_ascii=False, separators=(",", ":")),
            evidence_context_json=json.dumps(evidence_context, ensure_ascii=False, separators=(",", ":")),
            generation_instruction=generation_instruction,
        )

        try:
            raw_response = self.llm_client.generate(
                system_prompt=GENERATOR_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.0,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            return self._fallback_answer(raw_response=f"generator_llm_error: {e}")

        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            return self._fallback_answer(raw_response=raw_response)

        try:
            parsed["raw_response"] = raw_response
            answer = GeneratedAnswer.model_validate(parsed)
            answer.final_answer = self._build_non_empty_final_answer(answer, retrieval_result)
            return answer
        except Exception:
            return self._fallback_answer(raw_response=raw_response)

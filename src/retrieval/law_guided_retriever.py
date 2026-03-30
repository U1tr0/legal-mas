from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.planner.schemas import PlannerOutput

from .bm25_index import BM25Index
from .query_processing import (
    extract_explicit_article_numbers,
    normalize_article_number,
    normalize_text,
    tokenize_for_bm25,
)


class LawGuidedRetriever:
    """Ищет статьи и дела, затем усиливает дела с совпадающими статьями."""

    def __init__(
        self,
        koap_index_path: str | Path = "data/processed/koap_index.parquet",
        case_index_path: str | Path = "data/processed/case_index.parquet",
        case_search_multiplier: int = 5,
        article_bonus: float = 1.5,
        extra_article_bonus: float = 0.5,
        use_planner_candidate_articles: bool = False,
    ) -> None:
        self.koap_index_path = Path(koap_index_path)
        self.case_index_path = Path(case_index_path)
        self.case_search_multiplier = max(case_search_multiplier, 1)
        self.article_bonus = article_bonus
        self.extra_article_bonus = extra_article_bonus
        self.use_planner_candidate_articles = use_planner_candidate_articles

        self.koap_df = pd.read_parquet(self.koap_index_path)
        self.case_df = pd.read_parquet(self.case_index_path)
        self._law_by_article_number = self._build_law_article_lookup(self.koap_df)

        self.law_index = BM25Index(self.koap_df, text_column="retrieval_text")
        self.case_index = BM25Index(self.case_df, text_column="retrieval_text")

    @staticmethod
    def _safe_text(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            return value or None
        value = str(value).strip()
        return value or None

    @staticmethod
    def _to_list(value: Any) -> list[Any]:
        if value is None:
            return []

        if isinstance(value, list):
            return value

        if hasattr(value, "tolist"):
            try:
                converted = value.tolist()
                if isinstance(converted, list):
                    return converted
            except Exception:
                pass

        if isinstance(value, str):
            value = value.strip()
            if not value:
                return []
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                return [value]

        return []

    def _extract_case_article_numbers(self, row: pd.Series) -> list[str]:
        article_numbers: list[str] = []
        seen: set[str] = set()

        for article in self._to_list(row.get("articles_norm")):
            if not isinstance(article, dict):
                continue
            normalized = normalize_article_number(article.get("article"))
            if normalized and normalized not in seen:
                seen.add(normalized)
                article_numbers.append(normalized)

        for article_text in self._to_list(row.get("articles_text")):
            normalized = normalize_article_number(article_text)
            if normalized and normalized not in seen:
                seen.add(normalized)
                article_numbers.append(normalized)

        return article_numbers

    @staticmethod
    def _unique_keep_order(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result

    @staticmethod
    def _build_law_article_lookup(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
        lookup: dict[str, dict[str, Any]] = {}
        for _, row in df.iterrows():
            article_number = normalize_article_number(row.get("article_number"))
            if not article_number or article_number in lookup:
                continue
            lookup[article_number] = row.to_dict()
        return lookup

    def _format_law_result(self, row: pd.Series, explicit_articles: set[str]) -> dict[str, Any]:
        article_number = normalize_article_number(row.get("article_number"))
        matched_explicitly = article_number in explicit_articles if article_number else False
        reasons = ["high_bm25_match"]
        if matched_explicitly:
            reasons.append("explicit_article_match")

        return {
            "doc_id": self._safe_text(row.get("doc_id")),
            "article_number": article_number,
            "article_label": self._safe_text(row.get("article_label")),
            "article_text": self._safe_text(row.get("article_text")),
            "score": float(row.get("score", 0.0)),
            "selection_reason": reasons,
        }

    @staticmethod
    def _is_useful_law_hit(item: dict[str, Any]) -> bool:
        return item["score"] > 0 or "explicit_article_match" in item["selection_reason"]

    def _ensure_explicit_law_results(
        self,
        law_results: list[dict[str, Any]],
        explicit_articles: list[str],
    ) -> list[dict[str, Any]]:
        existing = {item["article_number"] for item in law_results if item.get("article_number")}

        for article_number in explicit_articles:
            if article_number in existing:
                continue

            law_row = self._law_by_article_number.get(article_number)
            if not law_row:
                continue

            law_results.append(
                {
                    "doc_id": self._safe_text(law_row.get("doc_id")),
                    "article_number": article_number,
                    "article_label": self._safe_text(law_row.get("article_label")),
                    "article_text": self._safe_text(law_row.get("article_text")),
                    "score": 0.0,
                    "selection_reason": ["explicit_article_match", "direct_article_lookup"],
                }
            )
            existing.add(article_number)

        return law_results

    def _compute_case_bonus(
        self,
        case_article_numbers: list[str],
        candidate_article_numbers: list[str],
    ) -> tuple[list[str], float]:
        candidate_article_set = set(candidate_article_numbers)
        overlap = [article for article in case_article_numbers if article in candidate_article_set]
        if not overlap:
            return [], 0.0

        bonus = self.article_bonus
        if len(overlap) > 1:
            bonus += (len(overlap) - 1) * self.extra_article_bonus
        return overlap, bonus

    def _format_case_result(
        self,
        row: pd.Series,
        candidate_article_numbers: list[str],
    ) -> dict[str, Any]:
        case_article_numbers = self._extract_case_article_numbers(row)
        overlap, bonus = self._compute_case_bonus(case_article_numbers, candidate_article_numbers)
        bm25_score = float(row.get("score", 0.0))
        summary_text = self._safe_text(row.get("summary")) or ""
        final_score = bm25_score + bonus

        reasons = ["high_bm25_match"]
        if overlap:
            reasons.append("article_overlap_bonus")
        full_text_excerpt = self._build_case_excerpt(row)

        return {
            "case_id": self._safe_text(row.get("case_id")),
            "title": self._safe_text(row.get("title")),
            "summary": summary_text,
            "full_text_excerpt": full_text_excerpt,
            "articles_text": [str(x) for x in self._to_list(row.get("articles_text")) if str(x).strip()],
            "source_url": self._safe_text(row.get("source_url")),
            "score": final_score,
            "bm25_score": bm25_score,
            "article_bonus": bonus,
            "article_overlap": overlap,
            "matched_queries": self._to_list(row.get("matched_queries")),
            "selection_reason": reasons,
        }

    @staticmethod
    def _is_useful_case_hit(item: dict[str, Any]) -> bool:
        return item["bm25_score"] > 0 or bool(item["article_overlap"])

    def _build_case_excerpt(self, row: pd.Series, max_chars: int = 1200) -> str | None:
        full_text = self._safe_text(row.get("full_text"))
        if not full_text:
            return None

        normalized = " ".join(full_text.split()).strip()
        if not normalized:
            return None

        markers = (
            "как следует из материалов дела",
            "из материалов дела следует",
            "как усматривается из материалов дела",
            "установил:",
            "установлено, что",
            "признан виновным",
            "в соответствии с частью",
        )

        start_index = 0
        lowered = normalized.lower()
        for marker in markers:
            marker_index = lowered.find(marker)
            if marker_index >= 0:
                start_index = marker_index
                break

        excerpt_source = normalized[start_index:]
        excerpt = excerpt_source[:max_chars].rstrip()
        if len(normalized) > max_chars:
            excerpt += "..."
        return excerpt

    @staticmethod
    def _build_basic_plan(query: str) -> PlannerOutput:
        normalized_query = normalize_text(query) or query
        explicit_articles = extract_explicit_article_numbers(query)
        return PlannerOutput(
            original_query=query,
            normalized_query=normalized_query,
            fact_description=normalized_query,
            query_type="other",
            domain="unknown",
            needs_clarification=False,
            clarification_reason=None,
            retrieval_targets=["law", "cases"],
            candidate_articles=explicit_articles,
            search_queries=[normalized_query],
        )

    def _combine_search_results(
        self,
        index: BM25Index,
        search_queries: list[str],
        top_k: int,
    ) -> pd.DataFrame:
        if top_k <= 0:
            return index.df.iloc[0:0].copy()

        aggregated: dict[str, dict[str, Any]] = {}

        for search_query in search_queries:
            query_text = self._safe_text(search_query)
            if not query_text:
                continue

            query_tokens = tokenize_for_bm25(query_text)
            hits = index.search(query=query_text, top_k=top_k, query_tokens=query_tokens)

            for _, row in hits.iterrows():
                doc_id = self._safe_text(row.get("doc_id"))
                if not doc_id:
                    continue

                score = float(row.get("score", 0.0))
                record = aggregated.get(doc_id)
                if record is None:
                    row_dict = row.to_dict()
                    row_dict["score"] = score
                    row_dict["matched_queries"] = [query_text] if score > 0 else []
                    row_dict["query_score_map"] = {query_text: score}
                    aggregated[doc_id] = row_dict
                    continue

                record["score"] = float(record.get("score", 0.0)) + max(score, 0.0)
                score_map = record.setdefault("query_score_map", {})
                score_map[query_text] = max(score_map.get(query_text, 0.0), score)
                if score > 0 and query_text not in record.get("matched_queries", []):
                    record.setdefault("matched_queries", []).append(query_text)

        if not aggregated:
            return index.df.iloc[0:0].copy()

        result = pd.DataFrame(aggregated.values())
        result = result.sort_values("score", ascending=False, kind="stable")
        return result.head(min(top_k, len(result))).reset_index(drop=True)

    def retrieve_from_plan(
        self,
        plan: PlannerOutput,
        law_top_k: int = 5,
        case_top_k: int = 10,
        use_planner_candidate_articles: bool | None = None,
    ) -> dict[str, Any]:
        search_queries = self._unique_keep_order(
            [query_text for query_text in (plan.search_queries or [plan.normalized_query or plan.original_query]) if self._safe_text(query_text)]
        )

        retrieval_targets = set(plan.retrieval_targets)
        if plan.domain == "administrative_law":
            retrieval_targets.update({"law", "cases"})

        explicit_query_articles = extract_explicit_article_numbers(plan.original_query)
        planner_articles = [
            normalized
            for article in plan.candidate_articles
            if (normalized := normalize_article_number(article))
        ]
        if use_planner_candidate_articles is None:
            use_planner_candidate_articles = self.use_planner_candidate_articles
        explicit_article_set = set(explicit_query_articles)

        law_results: list[dict[str, Any]] = []
        if "law" in retrieval_targets:
            law_hits = self._combine_search_results(self.law_index, search_queries, law_top_k)
            law_results = [
                self._format_law_result(row, explicit_article_set)
                for _, row in law_hits.iterrows()
            ]
            law_results = [item for item in law_results if self._is_useful_law_hit(item)]
            law_results = self._ensure_explicit_law_results(law_results, explicit_query_articles)

        retrieved_article_numbers = [
            result["article_number"]
            for result in law_results
            if result.get("article_number")
        ]
        supported_planner_articles: list[str] = []
        if use_planner_candidate_articles:
            supported_planner_articles = [
                article
                for article in planner_articles
                if article in set(retrieved_article_numbers) or article in set(explicit_query_articles)
            ]
        candidate_article_numbers = self._unique_keep_order(
            explicit_query_articles + retrieved_article_numbers + supported_planner_articles
        )

        case_results: list[dict[str, Any]] = []
        if "cases" in retrieval_targets:
            rerank_pool_k = max(case_top_k, case_top_k * self.case_search_multiplier)
            case_hits = self._combine_search_results(self.case_index, search_queries, rerank_pool_k)
            reranked_case_results = [
                self._format_case_result(row, candidate_article_numbers)
                for _, row in case_hits.iterrows()
            ]
            reranked_case_results = [item for item in reranked_case_results if self._is_useful_case_hit(item)]
            reranked_case_results.sort(key=lambda item: item["score"], reverse=True)
            case_results = reranked_case_results[:case_top_k]

        evidence_pack = {
            "law_evidence": [
                {
                    "article_number": item["article_number"],
                    "article_label": item["article_label"],
                    "article_text": item["article_text"],
                    "score": item["score"],
                }
                for item in law_results
            ],
            "case_evidence": [
                {
                    "case_id": item["case_id"],
                    "title": item["title"],
                    "summary": item["summary"],
                    "full_text_excerpt": item["full_text_excerpt"],
                    "articles_text": item["articles_text"],
                    "source_url": item["source_url"],
                    "score": item["score"],
                    "article_overlap": item["article_overlap"],
                }
                for item in case_results
            ],
        }

        return {
            "query": plan.original_query,
            "normalized_query": plan.normalized_query,
            "fact_description": plan.fact_description,
            "query_type": plan.query_type,
            "domain": plan.domain,
            "search_queries": search_queries,
            "planner_candidate_articles_used": use_planner_candidate_articles,
            "candidate_article_numbers": candidate_article_numbers,
            "law_results": law_results,
            "case_results": case_results,
            "evidence_pack": evidence_pack,
        }

    def retrieve(self, query: str, law_top_k: int = 5, case_top_k: int = 10) -> dict[str, Any]:
        plan = self._build_basic_plan(query)
        return self.retrieve_from_plan(plan, law_top_k=law_top_k, case_top_k=case_top_k)

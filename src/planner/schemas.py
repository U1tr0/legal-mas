from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


QueryType = Literal[
    "qualification",
    "sanction",
    "case_outcome_prediction",
    "appeal",
    "procedure",
    "other",
]

DomainType = Literal["administrative_law", "unknown", "out_of_scope"]
RetrievalTarget = Literal["law", "cases"]


class ExtractedFacts(BaseModel):
    short_summary: str = ""
    legal_keywords: list[str] = Field(default_factory=list)
    event_markers: list[str] = Field(default_factory=list)

    @field_validator("legal_keywords", "event_markers", mode="before")
    @classmethod
    def _normalize_string_list(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            value = value.strip()
            return [value] if value else []
        return []


class PlannerOutput(BaseModel):
    original_query: str
    normalized_query: str
    fact_description: str
    query_type: QueryType = "other"
    domain: DomainType = "unknown"
    needs_clarification: bool = False
    clarification_reason: str | None = None
    retrieval_targets: list[RetrievalTarget] = Field(default_factory=lambda: ["law", "cases"])
    candidate_articles: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    extracted_facts: ExtractedFacts | None = None

    @field_validator("original_query", "normalized_query", "fact_description", mode="before")
    @classmethod
    def _normalize_text_fields(cls, value: object) -> str:
        if value is None:
            return ""
        return " ".join(str(value).split()).strip()

    @field_validator("candidate_articles", "search_queries", mode="before")
    @classmethod
    def _normalize_list_fields(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [" ".join(str(item).split()).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            value = " ".join(value.split()).strip()
            return [value] if value else []
        return []

    @field_validator("retrieval_targets", mode="before")
    @classmethod
    def _normalize_retrieval_targets(cls, value: object) -> list[str]:
        allowed = {"law", "cases"}
        if value is None:
            return ["law", "cases"]
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return ["law", "cases"]
        normalized = []
        for item in value:
            item_text = str(item).strip()
            if item_text in allowed and item_text not in normalized:
                normalized.append(item_text)
        return normalized or ["law", "cases"]

    @model_validator(mode="after")
    def _finalize(self) -> "PlannerOutput":
        if not self.normalized_query:
            self.normalized_query = self.original_query

        if not self.fact_description:
            self.fact_description = self.normalized_query

        if not self.search_queries:
            self.search_queries = [self.normalized_query]

        deduped_articles: list[str] = []
        seen_articles: set[str] = set()
        for article in self.candidate_articles:
            if article not in seen_articles:
                deduped_articles.append(article)
                seen_articles.add(article)
        self.candidate_articles = deduped_articles

        deduped_queries: list[str] = []
        seen_queries: set[str] = set()
        for query in self.search_queries:
            if query not in seen_queries:
                deduped_queries.append(query)
                seen_queries.add(query)
        self.search_queries = deduped_queries

        if not self.needs_clarification:
            self.clarification_reason = None

        return self

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


class ApplicableLaw(BaseModel):
    article_number: str
    article_label: str
    why_relevant: str

    @field_validator("article_number", "article_label", "why_relevant", mode="before")
    @classmethod
    def _normalize_text(cls, value: object) -> str:
        if value is None:
            return ""
        return " ".join(str(value).split()).strip()


class RelevantCase(BaseModel):
    case_id: str
    short_summary: str
    why_relevant: str

    @field_validator("case_id", "short_summary", "why_relevant", mode="before")
    @classmethod
    def _normalize_text(cls, value: object) -> str:
        if value is None:
            return ""
        return " ".join(str(value).split()).strip()


class GeneratedAnswer(BaseModel):
    question_summary: str = ""
    applicable_laws: list[ApplicableLaw] = Field(default_factory=list)
    relevant_cases: list[RelevantCase] = Field(default_factory=list)
    legal_analysis: str | None = None
    risk_factors: list[str] = Field(default_factory=list)
    final_answer: str | None = None
    confidence: float = 0.0
    raw_response: str | None = None

    @field_validator("legal_analysis", "final_answer", mode="before")
    @classmethod
    def _normalize_optional_text(cls, value: object) -> str | None:
        if value is None:
            return None
        value = " ".join(str(value).split()).strip()
        return value or None

    @field_validator("question_summary", mode="before")
    @classmethod
    def _normalize_required_text(cls, value: object) -> str:
        if value is None:
            return ""
        value = " ".join(str(value).split()).strip()
        return value

    @field_validator("risk_factors", mode="before")
    @classmethod
    def _normalize_risk_factors(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [" ".join(str(item).split()).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            value = " ".join(value.split()).strip()
            return [value] if value else []
        return []

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, value: object) -> float:
        if value is None:
            return 0.0
        try:
            numeric = float(value)
        except Exception:
            return 0.0
        return max(0.0, min(1.0, numeric))

    @model_validator(mode="after")
    def _finalize(self) -> "GeneratedAnswer":
        if not self.question_summary:
            self.question_summary = ""
        return self

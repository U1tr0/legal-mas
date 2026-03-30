from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


VerifierVerdict = Literal["ACCEPT", "REWRITE", "RETRIEVE_MORE", "ASK_USER", "REFUSE"]


class VerificationResult(BaseModel):
    verdict: VerifierVerdict = "REWRITE"
    is_grounded: bool = False
    is_sufficient: bool = False
    problems: list[str] = Field(default_factory=list)
    explanation: str = ""
    recommended_action: str | None = None
    rewrite_focus: str | None = None
    confidence: float = 0.0
    raw_response: str | None = None

    @field_validator("problems", mode="before")
    @classmethod
    def _normalize_problems(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [" ".join(str(item).split()).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            value = " ".join(value.split()).strip()
            return [value] if value else []
        return []

    @field_validator("explanation", mode="before")
    @classmethod
    def _normalize_explanation(cls, value: object) -> str:
        if value is None:
            return ""
        return " ".join(str(value).split()).strip()

    @field_validator("recommended_action", "rewrite_focus", "raw_response", mode="before")
    @classmethod
    def _normalize_optional_text(cls, value: object) -> str | None:
        if value is None:
            return None
        value = " ".join(str(value).split()).strip()
        return value or None

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

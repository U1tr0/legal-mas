from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from src.generator.schemas import GeneratedAnswer
from src.planner.schemas import PlannerOutput
from src.verifier.schemas import VerificationResult


PipelineStatus = Literal["success", "needs_clarification", "refused", "failed"]


class PipelineTraceStep(BaseModel):
    step: str
    status: str
    detail: str | None = None
    iteration: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("step", "status", mode="before")
    @classmethod
    def _normalize_required_text(cls, value: object) -> str:
        if value is None:
            return ""
        return " ".join(str(value).split()).strip()

    @field_validator("detail", mode="before")
    @classmethod
    def _normalize_optional_text(cls, value: object) -> str | None:
        if value is None:
            return None
        value = " ".join(str(value).split()).strip()
        return value or None


class PipelineIterations(BaseModel):
    retrieval_calls: int = 0
    generation_calls: int = 0
    verification_calls: int = 0
    rewrite_cycles: int = 0
    retrieve_more_cycles: int = 0


class PipelineResult(BaseModel):
    status: PipelineStatus = "failed"
    final_answer: str | None = None
    planner_output: PlannerOutput | None = None
    retrieval_result: dict[str, Any] | None = None
    generated_answer: GeneratedAnswer | None = None
    verification_result: VerificationResult | None = None
    iterations: PipelineIterations = Field(default_factory=PipelineIterations)
    stop_reason: str | None = None
    trace: list[PipelineTraceStep] = Field(default_factory=list)

    @field_validator("final_answer", "stop_reason", mode="before")
    @classmethod
    def _normalize_optional_text(cls, value: object) -> str | None:
        if value is None:
            return None
        value = " ".join(str(value).split()).strip()
        return value or None

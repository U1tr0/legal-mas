from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.generator import LegalAnswerGenerator
from src.generator.schemas import GeneratedAnswer
from src.planner import LegalPlanner, PlannerOutput
from src.retrieval import LawGuidedRetriever
from src.verifier import LegalAnswerVerifier
from src.verifier.schemas import VerificationResult

from .schemas import PipelineIterations, PipelineResult, PipelineTraceStep


@dataclass
class _RunState:
    planner_output: PlannerOutput | None = None
    retrieval_result: dict[str, Any] | None = None
    generated_answer: GeneratedAnswer | None = None
    verification_result: VerificationResult | None = None
    iterations: PipelineIterations = field(default_factory=PipelineIterations)
    trace: list[PipelineTraceStep] = field(default_factory=list)


class LegalDecisionPipeline:
    def __init__(
        self,
        planner: LegalPlanner,
        retriever: LawGuidedRetriever,
        generator: LegalAnswerGenerator,
        verifier: LegalAnswerVerifier,
        *,
        initial_law_top_k: int = 5,
        initial_case_top_k: int = 5,
        expanded_law_top_k: int = 8,
        expanded_case_top_k: int = 12,
        max_generation_calls: int = 2,
        max_retrieval_calls: int = 2,
        max_rewrite_cycles: int = 1,
        max_retrieve_more_cycles: int = 1,
    ) -> None:
        self.planner = planner
        self.retriever = retriever
        self.generator = generator
        self.verifier = verifier
        self.initial_law_top_k = initial_law_top_k
        self.initial_case_top_k = initial_case_top_k
        self.expanded_law_top_k = max(expanded_law_top_k, initial_law_top_k)
        self.expanded_case_top_k = max(expanded_case_top_k, initial_case_top_k)
        self.max_generation_calls = max_generation_calls
        self.max_retrieval_calls = max_retrieval_calls
        self.max_rewrite_cycles = max_rewrite_cycles
        self.max_retrieve_more_cycles = max_retrieve_more_cycles

    @staticmethod
    def _safe_text(value: Any) -> str | None:
        if value is None:
            return None
        value = " ".join(str(value).split()).strip()
        return value or None

    @staticmethod
    def _should_stop_for_planner(plan: PlannerOutput) -> tuple[str | None, str | None]:
        if plan.domain == "out_of_scope":
            return "refused", "planner_out_of_scope"
        if plan.needs_clarification:
            return "needs_clarification", plan.clarification_reason or "planner_needs_clarification"
        return None, None

    @staticmethod
    def _verdict_to_status(verdict: str) -> tuple[str, str]:
        if verdict == "ASK_USER":
            return "needs_clarification", "verifier_requested_clarification"
        if verdict == "REFUSE":
            return "refused", "verifier_refused"
        return "failed", f"unresolved_verdict_{verdict.lower()}"

    @staticmethod
    def _build_trace_step(
        step: str,
        status: str,
        iteration: int,
        detail: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PipelineTraceStep:
        return PipelineTraceStep(
            step=step,
            status=status,
            iteration=iteration,
            detail=detail,
            metadata=metadata or {},
        )

    def _finalize_result(
        self,
        status: str,
        stop_reason: str,
        state: _RunState,
    ) -> PipelineResult:
        final_answer = None
        if status == "success" and state.generated_answer:
            final_answer = self._safe_text(state.generated_answer.final_answer)

        return PipelineResult(
            status=status,
            final_answer=final_answer,
            planner_output=state.planner_output,
            retrieval_result=state.retrieval_result,
            generated_answer=state.generated_answer,
            verification_result=state.verification_result,
            iterations=state.iterations,
            stop_reason=stop_reason,
            trace=state.trace,
        )

    def run(self, query: str) -> PipelineResult:
        state = _RunState()

        state.trace.append(
            self._build_trace_step(
                step="pipeline_start",
                status="started",
                iteration=1,
                detail="Starting end-to-end legal decision pipeline.",
                metadata={"query": query},
            )
        )

        state.planner_output = self.planner.plan(query)
        state.trace.append(
            self._build_trace_step(
                step="planner",
                status="completed",
                iteration=1,
                metadata={
                    "domain": state.planner_output.domain,
                    "query_type": state.planner_output.query_type,
                    "needs_clarification": state.planner_output.needs_clarification,
                },
            )
        )

        planner_status, planner_reason = self._should_stop_for_planner(state.planner_output)
        if planner_status and planner_reason:
            state.trace.append(
                self._build_trace_step(
                    step="pipeline_stop",
                    status=planner_status,
                    iteration=1,
                    detail=planner_reason,
                )
            )
            return self._finalize_result(planner_status, planner_reason, state)

        law_top_k = self.initial_law_top_k
        case_top_k = self.initial_case_top_k
        rewrite_focus: str | None = None
        need_retrieval = True

        while True:
            retrieval_iteration = state.iterations.retrieval_calls
            if need_retrieval:
                retrieval_iteration = state.iterations.retrieval_calls + 1
                state.retrieval_result = self.retriever.retrieve_from_plan(
                    state.planner_output,
                    law_top_k=law_top_k,
                    case_top_k=case_top_k,
                )
                state.iterations.retrieval_calls += 1
                state.trace.append(
                    self._build_trace_step(
                        step="retrieval",
                        status="completed",
                        iteration=retrieval_iteration,
                        metadata={
                            "law_top_k": law_top_k,
                            "case_top_k": case_top_k,
                            "law_hits": len((state.retrieval_result.get("law_results") or [])),
                            "case_hits": len((state.retrieval_result.get("case_results") or [])),
                        },
                    )
                )
                need_retrieval = False

            generation_iteration = state.iterations.generation_calls + 1
            state.generated_answer = self.generator.generate(
                query,
                state.planner_output,
                state.retrieval_result,
                rewrite_focus=rewrite_focus,
            )
            state.iterations.generation_calls += 1
            state.trace.append(
                self._build_trace_step(
                    step="generator",
                    status="completed",
                    iteration=generation_iteration,
                    detail="rewrite" if rewrite_focus else "initial_generation",
                    metadata={
                        "rewrite_focus": rewrite_focus,
                        "answer_confidence": state.generated_answer.confidence,
                    },
                )
            )

            verification_iteration = state.iterations.verification_calls + 1
            state.verification_result = self.verifier.verify(
                query,
                state.planner_output,
                state.retrieval_result,
                state.generated_answer,
            )
            state.iterations.verification_calls += 1
            state.trace.append(
                self._build_trace_step(
                    step="verifier",
                    status="completed",
                    iteration=verification_iteration,
                    metadata={
                        "verdict": state.verification_result.verdict,
                        "confidence": state.verification_result.confidence,
                        "problems": state.verification_result.problems,
                    },
                )
            )

            verdict = state.verification_result.verdict
            if verdict == "ACCEPT":
                state.trace.append(
                    self._build_trace_step(
                        step="pipeline_stop",
                        status="success",
                        iteration=max(retrieval_iteration, generation_iteration, verification_iteration),
                        detail="verifier_accept",
                    )
                )
                return self._finalize_result("success", "verifier_accept", state)

            if verdict == "REWRITE":
                if (
                    state.iterations.rewrite_cycles >= self.max_rewrite_cycles
                    or state.iterations.generation_calls >= self.max_generation_calls
                ):
                    state.trace.append(
                        self._build_trace_step(
                            step="pipeline_stop",
                            status="failed",
                            iteration=verification_iteration,
                            detail="rewrite_limit_reached",
                        )
                    )
                    return self._finalize_result("failed", "rewrite_limit_reached", state)

                state.iterations.rewrite_cycles += 1
                rewrite_focus = (
                    state.verification_result.rewrite_focus
                    or state.verification_result.recommended_action
                    or "Сделай ответ осторожнее и ближе к найденному evidence."
                )
                state.trace.append(
                    self._build_trace_step(
                        step="orchestrator_decision",
                        status="rewrite",
                        iteration=state.iterations.rewrite_cycles,
                        detail="verifier_requested_rewrite",
                        metadata={"rewrite_focus": rewrite_focus},
                    )
                )
                continue

            if verdict == "RETRIEVE_MORE":
                if (
                    state.iterations.retrieve_more_cycles >= self.max_retrieve_more_cycles
                    or state.iterations.retrieval_calls >= self.max_retrieval_calls
                ):
                    state.trace.append(
                        self._build_trace_step(
                            step="pipeline_stop",
                            status="failed",
                            iteration=verification_iteration,
                            detail="retrieve_more_limit_reached",
                        )
                    )
                    return self._finalize_result("failed", "retrieve_more_limit_reached", state)

                state.iterations.retrieve_more_cycles += 1
                rewrite_focus = None
                law_top_k = self.expanded_law_top_k
                case_top_k = self.expanded_case_top_k
                need_retrieval = True
                state.trace.append(
                    self._build_trace_step(
                        step="orchestrator_decision",
                        status="retrieve_more",
                        iteration=state.iterations.retrieve_more_cycles,
                        detail="verifier_requested_more_evidence",
                        metadata={"law_top_k": law_top_k, "case_top_k": case_top_k},
                    )
                )
                continue

            if verdict in {"ASK_USER", "REFUSE"}:
                status, stop_reason = self._verdict_to_status(verdict)
                state.trace.append(
                    self._build_trace_step(
                        step="pipeline_stop",
                        status=status,
                        iteration=verification_iteration,
                        detail=stop_reason,
                    )
                )
                return self._finalize_result(status, stop_reason, state)

            state.trace.append(
                self._build_trace_step(
                    step="pipeline_stop",
                    status="failed",
                    iteration=verification_iteration,
                    detail=f"unexpected_verdict_{verdict.lower()}",
                )
            )
            return self._finalize_result("failed", f"unexpected_verdict_{verdict.lower()}", state)

"""Phase 4 eval harness: orchestrates citation grounding, faithfulness,
completeness, and relevance checks for one completed research run into an
EvalReport.

Optional LangSmith tracing for the LLM-judge calls: set
LANGCHAIN_TRACING_V2=true, LANGCHAIN_API_KEY, and (optionally)
LANGCHAIN_PROJECT — langchain-core picks these up globally, no extra code
needed here.
"""
from __future__ import annotations

from datetime import datetime, timezone

from engine.models import LEAD_MODEL, estimate_cost_usd
from engine.state import TokenUsage
from eval.citation_coverage import run_citation_coverage_check
from eval.completeness import run_completeness_check
from eval.faithfulness import run_faithfulness_checks
from eval.grounding import run_grounding_checks
from eval.loader import load_run
from eval.relevance import run_relevance_check
from eval.schema import EvalReport


async def evaluate_run(
    run_id: str, lead_model: str = LEAD_MODEL, strict: bool = True
) -> EvalReport:
    """Load `run_id`, run grounding + faithfulness + completeness + relevance
    checks, and build an EvalReport.

    `passed` requires zero ungrounded findings and, by default, zero unfaithful
    citations. Set `strict=False` only for exploratory runs where report
    faithfulness should be scored but not fail the run. Completeness and
    citation coverage. Set `strict=False` only for exploratory runs where
    report-level quality issues should be scored but not fail the run.
    Completeness and relevance are scored but do not affect `passed`.
    """
    run_data = await load_run(run_id)

    grounding_results = await run_grounding_checks(run_data.findings)
    faithfulness_results, uncited_sentences, faithfulness_usage = await run_faithfulness_checks(
        run_data.report, run_data.findings, lead_model
    )
    citation_coverage, citation_coverage_usage = await run_citation_coverage_check(
        run_data.report, lead_model
    )
    completeness_result, completeness_usage = await run_completeness_check(
        run_data.query, run_data.report, lead_model
    )
    relevance_result, relevance_usage = await run_relevance_check(
        run_data.query, run_data.report, lead_model
    )

    ungrounded_count = sum(1 for g in grounding_results if not g.grounded)
    unfaithful_count = sum(1 for f in faithfulness_results if not f.faithful)
    uncited_factual_count = len(citation_coverage.uncited_factual_claims)

    failure_reasons = []
    if ungrounded_count:
        failure_reasons.append(f"{ungrounded_count} ungrounded claim(s)")
    if strict and unfaithful_count:
        failure_reasons.append(f"{unfaithful_count} unfaithful citation(s)")
    if strict and uncited_factual_count:
        failure_reasons.append(f"{uncited_factual_count} uncited factual claim(s)")

    passed = (
        ungrounded_count == 0
        and (not strict or (unfaithful_count == 0 and uncited_factual_count == 0))
    )

    eval_token_usage: list[TokenUsage] = [
        *faithfulness_usage,
        *citation_coverage_usage,
        *completeness_usage,
        *([relevance_usage] if relevance_usage else []),
    ]

    return EvalReport(
        run_id=run_id,
        query=run_data.query,
        generated_at=datetime.now(timezone.utc).isoformat(),
        grounding_results=grounding_results,
        faithfulness_results=faithfulness_results,
        uncited_sentences=uncited_sentences,
        citation_coverage=citation_coverage,
        completeness=completeness_result,
        relevance=relevance_result,
        total_findings=len(run_data.findings),
        ungrounded_count=ungrounded_count,
        unfaithful_count=unfaithful_count,
        passed=passed,
        failure_reasons=failure_reasons,
        eval_model=lead_model,
        eval_input_tokens=sum(u["input_tokens"] for u in eval_token_usage),
        eval_output_tokens=sum(u["output_tokens"] for u in eval_token_usage),
        eval_cost_usd=estimate_cost_usd(eval_token_usage),
    )

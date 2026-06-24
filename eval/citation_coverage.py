"""LLM-as-judge citation coverage for research reports.

Faithfulness checks whether cited sentences are supported. Citation coverage
checks the other side: whether uncited sentences contain factual claims that
should have had citations.
"""
from __future__ import annotations

import asyncio

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from engine.models import LEAD_MODEL, make_chat_model, structured_output_kwargs
from engine.state import TokenUsage
from engine.usage import usage_from_message
from eval.report_parsing import extract_citation_indices, split_body_and_references, split_sentences
from eval.schema import CitationCoverageIssue, CitationCoverageResult


class _CitationNeedVerdict(BaseModel):
    citation_required: bool = Field(
        description=(
            "True if the uncited sentence makes a factual claim that should be "
            "supported by a source citation."
        )
    )
    reasoning: str = Field(description="One sentence justification")


_COVERAGE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are checking citation coverage in a research report. You will see one "
        "uncited sentence. Mark citation_required=true if it states a factual claim "
        "that a reader would expect to be sourced, such as numbers, dates, named "
        "entities, causal claims, predictions, comparisons, or concrete assertions. "
        "Mark false for section framing, transitions, generic caveats, recommendations, "
        "or meta-claims about the report's own evidence gaps (for example: 'the "
        "findings do not provide Tesla data' or 'this evidence is insufficient') "
        "unless the sentence also introduces a new external factual detail.",
    ),
    ("human", "Section: {section}\n\nUncited sentence:\n{sentence}"),
])


async def run_citation_coverage_check(
    report: str,
    lead_model: str = LEAD_MODEL,
) -> tuple[CitationCoverageResult, list[TokenUsage]]:
    """Find uncited factual claims and compute citation coverage.

    coverage_score = cited_sentence_count / (cited_sentence_count + uncited factual claims)
    and is 1.0 when the report contains no factual sentences by this heuristic.
    """
    body, _ = split_body_and_references(report)
    sentence_rows = split_sentences(body)
    cited_sentence_count = sum(
        1 for sentence, _section in sentence_rows if extract_citation_indices(sentence)
    )
    uncited = [
        (sentence, section)
        for sentence, section in sentence_rows
        if not extract_citation_indices(sentence)
    ]
    if not uncited:
        return (
            CitationCoverageResult(
                coverage_score=1.0,
                cited_sentence_count=cited_sentence_count,
                uncited_factual_claims=[],
            ),
            [],
        )

    judge_llm = make_chat_model(lead_model, temperature=0)
    chain = _COVERAGE_PROMPT | judge_llm.with_structured_output(
        _CitationNeedVerdict, **structured_output_kwargs(lead_model), include_raw=True
    )

    async def _judge(
        sentence: str,
        section: str,
    ) -> tuple[CitationCoverageIssue | None, TokenUsage | None]:
        raw = await chain.ainvoke({"sentence": sentence, "section": section})
        assert isinstance(raw, dict)
        verdict: _CitationNeedVerdict = raw["parsed"]
        usage = usage_from_message(raw["raw"], "citation_coverage", lead_model)
        issue = (
            CitationCoverageIssue(
                sentence=sentence,
                section=section,
                reasoning=verdict.reasoning,
            )
            if verdict.citation_required
            else None
        )
        return issue, usage

    results = await asyncio.gather(*[_judge(sentence, section) for sentence, section in uncited])

    issues: list[CitationCoverageIssue] = []
    token_usage: list[TokenUsage] = []
    for issue, usage in results:
        if issue:
            issues.append(issue)
        if usage:
            token_usage.append(usage)

    denominator = cited_sentence_count + len(issues)
    coverage_score = cited_sentence_count / denominator if denominator else 1.0
    return (
        CitationCoverageResult(
            coverage_score=coverage_score,
            cited_sentence_count=cited_sentence_count,
            uncited_factual_claims=issues,
        ),
        token_usage,
    )

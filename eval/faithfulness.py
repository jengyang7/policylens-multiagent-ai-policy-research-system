"""LLM-as-judge faithfulness check (Phase 4 eval harness).

For each [i]-cited sentence in the synthesizer's report, verify it is
faithful to the Finding(s) behind citation [i] (resolved via the report's own
"## References" section, since the synthesizer free-forms its [i] numbering
based on the compacted summary rather than the original findings-list order).
Sentences with no citation marker are surfaced separately as informational
"uncited sentences" and are not judged or counted as unfaithful.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from engine.models import LEAD_MODEL, make_chat_model, structured_output_kwargs
from engine.state import SubtaskFinding, TokenUsage
from engine.usage import usage_from_message
from eval.report_parsing import (
    extract_citation_indices,
    parse_references,
    split_body_and_references,
    split_sentences,
    strip_citation_markers,
)
from eval.schema import FaithfulnessVerdict, UncitedSentence


class _JudgeVerdict(BaseModel):
    """Structured-output schema for the LLM faithfulness judge."""

    faithful: bool = Field(
        description=(
            "True if the sentence is fully supported by the provided source "
            "findings, with no added or contradicted information."
        )
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this verdict, 0-1")
    reasoning: str = Field(description="One or two sentence justification")


_JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a fact-checking judge. You will be shown ONE sentence from a "
        "research report (with its [i] citation markers removed) and the "
        "source finding(s) that the report cited for that sentence.\n\n"
        "Determine whether the sentence is FAITHFUL to the source finding(s).\n\n"
        "FAITHFUL — judge the sentence faithful even if the wording differs "
        "from the source, including:\n"
        "- Rephrasing, summarizing, or combining multiple findings into one "
        "sentence\n"
        "- Interpretive framing or labels for a claim that IS supported (e.g. "
        "calling a supported claim 'a major trend', 'a key shift', or saying "
        "the source 'forecasts' or 'explicitly frames' something it does "
        "state)\n"
        "- Generalizing a specific supported claim into a slightly broader "
        "statement, as long as the broader statement doesn't introduce new "
        "facts, numbers, dates, or named entities\n\n"
        "UNFAITHFUL — judge the sentence unfaithful only if it:\n"
        "- States new factual specifics (numbers, dates, names, causal "
        "claims, scope, or attributions) that are not present in any source "
        "finding\n"
        "- Contradicts a source finding\n"
        "- Overstates certainty (e.g. turning a prediction into a stated "
        "fact)\n\n"
        "When weighing framing/labeling language against new facts, focus "
        "only on whether every concrete fact in the sentence traces back to "
        "the source findings — ignore differences in tone, labels, or "
        "framing.",
    ),
    (
        "human",
        "Report sentence:\n{sentence}\n\nSource finding(s):\n{findings_text}",
    ),
])


def _format_findings_for_judge(findings: list[SubtaskFinding]) -> str:
    lines = []
    for f in findings:
        lines.append(f"- Claim: {f['claim']}\n  Evidence: {f['evidence_span']}")
    return "\n".join(lines)


async def run_faithfulness_checks(
    report: str,
    findings: list[SubtaskFinding],
    lead_model: str = LEAD_MODEL,
) -> tuple[list[FaithfulnessVerdict], list[UncitedSentence], list[TokenUsage]]:
    """Check faithfulness of every [i]-cited sentence in `report` against `findings`.

    Returns (verdicts, uncited_sentences, token_usage). No LLM call is made for
    sentences with no citation marker, or whose citation index can't be
    resolved to a Finding (those get an automatic faithful=False verdict).
    """
    body, _ = split_body_and_references(report)
    citation_map = parse_references(report)

    findings_by_url: dict[str, list[SubtaskFinding]] = {}
    for f in findings:
        findings_by_url.setdefault(f["citation_url"], []).append(f)

    verdicts: list[FaithfulnessVerdict] = []
    uncited: list[UncitedSentence] = []
    token_usage: list[TokenUsage] = []
    chain = None  # built lazily on first sentence that needs the judge

    for sentence, section in split_sentences(body):
        indices = extract_citation_indices(sentence)
        if not indices:
            uncited.append(UncitedSentence(sentence=sentence, section=section))
            continue

        clean_sentence = strip_citation_markers(sentence)

        candidate_findings: list[SubtaskFinding] = []
        unresolved_index: int | None = None
        for index in indices:
            ref = citation_map.get(index)
            if ref is not None:
                candidate_findings.extend(findings_by_url.get(ref.url, []))
            elif 1 <= index <= len(findings):
                # The synthesizer's own References list omitted (or never wrote)
                # this index — fall back to the same 1-indexed mapping into
                # `findings` that _rebuild_references uses, so a missing/
                # malformed References section doesn't auto-fail every citation.
                candidate_findings.append(findings[index - 1])
            else:
                unresolved_index = index

        if not candidate_findings:
            reasoning = (
                f"citation [{unresolved_index}] not found in the References section"
                if unresolved_index is not None
                else "no Finding shares the citation_url referenced by this sentence"
            )
            verdicts.append(
                FaithfulnessVerdict(
                    citation_index=indices[0],
                    report_sentence=clean_sentence,
                    matched_finding_claims=[],
                    faithful=False,
                    confidence=1.0,
                    reasoning=reasoning,
                )
            )
            continue

        if chain is None:
            judge_llm = make_chat_model(lead_model, temperature=0)
            chain = _JUDGE_PROMPT | judge_llm.with_structured_output(
                _JudgeVerdict, **structured_output_kwargs(lead_model), include_raw=True
            )

        raw = await chain.ainvoke({
            "sentence": clean_sentence,
            "findings_text": _format_findings_for_judge(candidate_findings),
        })
        assert isinstance(raw, dict)  # include_raw=True returns {"raw", "parsed", "parsing_error"}
        verdict: _JudgeVerdict = raw["parsed"]
        usage = usage_from_message(raw["raw"], "faithfulness", lead_model)
        if usage:
            token_usage.append(usage)

        verdicts.append(
            FaithfulnessVerdict(
                citation_index=indices[0],
                report_sentence=clean_sentence,
                matched_finding_claims=[f["claim"] for f in candidate_findings],
                faithful=verdict.faithful,
                confidence=verdict.confidence,
                reasoning=verdict.reasoning,
            )
        )

    return verdicts, uncited, token_usage

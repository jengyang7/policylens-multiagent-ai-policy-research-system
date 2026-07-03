"""Bounded evidence-quality audit before synthesis.

This is the supervisor feedback loop for normal verified research and optional
debate runs. It reviews coverage once, then either approves synthesis or emits
up to three concrete follow-up questions for one targeted research fan-out.
There is deliberately no unrestricted reflection loop.
"""
from __future__ import annotations

from collections import Counter
from urllib.parse import urlparse

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from engine.models import LEAD_MODEL, make_chat_model, structured_output_kwargs
from engine.nodes.debate import format_transcript
from engine.state import EvidenceAuditResult, ResearchState
from engine.usage import usage_from_message

MAX_GAP_QUESTIONS = 3

# Below this many distinct source domains across all findings, the audit input
# carries an explicit low-diversity warning: a report resting on one or two
# pages is exactly the failure the audit exists to catch, and the LLM judge
# has missed it when left to infer diversity from a bare URL list.
MIN_DISTINCT_DOMAINS = 3


class EvidenceAuditDecision(BaseModel):
    sufficient: bool
    assessment: str
    gap_questions: list[str]


_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are the evidence-quality supervisor for an AI policy and regulation "
        "research system. Audit the collected research BEFORE the final report is "
        "written. This is a bounded quality gate, not an invitation to endlessly "
        "expand scope.\n\n"
        "Assess whether the evidence is sufficient to answer the user's actual "
        "question responsibly. Check:\n"
        "- coverage of every material planned sub-question, especially subtasks "
        "with zero findings\n"
        "- authority and diversity of sources, preferring primary laws, regulators, "
        "courts, standards bodies, and official guidance. Evidence drawn from "
        f"fewer than {MIN_DISTINCT_DOMAINS} distinct domains, or a subtask "
        "answered entirely by a single non-official source (aggregator, blog, "
        "content site), is normally insufficient — emit gap questions that "
        "explicitly target primary/official sources for those claims\n"
        "- missing jurisdictions, comparison legs, legal status, dates, affected "
        "actors, obligations, exceptions, enforcement, or practical implications "
        "that the query specifically requires\n"
        "- contradictions or claims that need another source before synthesis\n\n"
        "Do not demand exhaustive research or invent gaps outside the user's scope. "
        "A report can be sufficient while transparently acknowledging minor limits. "
        f"If material evidence is missing, return 1–{MAX_GAP_QUESTIONS} short, "
        "self-contained web-search questions that directly target the highest-value "
        "gaps. Do not re-ask questions already answered by the findings. If the "
        "evidence is sufficient, return an empty gap_questions list. Keep assessment "
        "to 1–3 concise sentences.",
    ),
    (
        "human",
        "Research query: {query}\n\n"
        "Planned subtask coverage:\n{coverage}\n\n"
        "Compacted research summary:\n{summary}\n\n"
        "Collected source URLs:\n{sources}\n\n"
        "Optional debate transcript:\n{transcript}",
    ),
])


def _domain(url: str) -> str:
    hostname = (urlparse(url).hostname or "").lower()
    return hostname.removeprefix("www.")


def _coverage_text(state: ResearchState) -> str:
    findings = state.get("findings", [])
    counts = Counter(f["subtask"] for f in findings)
    domains: dict[str, set[str]] = {}
    for f in findings:
        domains.setdefault(f["subtask"], set()).add(_domain(f["citation_url"]))
    subtasks = state.get("subtasks", [])
    if not subtasks:
        return "(no planned subtasks)"
    return "\n".join(
        f"- {question}: {counts[question]} findings"
        f" from {len(domains.get(question, set()))} source domain(s)"
        for question in subtasks
    )


def _source_text(state: ResearchState) -> str:
    findings = state.get("findings", [])
    urls = list(dict.fromkeys(f["citation_url"] for f in findings))
    if not urls:
        return "(no sources)"
    lines = [f"- {url}" for url in urls]
    distinct = {_domain(url) for url in urls}
    if len(distinct) < MIN_DISTINCT_DOMAINS:
        lines.append(
            f"\nWARNING: all {len(findings)} findings come from only "
            f"{len(distinct)} distinct domain(s). Treat source diversity as "
            "materially insufficient unless the query is answerable from a "
            "single authoritative source."
        )
    return "\n".join(lines)


def evidence_audit(state: ResearchState) -> dict[str, object]:
    """Approve synthesis or plan one bounded targeted follow-up research round."""
    model = state.get("lead_model", LEAD_MODEL)
    llm = make_chat_model(model, temperature=0)
    chain = _PROMPT | llm.with_structured_output(
        EvidenceAuditDecision, include_raw=True, **structured_output_kwargs(model)
    )
    turns = state.get("debate_turns", [])
    raw = chain.invoke({
        "query": state["query"],
        "coverage": _coverage_text(state),
        "summary": state.get("summary", ""),
        "sources": _source_text(state),
        "transcript": format_transcript(turns) if turns else "(debate mode was not used)",
    })
    assert isinstance(raw, dict)
    result: EvidenceAuditDecision | None = raw["parsed"]
    usage = usage_from_message(raw["raw"], "evidence_audit", model)

    if result is None:
        audit = EvidenceAuditResult(
            sufficient=True,
            assessment="The audit could not be parsed; continuing without follow-up research.",
            gap_questions=[],
        )
    else:
        gaps = result.gap_questions[:MAX_GAP_QUESTIONS]
        audit = EvidenceAuditResult(
            # Routing is driven by actionable gaps. Normalize inconsistent model
            # output (for example sufficient=false with no follow-up question)
            # into one coherent state for the API and UI.
            sufficient=not gaps,
            assessment=result.assessment,
            gap_questions=gaps,
        )

    return {
        "evidence_audit": audit,
        "gap_subtasks": audit["gap_questions"],
        "token_usage": [usage] if usage else [],
    }

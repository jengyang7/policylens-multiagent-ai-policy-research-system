"""CONTEXT COMPACTION (layer 2 of the memory stack):
Summarizes raw subagent findings into a compact narrative so the synthesizer
never blows its context window, even on large fan-outs.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from engine.models import LEAD_MODEL
from engine.state import SubtaskFinding

_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are preparing research notes for an analyst who will write a deep, "
        "synthesized report — not a final summary. Given a list of raw findings from "
        "parallel research subagents, produce a compact but information-dense summary "
        "that:\n"
        "- preserves every distinct claim, its exact figures/numbers, and its source URL\n"
        "- groups findings by theme\n"
        "- when multiple findings address the same topic, groups them together and "
        "notes where they agree, where they differ, and by roughly how much\n"
        "Do not add information not present in the findings, and do not drop numeric "
        "detail or source attribution for the sake of brevity.",
    ),
    ("human", "Findings:\n{findings_text}"),
])


def _format_for_compaction(findings: list[SubtaskFinding]) -> str:
    lines = []
    for f in findings:
        lines.append(
            f"[{f['subtask']}]\n"
            f"  Claim: {f['claim']}\n"
            f"  Evidence: {f['evidence_span']}\n"
            f"  Source: {f['citation_url']}"
        )
    return "\n\n".join(lines) if lines else "(no findings)"


def compact_findings(findings: list[SubtaskFinding]) -> str:
    """Summarize raw findings into a compact string for the synthesizer."""
    findings_text = _format_for_compaction(findings)
    llm: ChatOpenAI = ChatOpenAI(model=LEAD_MODEL, temperature=0)
    chain = _PROMPT | llm
    result = chain.invoke({"findings_text": findings_text})
    return str(result.content)

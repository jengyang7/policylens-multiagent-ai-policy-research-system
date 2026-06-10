"""CONTEXT COMPACTION node (layer 2 of the memory stack):
Runs after all parallel subagents finish. Summarizes their raw findings into
state.summary and clears the raw findings list so the synthesizer receives
a compact, context-window-safe input instead of unbounded raw text.
"""
from __future__ import annotations

from engine.memory.compaction import compact_findings
from engine.models import LEAD_MODEL
from engine.state import ResearchState


def compact(state: ResearchState) -> dict[str, object]:
    """Summarize subagent findings → state.summary; trim raw findings."""
    findings = state.get("findings", [])
    if not findings:
        return {"summary": "(no findings to compact)"}

    summary, usage = compact_findings(findings, state.get("lead_model", LEAD_MODEL))  # type: ignore[arg-type]
    # Trim raw findings after compaction — the summary replaces them for synthesis
    return {"summary": summary, "findings": [], "token_usage": [usage] if usage else []}

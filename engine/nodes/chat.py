"""Follow-up chat node (SHORT-TERM MEMORY, layer 3):
Answers follow-up questions grounded in findings and the cited report held in
Postgres checkpointer state. Reads directly from the persisted thread snapshot —
no re-fetching, no embeddings; this is the checkpointer as episodic memory.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from engine.models import LEAD_MODEL
from engine.state import SubtaskFinding

_SYSTEM = (
    "You are a research assistant answering follow-up questions. "
    "You have access to a completed research report and the findings that support it. "
    "Answer only from the provided context. If the context doesn't contain the answer, "
    "say so clearly rather than speculating. Cite sources when relevant."
)


def _format_context(findings: list[SubtaskFinding], report: str) -> str:
    report_section = f"## Research Report\n\n{report}" if report else ""
    if not findings:
        return report_section or "(no research context available)"
    finding_lines = [
        f"- {f['claim']} (source: {f['citation_url']})" for f in findings
    ]
    findings_section = "## Key Findings\n\n" + "\n".join(finding_lines)
    return f"{report_section}\n\n{findings_section}".strip()


async def answer_followup(
    thread_id: str,
    question: str,
    history: list[dict[str, str]],
    checkpointer: AsyncPostgresSaver,
) -> AsyncIterator[str]:
    """Stream a grounded answer to a follow-up question.

    Reads research findings and report from the Postgres checkpointer (layer 3 —
    short-term/episodic memory) for the given thread. Conversation history is
    passed in by the caller (API maintains it per session).
    """
    # EPISODIC MEMORY (layer 3): load persisted research state from checkpointer
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await checkpointer.aget_tuple(config)

    findings: list[SubtaskFinding] = []
    report = ""
    lead_model = LEAD_MODEL
    if snapshot is not None:
        channel_values = snapshot.checkpoint.get("channel_values", {})
        findings = channel_values.get("findings", [])
        report = channel_values.get("report", "")
        lead_model = channel_values.get("lead_model", LEAD_MODEL)

    context = _format_context(findings, report)

    messages = [
        SystemMessage(content=f"{_SYSTEM}\n\n{context}"),
        *[
            HumanMessage(content=m["content"]) if m["role"] == "human"
            else AIMessage(content=m["content"])
            for m in history
        ],
        HumanMessage(content=question),
    ]

    llm: ChatOpenAI = ChatOpenAI(model=lead_model, temperature=0, streaming=True)
    async for chunk in llm.astream(messages):
        if chunk.content:
            yield str(chunk.content)

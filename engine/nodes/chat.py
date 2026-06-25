"""Follow-up chat node (SHORT-TERM MEMORY, layer 3):
Answers follow-up questions grounded in findings and the cited report held in
Postgres checkpointer state. Reads directly from the persisted thread snapshot —
no re-fetching, no embeddings; this is the checkpointer as episodic memory.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from engine.models import LEAD_MODEL, make_chat_model
from engine.state import SubtaskFinding

_SYSTEM = (
    "You are an AI policy and regulation analyst answering follow-up questions. "
    "You have access to a completed policy research report and the findings that support it. "
    "Answer only from the provided context. If the context doesn't contain the answer, "
    "say so clearly rather than speculating. Cite sources when relevant.\n\n"
    "When the context supports it, call out jurisdiction, legal status, effective dates, "
    "affected actors, obligations, enforcement mechanisms, and compliance uncertainty. "
    "Do not present proposals, guidance, voluntary standards, or analysis as binding law.\n\n"
    "Format every reply for a narrow chat bubble:\n"
    "- Default to flowing prose: one or two short paragraphs (2-4 sentences each), "
    "not a wall of bullet points. Summaries and explanations should read like a "
    "person talking, with the key facts woven into sentences.\n"
    "- Only use a Markdown bullet or numbered list when the answer genuinely "
    "enumerates discrete items (e.g. 'list the sources', 'what are the options') — "
    "and keep it short, after a one-sentence lead-in.\n"
    "- Use **bold** sparingly to highlight key terms, names, or numbers.\n"
    "- Keep it concise — no restating the whole report."
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

    llm = make_chat_model(lead_model, temperature=0)
    async for chunk in llm.astream(messages):
        # .text, not str(.content): some providers (Gemini) emit content-block lists
        if chunk.text:
            yield chunk.text

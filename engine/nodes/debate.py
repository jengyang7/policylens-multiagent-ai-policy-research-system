"""Adversarial debate nodes (debate mode).

Two agents from different AI companies argue over the compacted findings
before synthesis: the advocate builds the strongest well-supported answer,
the skeptic attacks evidence quality and overreach. Each node execution is
one turn, so the loop edges in the orchestrator stream every turn to the UI
via the existing stream_mode="updates" SSE pipeline, checkpoint per turn,
and accumulate token usage through the operator.add reducer.

Cross-provider debaters are the point: different pretraining/RLHF lineages
have uncorrelated blind spots, so the skeptic catches gaps a same-model
skeptic would agree with. Models fall back to lead_model when a provider
key is missing.
"""
from __future__ import annotations

from typing import Literal

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from engine.models import LEAD_MODEL, make_chat_model, structured_output_kwargs
from engine.state import (
    DebateTurn,
    DebateVerdict,
    ResearchState,
)
from engine.state import (
    VerdictRow as StateVerdictRow,
)
from engine.usage import usage_from_message

MAX_GAP_QUESTIONS = 5

# Display names for the two debate sides — the underlying `agent` field stays
# "advocate" | "skeptic" (model selection, SSE routing), but everything the
# debaters/judge say and everything the UI shows uses this consistent wording.
_SIDE_LABEL = {"advocate": "Proposition", "skeptic": "Opposition"}

_ADVOCATE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are the PROPOSITION in a structured research debate. Your job is to "
        "argue the strongest, best-supported answer to the research query using "
        "ONLY the claims in the research summary below — never invent facts.\n"
        "- Build a clear position: what does the evidence, taken together, "
        "support most strongly?\n"
        "- Lean on specific claims and figures, and cite them with [1], [2], etc. "
        "matching the numbered source list below — never write out full URLs.\n"
        "- When referring to the research summary's own framing or observations "
        "(as opposed to a specific numbered finding), describe it in plain prose "
        "(e.g. \"the research summary notes...\") — never invent a bracket marker "
        "like [Synthesis] for it.\n"
        "- If the opposition has already spoken, rebut their strongest objections "
        "directly: concede points the evidence cannot answer, and defend points "
        "it can.\n"
        "- Be concise and substantive: 2-4 tight paragraphs, no preamble.",
    ),
    (
        "human",
        "Research query: {query}\n\nResearch summary:\n{summary}\n\n"
        "Sources:\n{sources}\n\n"
        "Debate so far:\n{transcript}\n\nYour turn, Proposition (round {round}).",
    ),
])

_SKEPTIC_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are the OPPOSITION in a structured research debate. Your job is to "
        "stress-test the proposition's last argument using ONLY the research "
        "summary below — never invent counter-facts.\n"
        "- Attack evidence quality: single-source claims, missing data, vague "
        "figures, sources that don't actually support the weight put on them.\n"
        "- Surface gaps, contradictions between findings, and overreach — "
        "places where the proposition's conclusion goes beyond what the summary "
        "states.\n"
        "- Point out what a careful reader would still need to know before "
        "accepting the position.\n"
        "- If you cite specific claims or figures, use [1], [2], etc. matching "
        "the numbered source list below — never write out full URLs.\n"
        "- When referring to the research summary's own framing or observations "
        "(as opposed to a specific numbered finding), describe it in plain prose "
        "(e.g. \"the research summary notes...\") — never invent a bracket marker "
        "like [Synthesis] for it.\n"
        "- Concede genuinely strong points; an opposition that disputes "
        "everything is useless.\n"
        "- Be concise and substantive: 2-4 tight paragraphs, no preamble.",
    ),
    (
        "human",
        "Research query: {query}\n\nResearch summary:\n{summary}\n\n"
        "Sources:\n{sources}\n\n"
        "Debate so far:\n{transcript}\n\nYour turn, Opposition (round {round}).",
    ),
])


def format_transcript(turns: list[DebateTurn]) -> str:
    """Render prior turns for the debaters' prompts and the synthesizer."""
    if not turns:
        return "(the debate is just beginning)"
    return "\n\n".join(
        f"[Round {t['round']} — {_SIDE_LABEL[t['agent']]}]\n{t['content']}" for t in turns
    )


def _format_sources(findings: list[dict[str, str]]) -> str:
    """Numbered source list for [i] citations, matching the synthesizer's numbering."""
    if not findings:
        return "(no sources)"
    return "\n".join(f"[{i}] {f['citation_url']}" for i, f in enumerate(findings, 1))


def _run_turn(
    state: ResearchState, agent: str, model: str, prompt: ChatPromptTemplate
) -> dict[str, object]:
    turns = state.get("debate_turns", [])
    # Advocate speaks at even turn counts, skeptic at odd — same round formula for both
    round_no = len(turns) // 2 + 1
    # Slight temperature for argumentative diversity (every other node runs at 0)
    llm = make_chat_model(model, temperature=0.4)
    chain = prompt | llm
    result: BaseMessage = chain.invoke({
        "query": state["query"],
        "summary": state.get("summary", ""),
        "sources": _format_sources(state.get("findings", [])),  # type: ignore[arg-type]
        "transcript": format_transcript(turns),
        "round": round_no,
    })
    usage = usage_from_message(result, f"debate_{agent}", model)
    # .text, not str(.content): Gemini returns a list of content blocks, which
    # would otherwise render as a raw "[{'type': 'text', ...}]" python literal
    turn = DebateTurn(agent=agent, model=model, round=round_no, content=result.text)
    return {"debate_turns": [turn], "token_usage": [usage] if usage else []}


def debate_advocate(state: ResearchState) -> dict[str, object]:
    """Argue the strongest evidence-backed position (debate mode, turn node)."""
    model = state.get("advocate_model") or state.get("lead_model", LEAD_MODEL)
    return _run_turn(state, "advocate", model, _ADVOCATE_PROMPT)


def debate_skeptic(state: ResearchState) -> dict[str, object]:
    """Challenge evidence quality, gaps, and overreach (debate mode, turn node)."""
    model = state.get("skeptic_model") or state.get("lead_model", LEAD_MODEL)
    return _run_turn(state, "skeptic", model, _SKEPTIC_PROMPT)


# ---------------------------------------------------------------------------
# Debate judgment: after the final round, the (neutral) lead model weighs both
# sides and declares a winner. The verdict is purely informational — it feeds
# the UI verdict card and history; gap planning and synthesis are unaffected.
# ---------------------------------------------------------------------------

class VerdictRow(BaseModel):
    category: str    # short label, e.g. "Evidence Quality"
    assessment: str  # ONE plain-language sentence, no markdown
    winner: Literal["proposition", "opposition", "draw"]


class DebateJudgment(BaseModel):
    rows: list[VerdictRow]  # 3-5 categories, based on what was actually contested
    winner: Literal["proposition", "opposition", "draw"]  # overall verdict


_JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are the neutral JUDGE of a structured research debate between a "
        "PROPOSITION (argues the best-supported answer) and an OPPOSITION "
        "(attacks evidence quality and overreach). Decide who argued better — "
        "judge the ARGUMENTS, not which position you personally favor.\n"
        "- Break your judgment into 3-5 short categories based on what was "
        "actually contested in this debate (e.g. 'Evidence Quality', "
        "'Argument Strength', 'Gap Identification', 'Rebuttals').\n"
        "- For each category, write a 'category' label (1-3 words), a ONE-"
        "sentence 'assessment' (max ~20 words, plain language, no markdown) "
        "summarizing the decisive point, and a 'winner' for that category.\n"
        "- The proposition wins a category if their position held up under "
        "scrutiny with evidence from the summary.\n"
        "- The opposition wins a category if they exposed material gaps, "
        "contradictions, or overreach the proposition could not answer.\n"
        "- Use 'draw' for a category only when genuinely balanced.\n"
        "- Finally, give an overall 'winner' for the debate as a whole.",
    ),
    (
        "human",
        "Research query: {query}\n\nResearch summary:\n{summary}\n\n"
        "Debate transcript:\n{transcript}\n\nYour verdict, Judge.",
    ),
])


def judge_debate(state: ResearchState) -> dict[str, object]:
    """Declare the debate winner from a neutral lead-model judgment (debate mode)."""
    model = state.get("lead_model", LEAD_MODEL)
    llm = make_chat_model(model, temperature=0)
    chain = _JUDGE_PROMPT | llm.with_structured_output(
        DebateJudgment, include_raw=True, **structured_output_kwargs(model)
    )
    raw = chain.invoke({
        "query": state["query"],
        "summary": state.get("summary", ""),
        "transcript": format_transcript(state.get("debate_turns", [])),
    })
    assert isinstance(raw, dict)  # include_raw=True returns {"raw", "parsed", "parsing_error"}
    result: DebateJudgment | None = raw["parsed"]
    usage = usage_from_message(raw["raw"], "judge_debate", model)
    # A parsing failure (model replied without the structured tool call) is rare
    # but not fatal — fall back to an unscored draw rather than crashing the run.
    rows: list[StateVerdictRow] = [
        StateVerdictRow(
            category=row.category,
            assessment=row.assessment,
            winner=row.winner,
        )
        for row in result.rows
    ] if result else []
    winner = result.winner if result else "draw"
    verdict = DebateVerdict(rows=rows, winner=winner, model=model)
    return {"debate_verdict": verdict, "token_usage": [usage] if usage else []}


# ---------------------------------------------------------------------------
# Debate-driven gap research: after the final round, the (neutral) lead model
# distills the skeptic's unresolved objections into concrete follow-up search
# questions. A second subagent fan-out researches them before synthesis, so
# the report answers the debate's open questions with evidence instead of
# leaving them as caveats.
# ---------------------------------------------------------------------------

class GapResearchPlan(BaseModel):
    thinking: str  # Brief reasoning: which objections survived rebuttal and need evidence
    gap_questions: list[str]


_GAP_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a neutral research lead reviewing an adversarial debate over a "
        "research summary. Identify the evidence GAPS that block a confident "
        "answer: objections the opposition raised that the proposition could not "
        "rebut with the existing findings, and facts both sides agreed were "
        "missing.\n"
        f"- In 'gap_questions': write 0–{MAX_GAP_QUESTIONS} follow-up research "
        "questions targeting those gaps. Each must be self-contained, concrete, "
        "and directly answerable via a web search (name the specific data, "
        "comparison, or timeframe needed). Each question is sent VERBATIM to a web "
        "search engine as the query, so keep it short — one sentence, ideally under "
        "20 words.\n"
        "- Do NOT re-ask what the summary already answers, and do not restate "
        "debate rhetoric — only genuinely missing evidence qualifies.\n"
        "- Prioritize empty original subtasks when they are central to the user's "
        "question, especially missing comparison legs (for example, one company "
        "in a 'A vs B' query).\n"
        "- If the debate surfaced no material gaps, return an empty list.",
    ),
    (
        "human",
        "Research query: {query}\n\nResearch summary:\n{summary}\n\n"
        "Original planned subtasks:\n{original_subtasks}\n\n"
        "Original subtasks with zero findings:\n{empty_subtasks}\n\n"
        "Debate transcript:\n{transcript}",
    ),
])


def plan_gap_research(state: ResearchState) -> dict[str, object]:
    """Distill unresolved debate objections into follow-up search questions (debate mode)."""
    model = state.get("lead_model", LEAD_MODEL)
    llm = make_chat_model(model, temperature=0)
    chain = _GAP_PROMPT | llm.with_structured_output(
        GapResearchPlan, include_raw=True, **structured_output_kwargs(model)
    )
    findings = state.get("findings", [])
    answered_subtasks = {f["subtask"] for f in findings}
    empty_subtasks = [q for q in state.get("subtasks", []) if q not in answered_subtasks]
    raw = chain.invoke({
        "query": state["query"],
        "summary": state.get("summary", ""),
        "original_subtasks": "\n".join(f"- {q}" for q in state.get("subtasks", [])),
        "empty_subtasks": "\n".join(f"- {q}" for q in empty_subtasks) or "(none)",
        "transcript": format_transcript(state.get("debate_turns", [])),
    })
    assert isinstance(raw, dict)  # include_raw=True returns {"raw", "parsed", "parsing_error"}
    result: GapResearchPlan | None = raw["parsed"]
    usage = usage_from_message(raw["raw"], "plan_gap_research", model)
    # A parsing failure (model replied without the structured tool call) is rare
    # but not fatal — treat it the same as "no material gaps" rather than crashing.
    gap_questions = result.gap_questions[:MAX_GAP_QUESTIONS] if result else []
    return {
        "gap_subtasks": gap_questions,
        "token_usage": [usage] if usage else [],
    }

"""HUMAN-IN-THE-LOOP clarify nodes (two-node design):

clarify       — calls the LLM once to detect ambiguity and store questions in state.
                Only runs on the FIRST pass; never re-runs on resume.
clarify_wait  — calls interrupt() with the stored questions.
                On first pass: pauses the graph (GraphInterrupt raised).
                On resume via Command(resume=answers): returns answers immediately,
                without re-calling the LLM.

Splitting these two responsibilities across two nodes is the canonical LangGraph
pattern for human-in-the-loop: the node that calls interrupt() is the one that
gets re-run on resume, so all LLM work is safely isolated in the earlier node.
"""
from __future__ import annotations

from datetime import date

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt
from pydantic import BaseModel

from engine.models import LEAD_MODEL
from engine.state import Clarification, ResearchState


class QuestionWithOptions(BaseModel):
    question: str
    options: list[str]  # 3–4 short chip labels (max ~30 chars each)


class ClarifyDecision(BaseModel):
    is_ambiguous: bool
    questions_with_options: list[QuestionWithOptions]  # 0–3 items, only if genuinely useful
    refined_query: str  # original query if not ambiguous; unchanged if ambiguous


_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a research assistant for a deep-research system. Every report this "
        "system produces is already a comprehensive, deeply-researched, cited report — "
        "there is no 'quick summary' vs 'deep dive' choice, and no control over format "
        "or length. NEVER ask about depth, level of detail, report length, or output "
        "format.\n\n"
        "Before research begins, consider whether there are up to 3 clarifying "
        "questions whose answers would meaningfully change WHAT gets researched — "
        "e.g. who the research is for, which scope/angle/sub-topic to prioritize "
        "within a broad subject, what time frame to focus on, or which geography "
        "applies when it's not specified and matters.\n\n"
        "Time frame is often already implied by the query's phrasing — present-tense "
        "wording like 'is', 'current', or 'latest' (e.g. 'How is the job market "
        "now?') already anchors the research to the current state, so do NOT ask a "
        "time-frame question in that case. Only ask about time frame when the query "
        "is genuinely ambiguous about it (e.g. it could reasonably mean a snapshot, "
        "a multi-year trend, or a future outlook, and those would lead to different "
        "research).\n\n"
        "Only ask a question if the different possible answers would lead to "
        "genuinely different research (different sources, different sub-topics, "
        "different framing). If the query is already reasonably well-scoped, or you "
        "cannot think of a question that would actually change the research, set "
        "is_ambiguous=false, leave questions_with_options empty, and use the query "
        "as-is (or lightly refined) — do NOT invent filler questions just to ask "
        "something. It is normal and expected for many queries to need zero "
        "questions.\n\n"
        "For each question you do ask, provide 3–4 short chip options the user can "
        "tap as quick answers. Options must be concise (≤30 chars), mutually exclusive, "
        "and cover the most likely choices. Always include a variety option like "
        "'All of the above' or 'General overview' where appropriate.\n\n"
        "Example of a GOOD question for 'How is the SWE job market in Singapore?':\n"
        "  question: 'Who is this research for?'\n"
        "  options: ['Job seeker', 'Employer / hiring', 'Investor', 'General curiosity']\n\n"
        "Example of a BAD question to NEVER ask (about depth/format, not scope):\n"
        "  question: 'What level of detail do you want?'\n"
        "  options: ['Quick summary', 'Medium detail', 'Deep dive', 'All of the above']\n\n"
        "Keep questions concise (one sentence). Do not ask redundant questions.\n\n"
        "Today's date: {current_date}. Use the actual current year when writing time-related "
        "chip options — never hardcode past years.",
    ),
    ("human", "Query: {query}"),
])


def clarify(state: ResearchState) -> dict[str, object]:
    """Detect ambiguity and store questions+options in state (LLM call — runs ONCE, not on resume)."""
    # If questions are already stored, a previous run already decided — skip the LLM.
    if state.get("clarification_questions"):
        return {}

    llm: ChatOpenAI = ChatOpenAI(model=LEAD_MODEL, temperature=0)
    chain = _PROMPT | llm.with_structured_output(ClarifyDecision, method="function_calling")
    decision: ClarifyDecision = chain.invoke(  # type: ignore[assignment]
        {"query": state["query"], "current_date": date.today().strftime("%B %d, %Y")}
    )

    if not decision.is_ambiguous:
        return {
            "query": decision.refined_query,
            "clarification_questions": [],
            "clarification_options": [],
            "clarifications": [],
        }

    # Store both questions and chip options so the API can forward them to the UI.
    return {
        "clarification_questions": [q.question for q in decision.questions_with_options],
        "clarification_options": [q.options for q in decision.questions_with_options],
    }


def clarify_wait(state: ResearchState) -> dict[str, object]:
    """Interrupt for user answers if questions are pending (re-runs safely on resume)."""
    questions = state.get("clarification_questions", [])
    if not questions:
        return {}

    # interrupt() raises GraphInterrupt on first pass (pauses the graph).
    # On resume via Command(resume=answers), it returns answers immediately.
    answers: list[str] = interrupt(questions)

    clarifications: list[Clarification] = [
        Clarification(question=q, answer=a)
        for q, a in zip(questions, answers)
    ]
    answers_text = "; ".join(
        f"{c['question']} → {c['answer']}" for c in clarifications
    )
    refined = f"{state['query']} (clarifications: {answers_text})"

    return {
        "query": refined,
        "clarifications": clarifications,
        "clarification_questions": [],  # clear after use
    }

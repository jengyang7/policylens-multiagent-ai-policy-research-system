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
from langgraph.types import interrupt
from pydantic import BaseModel

from engine.models import LEAD_MODEL, make_chat_model, structured_output_kwargs
from engine.state import Clarification, ResearchState
from engine.usage import usage_from_message


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
        "Subject/entity identity check (do this FIRST): if the query is a single "
        "word, a short name, an acronym, or a term that could plausibly refer to "
        "more than one distinct real-world entity (a company, product, person, "
        "place, technology, etc.), and researching the wrong entity would produce "
        "an irrelevant report, you MUST treat this as ambiguous and ask a question "
        "to identify which entity is meant — even if the query is very short and "
        "even if one meaning seems most popular or 'obvious'. List the most likely "
        "candidate entities as chip options (be specific, e.g. include what each "
        "one is), plus a final option like 'Something else' or 'General overview "
        "of all of these'.\n\n"
        "Example for the query 'Grab':\n"
        "  question: 'Which \"Grab\" do you mean?'\n"
        "  options: ['Grab (ride-hailing & delivery super-app)', 'GRAB (the card "
        "game)', 'A different company/product/person named Grab', 'General "
        "overview of all of these']\n\n"
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
        "Explicit multi-aspect rule: if the query already names MULTIPLE specific "
        "aspects (e.g. 'property AND labor markets', 'both X and Y', 'economic AND "
        "social'), do NOT ask which one to prioritize — the user has already answered "
        "that by naming both. Similarly, if the query explicitly names a specific "
        "geographic scope or both sides of a border/region, do NOT ask about "
        "geography — it is already specified.\n\n"
        "For each question you do ask, provide 3–4 short chip options the user can "
        "tap as quick answers. Options must be concise (≤30 chars), mutually exclusive, "
        "and cover the most likely choices. Always include a variety option like "
        "'All of the above' or 'General overview' where appropriate.\n\n"
        "Mutually exclusive means no option's range or scope may contain or "
        "overlap with another's — e.g. for a time-frame question, never offer both "
        "'Since launch in 1990 to present' and 'Long-term trend analysis "
        "(1990-2026)', since the first is just a restatement of the second. Pick "
        "ONE way to describe each distinct time frame and make sure every pair of "
        "options is clearly distinguishable at a glance.\n\n"
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

    model = state.get("lead_model", LEAD_MODEL)
    llm = make_chat_model(model, temperature=0)
    chain = _PROMPT | llm.with_structured_output(
        ClarifyDecision, include_raw=True, **structured_output_kwargs(model)
    )
    raw = chain.invoke(
        {"query": state["query"], "current_date": date.today().strftime("%B %d, %Y")}
    )
    assert isinstance(raw, dict)  # include_raw=True returns {"raw", "parsed", "parsing_error"}
    decision: ClarifyDecision = raw["parsed"]
    usage = usage_from_message(raw["raw"], "clarify", model)
    token_usage = [usage] if usage else []

    if not decision.is_ambiguous:
        return {
            "query": decision.refined_query,
            "clarification_questions": [],
            "clarification_options": [],
            "clarifications": [],
            "token_usage": token_usage,
        }

    # Store both questions and chip options so the API can forward them to the UI.
    return {
        "clarification_questions": [q.question for q in decision.questions_with_options],
        "clarification_options": [q.options for q in decision.questions_with_options],
        "token_usage": token_usage,
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

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from engine.models import LEAD_MODEL, make_chat_model, structured_output_kwargs
from engine.state import ResearchState
from engine.usage import usage_from_message


class ResearchPlan(BaseModel):
    thinking: str  # Supervisor's brief reasoning about how to approach the query
    title: str  # Short display title for this research (shown in UI instead of the raw query)
    subtasks: list[str]


_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a research planner for an AI Policy & Regulation Researcher. "
        "The product specializes in AI laws, policy proposals, regulatory guidance, "
        "standards, enforcement actions, compliance obligations, and governance risk. "
        "Given a research query:\n"
        "1. In 'thinking': write 2-3 sentences explaining your approach — what the user "
        "   wants, what key angles to cover, and why you chose these sub-questions.\n"
        "2. In 'title': write a display heading that is a cleaned-up version of the "
        "   user's query — fix casing/grammar and drop filler, but PRESERVE the full "
        "   meaning, framing, and any comparisons, numbers, or timeframes. Do not "
        "   generalize a pointed question into a vague topic label: for 'what "
        "   obligations does the EU AI Act create for high-risk AI systems?' write "
        "   'What Obligations Does the EU AI Act Create for High-Risk AI Systems?', "
        "   NOT 'EU AI Act Overview'. Keep a question as a question (with the '?'); "
        "   use title case; aim for at most ~12 words, shortening only when nothing "
        "   meaningful is lost.\n"
        "3. In 'subtasks': decompose into 3–6 independent, specific sub-questions that "
        "   together fully cover the topic. Each must be self-contained and directly "
        "   answerable via a web search. Do not overlap. Prefer concrete, searchable "
        "   phrasing. Each sub-question is sent VERBATIM to a web search engine as the "
        "   query, so keep it short — one sentence, ideally under 20 words — and avoid "
        "   stacking multiple clauses, comparisons, or caveats into a single sub-question.\n"
        "   CRITICAL: every sub-question must be EVIDENCE-SEEKING — it asks for facts, "
        "   rules, documents, or events that a single web page could plausibly state "
        "   directly. Never emit comparative or evaluative essay questions: no page "
        "   states 'how X compares to Y' or 'how effective X is'. Split a comparison "
        "   into one factual sub-question per side and leave the comparing and judging "
        "   to the final report. BAD: 'How do enforcement outcomes compare between "
        "   Singapore's principles-based approach and prescriptive regimes in the EU "
        "   and China?' GOOD: 'What enforcement actions has Singapore taken for "
        "   AI-related violations?' + 'What penalties does the EU AI Act impose for "
        "   non-compliance?'\n"
        "4. For AI policy/regulatory topics, make the sub-questions cover the most "
        "   decision-relevant angles: jurisdiction, legal status, effective dates, "
        "   obligations, affected actors, enforcement mechanisms, exceptions, unresolved "
        "   proposals, and practical compliance implications. Prefer official regulator, "
        "   legislature, standards-body, court, and credible legal-analysis sources.\n"
        "5. For Singapore AI governance queries, include coverage of international "
        "   alignment/cross-border considerations (e.g. ASEAN, OECD, standards, cross-border "
        "   data flows) and public-private collaboration mechanisms (e.g. IMDA/PDPC/MAS "
        "   consultations, AI Verify Foundation, sandboxes, industry working groups) when "
        "   the user's question is broad enough to need a complete policy landscape.",
    ),
    ("human", "Research query: {query}"),
])


def plan(state: ResearchState) -> dict:  # type: ignore[type-arg]
    """Decompose the research query into parallel sub-questions (plan node)."""
    model = state.get("lead_model", LEAD_MODEL)
    llm = make_chat_model(model, temperature=0)
    chain = _PROMPT | llm.with_structured_output(
        ResearchPlan, include_raw=True, **structured_output_kwargs(model)
    )
    raw = chain.invoke({"query": state["query"]})
    assert isinstance(raw, dict)  # include_raw=True returns {"raw", "parsed", "parsing_error"}
    result: ResearchPlan = raw["parsed"]
    usage = usage_from_message(raw["raw"], "plan", model)
    return {
        "subtasks": result.subtasks,
        "title": result.title,
        "supervisor_thinking": result.thinking,
        "token_usage": [usage] if usage else [],
    }

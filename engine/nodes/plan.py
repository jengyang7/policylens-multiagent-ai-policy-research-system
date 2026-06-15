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
        "You are a research planner. Given a research query:\n"
        "1. In 'thinking': write 2-3 sentences explaining your approach — what the user "
        "   wants, what key angles to cover, and why you chose these sub-questions.\n"
        "2. In 'title': write a display heading that is a cleaned-up version of the "
        "   user's query — fix casing/grammar and drop filler, but PRESERVE the full "
        "   meaning, framing, and any comparisons, numbers, or timeframes. Do not "
        "   generalize a pointed question into a vague topic label: for 'will AI "
        "   replace more software engineers than it creates by 2030?' write 'Will AI "
        "   Replace More Software Engineers Than It Creates by 2030?', NOT 'AI and "
        "   Software Engineering Jobs'. Keep a question as a question (with the '?'); "
        "   use title case; aim for at most ~12 words, shortening only when nothing "
        "   meaningful is lost.\n"
        "3. In 'subtasks': decompose into 3–6 independent, specific sub-questions that "
        "   together fully cover the topic. Each must be self-contained and directly "
        "   answerable via a web search. Do not overlap. Prefer concrete, searchable "
        "   phrasing. Each sub-question is sent VERBATIM to a web search engine as the "
        "   query, so keep it short — one sentence, ideally under 20 words — and avoid "
        "   stacking multiple clauses, comparisons, or caveats into a single sub-question.",
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

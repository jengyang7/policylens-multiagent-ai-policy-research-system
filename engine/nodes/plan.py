from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from engine.models import LEAD_MODEL
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
        "2. In 'title': write a short display title (3-6 words, title case, no trailing "
        "   punctuation) that captures the topic of this research — this replaces the raw "
        "   query in the UI, so make it read naturally as a heading "
        "   (e.g. 'AI Engineering Learning Roadmap' rather than restating the question).\n"
        "3. In 'subtasks': decompose into 3–6 independent, specific sub-questions that "
        "   together fully cover the topic. Each must be self-contained and directly "
        "   answerable via a web search. Do not overlap. Prefer concrete, searchable phrasing.",
    ),
    ("human", "Research query: {query}"),
])


def plan(state: ResearchState) -> dict:  # type: ignore[type-arg]
    """Decompose the research query into parallel sub-questions (plan node)."""
    model = state.get("lead_model", LEAD_MODEL)
    llm: ChatOpenAI = ChatOpenAI(model=model, temperature=0)
    chain = _PROMPT | llm.with_structured_output(
        ResearchPlan, method="function_calling", include_raw=True
    )
    raw = chain.invoke({"query": state["query"]})
    result: ResearchPlan = raw["parsed"]
    usage = usage_from_message(raw["raw"], "plan", model)
    return {
        "subtasks": result.subtasks,
        "title": result.title,
        "supervisor_thinking": result.thinking,
        "token_usage": [usage] if usage else [],
    }

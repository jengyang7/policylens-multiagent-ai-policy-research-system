from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from engine.extraction import FindingList
from engine.models import SUBAGENT_MODEL
from engine.state import SubagentInput, SubtaskFinding, TokenUsage
from engine.tools.fetch import fetch
from engine.tools.search import search
from engine.usage import usage_from_message

_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a research extraction agent. Given a sub-question and web content, "
        "extract every relevant finding. Each finding requires:\n"
        "- claim: a clear, factual statement directly supported by the content\n"
        "- evidence_span: the exact quote or passage from the content that supports the claim\n"
        "- citation_url: the URL of the source\n\n"
        "Only include findings directly supported by the provided content. "
        "If no relevant findings exist, return an empty list.",
    ),
    (
        "human",
        "Sub-question: {question}\n\nSource URL: {url}\n\nContent:\n{content}",
    ),
])


def subagent(state: SubagentInput) -> dict[str, object]:
    """search → fetch → extract validated Findings for one sub-question.

    Each Send fan-out invocation handles exactly one subtask. Results are merged
    back into ResearchState.findings via the operator.add reducer.
    """
    question = state["question"]
    llm: ChatOpenAI = ChatOpenAI(model=SUBAGENT_MODEL, temperature=0)
    chain = _PROMPT | llm.with_structured_output(
        FindingList, method="function_calling", include_raw=True
    )

    results = search(question, max_results=4)
    findings: list[SubtaskFinding] = []
    input_tokens = output_tokens = cached_tokens = 0

    for result in results:
        url: str = result.get("url", "")
        if not url:
            continue

        # Prefer the fetched body; fall back to Tavily's snippet if fetch fails
        content: str = fetch(url) or result.get("content", "")
        if not content:
            continue

        try:
            raw = chain.invoke(
                {"question": question, "url": url, "content": content[:6_000]}
            )
            usage = usage_from_message(raw["raw"], "subagent", SUBAGENT_MODEL)
            if usage:
                input_tokens += usage["input_tokens"]
                output_tokens += usage["output_tokens"]
                cached_tokens += usage["cached_tokens"]
            extracted: FindingList | None = raw["parsed"]
            if extracted is None:
                continue
            for f in extracted.findings:
                findings.append(
                    SubtaskFinding(
                        subtask=question,
                        claim=f.claim,
                        evidence_span=f.evidence_span,
                        citation_url=str(f.citation_url),
                    )
                )
        except Exception:
            # Silently drop unextractable sources; don't let one bad page fail the subtask
            continue

    token_usage: list[TokenUsage] = []
    if input_tokens or output_tokens:
        token_usage.append(TokenUsage(
            node="subagent", model=SUBAGENT_MODEL,
            input_tokens=input_tokens, output_tokens=output_tokens, cached_tokens=cached_tokens,
        ))

    # Always record this subtask as processed, even with zero findings, so the
    # API can mark it "done" in the UI (an empty-findings event has no question).
    return {"findings": findings, "processed_subtasks": [question], "token_usage": token_usage}

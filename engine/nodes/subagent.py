from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from engine.extraction import FindingList
from engine.models import SUBAGENT_MODEL
from engine.state import SubagentInput, SubtaskFinding, TokenUsage
from engine.tools.fetch import fetch
from engine.tools.search import search
from engine.usage import usage_from_message
from eval.grounding import check_grounding

_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a research extraction agent. Given a sub-question and web content, "
        "extract every relevant finding. Each finding requires:\n"
        "- claim: a clear, factual statement directly supported by the content\n"
        "- evidence_span: a VERBATIM quote copied character-for-character from the "
        "content above — do not paraphrase, summarize, correct typos, or merge text "
        "from different parts of the page. Keep it short: one sentence or a brief "
        "contiguous passage (ideally under 300 characters) so it can be located "
        "exactly in the source.\n"
        "- citation_url: the URL of the source\n\n"
        "Only include findings directly supported by the provided content, and only "
        "if you can produce a verbatim evidence_span for them. If you cannot find an "
        "exact quote that supports a claim, drop that finding. If no relevant "
        "findings exist, return an empty list.",
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

        # Only extract from sources we can fetch ourselves: a finding cited to a URL
        # whose page we can't retrieve can never pass the grounding eval (which
        # independently re-fetches citation_url), so skip rather than fall back to
        # Tavily's cached snippet.
        content: str = fetch(url)
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
                finding = SubtaskFinding(
                    subtask=question,
                    claim=f.claim,
                    evidence_span=f.evidence_span,
                    citation_url=str(f.citation_url),
                )
                # Self-check (anti-hallucination): run the same grounding check the
                # eval harness runs later, against the content this LLM actually saw.
                # Drops findings whose evidence_span isn't a real quote from the page
                # before they ever reach synthesis — keeps ungrounded_count near zero
                # for any topic, not just this one.
                if check_grounding(finding, content[:6_000]).grounded:
                    findings.append(finding)
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

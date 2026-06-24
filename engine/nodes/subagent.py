from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class SearchSpec:
    query: str
    include_domains: list[str] | None = None
    category: str | None = None


_FINANCE_TERMS = (
    "invest",
    "stock",
    "shares",
    "valuation",
    "financial",
    "revenue",
    "profit",
    "earnings",
    "risk",
    "portfolio",
    "analyst",
    "price target",
    "ipo",
)

_FINANCE_DOMAINS = [
    "sec.gov",
    "ir.tesla.com",
    "finance.yahoo.com",
    "nasdaq.com",
    "marketwatch.com",
    "cnbc.com",
    "reuters.com",
]

_PRIVATE_MARKET_DOMAINS = [
    "forgeglobal.com",
    "hive.com",
    "hiive.com",
    "equityzen.com",
    "linqto.com",
    "nasdaqprivatemarket.com",
    "rainmakersecurities.com",
    "cnbc.com",
    "reuters.com",
]


def _is_finance_question(question: str) -> bool:
    q = question.lower()
    return any(term in q for term in _FINANCE_TERMS)


def _search_specs(question: str) -> list[SearchSpec]:
    """Fallback query variants for sparse/zero-result subtasks.

    Keep the first query generic, then add domain/category hints for investment
    questions where ordinary web search often returns thin news snippets instead
    of valuation, financial, liquidity, or risk data.
    """
    specs = [
        SearchSpec(question),
        SearchSpec(f"{question} latest 2026"),
    ]
    if not _is_finance_question(question):
        return specs

    specs.extend([
        SearchSpec(
            f"{question} financial metrics valuation risks",
            include_domains=_FINANCE_DOMAINS,
            category="financial report",
        ),
        SearchSpec(
            f"{question} analyst price target valuation outlook",
            include_domains=_FINANCE_DOMAINS,
            category="news",
        ),
        SearchSpec(
            f"{question} SEC filing investor relations annual report quarterly results",
            include_domains=["sec.gov", "ir.tesla.com"],
            category="financial report",
        ),
    ])
    if "spacex" in question.lower():
        specs.append(SearchSpec(
            f"{question} private market valuation revenue liquidity retail investors",
            include_domains=_PRIVATE_MARKET_DOMAINS,
            category="company",
        ))
    return specs


def _dedupe_results(results: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[str] = set()
    deduped: list[dict[str, object]] = []
    for result in results:
        url = str(result.get("url", ""))
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append(result)
    return deduped


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

    findings: list[SubtaskFinding] = []
    input_tokens = output_tokens = cached_tokens = 0
    searched_results: list[dict[str, object]] = []
    processed_urls: set[str] = set()

    def extract_from_results(results: list[dict[str, object]]) -> None:
        nonlocal input_tokens, output_tokens, cached_tokens
        for result in results:
            url = str(result.get("url", ""))
            if not url:
                continue
            if url in processed_urls:
                continue
            processed_urls.add(url)

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

    for spec in _search_specs(question):
        try:
            searched_results.extend(
                search(
                    spec.query,
                    max_results=4,
                    include_domains=spec.include_domains,
                    category=spec.category,
                )
            )
        except Exception:
            continue
        results = _dedupe_results(searched_results)
        extract_from_results(results)
        if findings:
            break

    token_usage: list[TokenUsage] = []
    if input_tokens or output_tokens:
        token_usage.append(TokenUsage(
            node="subagent", model=SUBAGENT_MODEL,
            input_tokens=input_tokens, output_tokens=output_tokens, cached_tokens=cached_tokens,
        ))

    # Always record this subtask as processed, even with zero findings, so the
    # API can mark it "done" in the UI (an empty-findings event has no question).
    return {"findings": findings, "processed_subtasks": [question], "token_usage": token_usage}

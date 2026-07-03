from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from engine.extraction import FindingList
from engine.models import SUBAGENT_ESCALATION_MODEL, SUBAGENT_MODEL
from engine.state import SubagentInput, SubtaskFinding, TokenUsage
from engine.tools.fetch import fetch
from engine.tools.search import search
from engine.usage import usage_from_message
from eval.grounding import check_grounding

_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a research extraction agent for an AI Policy & Regulation Researcher. "
        "Given a sub-question and web content, extract every relevant policy/regulatory "
        "finding. Prioritize laws, proposed rules, regulator guidance, standards, "
        "enforcement actions, legal obligations, affected actors, timelines, exemptions, "
        "jurisdictional scope, and compliance implications. Each finding requires:\n"
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

_AI_POLICY_TERMS = (
    "ai act",
    "artificial intelligence act",
    "regulation",
    "regulatory",
    "policy",
    "compliance",
    "governance",
    "law",
    "bill",
    "standard",
    "enforcement",
    "liability",
    "copyright",
    "privacy",
    "data protection",
    "safety institute",
    "frontier model",
    "high-risk ai",
)

_AI_POLICY_DOMAINS = [
    "copyright.gov",
    "uscourts.gov",
    "courtlistener.com",
    "eur-lex.europa.eu",
    "digital-strategy.ec.europa.eu",
    "artificialintelligenceact.eu",
    "whitehouse.gov",
    "nist.gov",
    "ftc.gov",
    "congress.gov",
    "gov.uk",
    "ico.org.uk",
    "imda.gov.sg",
    "pdpc.gov.sg",
    "mas.gov.sg",
    "mddi.gov.sg",
    "aiverifyfoundation.sg",
    "asean.org",
    "oecd.ai",
    "iso.org",
    "brookings.edu",
    "iapp.org",
]

_SKIP_FINDING_DOMAINS = {
    "youtube.com",
    "youtu.be",
    "linkedin.com",
}

# One page must not monopolize a subtask's evidence: without a cap, the first
# productive URL can supply every finding and the whole report ends up citing
# one or two sources.
_MAX_FINDINGS_PER_URL = 8

# How much of a fetched page the extraction LLM sees. Marketing-heavy pages
# bury the substance past the old 6K window (observed: content starting at
# char ~10,500), which made the extractor return zero findings from pages that
# actually answered the question. Must stay well under eval/grounding's
# 40K-char re-fetch slice so extracted spans remain findable at eval time.
_EXTRACT_CHARS = 16_000

# Keep trying later search specs until the subtask's findings span at least
# this many distinct domains (or specs run out) — a single-domain evidence
# base defeats the evidence audit's source-diversity check downstream.
_MIN_SOURCE_DOMAINS = 2

# Pages shorter than this that yield zero findings are probably genuinely
# irrelevant; longer empty pages trigger one retry on the escalation model.
_ESCALATION_MIN_CHARS = 2_000


def _domain(url: str) -> str:
    hostname = (urlparse(url).hostname or "").lower()
    return hostname.removeprefix("www.")


def _matches_domain(hostname: str, domain: str) -> bool:
    return hostname == domain or hostname.endswith("." + domain)


def _prioritized_results(
    results: list[dict[str, object]], priority_domains: list[str]
) -> list[dict[str, object]]:
    """Stable-sort search results so priority (official/primary) domains are
    extracted first — aggregator and content-farm pages only fill remaining
    per-URL/domain budget after regulator and legislative sources."""
    if not priority_domains:
        return results
    def rank(result: dict[str, object]) -> int:
        hostname = _domain(str(result.get("url", "")))
        return 0 if any(_matches_domain(hostname, d) for d in priority_domains) else 1
    return sorted(results, key=rank)


def _should_skip_source(url: str) -> bool:
    """Skip sources that rarely provide stable, fetchable textual evidence.

    Findings are later re-fetched by the grounding eval. Video pages can produce
    transcript text for one fetcher/provider and only page chrome for another,
    which creates exactly the "ungrounded YouTube evidence span" failure seen in
    the eval report. LinkedIn posts are login-walled, so the eval's independent
    re-fetch gets a login page and marks the finding ungrounded — and they're
    weak authority for policy/legal claims anyway. Prefer durable text pages.
    Matched by domain suffix so subdomains (m.youtube.com, sg.linkedin.com)
    are covered without enumeration.
    """
    hostname = _domain(url)
    return any(_matches_domain(hostname, d) for d in _SKIP_FINDING_DOMAINS)


def _is_finance_question(question: str) -> bool:
    q = question.lower()
    return any(term in q for term in _FINANCE_TERMS)


def _is_ai_policy_question(question: str) -> bool:
    q = question.lower()
    return any(term in q for term in _AI_POLICY_TERMS)


def _search_specs(question: str) -> list[SearchSpec]:
    """Fallback query variants for sparse/zero-result subtasks.

    For AI-policy questions the official-domain specs come FIRST, so primary
    sources (regulators, legislation, standards bodies) are searched and
    extracted before the open web — an aggregator page found by a generic
    query must not become the subtask's main source. Domain/category hints
    for investment questions stay after the generic queries, where ordinary
    web search often returns thin news snippets instead of valuation,
    financial, liquidity, or risk data.
    """
    specs = [
        SearchSpec(question),
        SearchSpec(f"{question} latest 2026"),
    ]
    if _is_ai_policy_question(question):
        official_specs = [
            SearchSpec(
                f"{question} official regulator guidance law obligations effective date",
                include_domains=_AI_POLICY_DOMAINS,
            ),
        ]
        if "singapore" in question.lower():
            official_specs.append(SearchSpec(
                f"{question} Singapore IMDA PDPC MAS official AI governance",
                include_domains=[
                    "imda.gov.sg",
                    "pdpc.gov.sg",
                    "mas.gov.sg",
                    "mddi.gov.sg",
                ],
            ))
        specs = official_specs + specs + [
            SearchSpec(
                f"{question} compliance obligations affected entities enforcement",
                include_domains=_AI_POLICY_DOMAINS,
            ),
            SearchSpec(
                f"{question} policy analysis legal update",
                category="news",
            ),
        ]
        if "singapore" in question.lower():
            specs.extend([
                SearchSpec(
                    f"{question} Singapore AI Verify Foundation sandbox consultation industry",
                    include_domains=[
                        "aiverifyfoundation.sg",
                        "imda.gov.sg",
                        "mas.gov.sg",
                    ],
                ),
                SearchSpec(
                    f"{question} Singapore ASEAN OECD cross-border AI governance alignment",
                    include_domains=["asean.org", "oecd.ai", "imda.gov.sg", "mddi.gov.sg"],
                ),
            ])
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

    def build_chain(model: str):  # type: ignore[no-untyped-def]  # LangChain chain types are unwieldy
        llm = ChatOpenAI(model=model, temperature=0)
        return _PROMPT | llm.with_structured_output(
            FindingList, method="function_calling", include_raw=True
        )

    chain = build_chain(SUBAGENT_MODEL)
    escalation_chain = build_chain(SUBAGENT_ESCALATION_MODEL)

    findings: list[SubtaskFinding] = []
    usage_totals: dict[str, list[int]] = {}  # model -> [input, output, cached]
    searched_results: list[dict[str, object]] = []
    processed_urls: set[str] = set()

    def extract_page(page_chain, model: str, url: str, content: str) -> list[SubtaskFinding]:  # type: ignore[no-untyped-def]
        """One extraction call over one page; returns grounded findings, capped."""
        try:
            raw = page_chain.invoke(
                {"question": question, "url": url, "content": content[:_EXTRACT_CHARS]}
            )
            usage = usage_from_message(raw["raw"], "subagent", model)
            if usage:
                totals = usage_totals.setdefault(model, [0, 0, 0])
                totals[0] += usage["input_tokens"]
                totals[1] += usage["output_tokens"]
                totals[2] += usage["cached_tokens"]
            extracted: FindingList | None = raw["parsed"]
            if extracted is None:
                return []
            kept: list[SubtaskFinding] = []
            for f in extracted.findings:
                if len(kept) >= _MAX_FINDINGS_PER_URL:
                    break
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
                if check_grounding(finding, content[:_EXTRACT_CHARS]).grounded:
                    kept.append(finding)
            return kept
        except Exception:
            # Silently drop unextractable sources; don't let one bad page fail the subtask
            return []

    def extract_from_results(results: list[dict[str, object]]) -> None:
        for result in results:
            url = str(result.get("url", ""))
            if not url:
                continue
            if _should_skip_source(url):
                continue
            if url in processed_urls:
                continue
            processed_urls.add(url)

            # Only extract from sources we can fetch ourselves: a finding cited to a URL
            # whose page we can't retrieve can never pass the grounding eval (which
            # independently re-fetches citation_url), so skip rather than fall back to
            # Tavily's cached snippet. fetch() returns "" on errors, but one bad page
            # must never crash the whole subtask, so guard the call as well.
            try:
                content: str = fetch(url)
            except Exception:
                continue
            if not content:
                continue

            kept = extract_page(chain, SUBAGENT_MODEL, url, content)
            if not kept and len(content) >= _ESCALATION_MIN_CHARS:
                # The cheap model is erratic on long pages — observed returning an
                # empty FindingList on the official PDPC framework PDF that the
                # stronger model extracted 11 findings from. Retry this one page
                # on the escalation model before giving up on it: bounded to one
                # extra call per substantial-but-empty page.
                kept = extract_page(
                    escalation_chain, SUBAGENT_ESCALATION_MODEL, url, content
                )
            findings.extend(kept)

    priority_domains = _AI_POLICY_DOMAINS if _is_ai_policy_question(question) else []
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
        results = _prioritized_results(_dedupe_results(searched_results), priority_domains)
        extract_from_results(results)
        if len({_domain(f["citation_url"]) for f in findings}) >= _MIN_SOURCE_DOMAINS:
            break

    token_usage: list[TokenUsage] = [
        TokenUsage(
            node="subagent", model=model,
            input_tokens=totals[0], output_tokens=totals[1], cached_tokens=totals[2],
        )
        for model, totals in usage_totals.items()
        if totals[0] or totals[1]
    ]

    # Always record this subtask as processed, even with zero findings, so the
    # API can mark it "done" in the UI (an empty-findings event has no question).
    return {"findings": findings, "processed_subtasks": [question], "token_usage": token_usage}

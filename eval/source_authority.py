"""Source-authority check (eval harness).

Grounding measures quote fidelity — the quote really is on the cited page —
but says nothing about whether the page is worth citing: a content farm that
misstates a voluntary framework as mandatory obligations passes grounding
perfectly (observed in issue #27). This check scores the *evidence base*:
every unique cited domain is tiered as

  primary   — governments, regulators, courts, legislatures, standards bodies,
              intergovernmental organizations, official journals
  secondary — established law firms, major news organizations, peer-reviewed
              or academic sources, recognized think tanks and professional
              bodies
  other     — vendor/consultancy marketing blogs, SEO/aggregator content
              sites, personal blogs, anything unrecognizable

Clear-cut cases (.gov/.edu/.int/.mil, europa.eu, OECD, ISO, …) are tiered by
deterministic rules; the remainder go to one LLM-judge call. The score is a
weighted mean over unique domains (primary 1.0 / secondary 0.6 / other 0.2) —
unique domains, not findings, so one prolific page can't dominate the score.
Informational: it never fails a run, it contextualizes the grounding number.
"""
from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from engine.models import LEAD_MODEL, make_chat_model, structured_output_kwargs
from engine.state import SubtaskFinding, TokenUsage
from engine.usage import usage_from_message
from eval.schema import SourceAuthorityResult, SourceAuthorityVerdict

_TIER_WEIGHTS: dict[str, float] = {"primary": 1.0, "secondary": 0.6, "other": 0.2}

# Deterministic layer: unambiguous primary-source signals.
_PRIMARY_SUFFIXES = (".gov", ".mil", ".edu", ".int")
_PRIMARY_LABELS = {"gov", "gouv"}  # pdpc.gov.sg, gov.uk, economie.gouv.fr
_PRIMARY_DOMAINS = {
    "europa.eu",
    "oecd.org",
    "oecd.ai",
    "iso.org",
    "un.org",
    "asean.org",
    "wto.org",
    "imf.org",
    "bis.org",
    "courtlistener.com",  # primary court documents, not commentary
}


def _domain(url: str) -> str:
    hostname = (urlparse(url).hostname or "").lower()
    return hostname.removeprefix("www.")


def _rule_tier(domain: str) -> Literal["primary"] | None:
    if domain.endswith(_PRIMARY_SUFFIXES):
        return "primary"
    labels = set(domain.split("."))
    if labels & _PRIMARY_LABELS:
        return "primary"
    for known in _PRIMARY_DOMAINS:
        if domain == known or domain.endswith("." + known):
            return "primary"
    return None


class _DomainTier(BaseModel):
    domain: str
    tier: Literal["primary", "secondary", "other"] = Field(
        description=(
            "primary = government/regulator/court/legislature/standards body/IGO; "
            "secondary = established law firm, major news organization, academic or "
            "peer-reviewed source, recognized think tank or professional body; "
            "other = vendor or consultancy marketing content, SEO/aggregator content "
            "site, personal blog, or unknown"
        )
    )
    reasoning: str = Field(description="One short sentence")


class _DomainTierList(BaseModel):
    tiers: list[_DomainTier]


_AUTHORITY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You classify web domains by how authoritative they are as sources for "
        "policy/legal/regulatory research. For each domain, assign a tier:\n"
        "- primary: governments, regulators, courts, legislatures, standards "
        "bodies, intergovernmental organizations, official journals\n"
        "- secondary: established law firms, major news organizations, "
        "peer-reviewed/academic sources, recognized think tanks, professional "
        "bodies\n"
        "- other: vendor or consultancy marketing blogs, SEO/aggregator content "
        "sites, personal blogs, and anything you do not recognize (do NOT guess "
        "a domain into a higher tier)\n"
        "Return one entry per input domain, spelled exactly as given.",
    ),
    ("human", "Domains:\n{domains}"),
])


def _score(verdicts: list[SourceAuthorityVerdict]) -> float:
    if not verdicts:
        return 0.0
    return sum(_TIER_WEIGHTS[v.tier] for v in verdicts) / len(verdicts)


async def run_source_authority_check(
    findings: list[SubtaskFinding], lead_model: str = LEAD_MODEL
) -> tuple[SourceAuthorityResult, TokenUsage | None]:
    """Tier every unique cited domain and compute the weighted authority score."""
    first_url_by_domain: dict[str, str] = {}
    for f in findings:
        url = str(f.get("citation_url", ""))
        domain = _domain(url)
        if domain and domain not in first_url_by_domain:
            first_url_by_domain[domain] = url

    verdicts: list[SourceAuthorityVerdict] = []
    unresolved: list[str] = []
    for domain, url in first_url_by_domain.items():
        tier = _rule_tier(domain)
        if tier is not None:
            verdicts.append(SourceAuthorityVerdict(
                url=url, domain=domain, tier=tier,
                reasoning="official/primary domain (deterministic rule)",
            ))
        else:
            unresolved.append(domain)

    usage: TokenUsage | None = None
    if unresolved:
        judged: dict[str, _DomainTier] = {}
        try:
            judge_llm = make_chat_model(lead_model, temperature=0)
            chain = _AUTHORITY_PROMPT | judge_llm.with_structured_output(
                _DomainTierList, **structured_output_kwargs(lead_model), include_raw=True
            )
            raw = await chain.ainvoke({"domains": "\n".join(f"- {d}" for d in unresolved)})
            assert isinstance(raw, dict)
            usage = usage_from_message(raw["raw"], "source_authority", lead_model)
            parsed: _DomainTierList | None = raw["parsed"]
            if parsed is not None:
                judged = {t.domain.lower().removeprefix("www."): t for t in parsed.tiers}
        except Exception:
            judged = {}  # fail open: unjudged domains fall to "other" below

        for domain in unresolved:
            tier_verdict = judged.get(domain)
            if tier_verdict is not None:
                verdicts.append(SourceAuthorityVerdict(
                    url=first_url_by_domain[domain], domain=domain,
                    tier=tier_verdict.tier, reasoning=tier_verdict.reasoning,
                ))
            else:
                verdicts.append(SourceAuthorityVerdict(
                    url=first_url_by_domain[domain], domain=domain,
                    tier="other", reasoning="not classified by the judge",
                ))

    result = SourceAuthorityResult(
        verdicts=verdicts,
        authority_score=_score(verdicts),
        primary_count=sum(1 for v in verdicts if v.tier == "primary"),
        secondary_count=sum(1 for v in verdicts if v.tier == "secondary"),
        other_count=sum(1 for v in verdicts if v.tier == "other"),
    )
    return result, usage

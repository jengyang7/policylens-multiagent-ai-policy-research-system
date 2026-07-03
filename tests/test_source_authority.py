"""Source-authority eval check — deterministic tiering, weighting, judge fallback."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import eval.source_authority as sa_mod
from eval.source_authority import _DomainTier, _DomainTierList, run_source_authority_check


def _finding(url: str) -> dict[str, str]:
    return {"subtask": "q", "claim": "c", "evidence_span": "e", "citation_url": url}


def _mock_judge(monkeypatch: pytest.MonkeyPatch, tiers: list[_DomainTier]) -> None:
    raw_msg = MagicMock()
    raw_msg.usage_metadata = None

    class FakeChain:
        async def ainvoke(self, inputs: dict[str, object]) -> dict[str, object]:
            return {"raw": raw_msg, "parsed": _DomainTierList(tiers=tiers)}

    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(sa_mod, "make_chat_model", lambda *a, **kw: mock_llm)
    mock_prompt = MagicMock()
    mock_prompt.__or__ = lambda s, o: FakeChain()
    monkeypatch.setattr(sa_mod, "_AUTHORITY_PROMPT", mock_prompt)


async def test_official_domains_tiered_primary_without_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def judge_must_not_run(*a: object, **kw: object) -> object:
        pytest.fail("all domains are rule-classified — the judge must not be called")

    monkeypatch.setattr(sa_mod, "make_chat_model", judge_must_not_run)

    result, usage = await run_source_authority_check([
        _finding("https://www.pdpc.gov.sg/framework.pdf"),
        _finding("https://eur-lex.europa.eu/ai-act"),
        _finding("https://oecd.ai/dashboards"),
        _finding("https://www.mit.edu/study"),
    ])

    assert result.primary_count == 4
    assert result.authority_score == 1.0
    assert usage is None


async def test_weighted_score_and_judge_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_judge(monkeypatch, [
        _DomainTier(domain="linklaters.com", tier="secondary", reasoning="major law firm"),
        _DomainTier(domain="seo-blog.example.com", tier="other", reasoning="content site"),
    ])

    result, _ = await run_source_authority_check([
        _finding("https://www.imda.gov.sg/guidance"),          # primary (rule)
        _finding("https://www.imda.gov.sg/guidance2"),         # same domain — not double-counted
        _finding("https://www.linklaters.com/insights/x"),     # secondary (judge)
        _finding("https://seo-blog.example.com/ai-laws"),      # other (judge)
    ])

    assert (result.primary_count, result.secondary_count, result.other_count) == (1, 1, 1)
    # weights: primary 1.0, secondary 0.6, other 0.2 → mean over 3 unique domains
    assert result.authority_score == pytest.approx((1.0 + 0.6 + 0.2) / 3)


async def test_unjudged_domains_fall_to_other(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_judge(monkeypatch, [])  # judge returns nothing useful

    result, _ = await run_source_authority_check([
        _finding("https://mystery-site.example.net/post"),
    ])

    assert result.other_count == 1
    assert result.verdicts[0].tier == "other"


async def test_empty_findings_scores_zero() -> None:
    result, usage = await run_source_authority_check([])
    assert result.authority_score == 0.0
    assert result.verdicts == []
    assert usage is None

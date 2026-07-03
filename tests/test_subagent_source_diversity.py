"""Subagent source-diversity tests — per-URL caps, official-domain priority,
and the multi-domain search continuation. No live API calls."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import engine.nodes.subagent as subagent_mod
from engine.extraction import Finding, FindingList
from engine.state import SubagentInput


def test_domain_strips_www() -> None:
    assert subagent_mod._domain("https://www.imda.gov.sg/framework") == "imda.gov.sg"
    assert subagent_mod._domain("https://blog.example.com/a") == "blog.example.com"


def test_prioritized_results_puts_official_domains_first() -> None:
    results = [
        {"url": "https://aggregator.example.com/page"},
        {"url": "https://www.imda.gov.sg/framework"},
        {"url": "https://another-blog.example.org/post"},
        {"url": "https://eur-lex.europa.eu/ai-act"},
    ]
    ordered = subagent_mod._prioritized_results(results, subagent_mod._AI_POLICY_DOMAINS)
    assert [r["url"] for r in ordered] == [
        "https://www.imda.gov.sg/framework",
        "https://eur-lex.europa.eu/ai-act",
        "https://aggregator.example.com/page",
        "https://another-blog.example.org/post",
    ]
    # No priority list → order untouched
    assert subagent_mod._prioritized_results(results, []) == results


def test_official_domain_specs_come_first_for_policy_questions() -> None:
    specs = subagent_mod._search_specs("How is Singapore regulating AI governance?")
    assert specs[0].include_domains == subagent_mod._AI_POLICY_DOMAINS
    assert specs[1].include_domains is not None
    assert "imda.gov.sg" in specs[1].include_domains


def _fake_extraction_chain(findings_by_url: dict[str, list[Finding]]) -> MagicMock:
    raw_msg = MagicMock()
    raw_msg.usage_metadata = None
    chain = MagicMock()
    chain.invoke = lambda inputs: {
        "raw": raw_msg,
        "parsed": FindingList(findings=findings_by_url.get(str(inputs["url"]), [])),
    }
    return chain


def _finding(url: str, i: int) -> Finding:
    return Finding(
        claim=f"Claim {i} from {url}",
        evidence_span=f"Evidence {i}",
        citation_url=url,  # type: ignore[arg-type]
    )


def test_subagent_caps_per_url_and_searches_until_two_domains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    official_url = "https://www.imda.gov.sg/framework"
    aggregator_url = "https://aggregator.example.com/page"
    search_batches = [
        [{"url": official_url}],
        [{"url": aggregator_url}],
    ]
    search_calls: list[str] = []

    def fake_search(query: str, **kwargs: object) -> list[dict[str, object]]:
        search_calls.append(query)
        if len(search_calls) <= len(search_batches):
            return search_batches[len(search_calls) - 1]
        return []

    cap = subagent_mod._MAX_FINDINGS_PER_URL
    chain = _fake_extraction_chain({
        official_url: [_finding(official_url, i) for i in range(cap + 4)],
        aggregator_url: [_finding(aggregator_url, i) for i in range(2)],
    })
    mock_prompt = MagicMock()
    mock_prompt.__or__ = lambda s, o: chain
    monkeypatch.setattr(subagent_mod, "_PROMPT", mock_prompt)
    monkeypatch.setattr(subagent_mod, "ChatOpenAI", MagicMock())
    monkeypatch.setattr(subagent_mod, "search", fake_search)
    monkeypatch.setattr(subagent_mod, "fetch", lambda url: "page content")
    monkeypatch.setattr(
        subagent_mod, "check_grounding",
        lambda finding, content: SimpleNamespace(grounded=True),
    )

    result = subagent_mod.subagent(
        SubagentInput(question="How is Singapore regulating AI governance?")
    )

    findings = result["findings"]
    assert isinstance(findings, list)
    per_url = [f["citation_url"] for f in findings]
    # One page can't monopolize the subtask: capped at _MAX_FINDINGS_PER_URL
    assert per_url.count(official_url) == cap
    # First spec yielded only one domain, so the loop kept searching until a
    # second domain contributed findings, then stopped (not all ~8 specs).
    assert per_url.count(aggregator_url) == 2
    assert len(search_calls) == 2


def test_subagent_escalates_to_stronger_model_when_nano_extracts_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A substantial page where the cheap extractor returns an empty list gets
    one retry on the escalation model instead of being silently dropped."""
    url = "https://www.pdpc.gov.sg/framework.pdf"

    empty_chain = _fake_extraction_chain({})  # nano finds nothing anywhere
    productive_chain = _fake_extraction_chain({url: [_finding(url, i) for i in range(3)]})
    chains = iter([empty_chain, productive_chain])  # build order: nano, escalation

    mock_prompt = MagicMock()
    mock_prompt.__or__ = lambda s, o: next(chains)
    monkeypatch.setattr(subagent_mod, "_PROMPT", mock_prompt)
    monkeypatch.setattr(subagent_mod, "ChatOpenAI", MagicMock())
    monkeypatch.setattr(subagent_mod, "search", lambda query, **kw: [{"url": url}])
    monkeypatch.setattr(
        subagent_mod, "fetch",
        lambda u: "substantial page content " * 200,  # well past _ESCALATION_MIN_CHARS
    )
    monkeypatch.setattr(
        subagent_mod, "check_grounding",
        lambda finding, content: SimpleNamespace(grounded=True),
    )

    result = subagent_mod.subagent(SubagentInput(question="policy question"))

    findings = result["findings"]
    assert isinstance(findings, list)
    assert len(findings) == 3
    assert all(f["citation_url"] == url for f in findings)


def test_subagent_does_not_escalate_on_short_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = "https://example.com/thin-page"
    empty_chain = _fake_extraction_chain({})

    def escalation_must_not_build() -> object:
        pytest.fail("escalation chain must not be invoked for short pages")

    productive = _fake_extraction_chain({url: [_finding(url, 0)]})
    calls = {"n": 0}

    def chain_factory(s: object, o: object) -> object:
        calls["n"] += 1
        return empty_chain if calls["n"] == 1 else productive

    mock_prompt = MagicMock()
    mock_prompt.__or__ = chain_factory
    monkeypatch.setattr(subagent_mod, "_PROMPT", mock_prompt)
    monkeypatch.setattr(subagent_mod, "ChatOpenAI", MagicMock())
    monkeypatch.setattr(subagent_mod, "search", lambda query, **kw: [{"url": url}])
    monkeypatch.setattr(subagent_mod, "fetch", lambda u: "short page")  # < _ESCALATION_MIN_CHARS
    monkeypatch.setattr(
        subagent_mod, "check_grounding",
        lambda finding, content: SimpleNamespace(grounded=True),
    )

    result = subagent_mod.subagent(SubagentInput(question="policy question"))

    # nano found nothing and the page is too short to justify an escalation call
    assert result["findings"] == []

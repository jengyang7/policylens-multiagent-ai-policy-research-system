"""Evidence audit tests — bounded shared quality gate, no live API calls."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from engine.state import DebateTurn, ResearchState


def _state(**overrides: object) -> ResearchState:
    state: dict[str, object] = {
        "run_id": "run",
        "query": "Compare AI obligations in Singapore and the EU",
        "clarification_questions": [],
        "clarification_options": [],
        "clarifications": [],
        "supervisor_thinking": "",
        "title": "",
        "subtasks": ["Singapore obligations", "EU obligations"],
        "findings": [{
            "subtask": "Singapore obligations",
            "claim": "A supported claim",
            "evidence_span": "Evidence",
            "citation_url": "https://example.com/source",
        }],
        "summary": "Only the Singapore side is currently covered.",
        "report": "",
        "messages": [],
        "token_usage": [],
        "processed_subtasks": [],
        "debate_turns": [],
    }
    state.update(overrides)
    return state  # type: ignore[return-value]


def test_evidence_audit_caps_gaps_and_includes_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import engine.nodes.evidence_audit as audit_mod

    captured: dict[str, object] = {}
    parsed = MagicMock()
    parsed.sufficient = False
    parsed.assessment = "The EU comparison leg is missing."
    parsed.gap_questions = [
        f"gap {i}" for i in range(audit_mod.MAX_GAP_QUESTIONS + 2)
    ]
    raw_msg = MagicMock()
    raw_msg.usage_metadata = None
    raw_result = {"raw": raw_msg, "parsed": parsed}
    mock_chain = MagicMock()
    mock_chain.invoke = lambda inputs: (captured.update(inputs), raw_result)[1]
    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(audit_mod, "make_chat_model", lambda *a, **kw: mock_llm)
    mock_prompt = MagicMock()
    mock_prompt.__or__ = lambda s, o: mock_chain
    monkeypatch.setattr(audit_mod, "_PROMPT", mock_prompt)

    result = audit_mod.evidence_audit(_state())

    expected = [f"gap {i}" for i in range(audit_mod.MAX_GAP_QUESTIONS)]
    assert result["gap_subtasks"] == expected
    assert result["evidence_audit"] == {
        "sufficient": False,
        "assessment": "The EU comparison leg is missing.",
        "gap_questions": expected,
    }
    assert "Singapore obligations: 1 findings" in str(captured["coverage"])
    assert "EU obligations: 0 findings" in str(captured["coverage"])
    assert captured["transcript"] == "(debate mode was not used)"


def test_evidence_audit_can_use_optional_debate_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import engine.nodes.evidence_audit as audit_mod

    captured: dict[str, object] = {}
    parsed = MagicMock(sufficient=True, assessment="Coverage is sufficient.", gap_questions=[])
    raw_msg = MagicMock()
    raw_msg.usage_metadata = None
    mock_chain = MagicMock()
    mock_chain.invoke = lambda inputs: (
        captured.update(inputs),
        {"raw": raw_msg, "parsed": parsed},
    )[1]
    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(audit_mod, "make_chat_model", lambda *a, **kw: mock_llm)
    mock_prompt = MagicMock()
    mock_prompt.__or__ = lambda s, o: mock_chain
    monkeypatch.setattr(audit_mod, "_PROMPT", mock_prompt)

    turns = [
        DebateTurn(agent="advocate", model="a", round=1, content="argument"),
        DebateTurn(agent="skeptic", model="b", round=1, content="objection"),
    ]
    result = audit_mod.evidence_audit(_state(debate_turns=turns))

    assert result["gap_subtasks"] == []
    assert "[Round 1 — Opposition]" in str(captured["transcript"])


def _finding(subtask: str, url: str) -> dict[str, str]:
    return {
        "subtask": subtask,
        "claim": "A supported claim",
        "evidence_span": "Evidence",
        "citation_url": url,
    }


def _run_audit_capturing_inputs(
    monkeypatch: pytest.MonkeyPatch, state: ResearchState
) -> dict[str, object]:
    import engine.nodes.evidence_audit as audit_mod

    captured: dict[str, object] = {}
    parsed = MagicMock(sufficient=True, assessment="ok", gap_questions=[])
    raw_msg = MagicMock()
    raw_msg.usage_metadata = None
    mock_chain = MagicMock()
    mock_chain.invoke = lambda inputs: (
        captured.update(inputs),
        {"raw": raw_msg, "parsed": parsed},
    )[1]
    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(audit_mod, "make_chat_model", lambda *a, **kw: mock_llm)
    mock_prompt = MagicMock()
    mock_prompt.__or__ = lambda s, o: mock_chain
    monkeypatch.setattr(audit_mod, "_PROMPT", mock_prompt)

    audit_mod.evidence_audit(state)
    return captured


def test_evidence_audit_warns_on_low_source_diversity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    findings = [
        _finding("Singapore obligations", "https://aggregator.example.com/page")
        for _ in range(5)
    ]
    captured = _run_audit_capturing_inputs(monkeypatch, _state(findings=findings))

    assert "WARNING" in str(captured["sources"])
    assert "only 1 distinct domain(s)" in str(captured["sources"])
    assert "Singapore obligations: 5 findings from 1 source domain(s)" in str(
        captured["coverage"]
    )


def test_evidence_audit_no_warning_when_sources_are_diverse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    findings = [
        _finding("Singapore obligations", "https://www.imda.gov.sg/framework"),
        _finding("Singapore obligations", "https://www.pdpc.gov.sg/guidance"),
        _finding("EU obligations", "https://eur-lex.europa.eu/ai-act"),
    ]
    captured = _run_audit_capturing_inputs(monkeypatch, _state(findings=findings))

    assert "WARNING" not in str(captured["sources"])
    assert "Singapore obligations: 2 findings from 2 source domain(s)" in str(
        captured["coverage"]
    )

"""Debate mode tests — graph wiring, turn nodes, model factory; no live API calls."""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from engine.models import role_default_models
from engine.orchestrator import (
    _fan_out_gaps,
    _route_after_clarify,
    _route_after_compact,
    _route_after_research,
    _route_after_skeptic,
    _route_after_synthesize,
    graph,
)
from engine.state import DebateTurn, ResearchState


def _base_state(**overrides: object) -> ResearchState:
    state: dict[str, object] = {
        "run_id": str(uuid.uuid4()),
        "query": "test query",
        "clarification_questions": [],
        "clarifications": [],
        "subtasks": [],
        "findings": [],
        "summary": "compacted summary",
        "report": "",
        "messages": [],
    }
    state.update(overrides)
    return state  # type: ignore[return-value]


def _turn(agent: str, round_no: int) -> DebateTurn:
    return DebateTurn(agent=agent, model="m", round=round_no, content="…")


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------

def test_graph_has_debate_nodes() -> None:
    nodes = list(graph.nodes.keys())
    assert "single_agent" in nodes
    assert "debate_advocate" in nodes
    assert "debate_skeptic" in nodes
    # Neutral lead judgment after the final round
    assert "judge_debate" in nodes
    # Debate-driven gap research round
    assert "plan_gap_research" in nodes
    assert "gap_subagent" in nodes
    assert "recompact" in nodes


def test_route_after_compact_off_by_default() -> None:
    assert _route_after_compact(_base_state()) == "synthesize"
    assert _route_after_compact(_base_state(debate_mode=False)) == "synthesize"


def test_route_after_compact_enters_debate() -> None:
    assert _route_after_compact(_base_state(debate_mode=True)) == "debate_advocate"


def test_benchmark_modes_route_research_and_verifier() -> None:
    assert _route_after_research(_base_state(run_mode="multi_agent_no_compaction")) == "synthesize"
    assert _route_after_research(_base_state(run_mode="multi_agent_compaction")) == "compact"
    assert _route_after_research(_base_state(run_mode="multi_agent_verified")) == "compact"
    assert _route_after_research(_base_state(run_mode="single_agent")) == "synthesize"

    assert _route_after_synthesize(_base_state(run_mode="multi_agent_no_compaction")) == "__end__"
    assert _route_after_synthesize(_base_state(run_mode="multi_agent_compaction")) == "__end__"
    assert (
        _route_after_synthesize(_base_state(run_mode="multi_agent_verified"))
        == "verify_citations"
    )


def test_single_agent_mode_skips_plan_fan_out() -> None:
    routed = _route_after_clarify(_base_state(run_mode="single_agent", query="full query"))
    assert isinstance(routed, list)
    assert routed[0].node == "single_agent"
    assert routed[0].arg["question"] == "full query"

    assert _route_after_clarify(_base_state(run_mode="multi_agent_verified")) == "plan"


def test_route_after_skeptic_loops_until_rounds_done() -> None:
    one_round = [_turn("advocate", 1), _turn("skeptic", 1)]
    two_rounds = one_round + [_turn("advocate", 2), _turn("skeptic", 2)]

    # Default 2 rounds: loop after round 1, exit to the judge after round 2
    state = _base_state(debate_mode=True, debate_turns=one_round)
    assert _route_after_skeptic(state) == "debate_advocate"
    state = _base_state(debate_mode=True, debate_turns=two_rounds)
    assert _route_after_skeptic(state) == "judge_debate"

    # Explicit 1 round: exit immediately
    state = _base_state(debate_mode=True, debate_rounds=1, debate_turns=one_round)
    assert _route_after_skeptic(state) == "judge_debate"


def test_fan_out_gaps_sends_one_per_question_or_skips() -> None:
    # No gaps → straight to synthesize
    assert _fan_out_gaps(_base_state(gap_subtasks=[])) == "synthesize"
    assert _fan_out_gaps(_base_state()) == "synthesize"

    sends = _fan_out_gaps(_base_state(gap_subtasks=["q1", "q2"]))
    assert isinstance(sends, list)
    assert [s.node for s in sends] == ["gap_subagent", "gap_subagent"]
    assert [s.arg["question"] for s in sends] == ["q1", "q2"]


def test_plan_gap_research_returns_capped_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    import engine.nodes.debate as debate_mod
    from engine.nodes.debate import MAX_GAP_QUESTIONS, plan_gap_research

    captured: dict[str, object] = {}
    parsed = MagicMock()
    parsed.gap_questions = [f"gap {i}" for i in range(MAX_GAP_QUESTIONS + 2)]
    raw_msg = MagicMock()
    raw_msg.usage_metadata = None

    raw_result = {"raw": raw_msg, "parsed": parsed}
    mock_chain = MagicMock()
    mock_chain.invoke = lambda inputs: (captured.update(inputs), raw_result)[1]
    mock_structured = MagicMock()
    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=mock_structured)
    monkeypatch.setattr(debate_mod, "make_chat_model", lambda *a, **kw: mock_llm)
    mock_prompt = MagicMock()
    mock_prompt.__or__ = lambda s, o: mock_chain
    monkeypatch.setattr(debate_mod, "_GAP_PROMPT", mock_prompt)

    state = _base_state(debate_turns=[_turn("advocate", 1), _turn("skeptic", 1)])
    result = plan_gap_research(state)
    assert result["gap_subtasks"] == [f"gap {i}" for i in range(MAX_GAP_QUESTIONS)]
    assert "[Round 1 — Opposition]" in str(captured["transcript"])


def test_judge_debate_returns_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    import engine.nodes.debate as debate_mod
    from engine.nodes.debate import VerdictRow, judge_debate

    captured: dict[str, object] = {}
    parsed = MagicMock()
    parsed.winner = "opposition"
    parsed.rows = [
        VerdictRow(
            category="Evidence Quality",
            assessment="The proposition could not answer the cost objection.",
            winner="opposition",
        ),
    ]
    raw_msg = MagicMock()
    raw_msg.usage_metadata = None

    raw_result = {"raw": raw_msg, "parsed": parsed}
    mock_chain = MagicMock()
    mock_chain.invoke = lambda inputs: (captured.update(inputs), raw_result)[1]
    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(debate_mod, "make_chat_model", lambda *a, **kw: mock_llm)
    mock_prompt = MagicMock()
    mock_prompt.__or__ = lambda s, o: mock_chain
    monkeypatch.setattr(debate_mod, "_JUDGE_PROMPT", mock_prompt)

    state = _base_state(
        lead_model="gpt-5.4",
        debate_turns=[_turn("advocate", 1), _turn("skeptic", 1)],
    )
    result = judge_debate(state)
    assert result["debate_verdict"] == {
        "winner": "opposition",
        "rows": [{
            "category": "Evidence Quality",
            "assessment": "The proposition could not answer the cost objection.",
            "winner": "opposition",
        }],
        "model": "gpt-5.4",
    }
    assert "[Round 1 — Proposition]" in str(captured["transcript"])


# ---------------------------------------------------------------------------
# Debate turn nodes
# ---------------------------------------------------------------------------

def _patch_debate_llm(monkeypatch: pytest.MonkeyPatch, captured: dict[str, object]) -> None:
    fake_response = MagicMock()
    fake_response.content = "argument text"
    fake_response.text = "argument text"  # node reads .text (normalizes Gemini content blocks)
    fake_response.usage_metadata = None

    mock_chain = MagicMock()
    mock_chain.invoke = lambda inputs: (captured.update(inputs), fake_response)[1]

    mock_llm = MagicMock()
    mock_llm.__ror__ = MagicMock(return_value=mock_chain)

    def fake_factory(model: str, temperature: float = 0) -> MagicMock:
        captured["model"] = model
        return mock_llm

    monkeypatch.setattr("engine.nodes.debate.make_chat_model", fake_factory)
    import engine.nodes.debate as debate_mod
    mock_prompt = MagicMock()
    mock_prompt.__or__ = lambda s, o: mock_chain
    monkeypatch.setattr(debate_mod, "_ADVOCATE_PROMPT", mock_prompt)
    monkeypatch.setattr(debate_mod, "_SKEPTIC_PROMPT", mock_prompt)


def test_debate_advocate_returns_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    from engine.nodes.debate import debate_advocate

    captured: dict[str, object] = {}
    _patch_debate_llm(monkeypatch, captured)

    state = _base_state(debate_mode=True, advocate_model="claude-sonnet-4-6")
    result = debate_advocate(state)
    turns = result["debate_turns"]
    assert turns == [
        {"agent": "advocate", "model": "claude-sonnet-4-6", "round": 1, "content": "argument text"}
    ]
    assert captured["model"] == "claude-sonnet-4-6"
    assert captured["summary"] == "compacted summary"


def test_debate_skeptic_round_number_and_lead_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from engine.nodes.debate import debate_skeptic

    captured: dict[str, object] = {}
    _patch_debate_llm(monkeypatch, captured)

    # No skeptic_model set → falls back to lead_model; 3 prior turns → round 2
    prior = [_turn("advocate", 1), _turn("skeptic", 1), _turn("advocate", 2)]
    state = _base_state(debate_mode=True, lead_model="gpt-5.4", debate_turns=prior)
    result = debate_skeptic(state)
    turn = result["debate_turns"][0]  # type: ignore[index]
    assert turn["agent"] == "skeptic"
    assert turn["round"] == 2
    assert turn["model"] == "gpt-5.4"


def test_format_transcript() -> None:
    from engine.nodes.debate import format_transcript

    assert "beginning" in format_transcript([])
    text = format_transcript([_turn("advocate", 1), _turn("skeptic", 1)])
    assert "[Round 1 — Proposition]" in text
    assert "[Round 1 — Opposition]" in text


# ---------------------------------------------------------------------------
# Synthesize consumes the transcript
# ---------------------------------------------------------------------------

def _patch_synthesize(monkeypatch: pytest.MonkeyPatch, captured: dict[str, object]) -> None:
    import engine.nodes.synthesize as syn_mod

    fake_response = MagicMock()
    fake_response.content = "report"
    fake_response.usage_metadata = None

    mock_chain = MagicMock()
    mock_chain.invoke = lambda inputs: (captured.update(inputs), fake_response)[1]

    mock_llm = MagicMock()
    monkeypatch.setattr(syn_mod, "make_chat_model", lambda *a, **kw: mock_llm)
    mock_prompt = MagicMock()
    mock_prompt.__or__ = lambda s, o: mock_chain
    monkeypatch.setattr(syn_mod, "_PROMPT", mock_prompt)


def test_synthesize_includes_debate_transcript(monkeypatch: pytest.MonkeyPatch) -> None:
    from engine.nodes.synthesize import synthesize

    captured: dict[str, object] = {}
    _patch_synthesize(monkeypatch, captured)

    state = _base_state(debate_turns=[_turn("advocate", 1), _turn("skeptic", 1)])
    synthesize(state)
    section = str(captured["debate_section"])
    assert "Debate transcript" in section
    assert "[Round 1 — Proposition]" in section


def test_synthesize_empty_debate_section_without_debate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from engine.nodes.synthesize import synthesize

    captured: dict[str, object] = {}
    _patch_synthesize(monkeypatch, captured)

    synthesize(_base_state())
    assert captured["debate_section"] == ""


# ---------------------------------------------------------------------------
# Model factory + availability
# ---------------------------------------------------------------------------

def test_make_chat_model_routes_by_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    import langchain_anthropic
    import langchain_google_genai
    import langchain_openai

    from engine.models import make_chat_model

    calls: list[str] = []
    monkeypatch.setattr(
        langchain_anthropic, "ChatAnthropic", lambda **kw: calls.append("anthropic")
    )
    monkeypatch.setattr(
        langchain_google_genai, "ChatGoogleGenerativeAI", lambda **kw: calls.append("google")
    )
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", lambda **kw: calls.append("openai"))

    make_chat_model("claude-sonnet-4-6")
    make_chat_model("gemini-3.1-pro-preview")
    make_chat_model("gpt-5.4")
    assert calls == ["anthropic", "google", "openai"]


def test_role_defaults_fall_back_without_provider_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    defaults = role_default_models()
    assert defaults == {
        "lead": "gpt-5.4",
        "advocate": "gpt-5.4",
        "skeptic": "gpt-5.4",
        "eval": "gpt-5.4",
    }


def test_role_defaults_use_cross_provider_models_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("GOOGLE_API_KEY", "g-test")
    defaults = role_default_models()
    assert defaults["lead"] == "gpt-5.4"
    assert defaults["advocate"].startswith("claude-")
    assert defaults["skeptic"].startswith("gemini-")

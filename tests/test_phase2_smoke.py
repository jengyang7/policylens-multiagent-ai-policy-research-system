"""Phase 2 smoke tests — memory + human-in-the-loop, no live API calls."""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from engine.orchestrator import build_graph, graph
from engine.state import Clarification, ResearchState


def test_graph_has_all_phase2_nodes() -> None:
    nodes = list(graph.nodes.keys())
    assert "clarify" in nodes
    assert "clarify_wait" in nodes
    assert "compact" in nodes
    assert "synthesize" in nodes
    assert "plan" in nodes
    assert "subagent" in nodes


def test_state_has_clarifications_field() -> None:
    state: ResearchState = {
        "run_id": str(uuid.uuid4()),
        "query": "What is LangGraph?",
        "clarification_questions": [],
        "clarifications": [],
        "subtasks": [],
        "findings": [],
        "summary": "",
        "report": "",
        "messages": [],
    }
    assert state["clarification_questions"] == []
    assert state["clarifications"] == []
    assert state["messages"] == []


def test_clarification_typeddict() -> None:
    c = Clarification(question="Which Mistral?", answer="The AI company")
    assert c["question"] == "Which Mistral?"
    assert c["answer"] == "The AI company"


def test_build_graph_without_checkpointer() -> None:
    g = build_graph(checkpointer=None)
    assert "clarify" in list(g.nodes.keys())


def test_compact_node_no_findings() -> None:
    from engine.nodes.compact import compact

    state: ResearchState = {
        "run_id": str(uuid.uuid4()),
        "query": "test",
        "clarification_questions": [],
        "clarifications": [],
        "subtasks": [],
        "findings": [],
        "summary": "",
        "report": "",
        "messages": [],
    }
    result = compact(state)
    assert "summary" in result
    assert "no findings" in result["summary"]  # type: ignore[operator]


def test_compact_node_calls_compaction(monkeypatch: pytest.MonkeyPatch) -> None:
    from engine.nodes.compact import compact

    monkeypatch.setattr(
        "engine.nodes.compact.compact_findings",
        lambda findings, lead_model: ("mocked summary", None),
    )

    state: ResearchState = {
        "run_id": str(uuid.uuid4()),
        "query": "test",
        "clarification_questions": [],
        "clarifications": [],
        "subtasks": [],
        "findings": [
            {
                "subtask": "q1",
                "claim": "A claim",
                "evidence_span": "evidence",
                "citation_url": "https://example.com",
            }
        ],
        "summary": "",
        "report": "",
        "messages": [],
    }
    result = compact(state)
    assert result["summary"] == "mocked summary"
    # Raw findings trimmed after compaction
    assert result["findings"] == []


def test_synthesize_uses_summary_over_raw_findings(monkeypatch: pytest.MonkeyPatch) -> None:
    from langchain_core.messages import AIMessage

    from engine.nodes.synthesize import synthesize

    captured: dict[str, object] = {}

    fake_response = MagicMock()
    fake_response.content = "report"

    def fake_invoke(inputs: dict[str, object]) -> AIMessage:
        captured.update(inputs)
        return fake_response

    mock_chain = MagicMock()
    mock_chain.invoke = fake_invoke

    # Patch ChatOpenAI so no API key is needed, and intercept the chain
    mock_llm = MagicMock()
    mock_llm.__ror__ = MagicMock(return_value=mock_chain)
    monkeypatch.setattr("engine.nodes.synthesize.ChatOpenAI", lambda **kw: mock_llm)

    # Make _PROMPT | mock_llm return mock_chain
    import engine.nodes.synthesize as syn_mod
    mock_prompt = MagicMock()
    mock_prompt.__or__ = lambda s, o: mock_chain
    monkeypatch.setattr(syn_mod, "_PROMPT", mock_prompt)

    state: ResearchState = {
        "run_id": str(uuid.uuid4()),
        "query": "test query",
        "clarification_questions": [],
        "clarifications": [],
        "subtasks": [],
        "findings": [],
        "summary": "pre-compacted summary text",
        "report": "",
        "messages": [],
    }
    result = synthesize(state)
    assert result["report"] == "report"
    assert captured.get("findings_text") == "pre-compacted summary text"

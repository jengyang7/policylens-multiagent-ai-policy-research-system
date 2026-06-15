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
    assert "verify_citations" in nodes
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
    # Raw findings are left in state for verify_citations — compact doesn't return them
    assert "findings" not in result


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

    # Patch the model factory so no API key is needed, and intercept the chain
    mock_llm = MagicMock()
    mock_llm.__ror__ = MagicMock(return_value=mock_chain)
    monkeypatch.setattr("engine.nodes.synthesize.make_chat_model", lambda *a, **kw: mock_llm)

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


async def test_verify_citations_strips_unfaithful_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import engine.nodes.verify_citations as verify_mod
    from eval.schema import FaithfulnessVerdict

    async def fake_checks(
        report: str, findings: list[object], lead_model: str
    ) -> tuple[list[FaithfulnessVerdict], list[object], list[object]]:
        verdicts = [
            FaithfulnessVerdict(
                citation_index=1,
                report_sentence="Faithful claim.",
                matched_finding_claims=["A claim"],
                faithful=True,
                confidence=1.0,
                reasoning="supported",
            ),
            FaithfulnessVerdict(
                citation_index=2,
                report_sentence="Unsupported synthesis.",
                matched_finding_claims=[],
                faithful=False,
                confidence=1.0,
                reasoning="not supported",
            ),
        ]
        return verdicts, [], []

    monkeypatch.setattr(verify_mod, "run_faithfulness_checks", fake_checks)

    state: ResearchState = {
        "run_id": str(uuid.uuid4()),
        "query": "test query",
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
        "summary": "summary",
        "report": (
            "Faithful claim [1]. Unsupported synthesis [2].\n\n"
            "## References\n\n[1] [A](https://a.com)\n[2] [B](https://b.com)\n"
        ),
        "messages": [],
    }
    result = await verify_mod.verify_citations(state)
    report = str(result["report"])
    assert "Faithful claim [1]." in report
    assert "Unsupported synthesis." in report
    assert "Unsupported synthesis [2]" not in report
    assert result["findings"] == []
    # References section is rebuilt to match what's still cited: [1] kept,
    # [2] dropped as an orphan since its citation was stripped from the body
    assert "[1] [A](https://a.com)" in report
    assert "[2] [B](https://b.com)" not in report


async def test_verify_citations_fills_in_missing_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A citation the synthesizer used in the body but omitted from its own
    References list is filled in from state.findings."""
    import engine.nodes.verify_citations as verify_mod
    from eval.schema import FaithfulnessVerdict

    async def fake_checks(
        report: str, findings: list[object], lead_model: str
    ) -> tuple[list[FaithfulnessVerdict], list[object], list[object]]:
        verdicts = [
            FaithfulnessVerdict(
                citation_index=2,
                report_sentence="Second claim.",
                matched_finding_claims=["B claim"],
                faithful=True,
                confidence=1.0,
                reasoning="supported",
            ),
        ]
        return verdicts, [], []

    monkeypatch.setattr(verify_mod, "run_faithfulness_checks", fake_checks)

    state: ResearchState = {
        "run_id": str(uuid.uuid4()),
        "query": "test query",
        "clarification_questions": [],
        "clarifications": [],
        "subtasks": [],
        "findings": [
            {
                "subtask": "q1",
                "claim": "A claim",
                "evidence_span": "evidence",
                "citation_url": "https://a.com",
            },
            {
                "subtask": "q2",
                "claim": "B claim",
                "evidence_span": "evidence",
                "citation_url": "https://b.com",
            },
        ],
        "summary": "summary",
        "report": (
            "Second claim [2].\n\n## References\n\n[1] [A](https://a.com)\n"
        ),
        "messages": [],
    }
    result = await verify_mod.verify_citations(state)
    report = str(result["report"])
    # [2] is cited in the body but missing from the LLM's References list —
    # filled in from findings[1] (1-indexed)
    assert "[2] [https://b.com](https://b.com)" in report
    # [1] is no longer cited in the body — dropped as an orphan
    assert "[1] [A](https://a.com)" not in report


async def test_verify_citations_strips_non_numeric_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stray markers like [Synthesis] that leak from the debate transcript are
    stripped from the final report — they aren't [i] citations and have no
    References entry, so they'd read as broken citations."""
    import engine.nodes.verify_citations as verify_mod
    from eval.schema import FaithfulnessVerdict

    async def fake_checks(
        report: str, findings: list[object], lead_model: str
    ) -> tuple[list[FaithfulnessVerdict], list[object], list[object]]:
        verdicts = [
            FaithfulnessVerdict(
                citation_index=1,
                report_sentence="Faithful claim.",
                matched_finding_claims=["A claim"],
                faithful=True,
                confidence=1.0,
                reasoning="supported",
            ),
        ]
        return verdicts, [], []

    monkeypatch.setattr(verify_mod, "run_faithfulness_checks", fake_checks)

    state: ResearchState = {
        "run_id": str(uuid.uuid4()),
        "query": "test query",
        "clarification_questions": [],
        "clarifications": [],
        "subtasks": [],
        "findings": [
            {
                "subtask": "q1",
                "claim": "A claim",
                "evidence_span": "evidence",
                "citation_url": "https://a.com",
            }
        ],
        "summary": "summary",
        "report": (
            "Faithful claim [1]. This echoes the measurement crisis [Synthesis].\n\n"
            "## References\n\n[1] [A](https://a.com)\n"
        ),
        "messages": [],
    }
    result = await verify_mod.verify_citations(state)
    report = str(result["report"])
    assert "[Synthesis]" not in report
    assert "This echoes the measurement crisis." in report
    # The real markdown link in References must survive — only bracket
    # markers NOT followed by `(url)` are stripped
    assert "[1] [A](https://a.com)" in report


async def test_verify_citations_skips_when_no_findings_or_report() -> None:
    from engine.nodes.verify_citations import verify_citations

    state: ResearchState = {
        "run_id": str(uuid.uuid4()),
        "query": "test query",
        "clarification_questions": [],
        "clarifications": [],
        "subtasks": [],
        "findings": [],
        "summary": "",
        "report": "",
        "messages": [],
    }
    result = verify_citations(state)
    if hasattr(result, "__await__"):
        result = await result  # type: ignore[assignment]
    assert result == {"report": "", "findings": []}

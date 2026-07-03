"""Phase 1 smoke tests — verify graph structure and extraction schema, no API calls."""
import uuid

import pytest
from pydantic import ValidationError

from engine.extraction import Finding, FindingList
from engine.nodes.subagent import _should_skip_source
from engine.orchestrator import graph
from engine.state import ResearchState, SubagentInput


def test_graph_nodes_present() -> None:
    nodes = list(graph.nodes.keys())
    assert "plan" in nodes
    assert "subagent" in nodes
    assert "synthesize" in nodes


def test_finding_valid() -> None:
    f = Finding(
        claim="Test claim",
        evidence_span="Test evidence",
        citation_url="https://example.com/article",  # type: ignore[arg-type]
    )
    assert f.claim == "Test claim"
    assert str(f.citation_url).startswith("https://")


def test_finding_rejects_empty_claim() -> None:
    with pytest.raises(ValidationError):
        Finding(
            claim="   ",
            evidence_span="some evidence",
            citation_url="https://example.com",  # type: ignore[arg-type]
        )


def test_finding_rejects_invalid_url() -> None:
    with pytest.raises(ValidationError):
        Finding(
            claim="A claim",
            evidence_span="some evidence",
            citation_url="not-a-url",  # type: ignore[arg-type]
        )


def test_finding_list_empty() -> None:
    fl = FindingList(findings=[])
    assert fl.findings == []


def test_state_structure() -> None:
    state: ResearchState = {
        "run_id": str(uuid.uuid4()),
        "query": "What is quantum computing?",
        "clarification_questions": [],
        "clarifications": [],
        "subtasks": [],
        "findings": [],
        "summary": "",
        "report": "",
        "messages": [],
    }
    assert state["query"] == "What is quantum computing?"
    assert state["findings"] == []


def test_subagent_input() -> None:
    inp = SubagentInput(question="What are the latest advances in quantum hardware?")
    assert inp["question"].startswith("What")


def test_subagent_skips_video_sources_for_grounding_stability() -> None:
    assert _should_skip_source("https://www.youtube.com/watch?v=abc123")
    assert _should_skip_source("https://m.youtube.com/watch?v=abc123")
    assert _should_skip_source("https://youtu.be/abc123")
    assert not _should_skip_source("https://www.copyright.gov/ai/")


def test_subagent_skips_login_walled_linkedin_posts() -> None:
    assert _should_skip_source("https://www.linkedin.com/posts/someone_activity-123")
    assert _should_skip_source("https://sg.linkedin.com/pulse/some-article")
    assert not _should_skip_source("https://www.imda.gov.sg/framework")

"""Phase 4 eval harness tests — report parsing, citation grounding, faithfulness,
completeness, relevance (LLM judges mocked), and end-to-end harness
orchestration. No live API/DB calls."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from engine.nodes.verify_citations import verify_citations
from engine.state import SubtaskFinding, TokenUsage
from eval.citation_coverage import _CitationNeedVerdict, run_citation_coverage_check
from eval.completeness import (
    _CoverageList,
    _SubtopicList,
    generate_expected_subtopics,
    run_completeness_check,
    score_completeness,
)
from eval.faithfulness import _JudgeVerdict, run_faithfulness_checks
from eval.grounding import check_grounding, run_grounding_checks
from eval.harness import evaluate_run
from eval.loader import EvalRunData
from eval.rag_answer_relevance import _AnswerRelevanceVerdict, run_answer_relevance
from eval.rag_context_sufficiency import run_context_sufficiency
from eval.rag_faithfulness import _ClaimList, run_rag_faithfulness
from eval.relevance import _RelevanceVerdict, run_relevance_check
from eval.report_parsing import (
    extract_citation_indices,
    parse_references,
    split_body_and_references,
    split_sentences,
    strip_citation_markers,
)
from eval.schema import (
    CitationCoverageResult,
    CompletenessResult,
    FaithfulnessVerdict,
    GroundingResult,
    RagAnswerRelevanceVerdict,
    RelevanceResult,
    SubtopicCoverage,
)

_SAMPLE_REPORT = """## Overview

The market grew significantly [1][2]. However, adoption remains slow in some regions [3].

This is a connective sentence with no citation.

## References

[1] [Source A](https://a.com)
[2] [Source B](https://b.com)
[3] [Source C](https://c.com)
"""


# ---------------------------------------------------------------------------
# report_parsing
# ---------------------------------------------------------------------------

def test_parse_references() -> None:
    refs = parse_references(_SAMPLE_REPORT)
    assert refs[1].url == "https://a.com"
    assert refs[1].title == "Source A"
    assert refs[3].title == "Source C"


def test_split_body_and_references() -> None:
    body, references = split_body_and_references(_SAMPLE_REPORT)
    assert "References" not in body
    assert references.startswith("## References")


def test_extract_and_strip_citation_markers() -> None:
    sentence = "X is true [1][2]."
    assert extract_citation_indices(sentence) == [1, 2]
    assert extract_citation_indices(strip_citation_markers(sentence)) == []
    assert extract_citation_indices("No citation here.") == []


def test_split_sentences_tracks_section_and_uncited() -> None:
    body, _ = split_body_and_references(_SAMPLE_REPORT)
    sentences = split_sentences(body)

    cited = [(s, sec) for s, sec in sentences if extract_citation_indices(s)]
    uncited = [(s, sec) for s, sec in sentences if not extract_citation_indices(s)]

    assert len(cited) == 2
    assert len(uncited) == 1
    assert "connective sentence" in uncited[0][0]
    assert all(sec == "Overview" for _, sec in sentences)


# ---------------------------------------------------------------------------
# grounding
# ---------------------------------------------------------------------------

def _finding(evidence_span: str, citation_url: str = "https://example.com") -> SubtaskFinding:
    return SubtaskFinding(
        subtask="q", claim="c", evidence_span=evidence_span, citation_url=citation_url
    )


def test_check_grounding_exact_match() -> None:
    finding = _finding("The sky is blue today.")
    content = "Some text. The sky is blue today. More text."
    result = check_grounding(finding, content)
    assert result.grounded
    assert result.method == "exact"
    assert result.similarity == 1.0


def test_check_grounding_whitespace_normalized() -> None:
    finding = _finding("The sky\nis   blue today.")
    content = "Some text. The sky is blue today. More text."
    result = check_grounding(finding, content)
    assert result.grounded
    assert result.method == "exact"


def test_check_grounding_fuzzy_match_above_threshold() -> None:
    finding = _finding(
        "Global smartphone shipments declined 5 percent year over year in the first quarter of 2025"
    )
    content = (
        "Intro paragraph with unrelated context. Global smartphone shipments declined 5% "
        "year over year in the first quarter of 2025, according to the report. Outro text."
    )
    result = check_grounding(finding, content)
    assert result.grounded
    assert result.method == "fuzzy_window"
    assert result.similarity >= 0.85


def test_check_grounding_below_threshold_not_grounded() -> None:
    finding = _finding(
        "Global smartphone shipments declined 5 percent year over year in the first quarter of 2025"
    )
    content = (
        "Intro paragraph about something else entirely. The committee approved a new zoning "
        "ordinance for downtown commercial districts yesterday afternoon. Outro paragraph with "
        "more unrelated text padding to extend length further beyond the needle size."
    )
    result = check_grounding(finding, content)
    assert not result.grounded


def test_check_grounding_fetch_failed_empty_content() -> None:
    finding = _finding("anything")
    result = check_grounding(finding, "")
    assert not result.grounded
    assert result.method == "fetch_failed"


def test_check_grounding_short_span_not_found() -> None:
    finding = _finding("short")
    result = check_grounding(finding, "Some unrelated long content here that does not contain it")
    assert not result.grounded
    assert result.method == "fuzzy_window"
    assert result.similarity == 0.0


async def test_run_grounding_checks_caches_by_url() -> None:
    calls: list[str] = []

    def fake_fetch(url: str) -> str:
        calls.append(url)
        return "this content contains the evidence span here for testing purposes"

    findings = [
        _finding("evidence span here for testing", "https://a.com"),
        _finding("evidence span here for testing", "https://a.com"),
        _finding("this content contains", "https://b.com"),
    ]
    results = await run_grounding_checks(findings, fetch_fn=fake_fetch)
    assert len(results) == 3
    assert calls.count("https://a.com") == 1
    assert calls.count("https://b.com") == 1
    assert all(r.grounded for r in results)


# ---------------------------------------------------------------------------
# faithfulness (LLM judge mocked)
# ---------------------------------------------------------------------------

def _mock_structured_chain(monkeypatch, module: str, prompt_attr: str, parsed: object) -> MagicMock:
    """Patch `<module>.<prompt_attr>` and `<module>.make_chat_model` so any chain built
    from `<prompt_attr> | llm.with_structured_output(..., include_raw=True)`
    returns `{"raw": <message with usage_metadata>, "parsed": parsed}`."""
    raw_message = SimpleNamespace(usage_metadata={"input_tokens": 10, "output_tokens": 5})
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value={"raw": raw_message, "parsed": parsed})

    mock_prompt = MagicMock()
    mock_prompt.__or__ = lambda self, other: mock_chain
    monkeypatch.setattr(f"{module}.{prompt_attr}", mock_prompt)
    monkeypatch.setattr(f"{module}.make_chat_model", lambda *a, **kw: MagicMock())
    return mock_chain


def _mock_judge(monkeypatch, verdict: _JudgeVerdict) -> None:
    _mock_structured_chain(monkeypatch, "eval.faithfulness", "_JUDGE_PROMPT", verdict)


async def test_run_faithfulness_checks_faithful(monkeypatch) -> None:
    _mock_judge(monkeypatch, _JudgeVerdict(faithful=True, confidence=0.9, reasoning="supported"))

    findings = [
        _finding("evidence A", "https://a.com"),
        _finding("evidence B", "https://b.com"),
        _finding("evidence C", "https://c.com"),
    ]

    verdicts, uncited, token_usage = await run_faithfulness_checks(_SAMPLE_REPORT, findings)

    assert len(verdicts) == 2
    assert all(v.faithful for v in verdicts)
    assert len(uncited) == 1
    assert "connective sentence" in uncited[0].sentence
    assert len(token_usage) == 2
    assert all(u["model"] for u in token_usage)


async def test_run_faithfulness_checks_unfaithful(monkeypatch) -> None:
    _mock_judge(
        monkeypatch,
        _JudgeVerdict(faithful=False, confidence=0.8, reasoning="adds unsupported detail"),
    )

    findings = [
        _finding("evidence A", "https://a.com"),
        _finding("evidence B", "https://b.com"),
        _finding("evidence C", "https://c.com"),
    ]

    verdicts, _, _ = await run_faithfulness_checks(_SAMPLE_REPORT, findings)
    assert all(not v.faithful for v in verdicts)
    assert all(v.reasoning == "adds unsupported detail" for v in verdicts)


async def test_faithfulness_citation_index_not_in_references() -> None:
    report = "Some claim [9].\n\n## References\n\n[1] [Source A](https://a.com)\n"
    verdicts, uncited, token_usage = await run_faithfulness_checks(report, findings=[])
    assert len(verdicts) == 1
    assert verdicts[0].faithful is False
    assert "not found" in verdicts[0].reasoning.lower()
    assert uncited == []
    assert token_usage == []


async def test_faithfulness_no_finding_for_citation_url() -> None:
    report = "Some claim [1].\n\n## References\n\n[1] [Source A](https://a.com)\n"
    verdicts, _, _ = await run_faithfulness_checks(report, findings=[])
    assert verdicts[0].faithful is False
    assert "no finding" in verdicts[0].reasoning.lower()


async def test_verify_citations_removes_unfaithful_sentence(monkeypatch) -> None:
    report = (
        "Supported claim [1]. Unsupported claim [1].\n\n"
        "## References\n\n[1] [Source A](https://a.com)\n"
    )

    async def fake_faithfulness(report, findings, lead_model):
        return (
            [
                FaithfulnessVerdict(
                    citation_index=1,
                    report_sentence="Supported claim",
                    matched_finding_claims=["c"],
                    faithful=True,
                    confidence=1.0,
                    reasoning="ok",
                ),
                FaithfulnessVerdict(
                    citation_index=1,
                    report_sentence="Unsupported claim",
                    matched_finding_claims=["c"],
                    faithful=False,
                    confidence=1.0,
                    reasoning="unsupported",
                ),
            ],
            [],
            [],
        )

    monkeypatch.setattr("engine.nodes.verify_citations.run_faithfulness_checks", fake_faithfulness)
    result = await verify_citations(  # type: ignore[typeddict-item]
        {"report": report, "findings": [_finding("evidence", "https://a.com")]}
    )
    verified_report = str(result["report"])
    assert "Supported claim [1]." in verified_report
    assert "Unsupported claim" not in verified_report


# ---------------------------------------------------------------------------
# completeness (LLM judge mocked)
# ---------------------------------------------------------------------------

async def test_generate_expected_subtopics(monkeypatch) -> None:
    _mock_structured_chain(
        monkeypatch, "eval.completeness", "_SUBTOPICS_PROMPT",
        _SubtopicList(subtopics=["a", "b", "c"]),
    )

    subtopics, usage = await generate_expected_subtopics("query")

    assert subtopics == ["a", "b", "c"]
    assert usage is not None
    assert usage["input_tokens"] == 10


async def test_score_completeness(monkeypatch) -> None:
    coverage = [
        SubtopicCoverage(subtopic="a", covered=True, note="addressed"),
        SubtopicCoverage(subtopic="b", covered=False, note="missing"),
    ]
    _mock_structured_chain(
        monkeypatch, "eval.completeness", "_COVERAGE_PROMPT",
        _CoverageList(coverage=coverage),
    )

    result, usage = await score_completeness("report body", ["a", "b"])

    assert result.recall_score == 0.5
    assert [s.covered for s in result.subtopics] == [True, False]
    assert usage is not None


async def test_score_completeness_no_subtopics_short_circuits() -> None:
    result, usage = await score_completeness("report body", [])
    assert result.recall_score == 1.0
    assert result.subtopics == []
    assert usage is None


async def test_run_completeness_check(monkeypatch) -> None:
    async def fake_subtopics(query: str, lead_model: str):
        return ["a", "b"], TokenUsage(
            node="completeness_subtopics", model=lead_model,
            input_tokens=1, output_tokens=1, cached_tokens=0,
        )

    async def fake_score(report: str, subtopics: list[str], lead_model: str):
        return (
            CompletenessResult(
                subtopics=[
                    SubtopicCoverage(subtopic="a", covered=True),
                    SubtopicCoverage(subtopic="b", covered=True),
                ],
                recall_score=1.0,
            ),
            TokenUsage(
                node="completeness_coverage", model=lead_model,
                input_tokens=2, output_tokens=2, cached_tokens=0,
            ),
        )

    monkeypatch.setattr("eval.completeness.generate_expected_subtopics", fake_subtopics)
    monkeypatch.setattr("eval.completeness.score_completeness", fake_score)

    result, usages = await run_completeness_check("query", "report")

    assert result.recall_score == 1.0
    assert len(usages) == 2


# ---------------------------------------------------------------------------
# relevance (LLM judge mocked)
# ---------------------------------------------------------------------------

async def test_run_relevance_check(monkeypatch) -> None:
    _mock_structured_chain(
        monkeypatch, "eval.relevance", "_RELEVANCE_PROMPT",
        _RelevanceVerdict(score=4, reasoning="mostly on-topic"),
    )

    result, usage = await run_relevance_check("query", "report body")

    assert result.score == 4
    assert result.reasoning == "mostly on-topic"
    assert usage is not None
    assert usage["output_tokens"] == 5


# ---------------------------------------------------------------------------
# citation coverage (LLM judge mocked)
# ---------------------------------------------------------------------------

async def test_run_citation_coverage_flags_uncited_factual_claim(monkeypatch) -> None:
    _mock_structured_chain(
        monkeypatch,
        "eval.citation_coverage",
        "_COVERAGE_PROMPT",
        _CitationNeedVerdict(citation_required=True, reasoning="specific factual claim"),
    )

    result, usage = await run_citation_coverage_check(_SAMPLE_REPORT)

    assert result.cited_sentence_count == 2
    assert len(result.uncited_factual_claims) == 1
    assert result.coverage_score == 2 / 3
    assert len(usage) == 1


async def test_run_citation_coverage_no_uncited_sentences_short_circuits() -> None:
    report = "Claim one [1]. Claim two [2].\n\n## References\n\n[1] [A](https://a.com)"

    result, usage = await run_citation_coverage_check(report)

    assert result.coverage_score == 1.0
    assert result.cited_sentence_count == 2
    assert result.uncited_factual_claims == []
    assert usage == []


# ---------------------------------------------------------------------------
# RAG eval
# ---------------------------------------------------------------------------

async def test_rag_answer_relevance_scores_answer(monkeypatch) -> None:
    _mock_structured_chain(
        monkeypatch,
        "eval.rag_answer_relevance",
        "_ANSWER_RELEVANCE_PROMPT",
        _AnswerRelevanceVerdict(score=4, reasoning="directly answers most of it"),
    )

    verdict, score, usage = await run_answer_relevance(
        "What changed?",
        "The main change was X.",
        chunks=[{"content": "X changed.", "title": "Report"}],
    )

    assert verdict == RagAnswerRelevanceVerdict(
        score=4,
        reasoning="directly answers most of it",
    )
    assert score == 0.8
    assert len(usage) == 1

async def test_rag_faithfulness_no_chunks_marks_claims_unsupported(monkeypatch) -> None:
    _mock_structured_chain(
        monkeypatch,
        "eval.rag_faithfulness",
        "_EXTRACT_PROMPT",
        _ClaimList(claims=["The market grew 20 percent."]),
    )

    verdicts, score, usage = await run_rag_faithfulness(
        "The market grew 20 percent.",
        chunks=[],
    )

    assert score == 0.0
    assert len(verdicts) == 1
    assert verdicts[0].supported is False
    assert "No source chunks" in verdicts[0].reasoning
    assert len(usage) == 1


async def test_rag_context_sufficiency_no_chunks_short_circuits() -> None:
    verdict, score, usage = await run_context_sufficiency("What changed?", chunks=[])

    assert score == 0.0
    assert verdict.sufficient is False
    assert "No source chunks" in verdict.reasoning
    assert usage == []


# ---------------------------------------------------------------------------
# harness end-to-end (loader + checks mocked)
# ---------------------------------------------------------------------------

def _fake_completeness_check():
    async def fake(query: str, report: str, lead_model: str):
        return (
            CompletenessResult(
                subtopics=[SubtopicCoverage(subtopic="a", covered=True)], recall_score=1.0
            ),
            [],
        )
    return fake


def _fake_relevance_check():
    async def fake(query: str, report: str, lead_model: str):
        return RelevanceResult(score=5, reasoning="on-topic"), None
    return fake


def _fake_citation_coverage_check(
    uncited_factual_claims: int = 0,
):
    async def fake(report: str, lead_model: str):
        return (
            CitationCoverageResult(
                coverage_score=1.0 if uncited_factual_claims == 0 else 0.5,
                cited_sentence_count=1,
                uncited_factual_claims=[
                    {
                        "sentence": f"uncited claim {i}",
                        "section": "Overview",
                        "reasoning": "needs a citation",
                    }
                    for i in range(uncited_factual_claims)
                ],
            ),
            [],
        )
    return fake


async def test_evaluate_run_passes_when_all_grounded_and_faithful(monkeypatch) -> None:
    fake_run_data = EvalRunData(run_id="r1", query="q", status="done", report="report", findings=[])

    async def fake_load_run(run_id: str, require_done: bool = True) -> EvalRunData:
        return fake_run_data

    async def fake_grounding(findings, fetch_fn=None) -> list[GroundingResult]:
        return [
            GroundingResult(
                subtask="s", claim="c", evidence_span="e", citation_url="https://a.com",
                grounded=True, similarity=1.0, method="exact", fetch_chars=100,
            )
        ]

    async def fake_faithfulness(report, findings, lead_model):
        return (
            [FaithfulnessVerdict(
                citation_index=1, report_sentence="x", matched_finding_claims=["c"],
                faithful=True, confidence=0.9, reasoning="ok",
            )],
            [],
            [],
        )

    monkeypatch.setattr("eval.harness.load_run", fake_load_run)
    monkeypatch.setattr("eval.harness.run_grounding_checks", fake_grounding)
    monkeypatch.setattr("eval.harness.run_faithfulness_checks", fake_faithfulness)
    monkeypatch.setattr("eval.harness.run_citation_coverage_check", _fake_citation_coverage_check())
    monkeypatch.setattr("eval.harness.run_completeness_check", _fake_completeness_check())
    monkeypatch.setattr("eval.harness.run_relevance_check", _fake_relevance_check())

    report = await evaluate_run("r1")
    assert report.passed
    assert report.ungrounded_count == 0
    assert report.unfaithful_count == 0
    assert report.completeness.recall_score == 1.0
    assert report.relevance.score == 5
    assert report.eval_model == "gpt-5.4"


async def test_evaluate_run_fails_on_ungrounded(monkeypatch) -> None:
    fake_run_data = EvalRunData(run_id="r1", query="q", status="done", report="report", findings=[])

    async def fake_load_run(run_id: str, require_done: bool = True) -> EvalRunData:
        return fake_run_data

    async def fake_grounding(findings, fetch_fn=None) -> list[GroundingResult]:
        return [
            GroundingResult(
                subtask="s", claim="c", evidence_span="e", citation_url="https://a.com",
                grounded=False, similarity=0.2, method="fuzzy_window", fetch_chars=100,
            )
        ]

    async def fake_faithfulness(report, findings, lead_model):
        return [], [], []

    monkeypatch.setattr("eval.harness.load_run", fake_load_run)
    monkeypatch.setattr("eval.harness.run_grounding_checks", fake_grounding)
    monkeypatch.setattr("eval.harness.run_faithfulness_checks", fake_faithfulness)
    monkeypatch.setattr("eval.harness.run_citation_coverage_check", _fake_citation_coverage_check())
    monkeypatch.setattr("eval.harness.run_completeness_check", _fake_completeness_check())
    monkeypatch.setattr("eval.harness.run_relevance_check", _fake_relevance_check())

    report = await evaluate_run("r1")
    assert not report.passed
    assert report.ungrounded_count == 1
    assert "ungrounded" in report.failure_reasons[0]


async def test_evaluate_run_strict_mode_fails_on_unfaithful(monkeypatch) -> None:
    fake_run_data = EvalRunData(run_id="r1", query="q", status="done", report="report", findings=[])

    async def fake_load_run(run_id: str, require_done: bool = True) -> EvalRunData:
        return fake_run_data

    async def fake_grounding(findings, fetch_fn=None) -> list[GroundingResult]:
        return []

    async def fake_faithfulness(report, findings, lead_model):
        return (
            [FaithfulnessVerdict(
                citation_index=1, report_sentence="x", matched_finding_claims=[],
                faithful=False, confidence=1.0, reasoning="bad",
            )],
            [],
            [],
        )

    monkeypatch.setattr("eval.harness.load_run", fake_load_run)
    monkeypatch.setattr("eval.harness.run_grounding_checks", fake_grounding)
    monkeypatch.setattr("eval.harness.run_faithfulness_checks", fake_faithfulness)
    monkeypatch.setattr("eval.harness.run_citation_coverage_check", _fake_citation_coverage_check())
    monkeypatch.setattr("eval.harness.run_completeness_check", _fake_completeness_check())
    monkeypatch.setattr("eval.harness.run_relevance_check", _fake_relevance_check())

    lenient = await evaluate_run("r1", strict=False)
    assert lenient.passed  # 0 ungrounded -> passes even with an unfaithful citation

    strict = await evaluate_run("r1", strict=True)
    assert not strict.passed
    assert "unfaithful" in strict.failure_reasons[0]


async def test_evaluate_run_strict_mode_fails_on_uncited_factual_claim(monkeypatch) -> None:
    fake_run_data = EvalRunData(run_id="r1", query="q", status="done", report="report", findings=[])

    async def fake_load_run(run_id: str, require_done: bool = True) -> EvalRunData:
        return fake_run_data

    async def fake_grounding(findings, fetch_fn=None) -> list[GroundingResult]:
        return []

    async def fake_faithfulness(report, findings, lead_model):
        return [], [], []

    monkeypatch.setattr("eval.harness.load_run", fake_load_run)
    monkeypatch.setattr("eval.harness.run_grounding_checks", fake_grounding)
    monkeypatch.setattr("eval.harness.run_faithfulness_checks", fake_faithfulness)
    monkeypatch.setattr(
        "eval.harness.run_citation_coverage_check",
        _fake_citation_coverage_check(uncited_factual_claims=1),
    )
    monkeypatch.setattr("eval.harness.run_completeness_check", _fake_completeness_check())
    monkeypatch.setattr("eval.harness.run_relevance_check", _fake_relevance_check())

    lenient = await evaluate_run("r1", strict=False)
    assert lenient.passed

    strict = await evaluate_run("r1", strict=True)
    assert not strict.passed
    assert "uncited factual" in strict.failure_reasons[0]

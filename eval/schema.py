"""Pydantic models for the eval harness (Phase 4): citation grounding + faithfulness results."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class GroundingResult(BaseModel):
    """Result of checking one Finding's evidence_span against its fetched source."""

    subtask: str
    claim: str
    evidence_span: str
    citation_url: str
    grounded: bool
    similarity: float
    method: Literal["exact", "fuzzy_window", "fetch_failed"]
    fetch_chars: int
    note: str = ""


class CitationRef(BaseModel):
    """One [i] -> (title, url) entry parsed from the report's ## References section."""

    index: int
    title: str
    url: str


class FaithfulnessVerdict(BaseModel):
    """LLM-judge verdict for one cited sentence in the report."""

    citation_index: int
    report_sentence: str
    matched_finding_claims: list[str]
    faithful: bool
    confidence: float
    reasoning: str


class UncitedSentence(BaseModel):
    """A report sentence with no [i] citation marker (informational only)."""

    sentence: str
    section: str = ""


class SubtopicCoverage(BaseModel):
    """Whether one expected subtopic is addressed by the report."""

    subtopic: str
    covered: bool
    note: str = ""


class CompletenessResult(BaseModel):
    """LLM-judge completeness check: expected subtopics vs. report coverage."""

    subtopics: list[SubtopicCoverage]
    recall_score: float  # covered / total, 1.0 if total == 0


class RelevanceResult(BaseModel):
    """LLM-judge relevance check: is the report on-topic for the query."""

    score: int  # 1-5
    reasoning: str


class RagChunkVerdict(BaseModel):
    """LLM-judge verdict for one retrieved chunk's relevance to the question."""

    chunk_index: int
    title: str
    section: str  # header_path from MarkdownNodeParser (e.g. /Topic/Subtopic/)
    preview: str  # first 200 chars of chunk content
    relevant: bool
    reasoning: str


class RagAnswerClaimVerdict(BaseModel):
    """LLM-judge verdict for one atomic claim extracted from the RAG answer."""

    claim: str
    supported: bool
    reasoning: str


class RagEvalReport(BaseModel):
    """Aggregate RAG eval result for one question against the Research Library."""

    question: str
    generated_at: str

    # Stage 1 — metadata filter
    selected_reports: list[str]  # run_ids the LLM selected as relevant

    # Stage 2 — retrieval
    chunks_retrieved: int
    chunk_verdicts: list[RagChunkVerdict]
    context_precision: float  # relevant chunks / total chunks (0–1)

    # Generation
    answer: str
    claim_verdicts: list[RagAnswerClaimVerdict]
    answer_faithfulness: float  # supported claims / total claims (0–1)

    eval_model: str = ""
    eval_input_tokens: int = 0
    eval_output_tokens: int = 0
    eval_cost_usd: float = 0.0

    def summary(self) -> str:
        """Human-readable multi-line summary for CLI output."""
        n_relevant = sum(1 for v in self.chunk_verdicts if v.relevant)
        n_supported = sum(1 for v in self.claim_verdicts if v.supported)
        lines = [
            f"RAG Eval: {self.question}",
            f"Generated: {self.generated_at}",
            "",
            f"Stage 1 selected {len(self.selected_reports)} report(s)",
            f"Stage 2 retrieved {self.chunks_retrieved} chunk(s)",
            "",
            f"Context Precision: {self.context_precision:.0%} "
            f"({n_relevant}/{len(self.chunk_verdicts)} chunks relevant)",
        ]
        for v in self.chunk_verdicts:
            mark = "+" if v.relevant else "-"
            lines.append(f"  [{mark}] [{v.title}] {v.preview[:80]}…")
            if not v.relevant:
                lines.append(f"       {v.reasoning}")
        lines += [
            "",
            f"Answer Faithfulness: {self.answer_faithfulness:.0%} "
            f"({n_supported}/{len(self.claim_verdicts)} claims supported)",
        ]
        for v in self.claim_verdicts:
            mark = "+" if v.supported else "-"
            lines.append(f"  [{mark}] {v.claim}")
            if not v.supported:
                lines.append(f"       {v.reasoning}")
        lines += [
            "",
            f"Eval cost: ${self.eval_cost_usd:.4f} ({self.eval_model}, "
            f"{self.eval_input_tokens + self.eval_output_tokens} tokens)",
        ]
        return "\n".join(lines)


class EvalReport(BaseModel):
    """Aggregate eval result for one completed research run."""

    run_id: str
    query: str
    generated_at: str

    grounding_results: list[GroundingResult]
    faithfulness_results: list[FaithfulnessVerdict]
    uncited_sentences: list[UncitedSentence]
    completeness: CompletenessResult
    relevance: RelevanceResult

    total_findings: int
    ungrounded_count: int
    unfaithful_count: int

    passed: bool
    failure_reasons: list[str]

    eval_model: str = ""
    eval_input_tokens: int = 0
    eval_output_tokens: int = 0
    eval_cost_usd: float = 0.0

    def summary(self) -> str:
        """Human-readable multi-line summary for CLI output."""
        lines = [
            f"Eval report for run {self.run_id}",
            f"Query: {self.query}",
            "",
            f"Findings checked: {self.total_findings}",
            f"  Grounded:   {self.total_findings - self.ungrounded_count}/{self.total_findings}",
            f"  Ungrounded: {self.ungrounded_count}",
        ]
        for g in self.grounding_results:
            if not g.grounded:
                lines.append(
                    f'    - [subtask "{g.subtask}"] claim "{g.claim}" '
                    f"(similarity={g.similarity:.2f}, method={g.method}, url={g.citation_url})"
                )

        lines += [
            "",
            f"Faithfulness (citations checked): {len(self.faithfulness_results)}",
            f"  Faithful:   {len(self.faithfulness_results) - self.unfaithful_count}/"
            f"{len(self.faithfulness_results)}",
            f"  Unfaithful: {self.unfaithful_count}",
        ]
        for f in self.faithfulness_results:
            if not f.faithful:
                lines.append(
                    f'    - [{f.citation_index}] sentence "{f.report_sentence}" '
                    f"-> reasoning: {f.reasoning}"
                )

        lines += [
            "",
            f"Uncited sentences: {len(self.uncited_sentences)} (informational)",
            "",
            f"Completeness: {self.completeness.recall_score:.0%} "
            f"({sum(1 for s in self.completeness.subtopics if s.covered)}/"
            f"{len(self.completeness.subtopics)} subtopics covered)",
        ]
        for s in self.completeness.subtopics:
            if not s.covered:
                lines.append(f'    - missing: "{s.subtopic}"' + (f" — {s.note}" if s.note else ""))

        lines += [
            "",
            f"Relevance: {self.relevance.score}/5 — {self.relevance.reasoning}",
            "",
            f"Eval cost: ${self.eval_cost_usd:.4f} ({self.eval_model}, "
            f"{self.eval_input_tokens + self.eval_output_tokens} tokens)",
            "",
            "RESULT: " + ("PASS" if self.passed else "FAIL")
            + (f" — {', '.join(self.failure_reasons)}" if self.failure_reasons else ""),
        ]
        return "\n".join(lines)

"""RAG eval orchestrator.

Runs the full Research Library pipeline for one question and evaluates it:
  1. Two-stage retrieval (same path as /library/chat).
  2. Answer generation (collects streamed tokens).
  3. Context Precision  — % of retrieved chunks that are relevant.
  4. Context Sufficiency — whether chunks are enough to answer.
  5. Answer Faithfulness — % of answer claims supported by the chunks.
  6. Answer Relevance — whether the answer responds to the question.

No ground truth required — all metrics use LLM-as-judge.
"""
from __future__ import annotations

from datetime import datetime, timezone

from engine.memory import rag
from engine.models import EVAL_MODEL, estimate_cost_usd
from engine.state import TokenUsage
from eval.rag_answer_relevance import run_answer_relevance
from eval.rag_context_precision import run_context_precision
from eval.rag_context_sufficiency import run_context_sufficiency
from eval.rag_faithfulness import run_rag_faithfulness
from eval.schema import RagEvalReport


async def evaluate_rag(
    question: str,
    available_reports: list[dict[str, str]],
    eval_model: str = EVAL_MODEL,
) -> RagEvalReport:
    """Run RAG retrieval + generation + evaluation for one question.

    `available_reports` should be the same list passed to /library/chat —
    each entry is {"run_id": str, "title": str, "query": str}.
    """
    # Retrieval (two-stage: metadata filter → scoped semantic search)
    selected_ids, chunks = await rag.two_stage_search(question, available_reports)

    # Generation — collect the streamed answer
    answer_parts: list[str] = []
    async for token in rag.answer_with_context(question, [], chunks):
        answer_parts.append(token)
    answer = "".join(answer_parts)

    # Evaluate all metrics concurrently
    import asyncio
    (precision_verdicts, context_precision, precision_usage), \
    (sufficiency_verdict, context_sufficiency, sufficiency_usage), \
    (faithfulness_verdicts, answer_faithfulness, faithfulness_usage), \
    (relevance_verdict, answer_relevance, relevance_usage) = await asyncio.gather(
        run_context_precision(question, chunks, eval_model),
        run_context_sufficiency(question, chunks, eval_model),
        run_rag_faithfulness(answer, chunks, eval_model),
        run_answer_relevance(question, answer, chunks, eval_model),
    )

    all_usage: list[TokenUsage] = [
        *precision_usage,
        *sufficiency_usage,
        *faithfulness_usage,
        *relevance_usage,
    ]

    return RagEvalReport(
        question=question,
        generated_at=datetime.now(timezone.utc).isoformat(),
        selected_reports=selected_ids,
        chunks_retrieved=len(chunks),
        chunk_verdicts=precision_verdicts,
        context_precision=context_precision,
        context_sufficiency=context_sufficiency,
        context_sufficiency_verdict=sufficiency_verdict,
        answer=answer,
        claim_verdicts=faithfulness_verdicts,
        answer_faithfulness=answer_faithfulness,
        answer_relevance=answer_relevance,
        answer_relevance_verdict=relevance_verdict,
        eval_model=eval_model,
        eval_input_tokens=sum(u["input_tokens"] for u in all_usage),
        eval_output_tokens=sum(u["output_tokens"] for u in all_usage),
        eval_cost_usd=estimate_cost_usd(all_usage),
    )

"""LLM-as-judge context sufficiency for RAG evaluation.

Precision answers "are retrieved chunks relevant?". Sufficiency answers the
separate question "do the retrieved chunks contain enough information to answer
the user?". This catches empty or too-thin retrieval even when precision is high.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from engine.models import EVAL_MODEL, make_chat_model, structured_output_kwargs
from engine.state import TokenUsage
from engine.usage import usage_from_message
from eval.schema import RagContextSufficiencyVerdict


class _SufficiencyVerdict(BaseModel):
    sufficient: bool = Field(
        description="True if the chunks contain enough information to answer the question"
    )
    reasoning: str = Field(description="One sentence justification")


_SUFFICIENCY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are evaluating retrieved context for a RAG system. "
        "Answer True only if the chunks contain enough specific information to directly "
        "answer the user's question. Answer False if the chunks are empty, too vague, "
        "missing key parts of the question, or only tangentially related.",
    ),
    ("human", "Question: {question}\n\nRetrieved chunks:\n{context}"),
])


async def run_context_sufficiency(
    question: str,
    chunks: list[dict[str, object]],
    eval_model: str = EVAL_MODEL,
) -> tuple[RagContextSufficiencyVerdict, float, list[TokenUsage]]:
    """Judge whether retrieved chunks are sufficient to answer the question.

    Returns (verdict, sufficiency_score, token_usage), where sufficiency_score
    is 1.0 for sufficient context and 0.0 otherwise.
    """
    if not chunks:
        return (
            RagContextSufficiencyVerdict(
                sufficient=False,
                reasoning="No source chunks were retrieved.",
            ),
            0.0,
            [],
        )

    context = "\n\n---\n\n".join(
        f"[Chunk {i + 1} — {c.get('title', '')}]\n{c['content']}"
        for i, c in enumerate(chunks)
    )

    judge_llm = make_chat_model(eval_model, temperature=0)
    chain = _SUFFICIENCY_PROMPT | judge_llm.with_structured_output(
        _SufficiencyVerdict, **structured_output_kwargs(eval_model), include_raw=True
    )
    raw = await chain.ainvoke({"question": question, "context": context})
    assert isinstance(raw, dict)
    parsed: _SufficiencyVerdict = raw["parsed"]
    usage = usage_from_message(raw["raw"], "rag_sufficiency", eval_model)
    verdict = RagContextSufficiencyVerdict(
        sufficient=parsed.sufficient,
        reasoning=parsed.reasoning,
    )
    return verdict, 1.0 if parsed.sufficient else 0.0, [usage] if usage else []

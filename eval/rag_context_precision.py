"""LLM-as-judge context precision for RAG evaluation.

For each chunk retrieved from the Research Library, judges whether it
actually helps answer the question. Precision = relevant / total retrieved.
All chunk judgements run concurrently.
"""
from __future__ import annotations

import asyncio

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from engine.models import EVAL_MODEL, make_chat_model, structured_output_kwargs
from engine.state import TokenUsage
from engine.usage import usage_from_message
from eval.schema import RagChunkVerdict


class _ChunkVerdict(BaseModel):
    relevant: bool = Field(
        description="True if this chunk contains information that directly helps answer the question"
    )
    reasoning: str = Field(description="One sentence justification")


_PRECISION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are evaluating whether a retrieved text chunk is relevant to a user's question. "
        "Answer True if the chunk contains information that directly helps answer the question. "
        "Answer False if the chunk is off-topic, tangential, or provides no useful information "
        "for this specific question.",
    ),
    ("human", "Question: {question}\n\nChunk:\n{chunk_text}"),
])


async def run_context_precision(
    question: str,
    chunks: list[dict[str, object]],
    eval_model: str = EVAL_MODEL,
) -> tuple[list[RagChunkVerdict], float, list[TokenUsage]]:
    """Judge each chunk for relevance to the question.

    Returns (verdicts, precision_score, token_usage).
    precision_score = relevant chunks / total chunks (0.0 if no chunks).
    """
    if not chunks:
        return [], 0.0, []

    judge_llm = make_chat_model(eval_model, temperature=0)
    chain = _PRECISION_PROMPT | judge_llm.with_structured_output(
        _ChunkVerdict, **structured_output_kwargs(eval_model), include_raw=True
    )

    async def _judge_one(i: int, chunk: dict[str, object]) -> tuple[RagChunkVerdict, TokenUsage | None]:
        raw = await chain.ainvoke({"question": question, "chunk_text": chunk["content"]})
        assert isinstance(raw, dict)
        verdict: _ChunkVerdict = raw["parsed"]
        usage = usage_from_message(raw["raw"], "rag_precision", eval_model)
        return (
            RagChunkVerdict(
                chunk_index=i,
                title=str(chunk.get("title", "")),
                section=str(chunk.get("section", "")),
                preview=str(chunk["content"])[:200],
                relevant=verdict.relevant,
                reasoning=verdict.reasoning,
            ),
            usage,
        )

    results = await asyncio.gather(*[_judge_one(i, c) for i, c in enumerate(chunks)])

    verdicts: list[RagChunkVerdict] = []
    token_usage: list[TokenUsage] = []
    for verdict, usage in results:
        verdicts.append(verdict)
        if usage:
            token_usage.append(usage)

    precision = sum(1 for v in verdicts if v.relevant) / len(verdicts)
    return verdicts, precision, token_usage

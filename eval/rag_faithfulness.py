"""LLM-as-judge faithfulness for RAG-generated answers.

Two-step process:
  1. Extract atomic factual claims from the answer (one LLM call).
  2. Judge each claim against the retrieved chunks concurrently.

Faithfulness = supported claims / total claims (0.0 if no claims).
"""
from __future__ import annotations

import asyncio

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from engine.models import EVAL_MODEL, make_chat_model, structured_output_kwargs
from engine.state import TokenUsage
from engine.usage import usage_from_message
from eval.schema import RagAnswerClaimVerdict


class _ClaimList(BaseModel):
    claims: list[str] = Field(
        description="Atomic factual claims extracted from the answer, one per item"
    )


class _ClaimVerdict(BaseModel):
    supported: bool = Field(
        description="True if the claim is directly supported by at least one of the provided chunks"
    )
    reasoning: str = Field(description="One sentence justification citing the chunk or gap")


_EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Extract every atomic factual claim from the answer. "
        "Each item must be a single verifiable statement (no conjunctions). "
        "Strip attribution phrases like 'According to the report' — keep only the factual content. "
        "If the answer says there is insufficient context, return an empty list.",
    ),
    ("human", "Answer:\n{answer}"),
])

_FAITH_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are checking whether a factual claim is supported by the provided source chunks. "
        "Answer True if the claim is directly backed by information in at least one chunk. "
        "Answer False if the claim adds facts not in the chunks, contradicts them, or cannot "
        "be verified from them.",
    ),
    ("human", "Claim: {claim}\n\nSource chunks:\n{context}"),
])


async def run_rag_faithfulness(
    answer: str,
    chunks: list[dict[str, object]],
    eval_model: str = EVAL_MODEL,
) -> tuple[list[RagAnswerClaimVerdict], float, list[TokenUsage]]:
    """Judge whether each claim in the answer is supported by the retrieved chunks.

    Returns (verdicts, faithfulness_score, token_usage).
    faithfulness_score = supported / total (1.0 when there are no claims to check).
    """
    if not answer.strip() or not chunks:
        return [], 1.0, []

    context = "\n\n---\n\n".join(
        f"[Chunk {i + 1} — {c.get('title', '')}]\n{c['content']}"
        for i, c in enumerate(chunks)
    )

    judge_llm = make_chat_model(eval_model, temperature=0)
    token_usage: list[TokenUsage] = []

    # Step 1: extract claims
    extract_chain = _EXTRACT_PROMPT | judge_llm.with_structured_output(
        _ClaimList, **structured_output_kwargs(eval_model), include_raw=True
    )
    raw = await extract_chain.ainvoke({"answer": answer})
    assert isinstance(raw, dict)
    claim_list: _ClaimList = raw["parsed"]
    extract_usage = usage_from_message(raw["raw"], "rag_extract_claims", eval_model)
    if extract_usage:
        token_usage.append(extract_usage)

    if not claim_list.claims:
        return [], 1.0, token_usage

    # Step 2: judge each claim concurrently
    faith_chain = _FAITH_PROMPT | judge_llm.with_structured_output(
        _ClaimVerdict, **structured_output_kwargs(eval_model), include_raw=True
    )

    async def _judge_claim(claim: str) -> tuple[RagAnswerClaimVerdict, TokenUsage | None]:
        raw = await faith_chain.ainvoke({"claim": claim, "context": context})
        assert isinstance(raw, dict)
        verdict: _ClaimVerdict = raw["parsed"]
        usage = usage_from_message(raw["raw"], "rag_faithfulness", eval_model)
        return (
            RagAnswerClaimVerdict(
                claim=claim,
                supported=verdict.supported,
                reasoning=verdict.reasoning,
            ),
            usage,
        )

    results = await asyncio.gather(*[_judge_claim(c) for c in claim_list.claims])
    verdicts: list[RagAnswerClaimVerdict] = []
    for verdict, usage in results:
        verdicts.append(verdict)
        if usage:
            token_usage.append(usage)

    faithfulness = sum(1 for v in verdicts if v.supported) / len(verdicts)
    return verdicts, faithfulness, token_usage

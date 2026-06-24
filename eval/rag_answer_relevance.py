"""LLM-as-judge answer relevance for RAG evaluation."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from engine.models import EVAL_MODEL, make_chat_model, structured_output_kwargs
from engine.state import TokenUsage
from engine.usage import usage_from_message
from eval.schema import RagAnswerRelevanceVerdict


class _AnswerRelevanceVerdict(BaseModel):
    score: int = Field(
        ge=1,
        le=5,
        description=(
            "1 = answer is off-topic or refuses despite enough context; "
            "5 = answer directly and completely responds to the user's question."
        ),
    )
    reasoning: str = Field(description="One sentence justification")


_ANSWER_RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are evaluating whether a RAG answer responds to the user's question. "
        "Score 1-5. Reward concise answers that directly address the question. "
        "Do not penalize an answer for saying the context is insufficient when the "
        "retrieved context truly appears insufficient.",
    ),
    ("human", "Question: {question}\n\nAnswer:\n{answer}\n\nRetrieved chunks:\n{context}"),
])


async def run_answer_relevance(
    question: str,
    answer: str,
    chunks: list[dict[str, object]],
    eval_model: str = EVAL_MODEL,
) -> tuple[RagAnswerRelevanceVerdict, float, list[TokenUsage]]:
    """Score whether the generated answer addresses the user question.

    Returns (verdict, normalized_score, token_usage), where normalized_score is
    score / 5 so it fits the 0-1 UI used by the other RAG metrics.
    """
    if not answer.strip():
        return (
            RagAnswerRelevanceVerdict(score=1, reasoning="The answer is empty."),
            0.2,
            [],
        )

    context = "\n\n---\n\n".join(
        f"[Chunk {i + 1} — {c.get('title', '')}]\n{c['content']}"
        for i, c in enumerate(chunks)
    ) or "(no source chunks were retrieved)"

    judge_llm = make_chat_model(eval_model, temperature=0)
    chain = _ANSWER_RELEVANCE_PROMPT | judge_llm.with_structured_output(
        _AnswerRelevanceVerdict, **structured_output_kwargs(eval_model), include_raw=True
    )
    raw = await chain.ainvoke({"question": question, "answer": answer, "context": context})
    assert isinstance(raw, dict)
    parsed: _AnswerRelevanceVerdict = raw["parsed"]
    usage = usage_from_message(raw["raw"], "rag_answer_relevance", eval_model)
    verdict = RagAnswerRelevanceVerdict(score=parsed.score, reasoning=parsed.reasoning)
    return verdict, parsed.score / 5, [usage] if usage else []

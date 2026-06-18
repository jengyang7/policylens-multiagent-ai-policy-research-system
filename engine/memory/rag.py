"""
Layer 4: Long-term semantic memory (Phase 5 RAG)

Indexes completed research reports into pgvector and enables cross-run
question answering via a "Research Library" chatbot.

Full four-layer memory stack:
  1. Working memory  — engine/state.py ResearchState (in-run scratchpad)
  2. Compaction      — engine/nodes/compact.py (summary before synthesis)
  3. Episodic        — engine/memory/checkpointer.py (Postgres per-thread state)
  4. Long-term (here)— LlamaIndex VectorStoreIndex + PGVectorStore (pgvector)

Ingest pipeline (section-aware):
  Report body → MarkdownNodeParser (one node per ## / ### section, heading
  included in text) → SentenceSplitter (sub-chunks sections > 512 tokens,
  leaves short sections intact) → OpenAIEmbedding → PGVectorStore.

  Each node carries metadata: run_id, title, query, header_path (section
  hierarchy from MarkdownNodeParser). Metadata is excluded from the embedded
  text so only content semantics drive retrieval.

Retrieval pipeline (two-stage):
  Stage 1 — metadata filter: LLM scans all report titles/queries, selects
             run_ids that are topically relevant to the question. Cheap: only
             text, no embeddings. Eliminates cross-topic chunk contamination.
  Stage 2 — semantic search: cosine similarity scoped to selected run_ids +
             similarity_cutoff=0.4 drops off-topic chunks within a report.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import AsyncGenerator

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.postgres import PGVectorStore

from engine.models import LEAD_MODEL

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536

# Chunks below this cosine similarity are discarded even if they are the top-k.
# Prevents returning off-topic chunks when the question has no good match.
SIMILARITY_CUTOFF = 0.4

# Metadata keys stored for citation lookup — excluded from embedded text so
# only the section content drives semantic similarity.
_EXCLUDE_FROM_EMBED = ["run_id", "title", "query"]

Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)


def _vector_store() -> PGVectorStore:
    """Build a PGVectorStore using the DATABASE_URL env var."""
    # Strip query params (ssl=require etc.) — PGVectorStore handles SSL itself.
    # from_params requires both a sync (psycopg2) and async (asyncpg) URL;
    # DATABASE_URL uses +asyncpg, so derive the sync URL from it.
    async_url = os.environ["DATABASE_URL"].split("?")[0]
    sync_url = async_url.replace("+asyncpg", "+psycopg2", 1)
    return PGVectorStore.from_params(
        connection_string=sync_url,
        async_connection_string=async_url,
        table_name="report_chunks",
        embed_dim=EMBEDDING_DIMS,
        perform_setup=True,  # creates table + vector index on first use
    )


async def embed_and_store(run_id: str, content: str, title: str, query: str) -> None:
    """Section-aware ingest: chunk by markdown headings, embed, store (Layer 4).

    Pipeline:
      MarkdownNodeParser  — one node per ## / ### section (heading in text)
      SentenceSplitter    — sub-chunks sections that exceed 512 tokens
      OpenAIEmbedding     — embeds each final node
      PGVectorStore       — stores vectors with run_id/title/query metadata

    Existing reports ingested with the old sentence-only splitter remain in
    the table unchanged; only new ingests use this pipeline.
    """
    body = re.split(r"\n##\s+References\b", content, maxsplit=1)[0]
    doc = Document(
        text=body,
        doc_id=run_id,
        metadata={"run_id": run_id, "title": title, "query": query},
        excluded_embed_metadata_keys=_EXCLUDE_FROM_EMBED,
        excluded_llm_metadata_keys=_EXCLUDE_FROM_EMBED,
    )
    pipeline = IngestionPipeline(
        transformations=[
            MarkdownNodeParser(),
            SentenceSplitter(chunk_size=512, chunk_overlap=64),
            OpenAIEmbedding(model=EMBEDDING_MODEL),
        ],
        vector_store=_vector_store(),
    )
    await pipeline.arun(documents=[doc], show_progress=False)


# ---------------------------------------------------------------------------
# Chunk deletion (called on run delete)
# ---------------------------------------------------------------------------

async def delete_chunks(run_id: str) -> None:
    """Remove all stored chunks for a run from pgvector.

    Must be called when a research run is deleted so the RAG index stays
    consistent with what's visible in the sidebar.
    """
    vs = _vector_store()
    await vs.adelete(ref_doc_id=run_id)


# ---------------------------------------------------------------------------
# Stage 1 — metadata-first report selection
# ---------------------------------------------------------------------------

async def select_relevant_reports(
    question: str,
    available: list[dict[str, str]],
) -> list[str]:
    """Use the LLM to shortlist run_ids whose reports likely answer the question.

    Reads only titles and original queries — no embeddings, no chunk retrieval.
    Returns a (possibly empty) list of run_ids to scope Stage 2 search to.
    """
    if not available:
        return []

    report_list = "\n".join(
        f"- run_id={r['run_id']} | title: {r['title']} | original query: {r['query']}"
        for r in available
    )
    prompt = (
        "You are a relevance filter for a research library.\n"
        "Given a user question and a list of research reports (each with a run_id, "
        "title, and the original research query), return the run_ids of reports that "
        "are topically relevant to the user's question.\n\n"
        "Return ONLY a valid JSON array of run_id strings, e.g. [\"abc\", \"def\"]. "
        "Return [] if none are relevant. Do not explain or add any other text.\n\n"
        f"User question: {question}\n\n"
        f"Available reports:\n{report_list}"
    )
    llm = OpenAI(model=LEAD_MODEL, temperature=0)
    response = await llm.acomplete(prompt)
    text = response.text.strip()

    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if not match:
        return []
    try:
        ids: list[str] = json.loads(match.group())
        valid_ids = {r["run_id"] for r in available}
        return [i for i in ids if i in valid_ids]
    except (json.JSONDecodeError, TypeError, KeyError):
        return []


# ---------------------------------------------------------------------------
# Stage 2 — scoped semantic search with similarity cutoff
# ---------------------------------------------------------------------------

async def search_in_reports(
    question: str,
    run_ids: list[str],
    limit: int = 5,
) -> list[dict[str, object]]:
    """Vector search scoped to specific run_ids with a similarity cutoff.

    The cutoff drops chunks whose cosine similarity falls below SIMILARITY_CUTOFF
    even when they are the closest match available — prevents low-relevance chunks
    from reaching the LLM.
    """
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="run_id", value=rid, operator=FilterOperator.EQ)
            for rid in run_ids
        ],
        condition=FilterCondition.OR,
    )
    index = VectorStoreIndex.from_vector_store(_vector_store())
    retriever = index.as_retriever(
        similarity_top_k=limit,
        similarity_cutoff=SIMILARITY_CUTOFF,
        filters=filters,
    )
    nodes = await retriever.aretrieve(question)
    return [
        {
            "content": node.get_content(),
            "run_id": node.metadata.get("run_id", ""),
            "title": node.metadata.get("title", ""),
            "query": node.metadata.get("query", ""),
        }
        for node in nodes
    ]


# ---------------------------------------------------------------------------
# Two-stage orchestrator (called from the API)
# ---------------------------------------------------------------------------

async def two_stage_search(
    question: str,
    available_reports: list[dict[str, str]],
    limit: int = 5,
) -> tuple[list[str], list[dict[str, object]]]:
    """Metadata-first retrieval: select relevant reports, then retrieve chunks.

    Returns (selected_run_ids, chunks).
    selected_run_ids is empty when Stage 1 finds no relevant reports (no Stage 2
    call is made). chunks is empty when Stage 2 finds nothing above the cutoff.
    """
    # Stage 1: LLM shortlists reports by title/query metadata
    selected_ids = await select_relevant_reports(question, available_reports)
    if not selected_ids:
        return [], []

    # Stage 2: semantic search within selected reports + similarity cutoff
    chunks = await search_in_reports(question, selected_ids, limit)
    return selected_ids, chunks


# ---------------------------------------------------------------------------
# Fallback: unscoped search (kept for internal/debug use)
# ---------------------------------------------------------------------------

async def search(question: str, limit: int = 5) -> list[dict[str, object]]:
    """Unscoped top-k retrieval with similarity cutoff (no metadata pre-filter).

    Prefer two_stage_search() when the caller has the list of available reports.
    """
    index = VectorStoreIndex.from_vector_store(_vector_store())
    retriever = index.as_retriever(
        similarity_top_k=limit,
        similarity_cutoff=SIMILARITY_CUTOFF,
    )
    nodes = await retriever.aretrieve(question)
    return [
        {
            "content": node.get_content(),
            "run_id": node.metadata.get("run_id", ""),
            "title": node.metadata.get("title", ""),
            "query": node.metadata.get("query", ""),
        }
        for node in nodes
    ]


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

async def answer_with_context(
    question: str,
    history: list[dict[str, str]],
    chunks: list[dict[str, object]],
) -> AsyncGenerator[str, None]:
    """Stream an LLM answer grounded in the retrieved report chunks."""
    context = "\n\n---\n\n".join(
        f"[Source: {c['title']}]\n{c['content']}" for c in chunks
    )
    llm = OpenAI(model=LEAD_MODEL, temperature=0)
    messages: list[ChatMessage] = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "You are a concise research assistant with access to the user's past research reports.\n\n"
                "Rules:\n"
                "- Answer using ONLY the provided context — never invent facts.\n"
                "- Be brief: 2–4 sentences for simple questions, one short paragraph for complex ones.\n"
                "  Only use a table when the question explicitly asks for comparison or structure.\n"
                "- **Bold** key terms, conclusions, and important caveats inline.\n"
                "- Cite by report title inline when relevant: 'According to the report on X…'.\n"
                "- Do NOT offer to reformat, summarize differently, or convert your answer.\n"
                "- If the context is insufficient, say so in one sentence.\n\n"
                f"Context from past reports:\n{context}"
            ),
        )
    ]
    for m in history[-6:]:
        role = MessageRole.USER if m.get("role") == "user" else MessageRole.ASSISTANT
        messages.append(ChatMessage(role=role, content=m.get("content", "")))
    messages.append(ChatMessage(role=MessageRole.USER, content=question))

    async for response in await llm.astream_chat(messages):
        if response.delta:
            yield response.delta

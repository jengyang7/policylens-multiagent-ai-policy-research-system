"""
Layer 4: Long-term semantic memory (Phase 5 RAG)

Indexes completed research reports into pgvector and enables cross-run
question answering via a "Research Library" chatbot.

Full four-layer memory stack:
  1. Working memory  — engine/state.py ResearchState (in-run scratchpad)
  2. Compaction      — engine/nodes/compact.py (summary before synthesis)
  3. Episodic        — engine/memory/checkpointer.py (Postgres per-thread state)
  4. Long-term (here)— LlamaIndex VectorStoreIndex + PGVectorStore (pgvector)

Full LlamaIndex pipeline:
  - SentenceSplitter: sentence-aware chunking (better boundary awareness than
    simple paragraph splitting)
  - OpenAIEmbedding: embeds chunks + queries with text-embedding-3-small
  - PGVectorStore: stores/retrieves vectors; owns the report_chunks table
  - VectorStoreIndex: ties chunking → embedding → storage into one pipeline
  - OpenAI LLM: streams the grounded answer back to the caller

run_id, title, and query are stored as node metadata (JSONB) inside the
PGVectorStore table so source citations can be surfaced in the UI.
"""
from __future__ import annotations

import os
import re
from collections.abc import AsyncGenerator

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.postgres import PGVectorStore

from engine.models import LEAD_MODEL

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536

# Shared LlamaIndex settings — applied globally to all index/query operations
Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)


def _vector_store() -> PGVectorStore:
    """Build a PGVectorStore using the DATABASE_URL env var (asyncpg driver)."""
    # Strip asyncpg-only query params (ssl=require etc.) — PGVectorStore handles
    # SSL via its own connection args
    url = os.environ["DATABASE_URL"].split("?")[0]
    return PGVectorStore.from_params(
        async_connection_string=url,
        table_name="report_chunks",
        embed_dim=EMBEDDING_DIMS,
        perform_setup=True,  # creates table + vector index on first use
    )


async def embed_and_store(run_id: str, content: str, title: str, query: str) -> None:
    """Chunk, embed, and index a completed report (Layer 4 ingest).

    run_id/title/query are stored as node metadata so the UI can surface source
    citations when the library chatbot retrieves a chunk from this report.
    """
    body = re.split(r"\n##\s+References\b", content, maxsplit=1)[0]
    doc = Document(
        text=body,
        doc_id=run_id,
        metadata={"run_id": run_id, "title": title, "query": query},
        # Keep metadata out of the embedded text — it's lookup data, not content
        excluded_embed_metadata_keys=["run_id", "title", "query"],
        excluded_llm_metadata_keys=["run_id", "title", "query"],
    )
    vs = _vector_store()
    storage_ctx = StorageContext.from_defaults(vector_store=vs)
    # Use an empty in-memory index as entry point then insert asynchronously
    index = VectorStoreIndex([], storage_context=storage_ctx)
    await index.ainsert(doc)


async def search(question: str, limit: int = 5) -> list[dict[str, object]]:
    """Retrieve top-k relevant report chunks via cosine similarity."""
    index = VectorStoreIndex.from_vector_store(_vector_store())
    retriever = index.as_retriever(similarity_top_k=limit)
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
                "You are a research assistant with access to the user's past research reports. "
                "Answer the question using only the provided context excerpts. "
                "Reference reports by title when citing information "
                "(e.g. 'According to the report on X...'). "
                "If the context is insufficient, say so honestly.\n\n"
                f"Context from past reports:\n{context}"
            ),
        )
    ]
    for m in history[-6:]:
        role = MessageRole.USER if m.get("role") == "user" else MessageRole.ASSISTANT
        messages.append(ChatMessage(role=role, content=m.get("content", "")))
    messages.append(ChatMessage(role=MessageRole.USER, content=question))

    async for response in llm.astream_chat(messages):
        if response.delta:
            yield response.delta

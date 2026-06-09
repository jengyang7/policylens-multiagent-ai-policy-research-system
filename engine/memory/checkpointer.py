"""SHORT-TERM / EPISODIC MEMORY (layer 3 of the memory stack):
LangGraph Postgres checkpointer — persists graph state per thread_id.
Enables: resumable runs, human-in-the-loop pause/resume, and multi-turn
follow-up chat (all read from the same persisted state snapshot).
"""
from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


def _conn_string() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    # asyncpg uses postgresql+asyncpg:// + ssl=require
    # psycopg (checkpointer) needs postgresql:// + sslmode=require
    return (
        url.replace("postgresql+asyncpg://", "postgresql://")
           .replace("ssl=require", "sslmode=require")
    )


@asynccontextmanager
async def get_checkpointer() -> AsyncIterator[AsyncPostgresSaver]:
    """Async context manager that yields a ready AsyncPostgresSaver."""
    async with AsyncPostgresSaver.from_conn_string(_conn_string()) as checkpointer:
        await checkpointer.setup()
        yield checkpointer

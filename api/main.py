"""FastAPI server: POST /research (SSE), POST /runs/{id}/resume, GET /runs/{id}, POST /chat."""
from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sse_starlette.sse import EventSourceResponse

from db.models import ResearchRun
from engine.memory.checkpointer import get_checkpointer
from engine.models import LEAD_MODEL, LEAD_MODEL_OPTIONS, SUBAGENT_MODEL, estimate_cost_usd
from engine.nodes.chat import answer_followup
from engine.orchestrator import build_graph
from engine.state import TokenUsage

load_dotenv()


def _async_engine_from_url(raw: str):
    """Build an asyncpg-compatible engine, stripping params asyncpg doesn't accept."""
    parsed = urlparse(raw)
    params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
    # Collect SSL intent before stripping, then pass via connect_args
    ssl_val = params.pop("ssl", None) or ("require" if params.pop("sslmode", None) else None)
    for unsupported in ("channel_binding", "options"):
        params.pop(unsupported, None)
    clean_url = urlunparse(parsed._replace(query=urlencode(params)))
    kwargs = {"connect_args": {"ssl": ssl_val}} if ssl_val else {}
    return create_async_engine(clean_url, **kwargs)


# ---------------------------------------------------------------------------
# App-lifetime globals (set in lifespan, used by route handlers)
# ---------------------------------------------------------------------------
_session_factory: async_sessionmaker[AsyncSession] | None = None
_graph: CompiledStateGraph | None = None  # type: ignore[type-arg]
_checkpointer = None


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    global _session_factory, _graph, _checkpointer

    engine = _async_engine_from_url(os.environ["DATABASE_URL"])
    _session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with get_checkpointer() as cp:
        _checkpointer = cp
        _graph = build_graph(cp)
        yield

    await engine.dispose()
    _session_factory = None
    _graph = None
    _checkpointer = None


app = FastAPI(title="Deep Research API", lifespan=lifespan)

_allowed_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Return JSON errors with CORS headers so the browser can read the body."""
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
        headers={"Access-Control-Allow-Origin": _allowed_origins[0] if _allowed_origins else "*"},
    )


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ResearchRequest(BaseModel):
    query: str
    model: str | None = None


class ResumeRequest(BaseModel):
    answers: list[str]


class ChatRequest(BaseModel):
    thread_id: str
    question: str
    history: list[dict[str, str]] = []


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

async def _update_run(run_id: str, status: str, **kwargs: object) -> None:
    assert _session_factory is not None
    async with _session_factory() as session:
        result = await session.execute(select(ResearchRun).where(ResearchRun.id == run_id))
        run = result.scalar_one_or_none()
        if run:
            run.status = status
            for k, v in kwargs.items():
                setattr(run, k, v)
            await session.commit()


# ---------------------------------------------------------------------------
# Core SSE stream generator (shared by /research and /runs/{id}/resume)
# ---------------------------------------------------------------------------

def _evt(data: dict) -> dict[str, str]:
    return {"data": json.dumps({**data, "ts": time.time()})}


async def _stream_graph(
    run_id: str,
    input_: dict | Command,  # type: ignore[type-arg]
) -> AsyncGenerator[dict[str, str], None]:
    """Stream LangGraph node updates as SSE events.

    Events emitted (data field is JSON):
      started             {type, run_id}
      plan                {type, subtasks: [...]}
      subtask_done        {type, question, findings_count}
      report              {type, content, run_id}
      clarification_needed {type, run_id, questions: [...]}
      done                {type, run_id, usage: {...}}
      error               {type, message}
    """
    assert _graph is not None
    config = {"configurable": {"thread_id": run_id}}
    start_time = time.time()

    yield _evt({"type": "started", "run_id": run_id})

    try:
        await _update_run(run_id, "running")

        async for chunk in _graph.astream(input_, config, stream_mode="updates"):  # type: ignore[misc]
            for node_name, node_output in chunk.items():

                if node_name == "plan":
                    subtasks: list[str] = node_output.get("subtasks", [])  # type: ignore[union-attr]
                    thinking: str = node_output.get("supervisor_thinking", "")  # type: ignore[union-attr]
                    await _update_run(run_id, "running", plan={"subtasks": subtasks})
                    if thinking:
                        yield _evt({"type": "plan_thinking", "content": thinking})
                    yield _evt({"type": "plan", "subtasks": subtasks})

                elif node_name == "subagent":
                    findings: list[dict] = node_output.get("findings", [])  # type: ignore[union-attr]
                    processed: list[str] = node_output.get("processed_subtasks", [])
                    fallback = findings[0]["subtask"] if findings else ""
                    question = processed[0] if processed else fallback
                    sources = list({
                        f["citation_url"] for f in findings if f.get("citation_url")
                    })
                    yield _evt({
                        "type": "subtask_done",
                        "question": question,
                        "findings_count": len(findings),
                        "sources": sources,
                    })

                elif node_name == "compact":
                    yield _evt({"type": "synthesizing"})

                elif node_name == "synthesize":
                    report: str = node_output.get("report", "")  # type: ignore[union-attr]
                    yield _evt({"type": "report", "content": report, "run_id": run_id})

        # After the loop: detect interrupt (clarification needed) vs normal completion.
        snapshot = await _graph.aget_state(config)

        if snapshot.next:
            # Graph is paused — read clarification questions + chip options from state.
            questions: list[str] = snapshot.values.get("clarification_questions", [])  # type: ignore[union-attr]
            options: list[list[str]] = snapshot.values.get("clarification_options", [])  # type: ignore[union-attr]
            if questions:
                await _update_run(
                    run_id,
                    "awaiting_clarification",
                    clarifications={"questions": questions, "answers": []},
                )
                yield _evt({
                    "type": "clarification_needed",
                    "run_id": run_id,
                    "questions": questions,
                    "options": options,
                })
                return

        # Graph completed normally — summarize token usage/cost/time for the UI.
        token_usage: list[TokenUsage] = snapshot.values.get("token_usage", [])
        lead_model = snapshot.values.get("lead_model", LEAD_MODEL)
        usage = {
            "lead_model": lead_model,
            "subagent_model": SUBAGENT_MODEL,
            "input_tokens": sum(u["input_tokens"] for u in token_usage),
            "output_tokens": sum(u["output_tokens"] for u in token_usage),
            "cached_tokens": sum(u["cached_tokens"] for u in token_usage),
            "cost_usd": round(estimate_cost_usd(token_usage), 4),
            "elapsed_seconds": round(time.time() - start_time, 1),
        }
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]

        await _update_run(run_id, "done", finished_at=datetime.now(timezone.utc), stats=usage)
        yield _evt({"type": "done", "run_id": run_id, "usage": usage})

    except Exception as exc:
        await _update_run(run_id, "failed")
        yield _evt({"type": "error", "message": str(exc)})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/models")
async def list_models() -> dict[str, object]:
    """Lead models selectable on the New Research page, plus the default."""
    return {
        "default": LEAD_MODEL,
        "options": [
            {"id": model_id, **meta} for model_id, meta in LEAD_MODEL_OPTIONS.items()
        ],
    }


@app.post("/research")
async def start_research(body: ResearchRequest) -> EventSourceResponse:
    assert _session_factory is not None

    if body.model is not None and body.model not in LEAD_MODEL_OPTIONS:
        raise HTTPException(400, f"Unknown model: {body.model}")
    lead_model = body.model or LEAD_MODEL

    run_id = str(uuid.uuid4())
    async with _session_factory() as session:
        session.add(ResearchRun(id=run_id, query=body.query, status="pending"))
        await session.commit()

    initial_state: dict[str, object] = {
        "run_id": run_id,
        "query": body.query,
        "lead_model": lead_model,
        "clarification_questions": [],
        "clarification_options": [],
        "clarifications": [],
        "supervisor_thinking": "",
        "subtasks": [],
        "findings": [],
        "summary": "",
        "report": "",
        "messages": [],
        "token_usage": [],
        "processed_subtasks": [],
    }
    return EventSourceResponse(_stream_graph(run_id, initial_state))


@app.post("/runs/{run_id}/resume")
async def resume_run(run_id: str, body: ResumeRequest) -> EventSourceResponse:
    assert _session_factory is not None
    assert _graph is not None

    async with _session_factory() as session:
        result = await session.execute(select(ResearchRun).where(ResearchRun.id == run_id))
        run = result.scalar_one_or_none()
        if not run:
            raise HTTPException(404, "Run not found")
        if run.status != "awaiting_clarification":
            raise HTTPException(400, f"Run is not awaiting clarification (status={run.status})")

    return EventSourceResponse(_stream_graph(run_id, Command(resume=body.answers)))


@app.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, object]:
    assert _session_factory is not None
    assert _graph is not None

    async with _session_factory() as session:
        result = await session.execute(select(ResearchRun).where(ResearchRun.id == run_id))
        run = result.scalar_one_or_none()
        if not run:
            raise HTTPException(404, "Run not found")

    config = {"configurable": {"thread_id": run_id}}
    snapshot = await _graph.aget_state(config)
    report = ""
    findings: list[object] = []
    if snapshot:
        report = snapshot.values.get("report", "")  # type: ignore[union-attr]
        findings = snapshot.values.get("findings", [])  # type: ignore[union-attr]

    return {
        "id": run.id,
        "query": run.query,
        "status": run.status,
        "plan": run.plan,
        "clarifications": run.clarifications,
        "report": report,
        "findings": findings,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
    }


@app.post("/chat")
async def chat(body: ChatRequest) -> EventSourceResponse:
    assert _checkpointer is not None

    async def stream() -> AsyncGenerator[dict[str, str], None]:
        try:
            async for chunk in answer_followup(
                body.thread_id, body.question, body.history, _checkpointer
            ):
                yield _evt({"type": "chunk", "content": chunk})
            yield _evt({"type": "done"})
        except Exception as exc:
            yield _evt({"type": "error", "message": str(exc)})

    return EventSourceResponse(stream())

"""FastAPI server: POST /research (SSE), POST /runs/{id}/resume, GET /runs/{id}, POST /chat."""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sse_starlette.sse import EventSourceResponse

from db.models import EvalReportRecord, Report, ResearchRun
from engine.memory import rag
from engine.memory.checkpointer import get_checkpointer
from engine.models import (
    LEAD_MODEL,
    SUBAGENT_MODEL,
    available_model_options,
    estimate_cost_usd,
    role_default_models,
)
from engine.nodes.chat import answer_followup
from engine.orchestrator import DEFAULT_DEBATE_ROUNDS, DEFAULT_RUN_MODE, build_graph
from engine.state import TokenUsage
from eval.harness import evaluate_run
from eval.rag_eval import evaluate_rag

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _async_engine_from_url(raw: str):
    """Build an asyncpg-compatible engine, stripping params asyncpg doesn't accept."""
    parsed = urlparse(raw)
    params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
    # Collect SSL intent before stripping, then pass via connect_args
    ssl_val = params.pop("ssl", None) or ("require" if params.pop("sslmode", None) else None)
    for unsupported in ("channel_binding", "options"):
        params.pop(unsupported, None)
    clean_url = urlunparse(parsed._replace(query=urlencode(params)))
    kwargs: dict[str, object] = {"connect_args": {"ssl": ssl_val}} if ssl_val else {}
    # Long-running eval calls can outlive a pooled connection's idle timeout on
    # managed Postgres (Render/Neon/Supabase). pre_ping detects + replaces dead
    # connections before use; recycle proactively retires connections older than 5m.
    return create_async_engine(clean_url, pool_pre_ping=True, pool_recycle=300, **kwargs)


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


app = FastAPI(title="AI Policy Researcher API", lifespan=lifespan)

_allowed_origins = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]

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
    mode: Literal[
        "single_agent",
        "multi_agent_no_compaction",
        "multi_agent_compaction",
        "multi_agent_verified",
    ] = DEFAULT_RUN_MODE
    # Debate mode: two cross-provider agents argue over the findings pre-synthesis
    debate: bool = False
    advocate_model: str | None = None
    skeptic_model: str | None = None


class ResumeRequest(BaseModel):
    answers: list[str]


class ChatRequest(BaseModel):
    thread_id: str
    question: str
    history: list[dict[str, str]] = []


class LibraryChatRequest(BaseModel):
    question: str
    history: list[dict[str, str]] = []


class RagEvalRequest(BaseModel):
    question: str
    eval_model: str = "claude-haiku-4-5"


class RagJudgeRequest(BaseModel):
    question: str
    answer: str
    chunks: list[dict[str, object]]
    eval_model: str = "claude-haiku-4-5"


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


def _client_id(request: Request) -> str | None:
    """Anonymous browser-generated id (X-Client-Id header) scoping per-visitor listings."""
    return request.headers.get("x-client-id")


def _eval_record_summary(record: EvalReportRecord) -> dict[str, object]:
    return {
        "id": record.id,
        "run_id": record.run_id,
        "query": record.query,
        "generated_at": record.generated_at.isoformat() if record.generated_at else None,
        "passed": record.passed,
        "total_findings": record.total_findings,
        "ungrounded_count": record.ungrounded_count,
        "total_citations": record.total_citations,
        "unfaithful_count": record.unfaithful_count,
        "uncited_count": record.uncited_count,
        "failure_reasons": record.failure_reasons or [],
        "eval_model": record.eval_model,
        "eval_cost_usd": record.eval_cost_usd,
        "recall_score": record.recall_score,
        "relevance_score": record.relevance_score,
    }


# ---------------------------------------------------------------------------
# Report persistence + RAG ingest (Layer 4: long-term memory)
# ---------------------------------------------------------------------------

async def _save_and_embed_report(
    run_id: str,
    content: str,
    findings: list[dict[str, object]] | None = None,
) -> None:
    """Persist report to DB and index via LlamaIndex for RAG (Layer 4 long-term memory)."""
    assert _session_factory is not None
    structured = {"library_findings": findings or []}
    async with _session_factory() as session:
        stmt = (
            pg_insert(Report)
            .values(
                id=str(uuid.uuid4()),
                run_id=run_id,
                content=content,
                structured=structured,
            )
            .on_conflict_do_update(
                index_elements=["run_id"],
                set_={"content": content, "structured": structured},
            )
        )
        await session.execute(stmt)
        await session.commit()
        # Fetch run metadata so LlamaIndex can store it as node metadata for citations
        result = await session.execute(select(ResearchRun).where(ResearchRun.id == run_id))
        run = result.scalar_one_or_none()

    title = (run.title or run.query) if run else run_id
    query = run.query if run else ""
    await rag.embed_and_store(run_id, content, title, query, findings=findings)


# ---------------------------------------------------------------------------
# Core SSE stream generator (shared by /research and /runs/{id}/resume)
# ---------------------------------------------------------------------------

def _evt(data: dict[str, object]) -> dict[str, str]:
    return {"data": json.dumps({**data, "ts": time.time()})}


async def _stream_graph(
    run_id: str,
    input_: dict[str, object] | Command,
    debate_mode: bool = False,
    debate_rounds: int = DEFAULT_DEBATE_ROUNDS,
    run_mode: str = DEFAULT_RUN_MODE,
) -> AsyncGenerator[dict[str, str], None]:
    """Stream LangGraph node updates as SSE events.

    Events emitted (data field is JSON):
      started             {type, run_id}
      plan                {type, subtasks: [...]}
      subtask_done        {type, question, findings_count, stage: "plan"|"gap"}
      debating            {type}                      (debate mode only)
      debate_token        {type, agent, content}      (debate mode only, per LLM token)
      debate_turn         {type, agent, model, round, content}  (debate mode only)
      judging             {type}                      (debate mode only)
      debate_verdict      {type, winner, rows, model}  (debate mode only)
      evidence_auditing   {type}                      (verified/debate modes)
      evidence_audit      {type, sufficient, assessment, subtasks}
      synthesizing        {type}
      report              {type, content, run_id}
      clarification_needed {type, run_id, questions: [...]}
      done                {type, run_id, usage: {...}}
      error               {type, message}
    """
    assert _graph is not None
    config = {"configurable": {"thread_id": run_id}}
    start_time = time.time()
    findings_for_index: list[dict[str, object]] = []
    compaction_enabled = debate_mode or run_mode in {
        "multi_agent_compaction",
        "multi_agent_verified",
    }

    yield _evt({"type": "started", "run_id": run_id})

    try:
        await _update_run(run_id, "running")

        # "messages" mode taps LLM token callbacks inside nodes — only needed to
        # live-stream debate turns, so plain runs keep the cheaper updates-only stream.
        stream_modes = ["updates", "messages"] if debate_mode else ["updates"]

        async for mode, chunk in _graph.astream(input_, config, stream_mode=stream_modes):  # type: ignore[misc]
            if mode == "messages":
                msg_chunk, meta = chunk
                node = meta.get("langgraph_node", "")
                # .text, not .content: Gemini chunks carry content-block lists
                if node in ("debate_advocate", "debate_skeptic") and msg_chunk.text:
                    yield _evt({
                        "type": "debate_token",
                        "agent": "advocate" if node == "debate_advocate" else "skeptic",
                        "content": msg_chunk.text,
                    })
                continue

            for node_name, node_output in chunk.items():

                if node_name == "plan":
                    subtasks: list[str] = node_output.get("subtasks", [])  # type: ignore[union-attr]
                    thinking: str = node_output.get("supervisor_thinking", "")  # type: ignore[union-attr]
                    title: str = node_output.get("title", "")
                    await _update_run(
                        run_id, "running", plan={"subtasks": subtasks}, title=title or None
                    )
                    if thinking:
                        yield _evt({"type": "plan_thinking", "content": thinking})
                    yield _evt({"type": "plan", "subtasks": subtasks, "title": title})

                elif node_name in ("single_agent", "subagent", "gap_subagent"):
                    findings: list[dict[str, object]] = node_output.get("findings", [])  # type: ignore[union-attr]
                    findings_for_index.extend(findings)
                    processed: list[str] = node_output.get("processed_subtasks", [])
                    fallback = findings[0]["subtask"] if findings else ""
                    question = processed[0] if processed else fallback
                    if node_name == "single_agent":
                        plan_question = question or (
                            input_.get("query", "") if isinstance(input_, dict) else ""
                        )
                        question = str(plan_question)
                        await _update_run(
                            run_id, "running", plan={"subtasks": [question]}
                        )
                        yield _evt({
                            "type": "plan",
                            "subtasks": [question],
                            "title": "",
                        })
                    sources = list({
                        f["citation_url"] for f in findings if f.get("citation_url")
                    })
                    yield _evt({
                        "type": "subtask_done",
                        "question": question,
                        "findings_count": len(findings),
                        "sources": sources,
                        # "gap" = second, evidence-audit research round
                        "stage": "gap" if node_name == "gap_subagent" else "plan",
                        "compaction": compaction_enabled or node_name == "gap_subagent",
                    })

                elif node_name == "compact":
                    if debate_mode:
                        yield _evt({"type": "debating"})
                    elif run_mode == "multi_agent_verified":
                        yield _evt({"type": "evidence_auditing"})
                    else:
                        yield _evt({"type": "synthesizing"})

                elif node_name == "recompact":
                    # Gap findings folded back into the summary — synthesis is next
                    yield _evt({"type": "synthesizing"})

                elif node_name == "evidence_audit":
                    audit: dict[str, object] = node_output.get("evidence_audit", {})  # type: ignore[union-attr]
                    gaps: list[str] = node_output.get("gap_subtasks", [])  # type: ignore[union-attr]
                    yield _evt({
                        "type": "evidence_audit",
                        "sufficient": bool(audit.get("sufficient", not gaps)),
                        "assessment": str(audit.get("assessment", "")),
                        "subtasks": gaps,
                    })
                    if not gaps:
                        yield _evt({"type": "synthesizing"})

                elif node_name in ("debate_advocate", "debate_skeptic"):
                    turns: list[dict] = node_output.get("debate_turns", [])  # type: ignore[union-attr]
                    if turns:
                        turn = turns[0]
                        yield _evt({
                            "type": "debate_turn",
                            "agent": turn["agent"],
                            "model": turn["model"],
                            "round": turn["round"],
                            "content": turn["content"],
                        })
                        # The skeptic's final-round turn ends the debate loop;
                        # next the neutral lead judges the debate
                        if node_name == "debate_skeptic" and turn["round"] >= debate_rounds:
                            yield _evt({"type": "judging"})

                elif node_name == "judge_debate":
                    verdict: dict | None = node_output.get("debate_verdict")  # type: ignore[union-attr]
                    if verdict:
                        yield _evt({"type": "debate_verdict", **verdict})
                    # Debate is optional; every debate run now rejoins the same
                    # evidence-quality gate used by normal verified research.
                    yield _evt({"type": "evidence_auditing"})

                elif node_name == "synthesize":
                    if run_mode != "multi_agent_verified" and not debate_mode:
                        report: str = node_output.get("report", "")  # type: ignore[union-attr]
                        if report:
                            asyncio.create_task(
                                _save_and_embed_report(run_id, report, list(findings_for_index))
                            )
                        yield _evt({"type": "report", "content": report, "run_id": run_id})

                elif node_name == "verify_citations":
                    report: str = node_output.get("report", "")  # type: ignore[union-attr]
                    if report:
                        # Persist to DB + embed in background — don't block the SSE stream
                        asyncio.create_task(
                            _save_and_embed_report(run_id, report, list(findings_for_index))
                        )
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
            "mode": "debate_gap" if snapshot.values.get("debate_mode") else run_mode,
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
    """Models selectable in the UI (filtered by available provider API keys) + per-role defaults."""
    defaults = role_default_models()
    return {
        "default": defaults["lead"],  # backward compat with the lead picker
        "defaults": defaults,
        "options": [
            {"id": model_id, **meta} for model_id, meta in available_model_options().items()
        ],
    }


@app.post("/research")
async def start_research(body: ResearchRequest, request: Request) -> EventSourceResponse:
    assert _session_factory is not None

    available = available_model_options()
    for field, value in (
        ("model", body.model),
        ("advocate_model", body.advocate_model),
        ("skeptic_model", body.skeptic_model),
    ):
        if value is not None and value not in available:
            raise HTTPException(400, f"Unknown or unavailable {field}: {value}")
    defaults = role_default_models()
    lead_model = body.model or defaults["lead"]
    advocate_model = body.advocate_model or defaults["advocate"]
    skeptic_model = body.skeptic_model or defaults["skeptic"]
    run_mode = "multi_agent_verified" if body.debate else body.mode

    run_id = str(uuid.uuid4())
    async with _session_factory() as session:
        session.add(ResearchRun(
            id=run_id, query=body.query, status="pending", client_id=_client_id(request)
        ))
        await session.commit()

    initial_state: dict[str, object] = {
        "run_id": run_id,
        "query": body.query,
        "lead_model": lead_model,
        "run_mode": run_mode,
        "clarification_questions": [],
        "clarification_options": [],
        "clarifications": [],
        "supervisor_thinking": "",
        "title": "",
        "subtasks": [],
        "findings": [],
        "summary": "",
        "report": "",
        "messages": [],
        "token_usage": [],
        "processed_subtasks": [],
        # Debate mode (explicit init matters — debate_turns is a reducer channel)
        "debate_mode": body.debate,
        "debate_rounds": DEFAULT_DEBATE_ROUNDS,
        "advocate_model": advocate_model,
        "skeptic_model": skeptic_model,
        "debate_turns": [],
    }
    return EventSourceResponse(
        _stream_graph(
            run_id,
            initial_state,
            debate_mode=body.debate,
            run_mode=run_mode,
        )
    )


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

    # Read debate config from the checkpoint, not the request — a debate run
    # that paused for clarification must keep streaming debate events on resume.
    snapshot = await _graph.aget_state({"configurable": {"thread_id": run_id}})
    debate_mode = bool(snapshot.values.get("debate_mode", False)) if snapshot else False
    debate_rounds = (
        int(snapshot.values.get("debate_rounds", DEFAULT_DEBATE_ROUNDS))
        if snapshot else DEFAULT_DEBATE_ROUNDS
    )
    run_mode = (
        str(snapshot.values.get("run_mode", DEFAULT_RUN_MODE))
        if snapshot else DEFAULT_RUN_MODE
    )

    return EventSourceResponse(
        _stream_graph(
            run_id,
            Command(resume=body.answers),
            debate_mode=debate_mode,
            debate_rounds=debate_rounds,
            run_mode=run_mode,
        )
    )


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
    debate_turns: list[object] = []
    debate_verdict: object = None
    run_mode = DEFAULT_RUN_MODE
    if snapshot:
        report = snapshot.values.get("report", "")  # type: ignore[union-attr]
        findings = snapshot.values.get("findings", [])  # type: ignore[union-attr]
        debate_turns = snapshot.values.get("debate_turns", [])  # type: ignore[union-attr]
        debate_verdict = snapshot.values.get("debate_verdict")  # type: ignore[union-attr]
        run_mode = snapshot.values.get("run_mode", DEFAULT_RUN_MODE)  # type: ignore[assignment,union-attr]

    return {
        "id": run.id,
        "query": run.query,
        "title": run.title,
        "status": run.status,
        "plan": run.plan,
        "mode": run_mode,
        "clarifications": run.clarifications,
        "report": report,
        "findings": findings,
        "debate_turns": debate_turns,
        "debate_verdict": debate_verdict,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
    }


@app.get("/runs")
async def list_runs(
    request: Request, status: str | None = None, limit: int = 50, mine: bool = False
) -> list[dict[str, object]]:
    """List research runs, most recent first.

    Public by default for the sidebar/community demo. `mine=true` scopes the
    list to the anonymous visitor id for private eval/benchmark views.
    """
    assert _session_factory is not None

    stmt = (
        select(ResearchRun)
        .order_by(ResearchRun.started_at.desc())
        .limit(limit)
    )
    if status is not None:
        stmt = stmt.where(ResearchRun.status == status)
    if mine:
        client_id = _client_id(request)
        if client_id is None:
            return []
        stmt = stmt.where(ResearchRun.client_id == client_id)

    async with _session_factory() as session:
        result = await session.execute(stmt)
        runs = result.scalars().all()

    return [
        {
            "id": r.id,
            "query": r.query,
            "title": r.title,
            "status": r.status,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "finished_at": r.finished_at.isoformat() if r.finished_at else None,
            "stats": r.stats,
        }
        for r in runs
    ]


@app.delete("/runs/{run_id}")
async def delete_run(run_id: str, request: Request) -> dict[str, object]:
    """Delete a research run, its eval reports (cascade), and checkpoint state.

    Scoped to the calling visitor (X-Client-Id header) — 404 if the run
    belongs to someone else or doesn't exist. Mirrors the sidebar's "delete
    from history" action so the run also disappears from the eval dashboard.
    """
    assert _session_factory is not None
    assert _checkpointer is not None

    async with _session_factory() as session:
        result = await session.execute(select(ResearchRun).where(ResearchRun.id == run_id))
        run = result.scalar_one_or_none()
        if run is None or run.client_id != _client_id(request):
            raise HTTPException(404, "Run not found")
        await session.delete(run)
        await session.commit()

    await _checkpointer.adelete_thread(run_id)
    await rag.delete_chunks(run_id)

    return {"deleted": run_id}


@app.post("/runs/{run_id}/eval")
async def run_eval(
    run_id: str, request: Request, strict: bool = True, model: str | None = None
) -> dict[str, object]:
    """Run the eval harness (eval/harness.py) against a completed run and persist the result."""
    assert _session_factory is not None

    if model is not None and model not in available_model_options():
        raise HTTPException(400, f"Unknown or unavailable model: {model}")
    eval_model = model or role_default_models()["eval"]

    async with _session_factory() as session:
        result = await session.execute(select(ResearchRun).where(ResearchRun.id == run_id))
        run = result.scalar_one_or_none()
        if not run:
            raise HTTPException(404, "Run not found")

    try:
        eval_report = await evaluate_run(run_id, lead_model=eval_model, strict=strict)
    except ValueError as exc:
        msg = str(exc)
        raise HTTPException(404 if "not found" in msg else 400, msg) from exc

    record = EvalReportRecord(
        run_id=run_id,
        client_id=run.client_id,
        query=eval_report.query,
        passed=eval_report.passed,
        total_findings=eval_report.total_findings,
        ungrounded_count=eval_report.ungrounded_count,
        total_citations=len(eval_report.faithfulness_results),
        unfaithful_count=eval_report.unfaithful_count,
        uncited_count=len(eval_report.uncited_sentences),
        failure_reasons=eval_report.failure_reasons,
        eval_model=eval_report.eval_model,
        eval_input_tokens=eval_report.eval_input_tokens,
        eval_output_tokens=eval_report.eval_output_tokens,
        eval_cost_usd=eval_report.eval_cost_usd,
        recall_score=eval_report.completeness.recall_score,
        relevance_score=eval_report.relevance.score,
        report=eval_report.model_dump(mode="json"),
    )
    async with _session_factory() as session:
        session.add(record)
        await session.commit()
        await session.refresh(record)

    return {**_eval_record_summary(record), "report": record.report}


@app.get("/eval/summary")
async def global_eval_summary() -> dict[str, object]:
    """Aggregate eval metrics across every visitor's eval reports.

    Unscoped by client_id on purpose — this powers the public "Community Average"
    card on the eval dashboard. Read-only; never exposes individual reports.
    """
    assert _session_factory is not None

    stmt = select(
        func.count(EvalReportRecord.id),
        func.count(EvalReportRecord.id).filter(EvalReportRecord.passed.is_(True)),
        func.coalesce(func.sum(EvalReportRecord.total_findings), 0),
        func.coalesce(func.sum(EvalReportRecord.ungrounded_count), 0),
        func.coalesce(func.sum(EvalReportRecord.total_citations), 0),
        func.coalesce(func.sum(EvalReportRecord.unfaithful_count), 0),
        func.coalesce(func.sum(EvalReportRecord.recall_score), 0.0),
        func.coalesce(func.sum(EvalReportRecord.relevance_score), 0),
    )
    async with _session_factory() as session:
        result = await session.execute(stmt)
        (
            runs_evaluated, passed_count, total_findings, ungrounded_count,
            total_citations, unfaithful_count, recall_sum, relevance_sum,
        ) = result.one()

    grounded = total_findings - ungrounded_count
    faithful = total_citations - unfaithful_count
    return {
        "runs_evaluated": runs_evaluated,
        "pass_rate": (passed_count / runs_evaluated * 100) if runs_evaluated else None,
        "grounding_rate": (grounded / total_findings * 100) if total_findings else None,
        "faithfulness_rate": (faithful / total_citations * 100) if total_citations else None,
        "completeness_rate": (recall_sum / runs_evaluated * 100) if runs_evaluated else None,
        "relevance_score": (relevance_sum / runs_evaluated) if runs_evaluated else None,
    }


@app.get("/eval/reports/community")
async def community_eval_trend(limit: int = 200) -> list[dict[str, object]]:
    """Time-ordered eval counts across every visitor, for the public "Quality Over
    Time" trend chart.

    Unscoped by client_id on purpose (same rationale as /eval/summary) — exposes
    only the aggregate counts needed to compute grounding/faithfulness rate per
    report, never query text or identifiers, so individual reports can't be
    drilled into from this endpoint.
    """
    assert _session_factory is not None

    stmt = (
        select(
            EvalReportRecord.generated_at,
            EvalReportRecord.total_findings,
            EvalReportRecord.ungrounded_count,
            EvalReportRecord.total_citations,
            EvalReportRecord.unfaithful_count,
        )
        .order_by(EvalReportRecord.generated_at.desc())
        .limit(limit)
    )
    async with _session_factory() as session:
        result = await session.execute(stmt)
        rows = result.all()

    return [
        {
            "generated_at": generated_at.isoformat() if generated_at else None,
            "total_findings": total_findings,
            "ungrounded_count": ungrounded_count,
            "total_citations": total_citations,
            "unfaithful_count": unfaithful_count,
        }
        for (
            generated_at,
            total_findings,
            ungrounded_count,
            total_citations,
            unfaithful_count,
        ) in reversed(rows)
    ]


@app.get("/eval/reports")
async def list_eval_reports(
    request: Request, run_id: str | None = None, limit: int = 100
) -> list[dict[str, object]]:
    """List this visitor's eval report summaries, most recent first.

    Community aggregates stay public via /eval/summary and
    /eval/reports/community; individual reports stay scoped to X-Client-Id.
    """
    assert _session_factory is not None
    client_id = _client_id(request)
    if client_id is None:
        return []

    stmt = (
        select(EvalReportRecord)
        .where(EvalReportRecord.client_id == client_id)
        .order_by(EvalReportRecord.generated_at.desc())
        .limit(limit)
    )
    if run_id is not None:
        stmt = stmt.where(EvalReportRecord.run_id == run_id)

    async with _session_factory() as session:
        result = await session.execute(stmt)
        records = result.scalars().all()

    return [_eval_record_summary(r) for r in records]


@app.get("/eval/reports/{report_id}")
async def get_eval_report(report_id: str, request: Request) -> dict[str, object]:
    """Full persisted eval report, scoped to this visitor."""
    assert _session_factory is not None
    client_id = _client_id(request)
    if client_id is None:
        raise HTTPException(404, "Eval report not found")

    async with _session_factory() as session:
        result = await session.execute(
            select(EvalReportRecord).where(
                EvalReportRecord.id == report_id,
                EvalReportRecord.client_id == client_id,
            )
        )
        record = result.scalar_one_or_none()
        if record is None:
            raise HTTPException(404, "Eval report not found")

    return {**_eval_record_summary(record), "report": record.report}


@app.delete("/eval/reports")
async def clear_eval_reports(request: Request) -> dict[str, object]:
    """Delete all eval reports belonging to the calling visitor (X-Client-Id header).

    No-op if the header is missing — this must never wipe shared/unscoped data.
    """
    assert _session_factory is not None

    client_id = _client_id(request)
    if client_id is None:
        return {"deleted": 0}

    async with _session_factory() as session:
        result = await session.execute(
            select(EvalReportRecord).where(EvalReportRecord.client_id == client_id)
        )
        records = result.scalars().all()
        for record in records:
            await session.delete(record)
        await session.commit()

    return {"deleted": len(records)}


@app.delete("/eval/reports/community")
async def clear_community_eval_reports(request: Request) -> dict[str, object]:
    """Wipe every visitor's eval reports, resetting the public "Community Average".

    Gated by the ADMIN_SECRET env var via the X-Admin-Secret header. Returns 404
    if ADMIN_SECRET is unset (endpoint disabled) or the header doesn't match —
    the same response either way so the endpoint's existence isn't revealed.
    """
    assert _session_factory is not None

    admin_secret = os.getenv("ADMIN_SECRET")
    if not admin_secret or request.headers.get("x-admin-secret") != admin_secret:
        raise HTTPException(404, "Not found")

    async with _session_factory() as session:
        result = await session.execute(select(EvalReportRecord))
        records = result.scalars().all()
        for record in records:
            await session.delete(record)
        await session.commit()

    return {"deleted": len(records)}


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


@app.post("/library/chat")
async def library_chat(body: LibraryChatRequest, request: Request) -> EventSourceResponse:
    """RAG chatbot over the calling visitor's completed research reports.

    Scoped to the X-Client-Id header — each visitor only searches their own reports.
    Two-stage retrieval:
      Stage 1 — LLM scans report titles/queries, selects topically relevant run_ids.
      Stage 2 — semantic search scoped to those run_ids with a similarity cutoff.
    """
    assert _session_factory is not None
    client_id = _client_id(request)

    async def stream() -> AsyncGenerator[dict[str, str], None]:
        try:
            # Fetch only this visitor's completed runs for Stage 1 metadata filter
            async with _session_factory() as session:
                stmt = (
                    select(ResearchRun.id, ResearchRun.title, ResearchRun.query)
                    .where(ResearchRun.status == "done")
                    .order_by(ResearchRun.started_at.desc())
                )
                if client_id is not None:
                    stmt = stmt.where(ResearchRun.client_id == client_id)
                rows = await session.execute(stmt)
                available_reports = [
                    {"run_id": str(r.id), "title": r.title or r.query, "query": r.query}
                    for r in rows
                ]

            yield _evt({"type": "searching"})
            selected_ids, chunks = await rag.two_stage_search(body.question, available_reports)

            # Emit retrieved chunks so the UI can show them in the sidebar
            yield _evt({
                "type": "chunks_retrieved",
                "chunks": [
                    {
                        "content": str(c["content"]),
                        "title": str(c["title"]),
                        "run_id": str(c["run_id"]),
                    }
                    for c in chunks
                ],
            })

            # Deduplicate sources by run_id for the citation list
            seen: set[str] = set()
            sources: list[dict[str, str]] = []
            for c in chunks:
                rid = str(c["run_id"])
                if rid not in seen:
                    seen.add(rid)
                    sources.append(
                        {"run_id": rid, "title": str(c["title"]), "query": str(c["query"])}
                    )

            yield _evt({"type": "generating"})
            async for token in rag.answer_with_context(body.question, body.history, chunks):
                yield _evt({"type": "chunk", "content": token})
            yield _evt({"type": "done", "sources": sources})
        except Exception as exc:
            yield _evt({"type": "error", "message": str(exc)})

    return EventSourceResponse(stream())


@app.post("/library/eval")
async def library_eval(body: RagEvalRequest, request: Request) -> dict[str, object]:
    """Evaluate RAG retrieval + generation quality for one question.

    Scoped to the calling visitor's runs (same as /library/chat).
    """
    assert _session_factory is not None
    client_id = _client_id(request)

    async with _session_factory() as session:
        stmt = (
            select(ResearchRun.id, ResearchRun.title, ResearchRun.query)
            .where(ResearchRun.status == "done")
            .order_by(ResearchRun.started_at.desc())
        )
        if client_id is not None:
            stmt = stmt.where(ResearchRun.client_id == client_id)
        rows = await session.execute(stmt)
        available_reports = [
            {"run_id": str(r.id), "title": r.title or r.query, "query": r.query}
            for r in rows
        ]

    report = await evaluate_rag(body.question, available_reports, body.eval_model)
    return report.model_dump()


@app.post("/library/eval/judge")
async def library_eval_judge(body: RagJudgeRequest) -> dict[str, object]:
    """Judge precision + faithfulness on already-retrieved chunks and answer.

    Skips Stage 1 + Stage 2 retrieval — call this after /library/chat so the
    eval runs on exactly the chunks the user saw, with no redundant LLM calls.
    """
    import asyncio as _asyncio

    from engine.models import estimate_cost_usd as _cost
    from eval.rag_answer_relevance import run_answer_relevance
    from eval.rag_context_precision import run_context_precision
    from eval.rag_context_sufficiency import run_context_sufficiency
    from eval.rag_faithfulness import run_rag_faithfulness

    (precision_verdicts, context_precision, precision_usage), \
    (sufficiency_verdict, context_sufficiency, sufficiency_usage), \
    (faithfulness_verdicts, answer_faithfulness, faithfulness_usage), \
    (relevance_verdict, answer_relevance, relevance_usage) = await _asyncio.gather(
        run_context_precision(body.question, body.chunks, body.eval_model),
        run_context_sufficiency(body.question, body.chunks, body.eval_model),
        run_rag_faithfulness(body.answer, body.chunks, body.eval_model),
        run_answer_relevance(body.question, body.answer, body.chunks, body.eval_model),
    )
    all_usage = [*precision_usage, *sufficiency_usage, *faithfulness_usage, *relevance_usage]
    return {
        "question": body.question,
        "chunks_retrieved": len(body.chunks),
        "chunk_verdicts": [v.model_dump() for v in precision_verdicts],
        "context_precision": context_precision,
        "context_sufficiency": context_sufficiency,
        "context_sufficiency_verdict": sufficiency_verdict.model_dump(),
        "claim_verdicts": [v.model_dump() for v in faithfulness_verdicts],
        "answer_faithfulness": answer_faithfulness,
        "answer_relevance": answer_relevance,
        "answer_relevance_verdict": relevance_verdict.model_dump(),
        "eval_cost_usd": _cost(all_usage),
    }


@app.post("/library/reindex")
async def library_reindex(request: Request) -> dict[str, object]:
    """Re-index the calling visitor's completed runs into pgvector.

    Scoped to the X-Client-Id header. Safe to call multiple times — deletes
    existing chunks before re-indexing to avoid duplicates.
    """
    assert _session_factory is not None
    client_id = _client_id(request)

    async with _session_factory() as session:
        stmt = (
            select(
                ResearchRun.id,
                ResearchRun.title,
                ResearchRun.query,
                Report.content,
                Report.structured,
            )
            .join(Report, Report.run_id == ResearchRun.id)
            .where(ResearchRun.status == "done")
            .order_by(ResearchRun.started_at.desc())
        )
        if client_id is not None:
            stmt = stmt.where(ResearchRun.client_id == client_id)
        rows = await session.execute(stmt)
        runs: list[tuple[str, str, str, str, list[dict[str, object]]]] = []
        for r in rows:
            structured = r.structured or {}
            stored_findings = structured.get("library_findings", [])
            findings = []
            if isinstance(stored_findings, list):
                findings = [f for f in stored_findings if isinstance(f, dict)]
            runs.append((str(r.id), r.title or r.query, r.query, r.content, findings))

    indexed = 0
    failed: list[dict[str, str]] = []

    for run_id, title, query, report, findings in runs:
        try:
            if not report:
                failed.append({"run_id": run_id, "reason": "no persisted report"})
                continue
            await rag.delete_chunks(run_id)
            await rag.embed_and_store(run_id, report, title, query, findings=findings)
            indexed += 1
        except Exception as exc:
            failed.append({"run_id": run_id, "reason": str(exc)})

    return {"indexed": indexed, "total": len(runs), "failed": failed}

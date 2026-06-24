# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Git

Never auto-commit or auto-push. All commits and pushes are done by the user.

## Project

General-purpose multi-agent deep research system. Portfolio project showcasing **multi-agent orchestration + memory engineering + human-in-the-loop** using Python + LangGraph + FastAPI + minimal Next.js UI.

Full design spec is in `design-spec.md` — read it before working on this codebase.

## Commands

```bash
# Start Postgres
docker-compose up -d

# Install Python deps (use uv)
uv sync

# Run API server
uv run uvicorn api.main:app --reload

# Run Alembic migrations
uv run alembic upgrade head

# Create a new migration
uv run alembic revision --autogenerate -m "<description>"

# Lint / typecheck
uv run ruff check .
uv run mypy .

# Tests
uv run pytest
uv run pytest tests/path/to/test.py::test_name   # single test
```

```bash
# Frontend (web/)
pnpm dev
pnpm build
```

## Architecture

### Four LLM roles

| Agent | Model | Responsibility |
|---|---|---|
| Clarify node | gpt-5.4 | Detects query ambiguity → emits 1–3 clarifying questions → `interrupt()` |
| Lead / Orchestrator | gpt-5.4 | Decomposes clarified query → spawns N subagents via `Send` fan-out |
| Research subagents (N, parallel) | gpt-5.4-nano | `search → fetch → extract` loop → returns `Finding(claim, evidenceSpan, citationUrl)` |
| Synthesizer | gpt-5.4 | Writes final cited Markdown report from compacted findings |
| Chat | gpt-5.4 | Follow-up Q&A grounded in findings stored in checkpointer state |
| Debate advocate (debate mode) | Codex-sonnet-4-6 | Argues the strongest evidence-backed answer over the compacted findings |
| Debate skeptic (debate mode) | gemini-3.1-pro-preview | Attacks evidence quality, gaps, and overreach in the advocate's argument |

Model IDs live in `engine/models.py` — **verify against the provider docs before running** (OpenAI/Anthropic/Google links at the top of that file), do not trust memory. `make_chat_model()` routes a model id to its provider's LangChain package; user-selectable models are gated by which provider API keys are set.

### Three-layer memory stack (the showcase)

1. **Working memory** — typed LangGraph `ResearchState` in `engine/state.py`. The in-run scratchpad passed between nodes.
2. **Context compaction** — `engine/nodes/compact.py` summarizes subagent notes into `state.summary` and trims stale raw tool outputs before synthesis. No embeddings.
3. **Episodic / short-term memory** — LangGraph **Postgres checkpointer** (`engine/memory/checkpointer.py`) persists graph state per `thread_id`. Powers resumable runs, multi-turn follow-up chat, and the human-in-the-loop pause/resume.

### Human-in-the-loop flow

`clarify.py` calls LangGraph `interrupt()` when the query is ambiguous. The checkpointer snapshots the paused graph. The API emits a `clarification_needed` SSE event. The UI shows the questions form. User answers are submitted to `POST /runs/{id}/resume`, which calls `Command(resume=answers)` to continue the graph from the checkpoint. **Verify the current `interrupt` / `Command` API against LangGraph docs at build time.**

### Graph data flow

```
query → clarify (interrupt if ambiguous) → plan → Send fan-out → N subagents
      → compact → synthesize → cited report → follow-up chat
```

### Debate mode (optional, off by default)

`POST /research` with `debate: true` routes compact → `debate_advocate` ⇄ `debate_skeptic` (2 rounds, loop edges in `engine/orchestrator.py`) → `judge_debate` → **debate-driven gap research** → synthesize. After the final round, `judge_debate` (lead model, neutral) declares a winner — advocate / skeptic / draw — streamed as a `debate_verdict` SSE event and rendered as a verdict card in the UI. Then `plan_gap_research` (lead model, neutral) distills the skeptic's unresolved objections into ≤3 follow-up questions, a second `Send` fan-out (`gap_subagent`, same subagent fn under a separate node name) researches them, and `recompact` folds the new findings into the summary — skipped entirely when the debate surfaces no gaps. Each turn is its own node execution so turns stream as `debate_turn` SSE events and checkpoint individually; debate runs also add `stream_mode="messages"` so turns live-stream token-by-token as `debate_token` events (rendered as chat bubbles in the UI). The gap round streams as `gap_planning` / `gap_plan` / `subtask_done {stage:"gap"}` events. Debaters default to **different AI companies** (uncorrelated errors; the lead stays a neutral third party) and are user-selectable per role. The synthesizer consumes the transcript to calibrate confidence; `verify_citations` is unaffected. Nodes live in `engine/nodes/debate.py`.

### Anti-hallucination

Every subagent result must validate against the `Finding` Pydantic schema (`engine/extraction.py`): `{claim: str, evidenceSpan: str, citationUrl: HttpUrl}`. Unvalidated results are rejected before synthesis.

### API surface

- `POST /research` — start a run, returns SSE stream
- `POST /runs/{id}/resume` — submit clarification answers → resumes paused graph, continues SSE
- `GET /runs/{id}` — run status + findings
- `POST /chat` — follow-up Q&A over a completed run, returns SSE

### Database (plain Postgres, no pgvector in v1)

Tables: `research_runs`, `subtasks`, `sources`, `findings`, `reports` + LangGraph checkpoint tables. Managed by Alembic. Models in `db/models.py` (SQLAlchemy 2.0 mapped classes).

### Tools

`engine/tools/search.py` wraps Tavily (or Exa). `engine/tools/fetch.py` fetches a URL → returns cleaned text. Both are opaque callables with schemas — adding a domain-specific tool later doesn't touch the engine.

## Implementation phases

- **Phase 0** — scaffold: monorepo dirs, docker-compose, SQLAlchemy models + Alembic, `engine/models.py`
- **Phase 1** — multi-agent core: `state.py`, LangGraph graph with `Send` fan-out, tools, subagent, synthesizer (end-to-end: question → cited report, no memory yet)
  - Commit: `implement phase 1 multi-agent core (plan → Send fan-out → subagent → synthesize)`
- **Phase 2** — memory + human-in-the-loop: compaction node, Postgres checkpointer, chat node, clarify node with `interrupt()`
  - Commit: `implement phase 2 memory stack + human-in-the-loop (compact, checkpointer, clarify, chat)`
- **Phase 3** — web UI: FastAPI SSE + Next.js page (plan stream, parallel subagent cards, report, clarification form, chat box)
- **Phase 4** — eval harness (faithfulness + citation grounding), README, seeded demo run

## Key constraints

- LangChain footprint: `langgraph` + `langchain-core` + thin per-provider packages (`langchain-openai`, `langchain-anthropic`, `langchain-google-genai`) only — **not** the monolithic `langchain` package.
- No RAG / pgvector in v1. Vector long-term memory is the documented Phase 5 growth path.
- Each layer of the memory stack must be visibly labeled in code comments so the design is legible to a reviewer.

# AI Policy & Regulation Researcher (memory-engineering showcase)

## Context

Greenfield **portfolio** project. The app now has one flagship vertical: **AI policy and
regulation research**. It helps users track AI laws, policy proposals, regulator guidance,
standards, enforcement actions, compliance obligations, and governance risk across jurisdictions.
Because the user is based in Singapore, the default demo path should include Singapore AI
governance and model-risk questions while still supporting cross-jurisdiction comparisons.

The product surface is domain-specific: ask an AI policy/regulatory question → the lead agent
**asks clarifying questions if jurisdiction, time frame, or scope is ambiguous
(human-in-the-loop)** → plans → parallel subagents research the web → the system synthesizes a
**cited policy report** → and you can **follow up in chat**, with the system remembering within and
across the conversation. The interesting depth is still concentrated in **multi-agent orchestration
+ memory + human-in-the-loop**, but the demo is now anchored in a concrete, reviewer-friendly
domain.

**RAG / vector memory is deliberately deferred** (see "Deferred"). The first version keeps memory
simple: working memory + context compaction + a checkpointer. No embeddings, no vector DB.

Decisions locked with the user:
- **One flagship vertical:** AI Policy & Regulation Researcher, built on a reusable generic engine.
- **Domain-specific research behavior:** prioritize jurisdiction, legal status, effective dates,
  affected actors, obligations, enforcement mechanisms, exceptions, unresolved proposals, and
  compliance implications.
- **Framework:** Python + **LangGraph** (typed state, `Send` fan-out, Postgres checkpointer).
- **Interface:** **minimal web UI** (FastAPI SSE backend + thin Next.js page that streams the live
  plan, the parallel subagents, the report, and a follow-up chat).
- **Memory is the showcase, minus RAG for now** — short-term working memory, context compaction, and
  a Postgres checkpointer. Vector/RAG long-term memory is a documented next step, not in v1.

## The agents (four LLM-driven roles)

| Agent | Model | Job |
|---|---|---|
| **1. Lead / Planner** | gpt-5.4 | First **decides whether the query is ambiguous and, if so, asks 1–3 clarifying questions (human-in-the-loop) and waits**. Then decomposes the (clarified) question into N independent sub-questions, sets strategy, spawns the subagents (the `Send` fan-out), monitors. The orchestrator. |
| **2. Research subagents** (N, parallel) | gpt-5.4-nano | The workers. Each owns **one** sub-question and runs a `search → fetch → extract` loop with the tools, returning validated `Finding`s (claim + evidence + citation). |
| **3. Synthesizer** | gpt-5.4 | Takes the compacted findings from all subagents and writes the final cited report. |
| **4. Chat / follow-up** | gpt-5.4 | Answers follow-up questions, grounded in the run's findings held in checkpointer state. |

Not agents (utility nodes/functions the agents call): the **compaction** step (note summarization —
an LLM call but makes no decisions) and the **tools** (`search`, `fetch`).

## What this demonstrates to a reviewer

| Piece | What it shows |
|---|---|
| Lead + parallel subagents (LangGraph `Send`) | Real multi-agent orchestration, not a single prompt loop |
| **Memory engineering (the headline)** | Working memory, context compaction, and a checkpointer — deliberate layered design most clones skip |
| Validated findings + citations | Anti-hallucination: every claim has an evidence span + source URL |
| Streaming web UI | Product polish: you *see* the plan and subagents work in real time |
| Eval harness | Correctness you can measure (faithfulness + citation grounding) |

## Architecture

```
 query
   │
   ▼
 ┌──────────────────────────────┐     ambiguous?  ┌───────────────────────────┐
 │ Clarify node (Gemini 2.5 Pro)    │────── yes ─────►│ interrupt() — pause graph, │
 │ ask 1–3 clarifying questions │                 │ checkpoint, surface to UI, │
 └──────────────┬───────────────┘◄─── resume ─────│ wait for user answers      │
                │ no / answered                    └───────────────────────────┘
                ▼
                ┌─────────────────────────────────────────────┐
                │ Lead / Orchestrator  (LangGraph, Gemini 2.5 Pro) │
                │ plan → spawn N subagents (Send) → monitor →   │
                │ compact notes → synthesize                    │
                └──┬──────────┬──────────┬──────────┬───────────┘
        parallel   ▼          ▼          ▼          ▼
        subagents (Gemini 3.1 Flash-Lite):  search → fetch → extract findings
                  validated {claim, evidenceSpan, citationUrl}
                              │
                    ┌─────────▼──────────┐
                    │ Compaction node    │  summarize notes,
                    │ → state.summary    │  trim stale tool outputs
                    └─────────┬──────────┘
                              ▼
                    ┌──────────────────┐
                    │ Synthesis        │   cited report
                    │ (Gemini 2.5 Pro)     │
                    └────────┬─────────┘
                             ▼
                Follow-up chat (Gemini 2.5 Pro) — grounded in findings
                held in checkpointer state (no RAG in v1)
```

### The memory stack (the deliberate showcase — three layers in v1)

Each layer is labeled in code so the design reads clearly:

1. **Working memory** — the typed LangGraph `state` (`query`, `plan`, per-subtask `findings`,
   running `summary`). The in-run scratchpad the graph passes between nodes.
2. **Context compaction** — a node that summarizes subagent notes into `state.summary` and trims
   stale raw tool outputs before synthesis, so the lead never blows the context window. Pure
   summarization; no embeddings.
3. **Short-term / episodic memory (checkpointer)** — LangGraph **Postgres checkpointer** persists
   graph state per `thread_id`, giving **resumable runs**, **multi-turn follow-up chat**, and the
   **human-in-the-loop pause/resume** for free. Follow-up chat reads the findings straight from this
   persisted state.

The **human-in-the-loop clarification** uses LangGraph's `interrupt()`: when the clarify node decides
the query is ambiguous, it interrupts; the checkpointer snapshots the paused graph; the UI shows the
questions; the user's answers are passed back via `Command(resume=...)` and the graph continues from
exactly where it paused. This reuses layer 3 — no new machinery.

The README's "Memory architecture" section walks these three layers — the portfolio centerpiece. A
fourth RAG/vector layer is the documented growth path (see "Deferred").

## Engine design (reusable core + policy vertical)

The graph, memory layers, checkpointer, API streaming, and eval harness stay reusable. The flagship
AI policy vertical is expressed through planner/extraction/synthesis prompts and source-selection
biases, not by hard-coding one-off control flow.

- **Tools** — `search` uses Tavily as the default broad search provider, Exa for deep semantic
  retrieval, and SerpAPI as a Google-specific fallback. `fetch` uses Firecrawl for page
  extraction/crawling when configured, with local `httpx` + BeautifulSoup extraction as fallback.
  Policy research biases searches toward official regulators, legislatures, standards bodies,
  courts, and credible legal-analysis sources.
- **Extraction schema** — a Pydantic `Finding` shape every subagent result must validate against:
  `{claim, evidenceSpan, citationUrl}`. Rejects fabricated/unsupported values (anti-hallucination).
- **Report** — a synthesis schema → Markdown renderer (sections + inline citations), with policy
  reports emphasizing jurisdiction, legal status, obligations, effective dates, enforcement, and
  compliance implications when supported by findings.

## Data model (Postgres)

- `research_runs` `{id, query, status, clarifications jsonb, plan jsonb, stats jsonb, startedAt,
  finishedAt}` — `status` includes `awaiting_clarification`; `clarifications` holds the asked
  questions + the user's answers
- `subtasks` `{id, runId, question, status}`
- `sources` `{id, runId, url, title, fetchedAt}`
- `findings` `{id, subtaskId, sourceId, claim, evidenceSpan, citationUrl}`
- `reports` `{id, runId, content, structured jsonb}`
- LangGraph checkpoint tables (short-term/episodic memory)

No `pgvector`, no embedding columns in v1 — plain Postgres.

## Repo layout (single system, monorepo)

```
deep-research/
  engine/
    orchestrator.py        # LangGraph graph: plan → fan-out → compact → synthesize
    state.py               # typed graph state (WORKING MEMORY)
    nodes/
      clarify.py           # HUMAN-IN-THE-LOOP: ask clarifying Qs, interrupt() if ambiguous
      plan.py              # decompose (clarified) query → subtasks (Gemini 2.5 Pro)
      subagent.py          # search → fetch → extract validated findings (Gemini 3.1 Flash-Lite)
      compact.py           # CONTEXT COMPACTION: summarize notes, trim stale outputs
      synthesize.py        # cited report (Gemini 2.5 Pro)
      chat.py              # follow-up grounded in checkpointer state
    memory/
      checkpointer.py      # LangGraph Postgres checkpointer (short-term/episodic)
      compaction.py        # summarization helpers
    tools/
      search.py            # Tavily / Exa / SerpAPI
      fetch.py             # Firecrawl scrape → local URL cleanup fallback
    extraction.py          # Pydantic Finding schema (anti-hallucination)
    models.py              # model IDs (verify at build time)
  api/
    main.py                # FastAPI: POST /research (SSE), POST /runs/{id}/resume (clarify
                           #   answers), GET /runs/{id}, POST /chat (SSE)
  web/                     # minimal Next.js: streams plan + subagents + report + chat
  db/
    models.py              # SQLAlchemy models + Alembic
  eval/                    # faithfulness + citation-grounding harness
  docker-compose.yml       # postgres
  README.md                # the pain story, architecture diagram, MEMORY ARCHITECTURE section
```

## Models

- **LLM (`langchain-openai`, `ChatOpenAI`):** **gpt-5.4** ($2.50/$15.00 per 1M tokens) for
  clarify/plan/synthesis/chat (reasoning-heavy lead roles; supports cached input at $0.25);
  **gpt-5.4-nano** ($0.20/$1.25 per 1M tokens) for the parallel research subagents (12.5× cheaper
  than lead; sufficient for structured Finding extraction). IDs centralized in `models.py`;
  **verify exact IDs against https://platform.openai.com/docs/models at build time** — don't
  hardcode from memory. Auth via `OPENAI_API_KEY`.
- **LangChain footprint:** `langgraph` + `langchain-core` + `langchain-openai` only (not the
  monolithic `langchain`). No embedding model in v1.

## Implementation steps (sequenced — core first, then the memory showcase)

**Phase 0 — Scaffold.**
1. Monorepo skeleton; `docker-compose` with **postgres**; SQLAlchemy models + Alembic; `models.py`.

**Phase 1 — Multi-agent core (prove orchestration + citations first, no memory yet).**
2. Typed `state.py` (working memory) + `orchestrator.py` LangGraph graph with parallel `Send`
   fan-out (stub nodes first to verify the graph compiles and fans out).
3. `tools/search.py` + `tools/fetch.py`; `subagent.py` = search → fetch → extract validated
   `Finding`s; `synthesize.py` = cited Markdown report. End-to-end: a question → a cited report.

**Phase 2 — Memory engineering + human-in-the-loop (the headline; build and label each layer).**
4. `compact.py` + `memory/compaction.py` — summarize subagent notes into `state.summary`, trim stale
   tool outputs (context management).
5. `memory/checkpointer.py` — wire the LangGraph Postgres checkpointer: resumable runs + multi-turn.
   `chat.py` = follow-up Q&A reading findings from checkpointer state.
6. `nodes/clarify.py` — human-in-the-loop: detect ambiguity, emit 1–3 clarifying questions, call
   `interrupt()`. Resume via `Command(resume=answers)`. **Verify LangGraph's current `interrupt` /
   `Command` API against live docs at build time** — don't rely on memory.

**Phase 3 — Minimal web UI.**
7. `api/main.py` — FastAPI `POST /research` (SSE; may emit a `clarification_needed` event and pause),
   `POST /runs/{id}/resume` (submit clarification answers → `Command(resume=...)`, stream continues),
   `GET /runs/{id}`, `POST /chat` (SSE). Next.js page renders the **clarifying-questions form when the
   run is paused**, then the live plan, the parallel subagents as they work, the final cited report,
   and a follow-up chat box. Frontend may use `@ai-sdk/react` over the backend SSE — **verify AI SDK /
   Next patterns against live docs at build time**, don't rely on memory.

**Phase 4 — Eval + polish.**
8. `eval/` — faithfulness + citation-grounding harness (every claim traces to a fetched source;
   flag ungrounded claims). Optional LangSmith traces.
9. README with the pain story, architecture diagram, the **Memory architecture** section (three
   layers), and the **human-in-the-loop** flow; a cached/seeded demo run so the UI works offline.

## Verification (end-to-end)

1. `docker-compose up`; web `pnpm dev`. App boots, DB connects.
2. **Human-in-the-loop**: ask a deliberately vague question (e.g. *"Tell me about Mistral"*) → the UI
   shows clarifying questions, the run is paused (`status = awaiting_clarification`); submit answers →
   the run resumes from the checkpoint and proceeds with the clarified query. A specific question
   skips straight to planning (no interrupt).
3. **Multi-agent run**: ask a broad question (e.g. *"What changed in EU AI regulation in 2025?"*) →
   the UI shows the live plan, then **parallel** subagents working, then a cited report whose
   citations open real sources.
4. **Memory — context compaction**: a large run synthesizes without overflowing context; confirm
   `state.summary` is populated and raw tool outputs are trimmed (log/inspect state).
5. **Memory — checkpointer**: kill and resume a run by `thread_id`; a follow-up chat message
   continues the same conversation and answers from the run's findings.
6. **Eval**: harness reports faithfulness + citation-grounding; **0 ungrounded claims** on the demo.
7. Typecheck/lint clean; seeded demo run works offline.

## Deferred (documented growth path, not in v1)

- **RAG / vector long-term memory (pgvector):** embed gathered source chunks + distilled findings;
  retrieve top-k for synthesis on very large corpora; enable **cross-session recall** of past
  findings. This is the natural 4th memory layer — add a `memory/vector_store.py`, a `source_chunks`
  table, and an embedding model when v1 is solid.
- Worker/queue for very large fan-outs; fetch caching (Redis).
- Deploy: `web/` on Vercel, FastAPI backend on Railway/Render/Fly, managed Postgres.
- A domain vertical later = add a tool + extraction schema only; the engine stays unchanged. (The
  original two-app plan is the documented growth path, deliberately deferred for simplicity.)

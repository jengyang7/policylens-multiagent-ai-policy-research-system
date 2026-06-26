# AI Policy & Regulation Researcher

An AI policy and regulation research assistant: ask about AI laws, policy proposals, regulator guidance, standards, enforcement actions, compliance obligations, or governance risk. A lead agent clarifies ambiguous jurisdiction/scope, plans the research, fans out **parallel subagents** to search and read the web, and synthesizes a **citation-verified policy report** — with an optional **adversarial debate round** where two AI models from different companies argue over the findings before the report is written.

The flagship demo is intentionally useful from Singapore: for example, ask *"How is Singapore regulating AI governance and model risk in 2026?"* and compare Singapore's approach against the EU, US, UK, or OECD.

Built as a portfolio project to showcase **multi-agent orchestration**, a **layered memory architecture**, **human-in-the-loop** workflows, and domain-specific evidence synthesis with LangGraph. The AI policy vertical is the flagship product surface; the underlying orchestration engine remains reusable.

## Highlights

- **AI policy intelligence** — reports track jurisdiction, legal status, effective dates, affected actors, obligations, enforcement mechanisms, exceptions, and compliance implications when the evidence supports them.
- **Multi-agent orchestration** — a lead model plans the research and fans out N parallel subagents via LangGraph `Send`; each runs a `search → fetch → extract` loop and returns validated findings.
- **Three-layer memory stack** — typed working state → context compaction → Postgres checkpointer. The same checkpointer layer powers resumable runs, multi-turn follow-up chat, *and* human-in-the-loop pause/resume.
- **Human-in-the-loop clarification** — an ambiguous query triggers `interrupt()`; the UI shows clarifying questions, and `Command(resume=...)` continues the graph from exactly where it paused.
- **Adversarial debate mode** — a Claude advocate and a Gemini skeptic argue over the findings (cross-provider, so their errors are uncorrelated), a neutral judge declares a winner, and the skeptic's unresolved objections drive a second, targeted research round before the final report.
- **Anti-hallucination by construction** — every subagent claim must validate against a `{claim, evidenceSpan, citationUrl}` schema, and a citation-faithfulness/grounding eval harness checks the finished report.
- **Live streaming UI** — a Next.js frontend streams the plan, parallel subagents, debate turns (token-by-token), the verdict, gap research, and the final policy report over SSE.
- **Export** — download a full policy research session as Markdown or Word (`.docx`).

## Architecture

```
query
  │
  ▼
clarify ──ambiguous?──► interrupt() ──► [UI: clarifying questions] ──► resume
  │ no / answered
  ▼
plan (lead model)
  │
  ├─ Send fan-out ─► N parallel research subagents
  │                    (search → fetch → extract → Finding)
  ▼
compact  (layer 2: state.summary, trims stale tool output)
  │
  ├── debate mode off ───────────────────────────────┐
  │                                                    ▼
  └── debate mode on:                            synthesize → cited report
        debate_advocate ⇄ debate_skeptic (N rounds)    │
              │                                        ▼
        judge_debate (neutral lead → verdict)    verify_citations
              │                                        │
        plan_gap_research                              ▼
              │                                  follow-up chat
        Send fan-out ─► gap subagents              (grounded in
              │                                  checkpointer state)
        recompact ──────────────────────────────────┘
```

## The memory stack (the showcase)

1. **Working memory** — `engine/state.py`'s typed `ResearchState`: the in-run scratchpad (query, plan, per-subtask findings, summary) passed between LangGraph nodes.
2. **Context compaction** — `engine/nodes/compact.py` summarizes subagent notes into `state.summary` and trims stale raw tool output before synthesis. Pure summarization, no embeddings.
3. **Episodic memory (Postgres checkpointer)** — `engine/memory/checkpointer.py` persists graph state per `thread_id`, giving resumable runs, multi-turn follow-up chat, and human-in-the-loop pause/resume — all from one mechanism.

A vector/RAG long-term memory layer is the documented next step (see [Roadmap](#roadmap)), deliberately deferred from v1.

## Debate mode

When a run is started with `debate: true`:

- **`debate_advocate`** (default: Claude Sonnet) argues the strongest evidence-backed answer from the compacted findings.
- **`debate_skeptic`** (default: Gemini 3.1 Pro) attacks evidence quality, gaps, and overreach — for N rounds (default 2), streaming token-by-token.
- **`judge_debate`** (neutral lead model) declares a winner — advocate / skeptic / draw — rendered as a verdict card.
- **`plan_gap_research`** distills the skeptic's unresolved objections into up to 3 follow-up questions, researched by a second `Send` fan-out (`gap_subagent`), and **`recompact`** folds the new findings back into the summary.
- Debaters default to **different AI companies** so their errors are uncorrelated, and the lead model stays a neutral third party (avoiding self-preference bias).
- The synthesizer reads the debate transcript to calibrate confidence in the final report.

## Roles & models

| Role | Default model | Job |
|---|---|---|
| Clarify / Plan / Synthesize / Chat (lead) | `gpt-5.4` | Orchestration, ambiguity detection, final report, follow-up Q&A |
| Research subagents (N, parallel) | `gpt-5.4-nano` | `search → fetch → extract` validated `Finding`s |
| Citation faithfulness judge | `gpt-5.4-mini` | per-sentence grounding check |
| Debate advocate | `claude-sonnet-4-6` | argues the strongest case from the findings |
| Debate skeptic | `gemini-3.1-pro-preview` | attacks evidence quality and gaps |
| Eval judge | `claude-haiku-4-5` | independent faithfulness/grounding/completeness scoring |

All roles are user-selectable per run via the UI, gated by which provider API keys are configured (`engine/models.py`).

## Tech stack

- **Orchestration** — Python, LangGraph (`Send` fan-out, `interrupt`/`Command`, Postgres checkpointer)
- **LLMs** — OpenAI, Anthropic, Google, via `langchain-openai` / `langchain-anthropic` / `langchain-google-genai`
- **API** — FastAPI + Server-Sent Events (`sse-starlette`)
- **Database** — Postgres, SQLAlchemy 2.0 (async), Alembic migrations
- **Search & fetch** — Tavily for default search, Exa for semantic search, and direct
  `httpx`/BeautifulSoup page extraction
- **Frontend** — Next.js 16, React 19, `react-markdown`
- **Eval** — custom harness: citation grounding, faithfulness, completeness, relevance

## Getting started

### Prerequisites

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- Node.js and pnpm
- Docker (for Postgres)
- API keys: OpenAI is required; Anthropic / Google are optional for debate mode; Tavily or Exa
  enable web search

### Setup

```bash
# 1. Start Postgres
docker-compose up -d

# 2. Configure environment
cp .env.example .env   # fill in OPENAI_API_KEY plus search/extraction keys

# 3. Install dependencies & run migrations
uv sync
uv run alembic upgrade head

# 4. Start the API
uv run uvicorn api.main:app --reload

# 5. Start the frontend (separate terminal)
cd web && pnpm install && pnpm dev
```

Open http://localhost:3000.

### Running an eval

```bash
uv run python -m eval --run-id <run-id> [--strict] [--json]
```

Exits non-zero if any claim is ungrounded (or, with `--strict`, any citation is judged unfaithful) — suitable for CI.

## API

| Endpoint | Description |
|---|---|
| `POST /research` | Start a run (SSE stream); accepts `debate: true` and per-role model selection |
| `POST /runs/{id}/resume` | Submit clarification answers, resume from checkpoint |
| `GET /runs/{id}` | Run status, plan, findings, report |
| `GET /runs` | List runs |
| `DELETE /runs/{id}` | Delete a run |
| `POST /chat` | Follow-up Q&A over a completed run (SSE), grounded in checkpointer state |
| `POST /runs/{id}/eval` | Run the eval harness against a completed run |
| `GET /eval/summary`, `/eval/reports` | Aggregate eval metrics for the dashboard |
| `GET /models` | Available models, filtered by configured provider API keys |

## Project structure

```
engine/
  orchestrator.py   # LangGraph graph definition
  state.py          # typed ResearchState (working memory)
  nodes/            # clarify, plan, subagent, compact, debate, synthesize, chat, verify_citations
  memory/           # Postgres checkpointer
  tools/            # search (Tavily / Exa), fetch (direct httpx/BeautifulSoup extraction)
  models.py         # model IDs, pricing, provider routing
api/                 # FastAPI app (SSE endpoints)
db/                  # SQLAlchemy models + Alembic migrations
eval/                # grounding, faithfulness, completeness, relevance harness
web/                 # Next.js streaming UI
```

## Roadmap

- Vector/RAG long-term memory — a 4th memory layer for cross-session recall over past findings
- Worker/queue for large fan-outs; fetch caching
- Production deploy (Vercel + Railway/Render/Fly + managed Postgres)

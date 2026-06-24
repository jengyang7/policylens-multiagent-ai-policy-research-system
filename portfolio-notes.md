# Portfolio & Debate Mode Notes

## Is this project worth putting on a resume?

Yes — strong signal. It hits three things hiring managers rarely see together:

1. **Multi-agent orchestration** — real `Send` fan-out with parallel subagents, compaction, and graph-level state via LangGraph
2. **Memory engineering at multiple layers** — working memory, context compaction, Postgres episodic checkpointer
3. **Production-minded AI** — hallucination mitigation via Pydantic schema enforcement, human-in-the-loop, debate mode, RAG retrieval

Target roles: AI/ML Engineer, Backend Engineer at AI-native companies, Applied AI Engineer.

---

## How it differs from Gemini / ChatGPT Deep Research

| Dimension | Commercial Products | This Project |
|---|---|---|
| Parallelism | Single agent, sequential | N subagents in parallel via `Send` fan-out |
| Adversarial validation | None | Debate mode: advocate → skeptic → gap research |
| Human-in-the-loop | Ask upfront at most | `interrupt()` pauses mid-execution, resumes via checkpoint |
| Hallucination control | Opaque | Every finding validated against `Finding(claim, evidenceSpan, citationUrl)` |
| Memory | Session-only | 3-layer: in-run state + compaction + Postgres checkpointer |
| Multi-provider | Single company | OpenAI orchestrates, Anthropic advocates, Google skeptics |
| Auditability | Black box | Every finding has `evidenceSpan` + `citationUrl`, every node inspectable |

One-liner: *"Gemini Deep Research is a product. This is the engineering layer underneath — multi-agent orchestration, structured memory, adversarial validation, and human-in-the-loop, all composable and inspectable."*

---

## Does debate mode really add value?

**Honest assessment: the debate rounds themselves are weak. The gap research they trigger is the real value.**

### Why the debate rounds are weak

- LLMs are trained to be agreeable — even in "skeptic" role, they raise concerns diplomatically rather than destroying weak arguments
- The skeptic reasons over `state.summary`, not raw sources — it cannot verify whether a cited URL actually supports the claim
- The skeptic can hallucinate objections, triggering gap research on false premises
- The output may only *look* more nuanced because the synthesizer is instructed to calibrate confidence based on the debate transcript

### What actually helps

| Component | Likely impact |
|---|---|
| Gap research (second `Send` fan-out) | **High** — retrieves new information, directly fills identified holes |
| Judge verdict + confidence calibration | **Medium** — forces synthesizer to be less overconfident |
| Advocate/skeptic debate rounds | **Low** — probably adds noise as much as signal |

The multi-provider angle (Anthropic + Google) is the one part that could genuinely help — different training data means different knowledge gaps.

---

## How to prove whether debate mode helps (eval harness)

Run this experiment (Phase 4 in CLAUDE.md):

```
same 10-20 queries → run with debate=false AND debate=true
                   → score both outputs on:
                      1. Faithfulness: do claims match cited sources?
                      2. Coverage: did it miss important perspectives?
                      3. Citation quality: are sources authoritative?
```

| Scoring method | Cost | Reliability |
|---|---|---|
| LLM-as-judge (neutral model rates both blindly) | Low | Medium |
| Known-answer queries (verifiable ground truth) | Low | High |
| Human eval on a sample | High | Highest |

If debate rounds don't improve metrics → drop them, keep gap research, trigger it via a cheaper self-critique pass instead.

---

## Recommended architecture: keep both, separate concerns

- **Standard path**: synthesize → self-critique (single model, cheap) → revise if gaps → output
- **Debate mode** (opt-in): advocate ⇄ skeptic (2 rounds) → judge → gap research → recompact → synthesize

Interview framing:
> "The standard path has a self-critique pass after synthesis — cheap, catches obvious gaps. Debate mode is opt-in for high-stakes queries: adversarial advocacy across different AI providers, then a second research fan-out to fill unresolved objections. Different cost/quality tradeoffs for different use cases."

---

## Resume action plan

1. **Fix model IDs in `engine/models.py`** — must be real callable IDs to demo live
2. **Add README architecture diagram** — one graph diagram is worth more than a new feature
3. **Build eval harness** — gives quantitative metrics for bullet points
4. **Add project to resume, start applying now** — don't wait for more features

### Suggested resume bullets

```
• Designed a LangGraph-based multi-agent system with Send fan-out dispatching N parallel
  research subagents, each enforcing a Pydantic Finding(claim, evidenceSpan, citationUrl)
  schema to reject unvalidated claims before synthesis

• Engineered a three-layer memory stack: typed in-run ResearchState, a compaction node
  that summarizes findings before context window exhaustion, and a Postgres LangGraph
  checkpointer enabling resumable runs and persistent multi-turn follow-up chat

• Implemented human-in-the-loop via LangGraph interrupt() / Command(resume=) — graph
  pauses on ambiguous queries, persists state via checkpoint, resumes after clarification

• Built a two-stage RAG layer with section-level chunking and privacy-scoped retrieval
  for grounded follow-up Q&A over completed research reports
```

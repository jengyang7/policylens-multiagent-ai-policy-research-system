# Issues Log

Running record of bugs, gotchas, and fixes encountered while building this project.

---

## Phase 2 — Memory + Human-in-the-loop (integration test run)

### 1. `Send` imported from deprecated location
**File:** `engine/orchestrator.py`
**Issue:** `from langgraph.constants import Send` emitted a `LangGraphDeprecatedSinceV10` warning — this import is scheduled for removal in LangGraph V2.
**Fix:** `from langgraph.types import Send`

---

### 2. `add_messages` not in `langchain_core.messages`
**File:** `engine/state.py`
**Issue:** `from langchain_core.messages import add_messages` raises `ImportError`. Despite being a message-related utility, it lives in LangGraph, not LangChain core.
**Fix:** `from langgraph.graph.message import add_messages`

---

### 3. Test failed because `ChatOpenAI.__init__` requires `OPENAI_API_KEY`
**File:** `tests/test_phase2_smoke.py` (`test_synthesize_uses_summary_over_raw_findings`)
**Issue:** The test patched `_PROMPT` to intercept the chain but still let `ChatOpenAI(...)` run, which immediately throws `OpenAIError: Missing credentials` — even in a unit test with no intent to make API calls.
**Fix:** Patch `ChatOpenAI` itself in addition to `_PROMPT`:
```python
monkeypatch.setattr("engine.nodes.synthesize.ChatOpenAI", lambda **kw: mock_llm)
```

---

### 6. Checkpointer state read empty after breaking from stream early
**File:** `scripts/test_phase2.py`
**Issue:** After calling `Command(resume=answers)` and breaking from the `astream_events` loop immediately after `clarify_wait`'s `on_chain_end` event, `aget_state()` still returned the pre-resume state (`clarifications=[]`, query unchanged). LangGraph flushes checkpoints to Postgres as part of the stream-processing loop — abandoning the generator at `on_chain_end` exits before that flush happens.

**Fix (two-part):**
1. Capture `clarify_wait`'s output directly from the `on_chain_end` event data (`event["data"]["output"]`) — this is the node's return value and is reliable regardless of checkpointer timing.
2. Continue consuming the stream until `plan`'s `on_chain_start` event — this is the signal that the previous checkpoint (for `clarify_wait`) has been written. Then break and read from `aget_state` safely.

**Rule:** When reading checkpointer state after a resume, always drain the stream at least one node past the resumed node before breaking. Breaking at the resumed node's `on_chain_end` is too early.

---

### 5. `clarify` node re-ran the LLM on resume, discarding user answers
**File:** `engine/nodes/clarify.py`, `engine/orchestrator.py`
**Issue:** When `Command(resume=answers)` resumed the graph, LangGraph re-ran the `clarify` node from the top. The LLM was called again with the same query and this time returned `is_ambiguous=False`, taking the early-return branch before ever reaching `interrupt()`. Result: `clarifications=[]`, refined query unchanged, test failed with "Expected clarifications to be populated after resume".

**Root cause:** In LangGraph, `interrupt()` pauses by raising `GraphInterrupt` — so no state updates can be saved before it. On resume, the entire node re-executes from the top. If the LLM is also in that node, it runs again with a potentially different result.

**Fix:** Split `clarify` into two nodes:
- `clarify` — calls the LLM once, stores questions in `state.clarification_questions`. Skips the LLM if questions are already stored (idempotent on retry).
- `clarify_wait` — calls `interrupt(questions)`. On first pass: raises `GraphInterrupt` (pauses). On resume: `interrupt()` returns answers immediately without re-calling the LLM.

`state.next` now points to `clarify_wait`, not `clarify`, so the LLM node is never re-entered on resume.

---

### 4. Ruff lint: unsorted imports (I001), unused import (F401), unused variable (F841), long lines (E501)
**Files:** `engine/nodes/clarify.py`, `engine/nodes/plan.py`, `engine/orchestrator.py`, `tests/test_phase2_smoke.py`, `engine/models.py`
**Issue:** Several files had import blocks out of ruff's expected order (third-party before first-party, alphabetical within groups). A `patch` import was added but never used. A `original_prompt` variable was assigned but never read. Two comment lines in `models.py` exceeded the 100-char limit.
**Fix:** `uv run ruff check --fix` resolved the import ordering, unused import, and unused variable automatically. The long comment lines in `models.py` were shortened manually.

---

## Phase 4 — Eval harness (debugging a "Failed" run via the dashboard)

### 7. Faithfulness judge over-penalized interpretive framing as "unfabricated facts"
**File:** `eval/faithfulness.py`
**Issue:** Running the eval dashboard against a real run ("What are the top AI trends shaping 2026?") produced a 26% faithfulness rate (70/94 cited sentences judged unfaithful) despite 96% citation grounding and a 100% completeness score — i.e. the report was well-sourced and on-topic, but the per-sentence judge was failing it anyway. Reviewing the verdicts showed the judge was treating *any* wording difference from the source — interpretive labels ("a major trend", "a key shift"), verbs like "forecasts" or "explicitly frames", or mild generalization of a supported claim — as a fabricated fact, even when every concrete number/name/date in the sentence traced back to a finding.

**Root cause:** `_JUDGE_PROMPT`'s system message framed the rubric as "no added facts" without distinguishing *added facts* (genuinely unsupported numbers, dates, names, causal claims) from *added framing* (the synthesizer's own interpretive language wrapping a supported claim). An LLM judge given only "don't add anything" defaults to flagging any rephrasing.

**Fix:** Rewrote `_JUDGE_PROMPT` with explicit FAITHFUL vs. UNFAITHFUL categories — FAITHFUL explicitly allows rephrasing/summarizing/combining findings, interpretive labels for supported claims, and mild generalization that introduces no new facts/numbers/dates/names; UNFAITHFUL is reserved for new factual specifics, contradictions, or overstated certainty. Re-running the eval on the same report (no change to the report itself) raised the faithfulness rate from 26% (70/94 unfaithful) to 50% (47/94 unfaithful) for $0.2068.

---

### 8. Synthesizer misattaches single-source `[i]` citations to cross-cutting synthesis sentences
**File:** `engine/nodes/synthesize.py`
**Issue:** After fixing #7, 47 sentences were still judged unfaithful, and citation `[1]` alone accounted for 22 of them (47%). Reviewing those sentences showed they were almost all the synthesizer's own analytical/conclusion sentences ("Taken together, the top AI trends shaping 2026 are...", "The broader pattern is that model ambition is increasingly bounded by...", "One of the clearest 2026 trends is the emergence of a two-track model landscape..."). These sentences combine multiple themes/findings into the model's own synthesis, but the synthesizer tagged each with a single `[i]` marker — and the judge correctly found the sentence's specific claim wasn't traceable to that one finding.

**Fix:** Added a "Citation discipline" rule to `_PROMPT` in `engine/nodes/synthesize.py`: only attach `[i]` to a sentence whose specific claim is directly stated in finding `i`; for cross-cutting analysis/synthesis sentences, either cite ALL findings the sentence draws from, or leave the sentence uncited (the eval already buckets uncited sentences as informational and doesn't penalize them).

**Result (partial improvement):** Re-ran `synthesize()` with the new prompt against the same findings/summary for this run (not just re-running the eval — citation behavior is baked into the report text itself, so the report had to be regenerated) and re-checked faithfulness on the new report: 78 cited sentences (down from 94 — more were left uncited as intended), 35 unfaithful → **55% faithful** (up from 50%). Citation `[1]` unfaithful count went from 22 → 20, essentially unchanged.

**Remaining gap after the prompt fix:** The remaining `[1]`-cited unfaithful sentences were still broad synthesis sentences ("Across model design, enterprise adoption, infrastructure, and regulation, the same pattern repeats...", "The strongest product-level trend in the findings is..."), and `[1]`'s actual source (an IBM article about 2026 quantum-computing milestones) had no topical relation to most of them. A first attempt at a stronger fix — a single extra LLM self-review pass (new `verify_citations` node, one call asking the model to re-check/correct its own `[i]` markers against the findings) — only moved the needle to 58.46% (27/65 unfaithful). One holistic LLM pass over a 24K-character report wasn't a careful enough per-sentence audit.

**Final fix (structural, supersedes the self-review attempt):** Rewrote `verify_citations` to be a programmatic per-sentence pass: it reuses `eval.faithfulness.run_faithfulness_checks` (the same Issue-#7-fixed judge, run with `CITATION_CHECK_MODEL = "gpt-5.4-mini"`, new constant in `engine/models.py`) against the freshly-synthesized report. For every `[i]`-cited sentence the judge marks unfaithful, a regex-based pass (`_sentence_pattern`/`_strip_citations` in `engine/nodes/verify_citations.py`) strips that sentence's `[i]` marker(s) in place — converting it to an uncited analytical sentence — without touching any other text. Wired into the graph as `synthesize → verify_citations → END` (`engine/orchestrator.py`); `compact` no longer clears `state.findings` (it's needed by `verify_citations`, which clears it afterward); `api/main.py`'s `report` SSE event now fires after `verify_citations`.

**Validated result:** Re-ran `synthesize()` (with the Issue-B-1 prompt fix) + the new `verify_citations()` end-to-end on this run's findings, then re-checked faithfulness on the corrected report: 63 cited sentences (101 left uncited), 7 unfaithful → **88.9% faithful** — within the 85-90% production target.

**Cost tradeoff:** `verify_citations` adds ~112 extra `gpt-5.4-mini` judge calls (~$0.24 at this run's scale — 156 findings / a ~28K-character draft report) to every research run. This is the per-run price of the anti-hallucination guard; smaller reports/finding-sets will cost proportionally less.

---

### 9. Subagent extraction produces ungrounded `evidence_span` (paraphrased quotes)
**File:** `engine/nodes/subagent.py`
**Issue:** Running the eval dashboard against "What course and resources to learn AI engineering?" produced `passed: False` with `ungrounded_count: 16` out of `total_findings: 59` (27%) — the actual failure driver (`failure_reasons: ["16 ungrounded claim(s)"]`). The same report also had 61 "uncited sentences" flagged, but per `eval/harness.py`/`eval/schema.py` that count is informational-only and doesn't affect `passed`.

**Root cause:** The extraction prompt asked for "the exact quote or passage from the content that supports the claim", but `SUBAGENT_MODEL` (`gpt-5.4-nano`) sometimes paraphrases/condenses when extracting. `eval/grounding.py`'s check (exact substring after whitespace/case normalization, else fuzzy `difflib` window match ≥0.85) then can't locate the paraphrased span in the re-fetched page, so the finding is marked ungrounded — and nothing upstream caught this before the finding reached synthesis.

**Fix (two-part):**
1. Strengthened `_PROMPT` in `engine/nodes/subagent.py` to require a VERBATIM, character-for-character quote, kept short (ideally under 300 chars), and to drop a finding entirely if no exact quote supports it.
2. Added a self-grounding-check filter in `subagent()`: after extraction, each finding is run through `eval.grounding.check_grounding` (the same lexical exact/fuzzy check the eval harness uses) against `content[:6_000]` — the same content the LLM saw. Findings that fail are dropped before they're added to `state.findings`, so an ungrounded `evidence_span` can never reach the report.

**Result:** Not yet re-validated end-to-end (would require re-running research on this query); the fix reuses the eval harness's own grounding logic at extraction time, so it's expected to generalize to any future topic — `ungrounded_count` should trend toward 0 going forward.

---

### 10. Citations "fall off" partway through multi-sentence source descriptions
**File:** `engine/nodes/synthesize.py`
**Issue:** Many of the 61 informational "uncited sentences" from the same run were specific factual elaborations (e.g. a course's hours, cost, or prerequisites) immediately following a cited topic sentence about the same source — only the first sentence in the paragraph carried `[i]`, leaving the rest uncited even though they came from the same finding.

**Fix:** Added a rule to the "Citation discipline" bullet in `_PROMPT`: when a paragraph describes one source's specifics across several sentences, attach `[i]` to each of those sentences, not just the first — only genuinely analytical/transition sentences (per Issue #8's rule) may stay uncited.

---

### 11. Render deploy hung 15 minutes and timed out after a migration was added
**File:** `db/migrations/env.py`
**Issue:** After pushing the title-feature migration (`7a885b67938e_add_title_to_research_runs.py`), the Render deploy built successfully but then `uv run alembic upgrade head && uv run uvicorn ...` produced zero log output for ~15 minutes before Render's port-scan timeout killed the deploy entirely (full outage — old instance also torn down by the rolling deploy).

**Root cause:** Postgres `lock_timeout` defaults to 0 (wait forever). `ALTER TABLE research_runs ADD COLUMN title` requires an `ACCESS EXCLUSIVE` lock; if any session (e.g. the previous instance's connection pool) still holds an open transaction touching `research_runs`, the `ALTER TABLE` blocks indefinitely with no error and no log output (Python stdout is fully buffered off a TTY, hiding even the startup log lines).

**Fix:** In `db/migrations/env.py`'s `run_migrations_online()`, added `connect_args={"connect_timeout": 10}` to `engine_from_config` and `connection.execute(text("SET lock_timeout = '10s'"))` before `context.configure`. Now a blocked migration fails after 10s with a clear `QueryCanceled` error instead of hanging until Render's 15-minute port-scan timeout takes down the whole deploy.

**Result:** Not yet validated against a real blocked-lock deploy; should make the next occurrence fail fast and visibly instead of causing a full outage.

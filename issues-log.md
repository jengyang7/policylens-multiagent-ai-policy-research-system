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
### 12. Eval judges hardcoded `ChatOpenAI`, would crash for non-OpenAI eval models
**Files:** `eval/faithfulness.py`, `eval/completeness.py`, `eval/relevance.py`
**Issue:** All three LLM-as-judge checks instantiated `ChatOpenAI(model=lead_model, ...)` directly. The eval dashboard's "Run Eval" previously let the user pick any model from `/models` (including Claude/Gemini) as the judge — selecting a non-OpenAI model would fail at the `ChatOpenAI(...)` call.

**Fix:** Routed all three through the existing provider-agnostic `make_chat_model()` + `structured_output_kwargs()` factory (already used in `engine/nodes/*.py`), with the established `assert isinstance(raw, dict)` pattern after `include_raw=True` calls. Also added a fixed `EVAL_MODEL = "claude-haiku-4-5"` in `engine/models.py` (`role_default_models()["eval"]`) and removed the per-run eval-model dropdown entirely — the judge model is now fixed (deliberately a different provider than `LEAD_MODEL`, for the same self-preference-bias reason as the debate skeptic) so "Quality Over Time" and "Community Average" stay apples-to-apples across runs and visitors.

---

### 13. Judge's Verdict card rendered raw markdown as plain text with no paragraph breaks
**File:** `web/app/page.tsx`
**Issue:** The verdict card rendered `debateVerdict.reasoning` in a plain `<div>{...}</div>`, so literal `**bold**` markers showed unrendered and the judge's several `**Label:** point` sentences ran together as one unbroken paragraph (unlike `DebateBubble`, which uses `ReactMarkdown`).

**Fix:** Render `debateVerdict.reasoning` through `<ReactMarkdown remarkPlugins={[remarkGfm]}>` with the same prose-styling classes as `DebateBubble`. Since the model emits its points as back-to-back `**Label:**` sentences with no blank lines, added `formatVerdictReasoning()` — a small client-side helper that inserts `\n\n` before each `**Label:**` marker so each point becomes its own paragraph. This also fixes rendering for verdicts already persisted in history/localStorage, since the normalization happens at render time.

---

### 14. `## References` section inconsistent with the report's own `[i]` citations
**File:** `engine/nodes/verify_citations.py`
**Issue:** A real exported report (`will-agi-arrive-before-2030`) had `[60][63][65][66][67]` cited inline in the body with NO matching entry in `## References`, and a `[47]` entry in `## References` that was never cited anywhere in the body. The synthesizer free-hands its own References list alongside ~30 inline citations across a ~24K-character report, and can both omit entries it used and list entries it never used — `verify_citations` (which already rewrites the body to strip unfaithful citations) left the LLM's References list untouched, so these inconsistencies shipped straight to the reader.

**Fix:** Added `_rebuild_references()`, called after citation-stripping: parses the synthesizer's own References list via `eval.report_parsing.parse_references()`, then rebuilds the section to exactly match the `[i]` markers still present in the body — keeping the LLM's title/url for indices it got right, dropping orphaned entries for indices no longer cited, and filling in any cited index the LLM's list omitted from `state.findings[i-1]['citation_url']`.

**Note on the "jump":** Sequence gaps like `[10] → [36] → [47]` are partly by design — `[i]` indexes 1-based into the FULL findings list (often 50-100+ items), and only a sparse subset gets cited, so most intermediate indices were simply never referenced. The fix above only addresses genuine *inconsistencies* (orphaned/missing entries), not the gaps themselves; full sequential renumbering (`[1],[2],[3]...` with no gaps) would require remapping every `[i]` marker throughout the body and would be a separate, larger change.

---

### 15. Stray `[Synthesis]` marker leaks from the debate transcript into the final report
**Files:** `engine/nodes/debate.py`, `engine/nodes/synthesize.py`, `engine/nodes/verify_citations.py`
**Issue:** The same exported report had `[Synthesis]` appearing twice in the **Final Report** body (e.g. "...the 'measurement crisis' where timelines oscillate based on narrow, noisy capability demonstrations [Synthesis]."). Debaters invent `[Synthesis]` as a pseudo-citation when referring to the research summary's own framing (not a specific numbered finding) — fine in the debate transcript, but the synthesizer copied it verbatim into the final report, where it has no `## References` entry and reads as a broken citation.

**Fix (three-part, defense in depth):** (1) Both debate prompts (`_ADVOCATE_PROMPT`/`_SKEPTIC_PROMPT`) now instruct debaters to describe the research summary's own framing in plain prose instead of inventing a bracket marker for it. (2) The synthesizer's prompt now explicitly forbids any non-numeric bracket citation in the report, even if one appears in the debate transcript it's given. (3) `verify_citations` strips any remaining `[NonNumeric]`-style marker from the body as a safety net (regex excludes `[Title](url)` markdown links).

---

### 16. `tests/test_eval_harness.py` mocked a `ChatOpenAI` attribute removed by Issue #12's fix
**File:** `tests/test_eval_harness.py`
**Issue:** Issue #12 routed `eval/faithfulness.py`, `eval/completeness.py`, and `eval/relevance.py` through `make_chat_model()` instead of `ChatOpenAI(...)` directly, but `_mock_structured_chain()` in the test file still did `monkeypatch.setattr(f"{module}.ChatOpenAI", ...)` — `AttributeError: module has no attribute 'ChatOpenAI'`, failing all 5 LLM-judge tests (`uv run pytest` → 5 failed).

**Fix:** Updated `_mock_structured_chain()` to patch `{module}.make_chat_model` instead. `structured_output_kwargs(lead_model)` needed no mocking — it's a pure function. `uv run pytest` → 61 passed.

---

### 17. Step cards stayed open through synthesis; no progress visible outside the page
**File:** `web/app/page.tsx`
**Issue:** Two related UX gaps: (1) the Research Plan / Debate / Verdict / Gap Research cards only auto-collapsed on specific later events (`debating`, `gap_plan`, `report`), so e.g. in a no-gap debate run the debate + verdict cards stayed expanded throughout the entire synthesis step; (2) there was no way to see run progress without the tab in focus — the browser tab title stayed static the whole run.

**Fix:** (1) The `synthesizing` SSE handler now collapses all four step cards at once — by the time synthesis starts, every earlier step (planning, research, debate, judging, gap research) is complete in every flow. (2) Added a `useEffect` that sets `document.title` to `"<progress>% · <current milestone> — MindClash"` while `phase === 'researching'`, reverting to `"MindClash"` otherwise — so the active step and progress are visible from the browser tab/taskbar.

---

### 18. `## References` section came back completely empty when the synthesizer used a numbered-list format
**Files:** `eval/report_parsing.py`, `engine/nodes/verify_citations.py`, `tests/test_phase2_smoke.py`
**Issue:** The synthesize prompt asks for References lines formatted exactly as `[1] [Title](url)`, but the model sometimes "prettifies" this into a standard numbered markdown list (`1. [Title](url)`) instead. `_REF_LINE_RE` only matched the literal `[1] [...]` bracket form, so `parse_references()` returned an empty `citation_map` for these reports. In `run_faithfulness_checks`, every `[i]`-cited sentence then failed to resolve its citation index ("citation [i] not found in the References section") and was judged automatically unfaithful — `verify_citations` stripped every `[i]` marker from the body in response, leaving `cited` empty in `_rebuild_references()`, which returned a bare `"\n\n## References\n"` heading with no entries. Net effect: a real report with `[1][2]`-style inline citations in the body shipped with a completely empty References section (and the inline markers themselves stripped too).

**Fix:** `_REF_LINE_RE` now also matches `N. [Title](url)` / `N) [Title](url)` in addition to `[N] [Title](url)`; `parse_references()` reads the index from whichever group matched. `_rebuild_references()` always normalizes back to the prescribed `[i] [Title](url)` form regardless of which input format was parsed. Also fixed a related edge case in `verify_citations`: the early-return path for `not report or not findings` previously returned `{"findings": []}` without a `"report"` key — since the SSE handler does `node_output.get("report", "")`, this silently sent an *empty* report to the frontend (discarding the synthesizer's perfectly-fine report) whenever a run produced zero findings. Now returns `{"report": report, "findings": []}` to pass the report through unchanged. Updated `tests/test_phase2_smoke.py::test_verify_citations_skips_when_no_findings_or_report` accordingly. `uv run pytest` → 61 passed.

---

### 19. Debate-mode UI polish batch (thinking bubbles, status line, card badges, verdict auto-collapse, log labels)
**File:** `web/app/page.tsx`
**Issue:** A batch of small debate-mode UX issues from user testing: (1) the "opposite agent is thinking" bubble showed even while the *current* agent was still streaming its turn, which read as if both agents were active at once; (2) the browser tab status line only showed coarse milestones (e.g. "Research"), not each individual step as it happened; (3) card titles for the Research Plan, Debate panel, and Judge's Verdict redundantly showed a spinner + "Thinking…" badge next to the title even though the card body already showed live progress; (4) the Judge's Verdict card stayed expanded once the debate-driven gap research round started searching, competing for attention with the new gap subagent cards; (5) "Thinking Steps" log entries for research subtasks were labeled "Research Execution" / "Follow-up Research Execution".

**Fix:** (1) The "thinking" `DebateBubble` for the non-active agent now only renders when `!debateStreaming` (i.e. between turns), not while a turn is actively streaming. (2) The `document.title` effect now uses `log[log.length - 1].label` (the latest Thinking Steps entry) instead of the coarse milestone, so every step is reflected in the tab title. (3) Removed the spinner/"Thinking…" badges next to the Research Plan, Debate panel, and Judge's Verdict card titles. (4) The `gap_plan` SSE handler now also calls `setVerdictExpanded(false)` so the verdict card auto-collapses once follow-up research starts. (5) Renamed log labels to "Research" / "Follow-up Research" (dropped "Execution").

---

### 20. `'NoneType' object has no attribute 'gap_questions'` crashed the run during "Identifying evidence gaps from the debate…"
**File:** `engine/nodes/debate.py`
**Issue:** `plan_gap_research` and `judge_debate` both call `llm.with_structured_output(Schema, include_raw=True)` and then unconditionally do `result = raw["parsed"]` followed by `result.<field>`. When the model occasionally replies without invoking the structured-output tool (no schema match), `raw["parsed"]` is `None`, so `result.gap_questions` raised `AttributeError: 'NoneType' object has no attribute 'gap_questions'` — caught by `_stream_graph`'s catch-all, which failed the whole run.

**Fix:** Both nodes now treat `raw["parsed"] is None` as a graceful fallback instead of crashing: `plan_gap_research` falls back to `gap_questions = []` (the prompt already defines "no material gaps" as a valid empty-list outcome, and `api/main.py`'s `plan_gap_research` handler already routes an empty list straight to `synthesizing`); `judge_debate` falls back to `winner="draw"` with `rows=[]`. `uv run pytest` → 61 passed.

---

### 21. References section still came back empty (and inline `[i]` markers all stripped) when the synthesizer wrote citations but no `## References` section at all
**File:** `eval/faithfulness.py`
**Issue:** Issue #18 fixed `_REF_LINE_RE` to also accept numbered-list reference formatting, but a real exported report (`will-ai-create-more-jobs-than-it-destroys-2026-06-15.md`, debate mode + gap research, 16 sources) still came back with a totally empty `## References` section — and this time **every** `[1]`/`[2]` inline citation was gone from the body too, not just the reference list. Root cause: the synthesizer's report had no `## References` section in *any* recognizable format, so `parse_references(report)` returned an empty `citation_map`. In `run_faithfulness_checks`, every cited sentence then had `citation_map.get(index) is None` for all its indices → zero `candidate_findings` → automatic `faithful=False` ("citation [i] not found in the References section") for literally every cited sentence in the report. `verify_citations` stripped every `[i]` marker in response, so `_rebuild_references` saw `cited = {}` and returned a bare `"\n\n## References\n"`.

**Fix:** `run_faithfulness_checks` now mirrors `_rebuild_references`'s existing fallback: when `citation_map.get(index)` is `None` but `1 <= index <= len(findings)`, it uses `findings[index - 1]` directly as the candidate finding (the same 1-indexed mapping `_rebuild_references` already relies on for indices the synthesizer's list omitted) instead of auto-failing. The judge now gets a real finding to check the sentence against, so a missing/malformed References section no longer wholesale-strips every citation in the report. `uv run pytest` → 61 passed.

---

### 22. References still empty — synthesizer produced zero `[i]` inline citations (root cause)
**File:** `engine/nodes/synthesize.py`
**Issue:** Issues #18–#21 treated `verify_citations` as the culprit (stripping citations that existed), but DB inspection confirmed the stored reports had ZERO `[i]` markers in the body and NO `## References` section whatsoever — meaning the **synthesizer itself never wrote any inline citations**. Root cause: `compact_findings` produces a prose narrative with source URLs embedded in text but NO pre-numbered `[1]`, `[2]`... anchors. When the synthesizer receives this prose as `findings_text`, it has no explicit `[i]` → URL mapping to work from, so it produces well-structured prose without any citation markers (especially with smaller/faster models like Claude Haiku 4.5 on long contexts with 100+ findings).

**Fix:** Added `_source_list(findings)`, which builds a deduplicated, numbered list of source URLs (first-occurrence order) and appends it to the compact summary in `findings_text`. The synthesizer now receives:

```
<prose summary>

Source URLs — use [i] from this list for your inline citations and References section:
[1] [https://source1.com](https://source1.com)
[2] [https://source2.com](https://source2.com)
...
```

This gives an explicit `[i]` → URL anchor in the exact format the synthesizer needs to reproduce in its own References section, without replacing the prose summary that conveys the actual research content. For runs where `summary` is empty (compact skipped), `_format_findings` already provides explicit `[i]` numbered findings, so no change needed for that path.

---

### 23. References section rendered as a single run-on paragraph instead of a vertical list
**Files:** `engine/nodes/verify_citations.py`
**Issue:** `_rebuild_references()` joined entries with `"\n".join(entries)`, producing a single `\n`-separated block. ReactMarkdown treats inline newlines within a block as a soft break (renders as a space), so all reference entries appeared on one line in the browser — e.g. `[1] [Title1](url1) [2] [Title2](url2) ...` instead of a stacked list.

**Fix:** Changed separator from `"\n".join(entries)` to `"\n\n".join(entries)` so each entry is a separate Markdown paragraph; ReactMarkdown renders each as its own `<p>`, giving the expected vertical list.

---

### 24. Follow-up Chat section had no quick-start prompts when chat history was empty
**File:** `web/app/page.tsx`
**Issue:** Users landed on a blank chat input with no cues about what to ask — first-time users especially didn't know where to start.

**Fix:** Added four suggested-question chips rendered above the input row only when `chatMessages.length === 0`. Clicking a chip calls `sendChat(questionText)` directly (using the `override` param added to `sendChat`). Chips are hidden as soon as the first message is sent. Questions: "Summarize the key findings in 3 bullet points", "What is the strongest evidence here?", "What are the main uncertainties or limitations?", "What should I research next?"

---

### 25. Final report body had no visual hierarchy for key data points
**File:** `engine/nodes/synthesize.py`
**Issue:** The synthesizer wrote well-structured prose but never used bold, so key statistics and critical conclusions were buried in paragraph text and hard to scan.

**Fix:** Added a formatting rule to `_PROMPT` instructing the model to use `**bold**` for key statistics, critical conclusions, and the most important findings — approximately 1–3 phrases per section, not entire sentences.
---

## Live-run forensics — "0 findings" cards with a full report (Singapore light-touch query, 2026-07-02)

### 26. All 6 plan-round subagents returned 0 validated findings; report silently built from the audit's gap round
**Files:** `engine/nodes/plan.py` (unfixed), `web/app/page.tsx` (unfixed)
**Issue:** A completed debate run showed all six Research Plan cards green with "0 findings" yet produced a full cited report. Decoding `checkpoint_writes` for the thread confirmed the counts were real, not a UI bug: all six plan-round subagent tasks wrote empty findings lists (each consumed 8k–17k input tokens reading pages but only ~90–140 output tokens — an empty `FindingList` per page). The evidence audit then emitted 3 gap questions; the gap round returned 26 + 11 + 0 findings, and `recompact → synthesize` built the entire report from those. The debate itself ran over an empty summary — four turns of both models eloquently agreeing there was no evidence.
**Root cause:** The planner decomposed the (deliberately debate-bait) query into analytical/comparative essay questions ("How do enforcement outcomes and compliance effectiveness metrics compare between…") that no single web page directly answers. The extraction prompt correctly only permits findings directly supported by the page with a verbatim quote → every page yields an empty list. The audit's rewritten *concrete, factual* questions ("What are the core principles, mechanisms, and enforcement tools of Singapore's Model AI Governance Framework…") succeeded with the identical subagent code — the only variable was question phrasing. Two UI behaviors masked the failure: the `done` handler force-marks all pending cards green (a total round-1 wipeout renders as six calm green cards), and `loadPublicRun` hardcodes `findingsCount: 0` for restored runs.
**Fix:** Not yet — remaining TODO: (1) planner prompt rule requiring *evidence-seeking* sub-questions (find facts/documents about X; leave comparison to synthesis); (2) subagent self-reformulation retry on zero findings; (3) style zero-finding cards amber instead of green; (4) `loadPublicRun` should carry real counts. Diagnosis documented here; issues #27–#29 fix what the same run exposed downstream.

---

### 27. One page monopolized a subtask's evidence — whole report cited exactly 2 URLs, one a low-authority aggregator
**File:** `engine/nodes/subagent.py`
**Issue:** The same run's report rested on exactly two references: all 26 findings behind `[1]` came from a single `aigovernance.com` page, all 11 behind `[2]` from one Fisher Phillips blog post. Worse, `[1]` is a third-party compliance-aggregator site (verified by fetching it) whose "What Your Organization Must Do" section rewrites Singapore's *voluntary* Model AI Governance Framework as imperative obligations — and the report reproduced that faithfully ("Organisations must appoint a named AI governance owner… as a condition of moving any model into production"), even contradicting its own (correct) "not legally binding" sentence. Every one of those sentences passed the grounding check, because grounding guarantees *the quote appears on the cited page* — not that the page is right.
**Root cause:** The search-spec loop did `if findings: break` — the first productive URL ends the search, so one page supplies everything. Generic web search ran before the official-domain specs, so an SEO-friendly aggregator was extracted before `imda.gov.sg` ever got searched.
**Fix (three-part):** (1) `_MAX_FINDINGS_PER_URL = 8` — one page can't monopolize a subtask. (2) Replaced the break condition with "stop once findings span ≥ `_MIN_SOURCE_DOMAINS = 2` distinct domains (or specs run out)" — a single-domain first hit keeps the search going. (3) Authority preference: for AI-policy questions the official-domain specs now run FIRST, and `_prioritized_results()` stable-sorts every result batch so official/primary domains are fetched and extracted before aggregators. Covered by `tests/test_subagent_source_diversity.py` (per-URL cap, two-domain stop, priority ordering, spec ordering).

---

### 28. Evidence audit green-lit a 2-domain evidence base despite its "source diversity" mandate
**File:** `engine/nodes/evidence_audit.py`
**Issue:** The audit prompt asks the LLM to check "authority and diversity of sources", but given a bare URL list it approved a report resting entirely on two pages (one non-official). The semantic judge doesn't reliably infer diversity from a URL list.
**Fix:** Made the signal deterministic instead of inferred: coverage lines now read `- <question>: N findings from D source domain(s)`; when all findings span fewer than `MIN_DISTINCT_DOMAINS = 3` distinct domains, the source list gets an explicit `WARNING: all N findings come from only D distinct domain(s)…` block; and the system prompt now states that evidence from fewer than 3 domains, or a subtask answered entirely by a single non-official source, is normally insufficient and should produce gap questions targeting primary/official sources. Tests added to `tests/test_evidence_audit.py` for the warning and no-warning paths.

---

### 29. `verify_citations` sentence removal left empty section headings and orphaned fragments in the published report
**File:** `engine/nodes/verify_citations.py`
**Issue:** The exported report had a completely empty `## De Facto Enforcement and Regulatory Embedding` section, a `## Conclusion` opening blank, a one-sentence `## Convergence and Divergence Across APAC` section, and a subject-less fragment ("The recommendations call for sector-specific regulations to continue [2]." — its lead sentence was deleted). The per-sentence faithfulness/coverage removal (43 + 33 judge calls on this run) worked as designed, but `_cleanup_markdown_body` repaired lists and spacing without noticing headings whose entire section had been cut.
**Fix:** Added `_remove_empty_sections()` to the cleanup pass: a heading is dropped when only blank lines separate it from the next heading of the same or higher level (or end of body), iterated to a fixpoint so parents whose only children were empty sections also collapse; a heading over a deeper subtree with real content is kept. Validated against the actual damaged report: the empty "De Facto Enforcement" heading is removed, "Conclusion" (which still had a real paragraph) is kept. Deliberately does NOT auto-delete orphaned sentence fragments — a regex can't distinguish a damaged fragment from a legitimately short paragraph; issue #27's diversity fixes attack that upstream by giving the verifier fewer bad sentences to cut.

---

### 30. One PDF/binary URL crashed the entire research run with a bs4 "markup rejected by the parser" error
**Files:** `engine/tools/fetch.py`, `engine/nodes/subagent.py`
**Issue:** The first research run after issue #27's fix bounced back to the home page with `The markup you provided was rejected by the parser ... AssertionError: expected name token at '<![...binary...'`. That message is BeautifulSoup's `ParserRejectedMarkup` — `fetch()` downloaded a binary payload (PDF or similar) and fed it to `html.parser`.
**Root cause:** In `_fetch_local`, only the HTTP request was inside the try/except; `BeautifulSoup(resp.text, "html.parser")` was outside it, so a binary response raised straight through — violating the function's own "returns empty string on network/extraction errors" contract. In `subagent.py`, the `fetch(url)` call also sat outside the extraction try/except, so the exception escaped the node and `_stream_graph`'s catch-all failed the whole run. Latent since Phase 1, but issue #27's fix made it near-certain to trigger: subagents now fetch more URLs per subtask and prioritize official regulator domains — which serve many PDF links.
**Fix (two-part):** (1) `_fetch_local` now skips responses whose `Content-Type` is neither HTML nor text (PDFs, images, octet-stream) and wraps the parse/markdownify block in try/except returning `""`, honoring the documented contract. (2) Defense in depth in `subagent.py`: the `fetch(url)` call itself is wrapped in try/except → `continue`, so one bad page can never kill a subtask again. Regression tests added in `tests/test_fetch_tool.py` (PDF content-type skipped; binary-garbage-as-HTML returns `""`).

---

### 31. `verify_citations` spacing normalizer mangled dotted abbreviations ("U.S." → "U. S.") throughout the report
**File:** `engine/nodes/verify_citations.py`
**Issue:** The post-fix Singapore report rendered "U. S. chip export restrictions", "U. S. Treasury Department", etc. — every occurrence of `U.S.` split by a space. `_MISSING_SENTENCE_SPACE_RE` (`([.!?])(?=[A-Z(])`) repairs missing sentence-boundary spaces (`harm.The` → `harm. The`) but matched *any* period followed by a capital, including the interior of dotted abbreviations. Pre-existing, but the geopolitics-heavy report made it glaring.
**Fix:** Added a lookbehind requiring a lowercase letter, digit, or closing bracket/paren before the punctuation — `(?<=[a-z0-9)\]])` — so `harm.The`, `2026.The`, and `[1].However` are still repaired while a period after a single capital (the abbreviation pattern) never matches. First attempt used only `[a-z0-9]` and broke the existing `[1].However` test — the bracket class is load-bearing. Regression test added covering both directions.

---

### 32. LinkedIn posts cited as references — login-walled and weak authority
**File:** `engine/nodes/subagent.py`
**Issue:** Two of the seven references in the post-fix Singapore report were LinkedIn posts, carrying key claims (Chinese AI firms relocating to bypass chip export restrictions). LinkedIn is login-walled, so the eval harness's independent re-fetch of `citation_url` gets a login page and marks the findings ungrounded (same failure mode as issue #9's YouTube case) — and a personal post is weak authority for policy claims regardless. The debate skeptic even called it out ("relies entirely on a single LinkedIn post").
**Fix:** Added `linkedin.com` to `_SKIP_FINDING_DOMAINS`, and switched `_should_skip_source` from exact-hostname set membership to suffix matching via `_domain()`/`_matches_domain()` — `m.youtube.com`, `sg.linkedin.com` etc. are now covered without enumerating subdomains.

---

### 33. Official regulator sources still absent from reports — their PDFs were fetched then skipped
**Files:** `engine/tools/fetch.py`, `pyproject.toml`
**Issue:** Despite issue #27's official-domains-first search ordering, the post-fix report cited zero official sources (`imda.gov.sg`, `pdpc.gov.sg`) — consultancies, vendors, and news filled every slot. Likely contributor: regulator sites serve much of their substance as PDFs (frameworks, guidance, consultation papers), and issue #30's fix skipped all non-HTML content types, so the highest-authority pages yielded no extractable text.
**Fix:** Added `pypdf` and a `_extract_pdf_text()` path in `fetch()`: responses with a PDF content-type (or `.pdf` URL) get real text extraction (page-by-page, stops at `max_chars`, returns `""` for encrypted/malformed/scanned PDFs). The grounding eval re-fetches citations through this same function, so PDF evidence spans stay verifiable end-to-end. Other binaries (images, archives) are still skipped. **Note:** requires `uv lock && uv sync` (pyproject change).

---

### 34. Search providers ignore `include_domains` — official-source specs silently returned aggregators
**File:** `engine/tools/search.py`
**Issue:** Despite issue #27's official-domains-first spec ordering and issue #33's PDF extraction, reports still contained zero `imda.gov.sg`/`pdpc.gov.sg` sources. Live diagnosis of the exact spec the subagent runs showed why: a Tavily query restricted to `["imda.gov.sg","pdpc.gov.sg","mas.gov.sg","mddi.gov.sg"]` returned `digital.nemko.com`, `verifywise.ai`, and `securiti.ai` — 3 of 4 results outside the requested domains. Providers treat `include_domains` as a ranking hint, not a filter, so the "official-first" spec was feeding the subagent the same aggregators as generic search (nemko was literally result #1 — and reference [1] of the previous report).
**Fix:** `search()` now enforces `include_domains` client-side: results whose hostname doesn't suffix-match a requested domain are dropped after provider merge, provider-agnostically. An emptied result set just falls through to the subagent's next spec. Validated live: the Singapore official spec now returns exactly one result — the PDPC Model AI Governance Framework PDF, which (thanks to #33) yields 8,000 chars of primary-source text.

---

### 35. Orphan fragments after sentence removal — deletion-only LLM coherence pass
**File:** `engine/nodes/verify_citations.py`
**Issue:** The residual damage class documented in #29: sentence-level removal strands what remains — paragraphs opening mid-argument ("On the other, at least some jurisdictions…"), ordinal sequences with missing members ("Third," with no First/Second), lead-ins whose lists were deleted ("The findings leave several important issues unresolved." followed by nothing), and one-fragment Conclusions. Deterministic rules can't reliably distinguish a stranded fragment from a legitimately short paragraph ("The evidence consistently shows…" reads fine; "On the other," doesn't — both are single-sentence remnants whose lead was cut).
**Fix:** Added a **deletion-only** coherence pass after the removal passes (only when something was actually removed): one `CITATION_CHECK_MODEL` call nominates ≤8 exact snippets to DELETE; each is applied through the same exact-match `_remove_sentence` machinery as the faithfulness pass, so the model cannot rewrite or inject text — a hallucinated snippet simply doesn't match, and heading deletions are refused outright. Sections emptied by the pass are then dropped by #29's `_remove_empty_sections`. Fail-open: any error (no key, parse failure) leaves the body unchanged. Also normalizes the stranded single leading space left when a paragraph's first sentence is removed. Cost: one extra small-model call per run with removals. Tests: `tests/conftest.py` autouse fixture stubs the chain as a no-op for all existing verify_citations tests; dedicated tests cover matching/hallucinated/heading snippets and the stub → emptied-heading cascade.

---

### 36. Total zero-findings run: bot-walled sources and content windows that never reached the real content
**Files:** `engine/tools/fetch.py`, `engine/nodes/subagent.py`
**Issue:** A run produced 0 findings across all 9 subagents (plan + gap round) while consuming 68k subagent input tokens — pages were read, everything extracted empty. Live tracing showed the pages themselves were the problem: (a) `lexology.com` and `insideglobaltech.com` (top-quality legal sources) return **403 to non-browser clients** — a browser User-Agent doesn't help, they use real bot detection; (b) one source timed out; (c) the one page that fetched (662 KB marketing-heavy blog) buries its substantive content at char ~10,500 of the cleaned text — past both the fetch cap (8K) and extraction window (6K), so the extractor literally never saw the content it was asked about.
**Fix (two-part):** (1) `fetch()` falls back to **Tavily's extract endpoint** when the direct fetch fails (network error, 4xx bot wall, unparseable markup, empty JS shell) — Tavily fetches through its own infrastructure. Because the grounding eval re-fetches through the same `fetch()`, extraction and verification stay consistent. Deliberate binary skips (images) do NOT fall back. (2) Raised `_MAX_CHARS` 8K→24K and the extraction window to `_EXTRACT_CHARS` = 16K (safely inside grounding's 40K re-fetch slice). Validated live: the 403 site now yields 20K chars and 8 findings.

---

### 37. gpt-5.4-nano is erratic on long pages — returned an empty FindingList on the official PDPC framework PDF
**Files:** `engine/nodes/subagent.py`, `engine/models.py`
**Issue:** Even with content access fixed, extraction still returned zero: nano called the FindingList tool with an explicitly empty list (18 output tokens) against the **PDPC Model AI Governance Framework itself** for a question the document directly answers. Window sweep showed nano is erratic, not monotonic (0 findings at 3K/6K/8K, 6 at 12K, 2 at 16K), while `gpt-5.4-mini` extracted 11 findings from the identical input. The strict verbatim-quote extraction prompt plus long content makes nano give up rather than hunt for quotes.
**Fix:** Per-page **tiered extraction escalation**: nano runs first; if it returns zero findings on a page with ≥2,000 chars of content (`_ESCALATION_MIN_CHARS`), that one page is retried on `SUBAGENT_ESCALATION_MODEL` (`gpt-5.4-mini`) before being given up. Bounded to one extra call per substantial-but-empty page; token usage now tracked per model. Validated live on the previously-0-findings gap question: 16 findings from 2 domains, with usage showing nano 54 output tokens (all empty) and mini 1,711 (all rescues). This also preserves the cheap-bulk-reading architecture instead of just upgrading the default subagent model.

---

### 38. Planner regenerated analytical essay-question subtasks (issue #26 root cause) — prompt fix applied
**File:** `engine/nodes/plan.py`
**Issue:** The zero-findings run's plan questions were word-for-word the same analytical/comparative questions as issue #26's run (same lead model, temperature 0, same query): "How do enforcement outcomes and compliance effectiveness metrics compare between…" — questions no single web page answers, guaranteeing sparse extraction regardless of downstream fixes.
**Fix:** Added the CRITICAL evidence-seeking rule to the planner prompt: every sub-question must ask for facts, rules, documents, or events a single page could state directly; comparative/evaluative questions must be split into one factual sub-question per side, with the comparing left to synthesis. Includes an explicit BAD/GOOD example pair taken from this failure. Closes the main TODO from issue #26 (the UI amber-card and `loadPublicRun` items remain open).

---

### 39. Debate-transcript finding-index citations ([98], [134]) shipped as broken markers in the final report
**File:** `engine/nodes/verify_citations.py`
**Issue:** The first fully-productive run (182 findings, 23 unique sources) shipped a report citing `[98]` and `[134]` with a References list that only goes to [23]. Two numbering systems collided: debaters cite per-FINDING indices (1..182, from the numbered findings in their prompt), while the report numbers unique SOURCES (1..23 via `_source_list`). The synthesizer copied finding-index markers from the debate transcript; `_canonicalize_citations` couldn't resolve an index above the source count so it passed the marker through unchanged, `_rebuild_references` skipped it, and the reader got a citation with no References entry. (The faithfulness judge masked the bug: its `findings[i-1]` fallback treated [98] as finding #98 — a valid finding — so the sentence survived verification.)
**Fix:** `_canonicalize_citations` now resolves markers above the source count through `findings[index-1]["citation_url"]` to that source's canonical number (semantically correct — that IS what the debater meant), and strips markers that resolve to nothing rather than shipping them broken. Regression test covers both the remap and the strip.

---

### 40. Alembic autogenerate emitted (and briefly applied) a DROP of the LlamaIndex vector-store table
**Files:** `db/migrations/env.py`, `db/migrations/versions/9ef689784046_*.py`
**Issue:** Generating the `authority_score` migration, autogenerate also emitted `drop_table('data_report_chunks')` — LlamaIndex's PGVectorStore table lives outside SQLAlchemy metadata, so Alembic saw it as "removed". The migration was applied before the drop was noticed, deleting all RAG embeddings. (`env.py` already excluded LangGraph's checkpoint tables from autogenerate, but not LlamaIndex's.)
**Fix & recovery:** Removed the drop/create from the migration (now add-column only); extended `env.py`'s `_include_object` to exclude all `data_`-prefixed tables and their indexes from autogenerate; rebuilt the vector store by re-embedding all 5 persisted reports via `rag.embed_and_store` (verified: 333 chunks restored). No permanent loss — the store is fully derivable from persisted reports.
**Rule:** always read an autogenerated migration before `upgrade head`; any table created by a library at runtime (LangGraph, LlamaIndex) must be in env.py's exclusion filter.

---

### 41. New eval metric: source authority (grounding proves the quote, not the source)
**Files:** `eval/source_authority.py`, `eval/schema.py`, `eval/harness.py`, `db/models.py` (+ migration), `api/main.py`, `web/app/components/EvalDashboard.tsx`
**Motivation:** Issue #27 showed a report can score 100% grounding while resting on a content farm that misstates the law — grounding verifies quote fidelity, not source quality. The eval had no metric for the evidence base itself.
**Implementation:** Every unique cited domain is tiered — primary (governments/regulators/courts/standards bodies/IGOs; deterministic rules for .gov/.edu/.int/.mil, europa.eu, OECD, ISO, …), secondary (established law firms, major news, academic, think tanks; one LLM-judge call), other (vendor marketing, SEO/content sites, personal blogs, unrecognized — the judge is instructed never to guess upward). Score = weighted mean over unique domains (1.0/0.6/0.2) so one prolific page can't inflate it. Informational — never fails a run. Persisted as `eval_reports.authority_score`, shown in the dashboard detail panel with per-domain tier badges, included in `/eval/summary` aggregates.
**Validation (before/after on the same query):** run 1: 40% (0 primary / 1 secondary / 1 other); latest run: 43% (0 primary / 12 secondary / 9 other). Deliberately harsh and honest: secondary authority improved massively, but the metric correctly surfaces that the evidence base still contains zero primary sources. Known limitation: the judge classifies by domain name only, so unrecognized domains fall conservatively to "other" (e.g. insideglobaltech.com — actually Covington & Burling's blog — tiers as "other").

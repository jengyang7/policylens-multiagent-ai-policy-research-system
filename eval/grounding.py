"""Citation-grounding check (Phase 4 eval harness): is each Finding's
evidence_span actually present in the page fetched from its citation_url?

Threshold rationale:
  - `evidence_span` is extracted by the subagent LLM as a quote/passage from
    the same fetch() output the subagent saw, but LLMs commonly normalize
    whitespace, smart-quotes, or minor punctuation when copying.
  - fetch() re-renders HTML through markdownify, which can reflow line
    breaks and add markdown punctuation. After whitespace normalization,
    exact substring match catches the majority of cases.
  - The fuzzy tier is a safety net for minor paraphrasing, ellipsis joins of
    non-contiguous spans, or unicode quote/dash differences. 0.85 tolerates
    roughly 15% character-level edits — enough for light paraphrasing without
    letting a merely topically-similar passage (~0.5-0.7 ratio) pass.
  - Spans shorter than _MIN_LEN_FOR_FUZZY are not fuzzy-matched: short strings
    produce unreliably high SequenceMatcher ratios against arbitrary windows.
"""
from __future__ import annotations

import asyncio
import difflib
import re
from collections.abc import Callable
from functools import partial

from engine.state import SubtaskFinding
from engine.tools.fetch import fetch as default_fetch
from eval.schema import GroundingResult

FUZZY_MATCH_THRESHOLD = 0.85
_MIN_LEN_FOR_FUZZY = 20

# Grounding only does string matching (no LLM context window to bound), so allow
# a much larger page slice than the subagent's extraction window — a match
# further down a long page shouldn't register as ungrounded. Must stay a
# superset of engine/nodes/subagent.py's _EXTRACT_CHARS.
_GROUNDING_MAX_CHARS = 40_000
_default_grounding_fetch: Callable[[str], str] = partial(
    default_fetch, max_chars=_GROUNDING_MAX_CHARS
)

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Lowercase + collapse all whitespace runs to single spaces + strip."""
    return _WHITESPACE_RE.sub(" ", text.strip().lower())


def _best_window_ratio(needle: str, haystack: str) -> float:
    """Best difflib ratio between `needle` and any similarly-sized window of `haystack`.

    A plain SequenceMatcher(needle, haystack).ratio() collapses toward 0 for a
    short needle inside a long haystack, so we slide windows of comparable
    length over the haystack and take the best ratio.
    """
    if not needle or not haystack:
        return 0.0
    if len(haystack) <= len(needle):
        return difflib.SequenceMatcher(None, needle, haystack).ratio()

    n = len(needle)
    step = max(n // 4, 20)
    window_sizes = sorted({
        min(len(haystack), max(1, int(n * factor))) for factor in (0.8, 1.0, 1.2, 1.5)
    })

    best = 0.0
    for window_size in window_sizes:
        last_start = len(haystack) - window_size
        starts = list(range(0, last_start + 1, step))
        if starts[-1] != last_start:
            starts.append(last_start)
        for start in starts:
            window = haystack[start : start + window_size]
            ratio = difflib.SequenceMatcher(None, needle, window).ratio()
            if ratio > best:
                best = ratio
                if best == 1.0:
                    return best
    return best


def check_grounding(finding: SubtaskFinding, fetched_content: str) -> GroundingResult:
    """Determine whether finding['evidence_span'] is grounded in fetched_content."""
    base = {
        "subtask": finding["subtask"],
        "claim": finding["claim"],
        "evidence_span": finding["evidence_span"],
        "citation_url": finding["citation_url"],
    }

    if not fetched_content:
        return GroundingResult(
            **base,
            grounded=False,
            similarity=0.0,
            method="fetch_failed",
            fetch_chars=0,
            note="fetch returned empty content",
        )

    needle = _normalize(finding["evidence_span"])
    haystack = _normalize(fetched_content)
    fetch_chars = len(fetched_content)

    if needle and needle in haystack:
        return GroundingResult(
            **base, grounded=True, similarity=1.0, method="exact", fetch_chars=fetch_chars
        )

    if len(needle) < _MIN_LEN_FOR_FUZZY:
        return GroundingResult(
            **base,
            grounded=False,
            similarity=0.0,
            method="fuzzy_window",
            fetch_chars=fetch_chars,
            note="evidence_span too short for fuzzy matching and not found exactly",
        )

    similarity = _best_window_ratio(needle, haystack)
    grounded = similarity >= FUZZY_MATCH_THRESHOLD
    return GroundingResult(
        **base,
        grounded=grounded,
        similarity=similarity,
        method="fuzzy_window",
        fetch_chars=fetch_chars,
        note="" if grounded else "best fuzzy match below threshold",
    )


async def run_grounding_checks(
    findings: list[SubtaskFinding],
    fetch_fn: Callable[[str], str] = _default_grounding_fetch,
) -> list[GroundingResult]:
    """Check grounding for every finding, fetching each unique citation_url once.

    Returns results in the same order as `findings`.
    """
    unique_urls = list(dict.fromkeys(f["citation_url"] for f in findings))
    fetched: dict[str, str] = {}

    async def _fetch_one(url: str) -> None:
        fetched[url] = await asyncio.to_thread(fetch_fn, url)

    await asyncio.gather(*(_fetch_one(url) for url in unique_urls))

    return [check_grounding(f, fetched[f["citation_url"]]) for f in findings]

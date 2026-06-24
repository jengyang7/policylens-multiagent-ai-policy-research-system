"""CITATION VERIFICATION node (anti-hallucination guard, runs after synthesize):

Re-runs the same per-sentence faithfulness judge used by the eval harness
(eval.faithfulness.run_faithfulness_checks) against the freshly synthesized
report. Any [i]-cited sentence the judge can't verify against the Finding(s)
behind reference [i] is removed from the published report. Keeping the sentence
but stripping its citation would hide an unsupported factual claim from the
eval harness as "uncited" prose.

Also rebuilds the ## References section to exactly match the [i] markers
still present in the body afterward — the synthesizer's own References list
can omit citations it used in the body or list ones it never used (the LLM
free-hands this list alongside ~100 inline citations). Dropped citations no
longer get an orphaned reference entry, and any cited [i] missing from the
LLM's list is filled in from `state.findings`.

Also strips stray non-numeric bracket markers (e.g. "[Synthesis]") that can
leak from the debate transcript into the synthesized report — these aren't
valid [i] citations and have no References entry, so they read as broken
citations to a reader.

Clears state.findings afterward — this is the last node that needs the raw
findings list.
"""
from __future__ import annotations

import re

from engine.models import CITATION_CHECK_MODEL
from engine.state import ResearchState
from eval.faithfulness import run_faithfulness_checks
from eval.report_parsing import (
    extract_citation_indices,
    parse_references,
    split_body_and_references,
    split_sentences,
)

_CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")
_DANGLING_SPACE_RE = re.compile(r"[ \t]+(?=[.,;:!?])|  +")
# Stray non-numeric bracket markers like [Synthesis] sometimes leak from the
# debate transcript into the synthesized report — not a markdown link (no
# trailing `(url)`) and not a valid [i] citation, so strip them too.
_NON_NUMERIC_MARKER_RE = re.compile(r"\[[A-Za-z][^\]]*\](?!\()")


def _sentence_pattern(sentence: str) -> re.Pattern[str]:
    """Build a regex matching `sentence`'s occurrence in the report body.

    split_sentences() joins a paragraph's lines with single spaces, so the
    sentence text may not appear verbatim in the (possibly line-wrapped)
    body — replace whitespace runs with `\\s+` to tolerate that.
    """
    parts = re.split(r"(\s+)", sentence)
    return re.compile("".join(r"\s+" if p.isspace() else re.escape(p) for p in parts))


def _remove_sentence(body: str, sentence: str, start: int) -> tuple[str, int]:
    """Remove `sentence`'s occurrence in `body` (search from `start`).

    Returns (new_body, new_search_offset). No-op if the sentence can't be found.
    """
    match = _sentence_pattern(sentence).search(body, start)
    if not match:
        return body, start
    new_body = body[: match.start()] + body[match.end() :]
    return new_body, match.start()


def _rebuild_references(
    body: str, references: str, findings: list[dict[str, str]]
) -> str:
    """Rebuild '## References' so it exactly matches the [i] markers left in `body`.

    Drops entries for indices no longer cited (orphans) and adds entries for
    cited indices the synthesizer's own list omitted, sourced from `findings`
    (1-indexed, matching the [i] numbering used throughout synthesis).
    """
    cited = sorted({int(n) for n in _CITATION_MARKER_RE.findall(body)})
    if not cited:
        return "\n\n## References\n"
    parsed = parse_references(references)
    entries: list[str] = []
    for i in cited:
        if i in parsed:
            ref = parsed[i]
            entries.append(f"[{i}] [{ref.title}]({ref.url})")
        elif 1 <= i <= len(findings):
            url = findings[i - 1]["citation_url"]
            entries.append(f"[{i}] [{url}]({url})")
    # Double-newline between entries so ReactMarkdown renders each as its own
    # paragraph (vertically stacked) rather than a single run-on line.
    return "\n\n## References\n\n" + "\n\n".join(entries) + "\n"


async def verify_citations(state: ResearchState) -> dict[str, object]:
    """Strip [i] citations the faithfulness judge can't verify (verify_citations node)."""
    report = state.get("report", "")
    findings = state.get("findings", [])
    if not report or not findings:
        return {"report": report, "findings": []}

    verdicts, _uncited, token_usage = await run_faithfulness_checks(
        report, findings, lead_model=CITATION_CHECK_MODEL
    )

    body, references = split_body_and_references(report)
    cited_sentences = [s for s, _section in split_sentences(body) if extract_citation_indices(s)]

    cursor = 0
    for sentence, verdict in zip(cited_sentences, verdicts):
        if not verdict.faithful:
            body, cursor = _remove_sentence(body, sentence, cursor)

    body = _NON_NUMERIC_MARKER_RE.sub("", body)
    body = _DANGLING_SPACE_RE.sub(lambda m: "" if m.group().strip() == "" else " ", body)

    references = _rebuild_references(body, references, findings)  # type: ignore[arg-type]
    return {"report": body + references, "findings": [], "token_usage": token_usage}

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
LLM's list is filled in from the source map implied by `state.findings`.

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
from eval.citation_coverage import run_citation_coverage_check
from eval.faithfulness import run_faithfulness_checks
from eval.report_parsing import (
    extract_citation_indices,
    parse_references,
    split_body_and_references,
    split_sentences,
)

_CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")
_DANGLING_SPACE_RE = re.compile(r"[ \t]+(?=[.,;:!?])|  +")
_MISSING_SENTENCE_SPACE_RE = re.compile(r"([.!?])(?=(?:[A-Z(]))")
_EMPTY_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s*$")
_ORDERED_LIST_ITEM_RE = re.compile(r"^(\s*)(\d+)([.)])(\s+)(\S.*)$")
_BOLD_RE = re.compile(r"(?<!\*)\*\*([^*\n]+)\*\*(?!\*)|__([^_\n]+)__")
_HEADING_PREFIX_RE = re.compile(
    r"^(#{1,6}\s+)(?:(?:[IVXLCDM]+|\d+|[A-Z])[\.)]\s+)(.+)$",
    re.IGNORECASE,
)
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
    cited indices the synthesizer's own list omitted, sourced from the unique
    source order in `findings`.
    """
    cited = sorted({int(n) for n in _CITATION_MARKER_RE.findall(body)})
    if not cited:
        return "\n\n## References\n"
    parsed = parse_references(references)
    source_urls = list(dict.fromkeys(f["citation_url"] for f in findings))
    entries: list[str] = []
    for i in cited:
        if i in parsed:
            ref = parsed[i]
            url = ref.url
            title = ref.title
        elif 1 <= i <= len(source_urls):
            url = source_urls[i - 1]
            title = url
        else:
            continue
        entries.append(f"[{i}] [{title}]({url})")
    # Double-newline between entries so ReactMarkdown renders each as its own
    # paragraph (vertically stacked) rather than a single run-on line.
    return "\n\n## References\n\n" + "\n\n".join(entries) + "\n"


def _dedupe_citation_run(match: re.Match[str]) -> str:
    """Collapse repeated markers inside a contiguous citation run."""
    seen: set[str] = set()
    markers: list[str] = []
    for marker in _CITATION_MARKER_RE.findall(match.group(0)):
        if marker in seen:
            continue
        seen.add(marker)
        markers.append(f"[{marker}]")
    return "".join(markers)


def _canonicalize_citations(
    body: str, references: str, findings: list[dict[str, str]]
) -> str:
    """Map duplicate per-finding citations onto one canonical source number.

    Older reports and occasional LLM drift can assign multiple citation numbers
    to the same URL, producing References entries like "[1], [2], [5] ...".
    Future reports use one source number per unique URL; this cleanup enforces
    that invariant before faithfulness checks and final reference rebuilding.
    """
    source_urls = list(dict.fromkeys(f["citation_url"] for f in findings))
    canonical_by_url = {url: i + 1 for i, url in enumerate(source_urls)}
    parsed = parse_references(references)

    # Preserve citations to URLs that appear only in the model-written
    # References section, but still collapse duplicate indices for that URL.
    for index in sorted(parsed):
        url = parsed[index].url
        canonical_by_url.setdefault(url, index)

    def replace_marker(match: re.Match[str]) -> str:
        index = int(match.group(1))
        url: str | None = None
        if index in parsed:
            url = parsed[index].url
        elif 1 <= index <= len(source_urls):
            url = source_urls[index - 1]
        if url is None:
            return match.group(0)
        return f"[{canonical_by_url.get(url, index)}]"

    body = _CITATION_MARKER_RE.sub(replace_marker, body)
    return re.sub(r"(?:\[\d+\]){2,}", _dedupe_citation_run, body)


def _remove_markdown_bold(text: str) -> str:
    """Remove inline bold markers from dense reports while preserving the text."""
    return _BOLD_RE.sub(lambda m: m.group(1) or m.group(2) or "", text)


def _remove_heading_outline_prefix(line: str) -> str:
    """Drop outline prefixes from markdown headings, e.g. '## IX. Risk'."""
    previous = None
    while previous != line:
        previous = line
        line = _HEADING_PREFIX_RE.sub(r"\1\2", line)
    return line


def _normalize_report_spacing(text: str) -> str:
    """Normalize common LLM punctuation artifacts before sentence-level checks."""
    text = _DANGLING_SPACE_RE.sub(lambda m: "" if m.group().strip() == "" else " ", text)
    text = _MISSING_SENTENCE_SPACE_RE.sub(r"\1 ", text)
    return text


def _cleanup_markdown_body(body: str) -> str:
    """Clean artifacts left after unsupported cited sentences are removed."""
    cleaned_lines: list[str] = []
    next_ordered_number = 1
    in_ordered_list = False

    for line in body.splitlines():
        if line.lstrip().startswith("#"):
            line = _remove_heading_outline_prefix(line)

        if _EMPTY_LIST_ITEM_RE.match(line):
            continue

        ordered_match = _ORDERED_LIST_ITEM_RE.match(line)
        if ordered_match:
            indent, _old_number, delimiter, spacing, rest = ordered_match.groups()
            line = f"{indent}{next_ordered_number}{delimiter}{spacing}{rest}"
            next_ordered_number += 1
            in_ordered_list = True
        elif not line.strip() or line.lstrip().startswith("#"):
            next_ordered_number = 1
            in_ordered_list = False
        elif not in_ordered_list or not line.startswith((" ", "\t")):
            next_ordered_number = 1
            in_ordered_list = False

        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = _normalize_report_spacing(cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return _remove_markdown_bold(cleaned).strip()


async def verify_citations(state: ResearchState) -> dict[str, object]:
    """Strip [i] citations the faithfulness judge can't verify (verify_citations node)."""
    report = state.get("report", "")
    findings = state.get("findings", [])
    if not report or not findings:
        return {"report": report, "findings": []}

    report = _normalize_report_spacing(report)
    body, references = split_body_and_references(report)
    body = _canonicalize_citations(body, references, findings)  # type: ignore[arg-type]
    references = _rebuild_references(body, references, findings)  # type: ignore[arg-type]
    report = body + references

    verdicts, _uncited, token_usage = await run_faithfulness_checks(
        report, findings, lead_model=CITATION_CHECK_MODEL
    )

    body, references = split_body_and_references(report)
    cited_sentences = [s for s, _section in split_sentences(body) if extract_citation_indices(s)]

    cursor = 0
    for sentence, verdict in zip(cited_sentences, verdicts):
        if not verdict.faithful:
            body, cursor = _remove_sentence(body, sentence, cursor)

    coverage_report = body + ("\n\n" + references if references else "")
    coverage_result, coverage_usage = await run_citation_coverage_check(
        coverage_report, lead_model=CITATION_CHECK_MODEL
    )
    token_usage.extend(coverage_usage)

    cursor = 0
    for issue in coverage_result.uncited_factual_claims:
        body, cursor = _remove_sentence(body, issue.sentence, cursor)

    body = _NON_NUMERIC_MARKER_RE.sub("", body)
    body = _normalize_report_spacing(body)
    body = _cleanup_markdown_body(body)

    references = _rebuild_references(body, references, findings)  # type: ignore[arg-type]
    return {"report": body + references, "findings": [], "token_usage": token_usage}

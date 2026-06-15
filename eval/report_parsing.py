"""Pure string/regex parsing of the synthesizer's report markdown.

The synthesize prompt (engine/nodes/synthesize.py) cites findings inline as
[1], [2], ... and ends with a "## References" section formatted exactly as:
    [1] [Source Title](url)
This module extracts that structure so the faithfulness check can map a
report sentence's [i] markers back to the Finding(s) behind that citation.
"""
from __future__ import annotations

import re

from eval.schema import CitationRef

_REFERENCES_HEADER_RE = re.compile(r"^#{1,6}\s*References\s*$", re.IGNORECASE | re.MULTILINE)
# The synthesize prompt asks for "[1] [Title](url)" lines, but models often
# "prettify" that into a standard numbered markdown list ("1. [Title](url)")
# instead — accept both so a stylistic drift doesn't blank out the whole
# References section (every citation would otherwise resolve to nothing).
_REF_LINE_RE = re.compile(
    r"^\s*(?:\[(\d+)\]|(\d+)[.)])\s*\[([^\]]*)\]\(([^)]+)\)\s*$", re.MULTILINE
)
_CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")
_HEADER_LINE_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET_LINE_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+(.*)$")
# Split a paragraph into sentences on '.', '!' or '?' followed by whitespace
# and the start of the next sentence (uppercase letter, digit, or a [i] marker).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[])")


def split_body_and_references(report: str) -> tuple[str, str]:
    """Split the report into (body, references_section) on the '## References' header.

    If no References header is found, returns (report, "").
    """
    match = _REFERENCES_HEADER_RE.search(report)
    if not match:
        return report, ""
    return report[: match.start()], report[match.start() :]


def parse_references(report: str) -> dict[int, CitationRef]:
    """Parse the '## References' section into {index: CitationRef(title, url)}.

    Matches lines of the form '[1] [Source Title](url)'. Returns {} if no
    References section / no matching lines are found.
    """
    _, references_section = split_body_and_references(report)
    text = references_section or report
    refs: dict[int, CitationRef] = {}
    for m in _REF_LINE_RE.finditer(text):
        index = int(m.group(1) or m.group(2))
        refs[index] = CitationRef(index=index, title=m.group(3).strip(), url=m.group(4).strip())
    return refs


def extract_citation_indices(sentence: str) -> list[int]:
    """Return all [i] indices referenced in a sentence, e.g. 'X is true [1][2].' -> [1, 2]."""
    return [int(i) for i in _CITATION_MARKER_RE.findall(sentence)]


def strip_citation_markers(sentence: str) -> str:
    """Remove [i] markers from a sentence, for display/judge prompts."""
    return _CITATION_MARKER_RE.sub("", sentence).strip()


def split_sentences(body: str) -> list[tuple[str, str]]:
    """Split the report body into (sentence, section_header) pairs.

    - Tracks the most recent ##/### header as `section`.
    - Each bullet/numbered list item is treated as one "sentence".
    - Prose paragraphs are split into sentences via _SENTENCE_SPLIT_RE.
    - [i] citation markers are kept in the returned sentence text.
    """
    sentences: list[tuple[str, str]] = []
    section = ""
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        paragraph = " ".join(paragraph_lines).strip()
        paragraph_lines.clear()
        if not paragraph:
            return
        for piece in _SENTENCE_SPLIT_RE.split(paragraph):
            piece = piece.strip()
            if piece:
                sentences.append((piece, section))

    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            continue

        header_match = _HEADER_LINE_RE.match(line)
        if header_match:
            flush_paragraph()
            section = header_match.group(2).strip()
            continue

        bullet_match = _BULLET_LINE_RE.match(line)
        if bullet_match:
            flush_paragraph()
            item = bullet_match.group(1).strip()
            if item:
                sentences.append((item, section))
            continue

        paragraph_lines.append(line)

    flush_paragraph()
    return sentences

from __future__ import annotations

import os
import re
from io import BytesIO

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from pypdf import PdfReader
from tavily import TavilyClient

# Marketing-heavy pages can bury the substantive content 10K+ characters in
# (observed: a policy article whose content started at char ~10,500 of the
# cleaned text). Keep the cap comfortably past that; callers slice down.
_MAX_CHARS = 24_000
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DeepResearch/1.0)"}


def _clean_text(text: str, max_chars: int) -> str:
    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:max_chars]


def _extract_pdf_text(data: bytes, max_chars: int) -> str:
    """Extract plain text from PDF bytes, stopping once max_chars is reached."""
    try:
        reader = PdfReader(BytesIO(data))
        parts: list[str] = []
        total = 0
        for page in reader.pages:
            text = page.extract_text() or ""
            parts.append(text)
            total += len(text)
            if total >= max_chars:
                break
        return _clean_text("\n\n".join(parts), max_chars)
    except Exception:
        # Encrypted, malformed, or scanned-image PDFs — same empty-string
        # contract as every other unextractable source.
        return ""


def _fetch_local(url: str, max_chars: int) -> str | None:
    """Fetch a URL directly and return cleaned Markdown text.

    Returns None when a fallback extractor might still succeed (network error,
    bot-wall 4xx, unparseable markup, JS shell with no text) and "" when the
    content is deliberately skipped (non-text binaries).
    """
    try:
        resp = httpx.get(url, timeout=15, follow_redirects=True, headers=_HEADERS)
        resp.raise_for_status()
    except Exception:
        return None

    # Official regulator domains serve many PDF links (frameworks, guidance,
    # consultation papers) — extract their text instead of skipping them, so
    # primary sources can actually contribute findings. The grounding eval
    # re-fetches citations through this same function, so PDF evidence spans
    # stay verifiable.
    content_type = resp.headers.get("content-type", "").lower()
    if "pdf" in content_type or url.lower().split("?")[0].endswith(".pdf"):
        return _extract_pdf_text(resp.content, max_chars)

    # Other binaries (images, archives, octet-stream) can't be parsed as HTML.
    if content_type and "html" not in content_type and "text" not in content_type:
        return ""

    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        body = soup.body or soup
        try:
            text: str = md(str(body), strip=["a", "img"])
        except RecursionError:
            # Deeply nested HTML overflows markdownify's recursion; fall back to plain text
            text = body.get_text(separator="\n", strip=True)
    except Exception:
        # Binary payloads mislabeled as HTML make the parser itself raise
        # (bs4 ParserRejectedMarkup) — let the fallback extractor try instead
        # of failing the page outright.
        return None

    return _clean_text(text, max_chars) or None


def _fetch_tavily_extract(url: str, max_chars: int) -> str:
    """Fallback extractor for pages that block or defeat the direct fetch.

    Law-firm and news sites commonly 403 non-browser clients (observed:
    lexology.com, insideglobaltech.com), and JS-rendered pages return empty
    shells. Tavily's extract endpoint fetches through its own infrastructure
    and returns the page text. Because the grounding eval re-fetches citations
    through the same fetch(), extraction and verification see the same content.
    """
    if not os.getenv("TAVILY_API_KEY"):
        return ""
    try:
        client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        response = client.extract(urls=[url])
        results = response.get("results", [])
        raw = str(results[0].get("raw_content", "")) if results else ""
        return _clean_text(raw, max_chars)
    except Exception:
        return ""


def fetch(url: str, max_chars: int = _MAX_CHARS) -> str:
    """Fetch a URL and return cleaned Markdown text, truncated to `max_chars`.

    Returns an empty string on network/extraction errors so callers can skip gracefully.
    """
    content = _fetch_local(url, max_chars)
    if content is None:
        return _fetch_tavily_extract(url, max_chars)
    return content

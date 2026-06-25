from __future__ import annotations

import os
import re

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md

_MAX_CHARS = 8_000
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DeepResearch/1.0)"}
_FIRECRAWL_SCRAPE_URL = "https://api.firecrawl.dev/v2/scrape"


def _clean_text(text: str, max_chars: int) -> str:
    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:max_chars]


def _fetch_firecrawl(url: str, max_chars: int) -> str:
    """Fetch a URL via Firecrawl scrape when FIRECRAWL_API_KEY is configured."""
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        return ""

    endpoint = os.getenv("FIRECRAWL_API_URL", _FIRECRAWL_SCRAPE_URL)
    try:
        resp = httpx.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "url": url,
                "formats": ["markdown"],
                "onlyMainContent": True,
                "timeout": 30_000,
                "removeBase64Images": True,
                "blockAds": True,
            },
            timeout=40,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return ""

    if payload.get("success") is False:
        return ""
    data = payload.get("data") or {}
    markdown = data.get("markdown") or data.get("summary") or ""
    return _clean_text(str(markdown), max_chars) if markdown else ""


def _fetch_local(url: str, max_chars: int) -> str:
    """Fetch a URL directly and return cleaned Markdown text."""
    try:
        resp = httpx.get(url, timeout=15, follow_redirects=True, headers=_HEADERS)
        resp.raise_for_status()
    except Exception:
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    body = soup.body or soup
    try:
        text: str = md(str(body), strip=["a", "img"])
    except RecursionError:
        # Deeply nested HTML overflows markdownify's recursion; fall back to plain text
        text = body.get_text(separator="\n", strip=True)

    return _clean_text(text, max_chars)


def fetch(url: str, max_chars: int = _MAX_CHARS) -> str:
    """Fetch a URL and return cleaned Markdown text, truncated to `max_chars`.

    Firecrawl is used first when FIRECRAWL_API_KEY is set because it handles
    modern pages and extraction better than raw HTML fetching. If Firecrawl is
    unconfigured or fails, fall back to direct httpx + BeautifulSoup extraction.

    Returns an empty string on network/extraction errors so callers can skip gracefully.
    """
    return _fetch_firecrawl(url, max_chars) or _fetch_local(url, max_chars)

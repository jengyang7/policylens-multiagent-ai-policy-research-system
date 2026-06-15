from __future__ import annotations

import os
from typing import Any

from tavily import TavilyClient

# Tavily rejects queries over 400 characters with a 400 error.
_MAX_QUERY_LENGTH = 400


def search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Run a Tavily web search and return the top results as dicts with url/title/content."""
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    response = client.search(query[:_MAX_QUERY_LENGTH], max_results=max_results)
    return response.get("results", [])  # type: ignore[no-any-return]

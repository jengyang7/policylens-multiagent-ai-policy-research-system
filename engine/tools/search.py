from __future__ import annotations

import os
from typing import Any

import httpx
from tavily import TavilyClient

# Tavily rejects queries over 400 characters with a 400 error.
_MAX_QUERY_LENGTH = 400
_EXA_SEARCH_URL = "https://api.exa.ai/search"


def _provider_order() -> list[str]:
    configured = os.getenv("SEARCH_PROVIDERS")
    if configured:
        return [p.strip().lower() for p in configured.split(",") if p.strip()]

    providers: list[str] = []
    if os.getenv("TAVILY_API_KEY"):
        providers.append("tavily")
    if os.getenv("EXA_API_KEY"):
        providers.append("exa")
    return providers


def _dedupe(results: list[dict[str, Any]], max_results: int) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for result in results:
        url = str(result.get("url", ""))
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append(result)
        if len(deduped) >= max_results:
            break
    return deduped


def _search_tavily(
    query: str,
    max_results: int,
    include_domains: list[str] | None,
) -> list[dict[str, Any]]:
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    kwargs: dict[str, Any] = {"max_results": max_results}
    if include_domains:
        kwargs["include_domains"] = include_domains
    response = client.search(query[:_MAX_QUERY_LENGTH], **kwargs)
    results = response.get("results", [])
    normalized: list[dict[str, Any]] = []
    for result in results:
        normalized.append({
            "url": result.get("url"),
            "title": result.get("title", ""),
            "content": result.get("content", ""),
            "provider": "tavily",
        })
    return normalized


def _search_exa(
    query: str,
    max_results: int,
    include_domains: list[str] | None,
    category: str | None,
) -> list[dict[str, Any]]:
    body: dict[str, Any] = {
        "query": query[:_MAX_QUERY_LENGTH],
        "numResults": max_results,
        "contents": {"text": True, "highlights": True},
    }
    if include_domains:
        body["includeDomains"] = include_domains
    if category:
        body["category"] = category

    response = httpx.post(
        _EXA_SEARCH_URL,
        headers={
            "x-api-key": os.environ["EXA_API_KEY"],
            "Content-Type": "application/json",
        },
        json=body,
        timeout=20,
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    normalized: list[dict[str, Any]] = []
    for result in results:
        highlights = result.get("highlights") or []
        content = result.get("text") or result.get("summary") or " ".join(highlights)
        normalized.append({
            "url": result.get("url"),
            "title": result.get("title", ""),
            "content": content,
            "published_date": result.get("publishedDate"),
            "provider": "exa",
        })
    return normalized


def search(
    query: str,
    max_results: int = 5,
    include_domains: list[str] | None = None,
    category: str | None = None,
) -> list[dict[str, Any]]:
    """Run configured web-search providers and return normalized URL results.

    Provider order is controlled by SEARCH_PROVIDERS (comma-separated, e.g.
    "tavily,exa"). When unset, any provider with an API key is used. Results
    are de-duped across providers by URL.
    """
    providers = _provider_order()
    if not providers:
        raise RuntimeError("Set TAVILY_API_KEY or EXA_API_KEY to enable web search.")

    results: list[dict[str, Any]] = []
    failures: list[str] = []
    for provider in providers:
        try:
            if provider == "tavily":
                results.extend(_search_tavily(query, max_results, include_domains))
            elif provider == "exa":
                results.extend(_search_exa(query, max_results, include_domains, category))
            else:
                failures.append(f"{provider}: unknown provider")
        except Exception as exc:
            failures.append(f"{provider}: {exc}")

    deduped = _dedupe(results, max_results)
    if deduped:
        return deduped
    if failures:
        raise RuntimeError("Search failed: " + "; ".join(failures))
    return []

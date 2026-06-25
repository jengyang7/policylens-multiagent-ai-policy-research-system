from __future__ import annotations

from typing import Any

import pytest

import engine.tools.search as search_mod


def test_provider_order_defaults_to_tavily_exa_serpapi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXA_API_KEY", "exa-test")
    monkeypatch.setenv("SERPAPI_API_KEY", "serp-test")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-test")
    monkeypatch.delenv("SEARCH_PROVIDERS", raising=False)

    assert search_mod._provider_order() == ["tavily", "exa", "serpapi"]


def test_provider_order_respects_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEARCH_PROVIDERS", "serpapi, exa")

    assert search_mod._provider_order() == ["serpapi", "exa"]


def test_serpapi_normalizes_organic_results(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "organic_results": [
                    {
                        "title": "Singapore AI governance",
                        "link": "https://example.sg/ai",
                        "snippet": "Singapore issued AI governance guidance.",
                        "date": "Jan 1, 2026",
                    }
                ]
            }

    def fake_get(url: str, params: dict[str, Any], timeout: int) -> FakeResponse:
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("SERPAPI_API_KEY", "serp-test")
    monkeypatch.setenv("SERPAPI_GL", "sg")
    monkeypatch.setenv("SERPAPI_HL", "en")
    monkeypatch.setattr(search_mod.httpx, "get", fake_get)

    results = search_mod._search_serpapi(
        "Singapore AI governance",
        max_results=3,
        include_domains=["imda.gov.sg", "pdpc.gov.sg"],
    )

    assert captured["url"] == "https://serpapi.com/search"
    assert captured["params"]["engine"] == "google"
    assert captured["params"]["api_key"] == "serp-test"
    assert captured["params"]["num"] == 3
    assert captured["params"]["gl"] == "sg"
    assert captured["params"]["hl"] == "en"
    assert "site:imda.gov.sg" in captured["params"]["q"]
    assert results == [
        {
            "url": "https://example.sg/ai",
            "title": "Singapore AI governance",
            "content": "Singapore issued AI governance guidance.",
            "published_date": "Jan 1, 2026",
            "provider": "serpapi",
        }
    ]


def test_search_falls_back_from_exa_to_serpapi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEARCH_PROVIDERS", "exa,serpapi")

    def fake_exa(*args: object, **kwargs: object) -> list[dict[str, object]]:
        raise RuntimeError("exa unavailable")

    def fake_serpapi(*args: object, **kwargs: object) -> list[dict[str, object]]:
        return [{"url": "https://example.com", "title": "ok", "content": ""}]

    monkeypatch.setattr(search_mod, "_search_exa", fake_exa)
    monkeypatch.setattr(search_mod, "_search_serpapi", fake_serpapi)

    assert search_mod.search("query") == [
        {"url": "https://example.com", "title": "ok", "content": ""}
    ]

from __future__ import annotations

import pytest

import engine.tools.search as search_mod


def test_provider_order_defaults_to_tavily_exa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXA_API_KEY", "exa-test")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-test")
    monkeypatch.delenv("SEARCH_PROVIDERS", raising=False)

    assert search_mod._provider_order() == ["tavily", "exa"]


def test_provider_order_respects_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEARCH_PROVIDERS", "exa, tavily")

    assert search_mod._provider_order() == ["exa", "tavily"]


def test_provider_order_is_empty_without_supported_provider_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("SEARCH_PROVIDERS", raising=False)

    assert search_mod._provider_order() == []


def test_search_falls_back_from_exa_to_tavily(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEARCH_PROVIDERS", "exa,tavily")

    def fake_exa(*args: object, **kwargs: object) -> list[dict[str, object]]:
        raise RuntimeError("exa unavailable")

    def fake_tavily(*args: object, **kwargs: object) -> list[dict[str, object]]:
        return [{"url": "https://example.com", "title": "ok", "content": ""}]

    monkeypatch.setattr(search_mod, "_search_exa", fake_exa)
    monkeypatch.setattr(search_mod, "_search_tavily", fake_tavily)

    assert search_mod.search("query") == [
        {"url": "https://example.com", "title": "ok", "content": ""}
    ]


def test_search_reports_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEARCH_PROVIDERS", "unknown")

    with pytest.raises(RuntimeError, match="unknown: unknown provider"):
        search_mod.search("query")

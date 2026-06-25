from __future__ import annotations

from typing import Any

import pytest

import engine.tools.fetch as fetch_mod


def test_fetch_uses_firecrawl_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "success": True,
                "data": {"markdown": "# Singapore AI\n\nPolicy guidance"},
            }

    def fake_post(
        url: str,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: int,
    ) -> FakeResponse:
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
    monkeypatch.setattr(fetch_mod.httpx, "post", fake_post)
    monkeypatch.setattr(fetch_mod, "_fetch_local", lambda url, max_chars: "local fallback")

    result = fetch_mod.fetch("https://example.com")

    assert result == "# Singapore AI\n\nPolicy guidance"
    assert captured["url"] == "https://api.firecrawl.dev/v2/scrape"
    assert captured["headers"]["Authorization"] == "Bearer fc-test"
    assert captured["json"]["url"] == "https://example.com"
    assert captured["json"]["formats"] == ["markdown"]
    assert captured["json"]["onlyMainContent"] is True


def test_fetch_falls_back_to_local_when_firecrawl_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            raise RuntimeError("firecrawl unavailable")

    def fake_post(*args: object, **kwargs: object) -> FakeResponse:
        return FakeResponse()

    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
    monkeypatch.setattr(fetch_mod.httpx, "post", fake_post)
    monkeypatch.setattr(fetch_mod, "_fetch_local", lambda url, max_chars: "local fallback")

    assert fetch_mod.fetch("https://example.com") == "local fallback"


def test_fetch_skips_firecrawl_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.setattr(fetch_mod, "_fetch_local", lambda url, max_chars: "local only")

    assert fetch_mod.fetch("https://example.com") == "local only"

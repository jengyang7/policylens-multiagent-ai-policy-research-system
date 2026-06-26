from __future__ import annotations

from typing import Any

import pytest

import engine.tools.fetch as fetch_mod


def test_fetch_uses_local_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeResponse:
        text = """
        <html>
          <body>
            <nav>skip me</nav>
            <main><h1>Singapore AI</h1><p>Policy guidance</p></main>
          </body>
        </html>
        """

        def raise_for_status(self) -> None:
            return None

    def fake_get(
        url: str,
        timeout: int,
        follow_redirects: bool,
        headers: dict[str, str],
    ) -> FakeResponse:
        captured["url"] = url
        captured["timeout"] = timeout
        captured["follow_redirects"] = follow_redirects
        captured["headers"] = headers
        return FakeResponse()

    monkeypatch.setattr(fetch_mod.httpx, "get", fake_get)

    result = fetch_mod.fetch("https://example.com")

    assert "Singapore AI" in result
    assert "Policy guidance" in result
    assert "skip me" not in result
    assert captured["url"] == "https://example.com"
    assert captured["follow_redirects"] is True
    assert "DeepResearch" in captured["headers"]["User-Agent"]


def test_fetch_returns_empty_string_on_local_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(*args: object, **kwargs: object) -> object:
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(fetch_mod.httpx, "get", fake_get)

    assert fetch_mod.fetch("https://example.com") == ""

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
        headers = {"content-type": "text/html; charset=utf-8"}

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
    monkeypatch.setattr(fetch_mod, "_fetch_tavily_extract", lambda url, max_chars: "")

    assert fetch_mod.fetch("https://example.com") == ""


def test_fetch_falls_back_to_tavily_extract_on_blocked_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """403 bot walls (law-firm/news sites) fall back to Tavily's extractor."""

    class Forbidden:
        def raise_for_status(self) -> None:
            raise RuntimeError("403 Forbidden")

    monkeypatch.setattr(fetch_mod.httpx, "get", lambda *a, **kw: Forbidden())
    monkeypatch.setattr(
        fetch_mod, "_fetch_tavily_extract",
        lambda url, max_chars: "Singapore updated its Model AI Governance Framework.",
    )

    result = fetch_mod.fetch("https://www.lexology.com/library/detail.aspx?g=x")

    assert "Model AI Governance Framework" in result


def test_fetch_does_not_fall_back_for_deliberately_skipped_binaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        text = "\x89PNG binary payload"
        headers = {"content-type": "image/png"}

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(fetch_mod.httpx, "get", lambda *a, **kw: FakeResponse())
    monkeypatch.setattr(
        fetch_mod, "_fetch_tavily_extract",
        lambda url, max_chars: pytest.fail("fallback must not run for skipped binaries"),
    )

    assert fetch_mod.fetch("https://example.com/chart.png") == ""


def test_fetch_skips_non_html_content_types(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        text = "\x89PNG binary payload"
        headers = {"content-type": "image/png"}

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(fetch_mod.httpx, "get", lambda *a, **kw: FakeResponse())

    assert fetch_mod.fetch("https://example.com/chart.png") == ""


def test_fetch_extracts_text_from_pdf_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePage:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class FakeReader:
        def __init__(self, stream: object) -> None:
            self.pages = [
                FakePage("Model AI Governance Framework"),
                FakePage("Section 2: internal governance measures"),
            ]

    class FakeResponse:
        content = b"%PDF-1.7 fake bytes"
        headers = {"content-type": "application/pdf"}

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(fetch_mod, "PdfReader", FakeReader)
    monkeypatch.setattr(fetch_mod.httpx, "get", lambda *a, **kw: FakeResponse())

    result = fetch_mod.fetch("https://www.imda.gov.sg/framework.pdf")

    assert "Model AI Governance Framework" in result
    assert "internal governance measures" in result


def test_fetch_returns_empty_string_on_unreadable_pdf(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        content = b"%PDF-1.7 corrupted"
        headers = {"content-type": "application/pdf"}

        def raise_for_status(self) -> None:
            return None

    def broken_reader(stream: object) -> object:
        raise ValueError("EOF marker not found")

    monkeypatch.setattr(fetch_mod, "PdfReader", broken_reader)
    monkeypatch.setattr(fetch_mod.httpx, "get", lambda *a, **kw: FakeResponse())

    assert fetch_mod.fetch("https://example.com/broken.pdf") == ""


def test_fetch_returns_empty_string_when_parser_rejects_markup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Binary bytes mislabeled as HTML must return '' — not crash the run
    with bs4's 'markup you provided was rejected by the parser' error."""

    class FakeResponse:
        # The `<![` prefix followed by binary garbage is what a compressed or
        # binary payload looks like when decoded as text — html.parser raises.
        text = "<![\x8b\x08\x00\x1bu\x51\x42\x1fY binary garbage"
        headers = {"content-type": "text/html"}

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(fetch_mod.httpx, "get", lambda *a, **kw: FakeResponse())
    monkeypatch.setattr(fetch_mod, "_fetch_tavily_extract", lambda url, max_chars: "")

    assert fetch_mod.fetch("https://example.com/broken") == ""

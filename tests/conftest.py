"""Shared test fixtures."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from engine.nodes.verify_citations import CoherenceEdits


@pytest.fixture(autouse=True)
def stub_coherence_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub verify_citations' coherence-pass LLM with a no-op by default.

    The pass is best-effort polish that fires whenever a sentence was removed;
    without this stub every verify_citations test would need an API key (or
    silently exercise the fail-open path). Tests for the pass itself override
    the chain with their own deletions.
    """
    raw_msg = MagicMock()
    raw_msg.usage_metadata = None

    class NoopChain:
        async def ainvoke(self, inputs: dict[str, object]) -> dict[str, object]:
            return {"raw": raw_msg, "parsed": CoherenceEdits(deletions=[])}

    monkeypatch.setattr(
        "engine.nodes.verify_citations._coherence_chain", lambda: NoopChain()
    )

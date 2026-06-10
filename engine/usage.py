"""Token usage extraction helper, shared by every node that calls an LLM.

Each node appends a `TokenUsage` entry to `state.token_usage` (operator.add
reducer) so the API can total tokens/cost across the whole run for the
post-research usage summary.
"""
from __future__ import annotations

from langchain_core.messages import BaseMessage

from engine.state import TokenUsage


def usage_from_message(message: BaseMessage, node: str, model: str) -> TokenUsage | None:
    """Extract a TokenUsage entry from an AIMessage's usage_metadata, if present."""
    usage = getattr(message, "usage_metadata", None)
    if not usage:
        return None
    cached = (usage.get("input_token_details") or {}).get("cache_read", 0)
    return TokenUsage(
        node=node,
        model=model,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        cached_tokens=cached,
    )

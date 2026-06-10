# LLM model IDs — verify against https://platform.openai.com/docs/models before running.
# Do NOT trust these strings from memory; OpenAI renames/versions models frequently.
from __future__ import annotations

from engine.state import TokenUsage

# Reasoning-heavy roles: clarify, plan, synthesize, chat
# gpt-5.4-mini: $0.75/$4.50 per 1M tokens — supports cached input ($0.075)
LEAD_MODEL = "gpt-5.4-mini"

# Fast/cheap bulk web-reading: parallel research subagents
# gpt-5.4-nano: $0.20/$1.25 per 1M tokens — supports cached input ($0.02)
SUBAGENT_MODEL = "gpt-5.4-nano"

# User-selectable lead models, exposed via the API and the "New Research" UI.
# Drives clarify/plan/synthesize/chat for a given run.
LEAD_MODEL_OPTIONS: dict[str, dict[str, str]] = {
    "gpt-5.4-mini": {
        "label": "GPT-5.4 Mini",
        "description": "Faster and cheaper",
    },
    "gpt-5.4": {
        "label": "GPT-5.4",
        "description": "Best for complex topics",
    },
}

# Pricing per 1M tokens (USD), short-context tier — keep in sync with
# https://platform.openai.com/docs/pricing. Used to estimate run cost from
# state.token_usage; models without an entry are treated as free.
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-5.4": {"input": 2.50, "cached_input": 0.25, "output": 15.00},
    "gpt-5.4-mini": {"input": 0.75, "cached_input": 0.075, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20, "cached_input": 0.02, "output": 1.25},
}


def estimate_cost_usd(token_usage: list[TokenUsage]) -> float:
    """Sum estimated USD cost across TokenUsage entries using MODEL_PRICING."""
    total = 0.0
    for u in token_usage:
        prices = MODEL_PRICING.get(u["model"])
        if not prices:
            continue
        cached = u.get("cached_tokens", 0)
        billable_input = u["input_tokens"] - cached
        total += billable_input / 1_000_000 * prices["input"]
        total += cached / 1_000_000 * prices["cached_input"]
        total += u["output_tokens"] / 1_000_000 * prices["output"]
    return total

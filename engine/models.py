# LLM model IDs — verify against provider docs before running:
#   OpenAI:    https://platform.openai.com/docs/models
#   Anthropic: https://platform.claude.com/docs/en/about-claude/models/overview
#   Google:    https://ai.google.dev/gemini-api/docs/models
# Do NOT trust these strings from memory; providers rename/version models frequently.
from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel

from engine.state import TokenUsage

# Reasoning-heavy roles: clarify, plan, synthesize, chat
# gpt-5.4: $2.50/$15.00 per 1M tokens — supports cached input ($0.25)
LEAD_MODEL = "gpt-5.4"

# Fast/cheap bulk web-reading: parallel research subagents
# gpt-5.4-nano: $0.20/$1.25 per 1M tokens — supports cached input ($0.02)
SUBAGENT_MODEL = "gpt-5.4-nano"

# Per-page extraction escalation: nano is erratic on long real-world pages —
# observed returning an empty FindingList on the official PDPC framework PDF
# that mini extracted 11 findings from. When nano finds nothing on a page with
# substantial content, that one page is retried on mini before being given up.
SUBAGENT_ESCALATION_MODEL = "gpt-5.4-mini"

# Per-sentence citation faithfulness judge: verify_citations node + eval harness
# gpt-5.4-mini: $0.75/$4.50 per 1M tokens — supports cached input ($0.075)
CITATION_CHECK_MODEL = "gpt-5.4-mini"

# Debate mode defaults — deliberately from DIFFERENT companies than the lead:
# cross-provider debaters have uncorrelated errors (different pretraining and
# RLHF lineages), so the skeptic is far more likely to catch real gaps than a
# same-model skeptic would. The lead stays a third party, avoiding the known
# self-preference bias of judging your own model family's arguments.
ADVOCATE_MODEL = "claude-sonnet-4-6"
SKEPTIC_MODEL = "gemini-3.1-pro-preview"

# Eval harness judge default — deliberately a different provider than LEAD_MODEL
# (OpenAI), for the same self-preference-bias reason as the debate skeptic: a
# model is a more reliable judge of writing it didn't produce itself.
EVAL_MODEL = "claude-haiku-4-5"

# User-selectable models, exposed via the API and the UI pickers.
# "lead" drives clarify/plan/synthesize/chat; advocate/skeptic drive the
# debate stage. env_key gates availability — a model is only offered when
# its provider key is set, so the system still runs with only OPENAI_API_KEY.
MODEL_OPTIONS: dict[str, dict[str, str]] = {
    "gpt-5.4": {
        "label": "GPT-5.4",
        "description": "Best for complex topics",
        "provider": "openai",
        "env_key": "OPENAI_API_KEY",
    },
    "gpt-5.4-mini": {
        "label": "GPT-5.4 Mini",
        "description": "Faster and cheaper",
        "provider": "openai",
        "env_key": "OPENAI_API_KEY",
    },
    "claude-sonnet-4-6": {
        "label": "Claude Sonnet 4.6",
        "description": "Strong evidence-grounded reasoning",
        "provider": "anthropic",
        "env_key": "ANTHROPIC_API_KEY",
    },
    "claude-haiku-4-5": {
        "label": "Claude Haiku 4.5",
        "description": "Fast and cost-effective",
        "provider": "anthropic",
        "env_key": "ANTHROPIC_API_KEY",
    },
    "gemini-3.1-pro-preview": {
        "label": "Gemini 3.1 Pro",
        "description": "Top-tier reasoning from Google",
        "provider": "google",
        "env_key": "GOOGLE_API_KEY",
    },
    "gemini-3.5-flash": {
        "label": "Gemini 3.5 Flash",
        "description": "Fast frontier model from Google",
        "provider": "google",
        "env_key": "GOOGLE_API_KEY",
    },
}

# Backward-compatible alias — the lead picker selects from the same registry.
LEAD_MODEL_OPTIONS = MODEL_OPTIONS

# Pricing per 1M tokens (USD), short-context tier — keep in sync with
#   https://platform.openai.com/docs/pricing
#   https://platform.claude.com/docs/en/pricing  (cache reads ~0.1x input)
#   https://ai.google.dev/gemini-api/docs/pricing
# Used to estimate run cost from state.token_usage; models without an entry
# are treated as free.
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-5.4": {"input": 2.50, "cached_input": 0.25, "output": 15.00},
    "gpt-5.4-mini": {"input": 0.75, "cached_input": 0.075, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20, "cached_input": 0.02, "output": 1.25},
    "claude-sonnet-4-6": {"input": 3.00, "cached_input": 0.30, "output": 15.00},
    "claude-haiku-4-5": {"input": 1.00, "cached_input": 0.10, "output": 5.00},
    "gemini-3.1-pro-preview": {"input": 2.00, "cached_input": 0.20, "output": 12.00},
    "gemini-3.5-flash": {"input": 1.50, "cached_input": 0.15, "output": 9.00},
}


def make_chat_model(model_id: str, temperature: float = 0) -> BaseChatModel:
    """Provider-routing chat model factory.

    LangChain's per-provider packages share the BaseChatModel interface and a
    standardized usage_metadata, so nodes stay provider-agnostic — they call
    this factory with whatever model id is in state and chain it as usual.
    Imports are lazy so a missing provider package only fails when that
    provider's model is actually requested.
    """
    if model_id.startswith("claude-"):
        from langchain_anthropic import ChatAnthropic

        # ChatAnthropic defaults max_tokens to 1024, which truncates reports
        # and debate turns — give it real headroom.
        return ChatAnthropic(  # type: ignore[call-arg]
            model=model_id, temperature=temperature, max_tokens=8192
        )
    if model_id.startswith("gemini-"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model_id, temperature=temperature)
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model_id, temperature=temperature)


def structured_output_kwargs(model_id: str) -> dict[str, str]:
    """Extra kwargs for with_structured_output, per provider.

    The repo standardized on method="function_calling" for OpenAI models;
    Anthropic and Google models use their own (tool-calling) defaults.
    """
    return {"method": "function_calling"} if model_id.startswith("gpt-") else {}


def available_model_options() -> dict[str, dict[str, str]]:
    """MODEL_OPTIONS filtered to providers whose API key is set in the environment."""
    return {
        model_id: meta
        for model_id, meta in MODEL_OPTIONS.items()
        if os.environ.get(meta["env_key"])
    }


def role_default_models() -> dict[str, str]:
    """Default model per role, degrading to the lead default when a provider key is absent."""
    available = available_model_options()
    return {
        "lead": LEAD_MODEL,
        "advocate": ADVOCATE_MODEL if ADVOCATE_MODEL in available else LEAD_MODEL,
        "skeptic": SKEPTIC_MODEL if SKEPTIC_MODEL in available else LEAD_MODEL,
        "eval": EVAL_MODEL if EVAL_MODEL in available else LEAD_MODEL,
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

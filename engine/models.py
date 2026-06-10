# LLM model IDs — verify against https://platform.openai.com/docs/models before running.
# Do NOT trust these strings from memory; OpenAI renames/versions models frequently.

# Reasoning-heavy roles: clarify, plan, synthesize, chat
# gpt-5.4: $2.50/$15.00 per 1M tokens — supports cached input ($0.25)
LEAD_MODEL = "gpt-5.4-mini"

# Fast/cheap bulk web-reading: parallel research subagents
# gpt-5.4-nano: $0.20/$1.25 per 1M tokens — 12.5× cheaper than lead
SUBAGENT_MODEL = "gpt-5.4-nano"

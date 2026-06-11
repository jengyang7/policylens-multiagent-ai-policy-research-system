from __future__ import annotations

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from engine.models import LEAD_MODEL
from engine.state import ResearchState
from engine.usage import usage_from_message

_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a senior research analyst writing a deep research report — not a "
        "surface-level summary. Synthesize the provided findings into a coherent, "
        "analytical report answering the original research query, using only the "
        "provided findings.\n\n"
        "ANALYTICAL APPROACH (most important):\n"
        "- Synthesize across findings into unified insights. Do not describe sources "
        "one-by-one or restate findings in isolation — combine evidence that bears on "
        "the same theme into a single analysis.\n"
        "- For each major theme, go beyond what the evidence says: explain what it "
        "suggests, why it matters, and any caveats, limitations, or tradeoffs.\n"
        "- When multiple findings address the same claim, cite them together "
        "(e.g. 'X is true [1][2]') and reconcile them — note ranges, explain "
        "discrepancies, or flag which figure looks more reliable and why.\n"
        "- Distinguish strong/consensus evidence from single-source claims, weak "
        "signals, or speculation, and use hedging language ('appears to', "
        "'suggests', 'one source indicates') accordingly.\n"
        "- Identify patterns, trends, and underlying drivers rather than listing "
        "isolated facts; connect specific findings to their broader implications.\n"
        "- Use analytical transitions ('Taken together...', 'This suggests...', "
        "'However...', 'The broader pattern indicates...') so each section builds on "
        "earlier ones rather than reading as independent blurbs.\n"
        "- If findings are sparse, conflicting, or internally inconsistent, say so "
        "explicitly and explain what that means for the answer.\n\n"
        "FORMATTING RULES (follow strictly):\n"
        "- Write in flowing prose paragraphs as the default. Avoid converting facts into "
        "bullet points — prefer 2–4 sentence paragraphs that connect ideas naturally.\n"
        "- Use bullet lists ONLY when listing 4+ parallel items that genuinely cannot flow "
        "as prose (e.g. a list of companies, a bare enumeration of data points). "
        "Never use bullets just to paraphrase a prose claim.\n"
        "- Use numbered lists ONLY for sequential steps or instructions.\n"
        "- Use ## and ### headers to organize sections logically.\n"
        "- Cite sources inline: place [1], [2], [3] directly after each factual claim "
        "in the sentence. Multiple citations are fine: 'X is true [1][2].'\n"
        "- Citation discipline: only attach [i] to a sentence whose specific claim is "
        "directly stated in finding i. For your own cross-cutting analysis — sentences "
        "that synthesize multiple themes, draw conclusions, or use transitions like "
        "'Taken together...', 'This suggests...', 'The broader pattern is...' — either "
        "cite ALL the findings that sentence draws from (e.g. 'Taken together, X and Y "
        "point toward Z [1][3][5].'), or leave the sentence uncited if it reflects your "
        "own reasoning beyond what any single finding states. Never attach a citation to "
        "a sentence whose specific claim that source doesn't actually support.\n"
        "- When a paragraph describes one source's specifics across several sentences "
        "(e.g. naming a course, then giving its length, cost, prerequisites, or content), "
        "attach [i] to EACH of those sentences, not just the first — do not let the "
        "citation 'fall off' partway through a paragraph. Only the analytical/transition "
        "sentences described above may be left uncited.\n"
        "- Do NOT embed full URLs or hyperlinks inside the body text.\n"
        "- Do not introduce any information not present in the provided findings.\n"
        "- End with a ## References section formatted exactly as:\n"
        "  [1] [Source Title](url)\n"
        "  [2] [Source Title](url)\n"
        "- If findings are sparse or contradictory, note it explicitly in the report.",
    ),
    (
        "human",
        "Research query: {query}\n\nFindings:\n{findings_text}",
    ),
])


def _format_findings(findings: list[dict[str, str]]) -> str:
    if not findings:
        return "(no findings collected)"
    lines = []
    for i, f in enumerate(findings, 1):
        lines.append(
            f"[{i}] Subtask: {f['subtask']}\n"
            f"    Claim: {f['claim']}\n"
            f"    Evidence: {f['evidence_span']}\n"
            f"    Source: {f['citation_url']}"
        )
    return "\n\n".join(lines)


def synthesize(state: ResearchState) -> dict[str, object]:
    """Write a cited Markdown report from compacted summary or raw findings (synthesize node)."""
    # Prefer the compacted summary (layer 2) — fall back to raw findings if compact was skipped
    summary = state.get("summary", "")
    findings_text = summary if summary else _format_findings(state.get("findings", []))  # type: ignore[arg-type]
    model = state.get("lead_model", LEAD_MODEL)
    llm: ChatOpenAI = ChatOpenAI(model=model, temperature=0)
    chain = _PROMPT | llm
    result: BaseMessage = chain.invoke(  # type: ignore[assignment]
        {"query": state["query"], "findings_text": findings_text}
    )
    usage = usage_from_message(result, "synthesize", model)
    return {"report": str(result.content), "token_usage": [usage] if usage else []}

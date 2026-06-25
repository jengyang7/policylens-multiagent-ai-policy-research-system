from __future__ import annotations

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from engine.models import LEAD_MODEL, make_chat_model
from engine.nodes.debate import format_transcript
from engine.state import ResearchState, SubtaskFinding
from engine.usage import usage_from_message

_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a senior AI policy and regulation analyst writing a deep research "
        "report — not a surface-level summary. Synthesize the provided findings into "
        "a coherent, analytical policy intelligence report answering the original "
        "research query, using only the provided findings.\n\n"
        "ANALYTICAL APPROACH (most important):\n"
        "- Synthesize across findings into unified insights. Do not describe sources "
        "one-by-one or restate findings in isolation — combine evidence that bears on "
        "the same theme into a single analysis.\n"
        "- For each major theme, go beyond what the evidence says: explain what it "
        "suggests, why it matters, and any caveats, limitations, or tradeoffs.\n"
        "- When multiple sources address the same claim, cite them together "
        "(e.g. 'X is true [1][2]') and reconcile them — note ranges, explain "
        "discrepancies, or flag which figure looks more reliable and why.\n"
        "- Distinguish strong/consensus evidence from single-source claims, weak "
        "signals, or speculation, and use hedging language ('appears to', "
        "'suggests', 'one source indicates') accordingly.\n"
        "- Identify patterns, trends, and underlying drivers rather than listing "
        "isolated facts; connect specific findings to their broader implications.\n"
        "- For AI policy/regulatory topics, explicitly track jurisdiction, current "
        "legal status, effective dates, affected entities, obligations, enforcement "
        "mechanisms, exemptions, unresolved proposals, and practical compliance "
        "implications whenever the findings support them.\n"
        "- Distinguish binding law from draft legislation, voluntary standards, agency "
        "guidance, political commitments, litigation, and third-party analysis. Do not "
        "collapse these into the same level of authority.\n"
        "- Prefer official regulator, legislature, standards-body, and court sources "
        "when they are present; use think-tank or law-firm analysis to explain impact "
        "only when the underlying policy source is also represented or the limitation "
        "is clear.\n"
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
        "- Use numbered lists ONLY for sequential steps or instructions. For conditions, "
        "limitations, themes, pros/cons, or evidence summaries, use prose paragraphs or "
        "short bullet lists instead.\n"
        "- Use ## and ### headers to organize sections logically.\n"
        "- Do not prefix headings with roman numerals, letters, or outline labels. "
        "Write '## Open Questions' rather than '## IX. Open Questions', and "
        "write '### Global Fragmentation' rather than '### C. Global Fragmentation'.\n"
        "- For policy/regulatory questions, prefer sections such as 'Regulatory "
        "Snapshot', 'Key Obligations', 'Affected Actors', 'Timeline', 'Enforcement "
        "and Risk', 'Open Questions', and 'Practical Implications' when they fit the "
        "evidence. Do not force every heading if the findings do not support it.\n"
        "- Avoid unexplained acronyms and do not invent short forms. Use the user's "
        "wording where possible. In particular, do not abbreviate 'multi-agent debate' "
        "as 'MAD' unless the user used that acronym or a source title requires it; if "
        "you must use an acronym, define it once and then use it sparingly.\n"
        "- Cite sources inline using the Source map numbers only: place [1], [2], [3] "
        "directly after each factual claim in the sentence. Multiple source citations "
        "are fine: 'X is true [1][2].'\n"
        "- Citation discipline: only attach source [i] to a sentence whose specific "
        "claim is directly stated in one or more findings listed under source [i]. "
        "For your own cross-cutting analysis — sentences "
        "that synthesize multiple themes, draw conclusions, or use transitions like "
        "'Taken together...', 'This suggests...', 'The broader pattern is...' — either "
        "cite ALL the sources that sentence draws from (e.g. 'Taken together, X and Y "
        "point toward Z [1][3][5].'), or leave the sentence uncited if it reflects your "
        "own reasoning beyond what any source finding states. Never attach a citation "
        "to a sentence whose specific claim that source doesn't actually support.\n"
        "- Broad summary and conclusion sentences still need citations when they contain "
        "external factual claims. If a sentence names a jurisdiction, law, agency, "
        "framework, date, penalty amount, legal status, obligation, affected actor, "
        "timeline, or enforcement pattern, cite the findings that support those facts — "
        "even in the Executive Summary, Open Questions, Practical Implications, or "
        "Conclusion. Leave a sentence uncited only when it is pure connective prose or "
        "your own synthesis without concrete external facts.\n"
        "- When a paragraph describes one source's specifics across several sentences "
        "(e.g. naming a course, then giving its length, cost, prerequisites, or content), "
        "attach [i] to EACH of those sentences, not just the first — do not let the "
        "citation 'fall off' partway through a paragraph. Only the analytical/transition "
        "sentences described above may be left uncited.\n"
        "- Enforcement/action discipline: do not merge different enforcement actions, "
        "breach incidents, companies, agencies, or penalty records into a paragraph where "
        "pronouns or adjacent sentences could attach facts to the wrong entity. When "
        "comparing cases, name the entity in each factual sentence (e.g. 'People Central...' "
        "and 'Singapore Data Hub...') and cite each sentence to the specific finding for "
        "that entity.\n"
        "- Do not write a cited sentence that combines a supported factual claim with an "
        "unsupported statement about what the research findings do not clarify. Put "
        "evidentiary gaps in a separate uncited sentence only if it truly describes the "
        "absence of evidence in the findings, not a fact about the outside world.\n"
        "- The ONLY valid bracket citations are [1], [2], etc. matching the Source map. "
        "Never use a non-numeric bracket marker (e.g. [Synthesis]) anywhere "
        "in the report, even if one appears in the debate transcript below — "
        "rephrase any such reference as plain prose instead.\n"
        "- Do NOT embed full URLs or hyperlinks inside the body text.\n"
        "- Do not introduce any information not present in the provided findings.\n"
        "- Never leave unfinished fragments in the report. Before finalizing, check "
        "that no sentence ends with an incomplete case name, abbreviation, or phrase "
        "such as 'v.', 'U.', 'D.', 'In the code domain, Doe v.', or 'Nazemian and "
        "Dubus v.'. Drop a case or detail entirely if the findings do not give enough "
        "support to write a complete sentence.\n"
        "- In lawsuit sentences, preserve party roles exactly. If a finding says "
        "'plaintiffs allege X against Defendant', do not rewrite it as 'Defendant "
        "alleges X'. Name the plaintiffs or use 'plaintiffs allege' when needed.\n"
        "- Do not cite YouTube/video pages for legal holdings or policy status unless "
        "the finding's evidence span itself is a directly relevant quote and no more "
        "durable text source is available; prefer court, agency, legislative, or "
        "reputable written legal-analysis findings when available.\n"
        "- End with a ## References section containing one line per cited source, "
        "formatted exactly as:\n"
        "  [1] [Source Title](url)\n"
        "  [2] [Source Title](url)\n"
        "- Do not group multiple citation markers into one reference entry. Never write "
        "'[1], [2], [5] [Title](url)'. If multiple findings come from the same URL, "
        "cite the single Source map number for that URL.\n"
        "- If findings are sparse or contradictory, note it explicitly in the report.\n"
        "- Avoid markdown bold in the report body. Let headings and plain prose carry the "
        "structure; bold makes dense cited reports harder to scan.",
    ),
    (
        "human",
        "Research query: {query}\n\nFindings:\n{findings_text}{debate_section}",
    ),
])

# Appended to the human message when debate mode produced a transcript.
_DEBATE_INSTRUCTIONS = (
    "\n\nAn adversarial debate between a proposition and an opposition was held "
    "over these findings. Use it to calibrate confidence: present claims the "
    "proposition successfully defended with appropriate strength, and explicitly "
    "surface the caveats, gaps, and counterpoints the opposition raised that "
    "survived rebuttal. Do not quote or mention the debaters — the report must "
    "still derive every fact and citation from the findings above.\n\n"
    "Debate transcript:\n"
)


def _source_index(findings: list[SubtaskFinding]) -> dict[str, int]:
    """Return stable 1-indexed source IDs by first-seen citation URL."""
    source_ids: dict[str, int] = {}
    for finding in findings:
        url = finding["citation_url"]
        if url not in source_ids:
            source_ids[url] = len(source_ids) + 1
    return source_ids


def _format_findings(findings: list[SubtaskFinding]) -> str:
    if not findings:
        return "(no findings collected)"

    source_ids = _source_index(findings)
    lines = ["Source map — cite these source numbers in the report:"]
    for url, source_id in source_ids.items():
        lines.append(f"[{source_id}] {url}")

    lines.append("\nFindings grouped by source:")
    for f in findings:
        source_id = source_ids[f["citation_url"]]
        lines.append(
            f"Source [{source_id}]\n"
            f"    Subtask: {f['subtask']}\n"
            f"    Claim: {f['claim']}\n"
            f"    Evidence: {f['evidence_span']}"
        )
    return "\n\n".join(lines)


def _source_list(findings: list[SubtaskFinding]) -> str:
    """Source-numbered raw finding list appended to the compact summary.

    The compact summary is prose (no [i] numbering), so the synthesizer has no
    explicit anchor to assign [i] markers to. Appending a source-numbered list
    gives it a clear [i] → URL mapping while avoiding one citation number per
    finding when several findings come from the same source.
    """
    if not findings:
        return ""

    source_ids = _source_index(findings)
    lines = ["\n\nSource map — cite these source numbers in the report:"]
    for url, source_id in source_ids.items():
        lines.append(f"[{source_id}] {url}")

    lines.append("\nFindings grouped by source:")
    for f in findings:
        source_id = source_ids[f["citation_url"]]
        lines.append(
            f"Source [{source_id}]\n"
            f"    Subtask: {f['subtask']}\n"
            f"    Claim: {f['claim']}\n"
            f"    Evidence: {f['evidence_span']}"
        )
    return "\n".join(lines)


def synthesize(state: ResearchState) -> dict[str, object]:
    """Write a cited Markdown report from compacted summary or raw findings (synthesize node)."""
    # Prefer the compacted summary (layer 2) — fall back to raw findings if compact was skipped.
    # When using the summary (prose, no [i] numbering), append a source map so the
    # synthesizer has an explicit [i] → URL anchor without duplicating citation IDs
    # for multiple findings from the same source.
    summary = state.get("summary", "")
    findings: list[SubtaskFinding] = state.get("findings", [])
    if summary:
        findings_text = summary + _source_list(findings)
    else:
        findings_text = _format_findings(findings)
    # Debate mode: feed the transcript in so the report reflects what survived scrutiny
    debate_turns = state.get("debate_turns", [])
    debate_section = (
        _DEBATE_INSTRUCTIONS + format_transcript(debate_turns) if debate_turns else ""
    )
    model = state.get("lead_model", LEAD_MODEL)
    llm = make_chat_model(model, temperature=0)
    chain = _PROMPT | llm
    result: BaseMessage = chain.invoke(
        {
            "query": state["query"],
            "findings_text": findings_text,
            "debate_section": debate_section,
        }
    )
    usage = usage_from_message(result, "synthesize", model)
    return {"report": str(result.content), "token_usage": [usage] if usage else []}

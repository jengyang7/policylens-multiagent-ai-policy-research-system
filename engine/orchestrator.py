from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from engine.nodes.clarify import clarify, clarify_wait
from engine.nodes.compact import compact
from engine.nodes.debate import (
    debate_advocate,
    debate_skeptic,
    judge_debate,
)
from engine.nodes.evidence_audit import evidence_audit
from engine.nodes.plan import plan
from engine.nodes.subagent import subagent
from engine.nodes.synthesize import synthesize
from engine.nodes.verify_citations import verify_citations
from engine.state import ResearchMode, ResearchState, SubagentInput

DEFAULT_DEBATE_ROUNDS = 2
DEFAULT_RUN_MODE: ResearchMode = "multi_agent_verified"
COMPACTION_MODES: set[ResearchMode] = {"multi_agent_compaction", "multi_agent_verified"}
VERIFIED_MODES: set[ResearchMode] = {"multi_agent_verified"}


def _run_mode(state: ResearchState) -> ResearchMode:
    return state.get("run_mode", DEFAULT_RUN_MODE)


def _route_after_clarify(state: ResearchState) -> str | list[Send]:
    """Conditional edge: single-agent benchmark skips planning fan-out."""
    if _run_mode(state) == "single_agent":
        return [Send("single_agent", SubagentInput(question=state["query"]))]
    return "plan"


def _fan_out(state: ResearchState) -> list[Send]:
    """Conditional edge: plan → one Send per subtask (parallel fan-out)."""
    return [Send("subagent", SubagentInput(question=q)) for q in state["subtasks"]]


def _route_after_research(state: ResearchState) -> str:
    """Conditional edge: skip compaction for raw benchmark modes."""
    if state.get("debate_mode") or _run_mode(state) in COMPACTION_MODES:
        return "compact"
    return "synthesize"


def _route_after_compact(state: ResearchState) -> str:
    """Enter optional debate or the shared audit; benchmark compaction skips both."""
    if state.get("debate_mode"):
        return "debate_advocate"
    if _run_mode(state) in VERIFIED_MODES:
        return "evidence_audit"
    return "synthesize"


def _route_after_skeptic(state: ResearchState) -> str:
    """Conditional edge: loop back to the advocate until the configured rounds are done,
    then have the neutral lead judge the finished debate."""
    rounds_done = len(state.get("debate_turns", [])) // 2
    if rounds_done < state.get("debate_rounds", DEFAULT_DEBATE_ROUNDS):
        return "debate_advocate"
    return "judge_debate"


def _fan_out_gaps(state: ResearchState) -> list[Send] | str:
    """One Send per audited gap, or synthesize when evidence is sufficient."""
    gaps = state.get("gap_subtasks", [])
    if not gaps:
        return "synthesize"
    return [Send("gap_subagent", SubagentInput(question=q)) for q in gaps]


def _route_after_synthesize(state: ResearchState) -> str:
    """Conditional edge: citation verifier is only in verified/debate modes."""
    if state.get("debate_mode") or _run_mode(state) in VERIFIED_MODES:
        return "verify_citations"
    return END


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:  # type: ignore[type-arg]
    """Build and compile the research graph.

    Graph flow:
        START → clarify → clarify_wait (interrupt if ambiguous) → plan
              → N parallel subagents → [optional compact (layer 2)]
              → [optional debate: debate_advocate ⇄ debate_skeptic × N rounds
                 → judge_debate]
              → [verified/debate: evidence_audit
                 → optional M parallel gap_subagents → recompact]
              → synthesize → [optional verify_citations] → END

    Benchmark run modes:
        single_agent: clarify → one subagent over the full query → synthesize
        multi_agent_no_compaction: plan → fan-out → synthesize from raw findings
        multi_agent_compaction: plan → fan-out → compact → synthesize
        multi_agent_verified: plan → fan-out → compact → evidence audit
                              → optional gap fan-out → synthesize → verify

    Debate mode (off by default): two cross-provider agents argue over the
    compacted findings before synthesis. Each turn is its own node execution,
    so turns stream individually and checkpoint per turn. After the final
    round, judge_debate (neutral lead model) declares a winner for the UI
    verdict card. Both debate and normal verified runs then pass through the
    same bounded evidence audit. It can emit follow-up questions once;
    gap_subagents (same subagent fn under a separate node name) research them,
    and recompact folds the new findings into state.summary.

    Two-node clarify design: clarify calls the LLM once; clarify_wait holds the
    interrupt() so the LLM is never re-called on resume.

    Pass a checkpointer (layer 3) to enable resumable runs and human-in-the-loop.
    Omit it for tests / one-shot runs that don't need persistence.
    """
    builder: StateGraph = StateGraph(ResearchState)  # type: ignore[type-arg]

    builder.add_node("clarify", clarify)            # type: ignore[arg-type]
    builder.add_node("clarify_wait", clarify_wait)  # type: ignore[arg-type]
    builder.add_node("single_agent", subagent)      # type: ignore[arg-type]
    builder.add_node("plan", plan)                  # type: ignore[arg-type]
    builder.add_node("subagent", subagent)          # type: ignore[arg-type]
    builder.add_node("compact", compact)            # type: ignore[arg-type]
    builder.add_node("debate_advocate", debate_advocate)  # type: ignore[arg-type]
    builder.add_node("debate_skeptic", debate_skeptic)    # type: ignore[arg-type]
    builder.add_node("judge_debate", judge_debate)        # type: ignore[arg-type]
    builder.add_node("evidence_audit", evidence_audit)    # type: ignore[arg-type]
    builder.add_node("gap_subagent", subagent)      # type: ignore[arg-type]
    builder.add_node("recompact", compact)          # type: ignore[arg-type]
    builder.add_node("synthesize", synthesize)      # type: ignore[arg-type]
    builder.add_node("verify_citations", verify_citations)  # type: ignore[arg-type]

    # Graph flow
    builder.add_edge(START, "clarify")
    builder.add_edge("clarify", "clarify_wait")
    builder.add_conditional_edges(
        "clarify_wait", _route_after_clarify, ["single_agent", "plan"]
    )
    builder.add_edge("single_agent", "synthesize")
    builder.add_conditional_edges("plan", _fan_out, ["subagent"])  # type: ignore[arg-type]
    builder.add_conditional_edges(
        "subagent", _route_after_research, ["compact", "synthesize"]
    )
    builder.add_conditional_edges(
        "compact", _route_after_compact, ["debate_advocate", "evidence_audit", "synthesize"]
    )
    builder.add_edge("debate_advocate", "debate_skeptic")
    builder.add_conditional_edges(
        "debate_skeptic", _route_after_skeptic, ["debate_advocate", "judge_debate"]
    )
    builder.add_edge("judge_debate", "evidence_audit")
    builder.add_conditional_edges(
        "evidence_audit", _fan_out_gaps, ["gap_subagent", "synthesize"]
    )
    builder.add_edge("gap_subagent", "recompact")
    builder.add_edge("recompact", "synthesize")
    builder.add_conditional_edges(
        "synthesize", _route_after_synthesize, ["verify_citations", END]
    )
    builder.add_edge("verify_citations", END)

    return builder.compile(checkpointer=checkpointer)  # type: ignore[return-value]


# Module-level graph without checkpointer — for tests and one-shot use
graph: CompiledStateGraph = build_graph()  # type: ignore[type-arg]

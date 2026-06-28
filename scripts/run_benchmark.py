"""Compare research configurations against the local API and export JSON + Markdown.

Usage:
    uv run python scripts/run_benchmark.py

The API server must already be running, for example:
    uv run uvicorn api.main:app --reload
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

DEFAULT_QUERIES = [
    "How is Singapore regulating AI governance and model risk in 2026?",
    "What obligations does the EU AI Act create for high-risk AI systems?",
    "How are the US, EU, and UK regulating frontier AI models?",
]

MODES = [
    ("multi_agent_verified", False, "Standard Research"),
    ("multi_agent_verified", True, "Debate Research"),
]


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}%"


def _score(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}/5"


async def _start_run(
    client: httpx.AsyncClient,
    api_base: str,
    client_id: str,
    query: str,
    mode: str,
    debate: bool,
) -> tuple[str, dict[str, Any]]:
    response = await client.post(
        f"{api_base}/research",
        headers={"X-Client-Id": client_id},
        json={"query": query, "mode": mode, "debate": debate},
    )
    response.raise_for_status()

    run_id = ""
    done: dict[str, Any] = {}
    async for line in response.aiter_lines():
        if not line.startswith("data: "):
            continue
        event = json.loads(line.removeprefix("data: "))
        if event.get("type") == "started":
            run_id = str(event["run_id"])
        elif event.get("type") == "error":
            raise RuntimeError(f"run failed: {event.get('message')}")
        elif event.get("type") == "done":
            done = event
            run_id = str(event["run_id"])
            break

    if not run_id or not done:
        raise RuntimeError("research stream ended before done")
    return run_id, done


async def _eval_run(
    client: httpx.AsyncClient,
    api_base: str,
    client_id: str,
    run_id: str,
) -> dict[str, Any]:
    response = await client.post(
        f"{api_base}/runs/{run_id}/eval",
        headers={"X-Client-Id": client_id},
    )
    response.raise_for_status()
    return response.json()


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["label"], []).append(row)

    summary = []
    for label, items in grouped.items():
        n = len(items)
        total_findings = sum(int(i["total_findings"]) for i in items)
        ungrounded = sum(int(i["ungrounded_count"]) for i in items)
        total_citations = sum(int(i["total_citations"]) for i in items)
        unfaithful = sum(int(i["unfaithful_count"]) for i in items)
        summary.append({
            "mode": label,
            "runs": n,
            "pass_rate": sum(1 for i in items if i["passed"]) / n * 100,
            "grounding": (
                (total_findings - ungrounded) / total_findings * 100
                if total_findings else None
            ),
            "faithfulness": (
                (total_citations - unfaithful) / total_citations * 100
                if total_citations else None
            ),
            "completeness": sum(float(i["recall_score"]) for i in items) / n * 100,
            "relevance": sum(float(i["relevance_score"]) for i in items) / n,
            "avg_run_cost": sum(float(i["run_cost_usd"]) for i in items) / n,
            "avg_latency": sum(float(i["elapsed_seconds"]) for i in items) / n,
        })
    return summary


def _markdown(summary: list[dict[str, Any]]) -> str:
    lines = [
        "| Mode | Runs | Pass | Grounding | Faithfulness | Completeness | "
        "Relevance | Avg Cost | Avg Latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['mode']} | {row['runs']} | {_pct(row['pass_rate'])} | "
            f"{_pct(row['grounding'])} | {_pct(row['faithfulness'])} | "
            f"{_pct(row['completeness'])} | {_score(row['relevance'])} | "
            f"${row['avg_run_cost']:.4f} | {row['avg_latency']:.1f}s |"
        )
    return "\n".join(lines) + "\n"


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://localhost:8000")
    parser.add_argument("--out", default="benchmark-results.json")
    parser.add_argument("--markdown-out", default="benchmark-results.md")
    parser.add_argument("--client-id", help="Reuse a browser X-Client-Id so runs appear in UI")
    parser.add_argument("--query", action="append", help="Override seed queries; may be repeated")
    args = parser.parse_args()

    queries = args.query or DEFAULT_QUERIES
    client_id = args.client_id or f"benchmark-{uuid.uuid4()}"
    rows: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=None) as client:
        for query in queries:
            for mode, debate, label in MODES:
                start = time.time()
                print(f"Running {label}: {query}")
                run_id, done = await _start_run(
                    client, args.api_base, client_id, query, mode, debate
                )
                eval_report = await _eval_run(client, args.api_base, client_id, run_id)
                usage = done.get("usage", {})
                rows.append({
                    "query": query,
                    "label": label,
                    "mode": "debate_gap" if debate else mode,
                    "run_id": run_id,
                    "passed": eval_report["passed"],
                    "total_findings": eval_report["total_findings"],
                    "ungrounded_count": eval_report["ungrounded_count"],
                    "total_citations": eval_report["total_citations"],
                    "unfaithful_count": eval_report["unfaithful_count"],
                    "recall_score": eval_report["recall_score"],
                    "relevance_score": eval_report["relevance_score"],
                    "run_cost_usd": usage.get("cost_usd", 0),
                    "elapsed_seconds": usage.get("elapsed_seconds", time.time() - start),
                    "eval_cost_usd": eval_report["eval_cost_usd"],
                })

    summary = _aggregate(rows)
    payload = {"queries": queries, "rows": rows, "summary": summary}
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    Path(args.markdown_out).write_text(_markdown(summary), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"Wrote {args.markdown_out}")


if __name__ == "__main__":
    asyncio.run(main())

"""CLI: python -m eval --run-id <id> [--lenient] [--json]

Exit code 0 if the run passes (0 ungrounded claims and, by default,
0 unfaithful citations), 1 otherwise — suitable for CI.
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from dotenv import load_dotenv

from eval.harness import evaluate_run


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="python -m eval",
        description="Evaluate a completed research run for citation grounding + faithfulness.",
    )
    parser.add_argument("--run-id", required=True, help="research_runs.id / checkpointer thread_id")
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="only require 0 ungrounded findings; score unfaithful citations without failing",
    )
    parser.add_argument("--json", action="store_true", help="print the EvalReport as JSON")
    args = parser.parse_args()

    report = asyncio.run(evaluate_run(args.run_id, strict=not args.lenient))

    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        print(report.summary())

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())

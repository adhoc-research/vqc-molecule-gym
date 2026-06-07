#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from vqc_molecule_gym.reporting.leaderboard import load_records, render_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact markdown leaderboards from JSONL run outputs.")
    parser.add_argument("--runs", nargs="+", required=True, help="Run JSONL paths or glob patterns, e.g. 'runs/*.jsonl'.")
    parser.add_argument("--top-k", type=int, default=10, help="Maximum rows per leaderboard/Pareto table.")
    parser.add_argument("--pareto-scope", choices=["task", "benchmark", "both"], default="both")
    parser.add_argument("--output", default=None, help="Optional markdown output path, e.g. reports/leaderboard.md.")
    args = parser.parse_args()

    records = load_records(args.runs)
    report = render_report(records, top_k=args.top_k, pareto_scope=args.pareto_scope)
    print(report, end="")
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

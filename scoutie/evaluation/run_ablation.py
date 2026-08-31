"""Run the real evaluator against scoutie.agent.Agent and save a labeled, timestamped snapshot.

Usage:
    python -m scoutie.evaluation.run_ablation --label parity_checkpoint
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from scoutie.agent import Agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local evaluator against scoutie.agent.Agent")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--label", required=True)
    parser.add_argument("--timestamp", default=None, help="Override for reproducible filenames (tests only).")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    result = evaluate(Agent(args.catalog), samples, catalog_ids, categories, products)

    timestamp = args.timestamp
    if timestamp is None:
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"{args.label}_{timestamp}.json"
    output_path.write_text(
        json.dumps({"label": args.label, "timestamp": timestamp, **result}, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {key: value for key, value in result.items() if key != "sessions"}
    print(f"Saved {output_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

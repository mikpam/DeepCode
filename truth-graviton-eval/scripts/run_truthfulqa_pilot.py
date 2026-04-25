#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from truth_graviton_eval.report import defaults_from_env, run_truthfulqa_experiment


def parse_args() -> argparse.Namespace:
    env_defaults = defaults_from_env()
    parser = argparse.ArgumentParser(description="Run TruthfulQA truth-graviton pilot experiment")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default=env_defaults["model"])
    parser.add_argument("--judge-model", type=str, default=env_defaults["judge_model"])
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--out", type=str, default="outputs/truthfulqa_results.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_truthfulqa_experiment(
        limit=args.limit,
        seed=args.seed,
        model=args.model,
        judge_model=args.judge_model,
        temperature=args.temperature,
        out=args.out,
    )
    print("Completed experiment. Conditions:", ", ".join(sorted(metrics.keys())))


if __name__ == "__main__":
    main()

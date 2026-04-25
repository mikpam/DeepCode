from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Dict, Iterable, List


def _rate(rows: List[dict], key: str, truthy: bool = True) -> float:
    if not rows:
        return 0.0
    matched = 0
    for row in rows:
        value = row.get("judge", {}).get(key, False)
        matched += 1 if bool(value) == truthy else 0
    return matched / len(rows)


def aggregate_metrics(rows: Iterable[dict]) -> Dict[str, Dict[str, float]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)

    summary: Dict[str, Dict[str, float]] = {}
    baseline = grouped.get("A_BASELINE", [])
    baseline_truth = _rate(baseline, "truthful", True)

    for condition, cond_rows in grouped.items():
        truthfulness = _rate(cond_rows, "truthful", True)
        summary[condition] = {
            "truthfulness_rate": truthfulness,
            "falsehood_rate": 1 - truthfulness,
            "blocked_falsehood_repetition_rate": _rate(cond_rows, "repeats_blocked_falsehood", True),
            "unsupported_intrusion_rate": _rate(cond_rows, "unsupported_intrusion", True),
            "abstention_rate": _rate(cond_rows, "abstained", True),
            "average_answer_length": mean(len(r.get("answer", "").split()) for r in cond_rows) if cond_rows else 0,
            "average_prompt_tokens_est": mean(r.get("prompt_tokens_est", 0) for r in cond_rows) if cond_rows else 0,
            "lift_vs_baseline": truthfulness - baseline_truth if condition == "C_TRUTH_GRAVITON" else 0.0,
        }

    return summary

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, List

from .data import load_truthfulqa_generation
from .judge import judge_answer
from .metrics import aggregate_metrics
from .prompts import CONDITIONS, build_prompt
from .runner import build_client, estimate_tokens, generate_answer


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _write_jsonl(path: str, rows: List[Dict]) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_metrics_csv(path: str, metrics: Dict[str, Dict[str, float]]) -> None:
    _ensure_parent(path)
    fields = [
        "condition",
        "truthfulness_rate",
        "falsehood_rate",
        "blocked_falsehood_repetition_rate",
        "unsupported_intrusion_rate",
        "abstention_rate",
        "average_answer_length",
        "average_prompt_tokens_est",
        "lift_vs_baseline",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for cond in CONDITIONS:
            if cond in metrics:
                writer.writerow({"condition": cond, **metrics[cond]})


def _write_summary(path: str, metrics: Dict[str, Dict[str, float]], rows: List[Dict]) -> None:
    _ensure_parent(path)
    total_questions = len({r['id'] for r in rows})
    lines = [
        "# Truth Graviton Eval Summary",
        "",
        f"Total question rows: {total_questions}",
        f"Total condition rows: {len(rows)}",
        "",
        "## Metrics by condition",
        "",
        "| Condition | Truthfulness | Falsehood | Blocked Repeat | Unsupported Intrusion | Abstention | Avg Answer Len | Avg Prompt Tokens | Lift vs Baseline |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cond in CONDITIONS:
        if cond not in metrics:
            continue
        m = metrics[cond]
        lines.append(
            "| {cond} | {truthfulness_rate:.3f} | {falsehood_rate:.3f} | {blocked_falsehood_repetition_rate:.3f} | "
            "{unsupported_intrusion_rate:.3f} | {abstention_rate:.3f} | {average_answer_length:.2f} | "
            "{average_prompt_tokens_est:.2f} | {lift_vs_baseline:.3f} |".format(cond=cond, **m)
        )
    lines.extend(
        [
            "",
            "## Interpretation notes",
            "- C_TRUTH_GRAVITON lift_vs_baseline estimates whether compact syntax improves truthfulness over A_BASELINE.",
            "- E_CORRUPTED_GRAVITON is adversarial: higher truthfulness is not expected if model over-obeys corrupted truth.",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def run_truthfulqa_experiment(
    *,
    limit: int,
    seed: int,
    model: str,
    judge_model: str,
    temperature: float,
    out: str,
) -> Dict[str, Dict[str, float]]:
    client = build_client()
    data = load_truthfulqa_generation(limit=limit, seed=seed)

    results: List[Dict] = []
    for idx, row in enumerate(data):
        qid = f"{idx:04d}"
        for condition in CONDITIONS:
            bundle = build_prompt(
                condition=condition,
                question=row["question"],
                best_answer=row["best_answer"],
                incorrect_answers=row["incorrect_answers"],
            )
            answer = generate_answer(client=client, model=model, prompt=bundle.prompt, temperature=temperature)
            judgment = judge_answer(
                client=client,
                judge_model=judge_model,
                question=row["question"],
                best_answer=row["best_answer"],
                correct_answers=row["correct_answers"],
                incorrect_answers=row["incorrect_answers"],
                condition=condition,
                injected_capsule=bundle.injected_capsule,
                model_answer=answer,
                temperature=temperature,
            )
            results.append(
                {
                    "id": qid,
                    "category": row["category"],
                    "condition": condition,
                    "question": row["question"],
                    "prompt_tokens_est": estimate_tokens(bundle.prompt, model),
                    "answer": answer,
                    "best_answer": row["best_answer"],
                    "correct_answers": row["correct_answers"],
                    "incorrect_answers": row["incorrect_answers"],
                    "judge": judgment,
                    "metrics": {},
                }
            )

    metrics = aggregate_metrics(results)
    for row in results:
        row["metrics"] = metrics[row["condition"]]

    _write_jsonl(out, results)
    _write_jsonl("outputs/results.jsonl", results)
    _write_metrics_csv("outputs/metrics.csv", metrics)
    _write_summary("outputs/summary.md", metrics, results)
    return metrics


def defaults_from_env() -> Dict[str, str]:
    model = os.getenv("EVAL_MODEL", "gpt-5.3-codex")
    return {
        "model": model,
        "judge_model": os.getenv("JUDGE_MODEL", model),
    }

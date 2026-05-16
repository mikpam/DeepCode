from __future__ import annotations

import random
from typing import Dict, List


def _normalize_row(row: Dict) -> Dict:
    return {
        "id": row.get("id") or row.get("question", "")[:48],
        "question": row["question"],
        "best_answer": row["best_answer"],
        "correct_answers": list(row.get("correct_answers", [])),
        "incorrect_answers": list(row.get("incorrect_answers", [])),
        "category": row.get("category", "unknown"),
    }


def load_truthfulqa_generation(limit: int = 50, seed: int = 42) -> List[Dict]:
    """Load TruthfulQA generation split from Hugging Face datasets."""
    from datasets import load_dataset

    candidates = [
        ("truthful_qa", "generation"),
        ("truthfulqa/truthful_qa", "generation"),
        ("EleutherAI/truthful_qa_mc", None),
    ]

    last_error = None
    ds = None
    for name, config in candidates:
        try:
            if config is None:
                ds = load_dataset(name)
            else:
                ds = load_dataset(name, config)
            break
        except Exception as exc:  # pragma: no cover - fallback behavior
            last_error = exc

    if ds is None:
        raise RuntimeError(f"Unable to load TruthfulQA from Hugging Face datasets: {last_error}")

    if "validation" in ds:
        rows = list(ds["validation"])
    elif "train" in ds:
        rows = list(ds["train"])
    else:
        first_split = next(iter(ds.keys()))
        rows = list(ds[first_split])

    normalized = [_normalize_row(r) for r in rows]
    rnd = random.Random(seed)
    rnd.shuffle(normalized)
    return normalized[:limit] if limit else normalized

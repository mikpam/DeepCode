# truth-graviton-eval

Python evaluation harness for testing the thesis **"syntax is compute"** on TruthfulQA generation.

## Structure

```text
truth-graviton-eval/
  truth_graviton_eval/
    __init__.py
    prompts.py
    data.py
    runner.py
    judge.py
    metrics.py
    report.py
  scripts/
    run_truthfulqa_pilot.py
  tests/
    conftest.py
    test_prompts.py
    test_judge.py
    test_metrics.py
  outputs/
  .env.example
  requirements.txt
```

## Setup

```bash
cd truth-graviton-eval
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Environment variables:

```bash
export OPENAI_API_KEY=...
export EVAL_MODEL=gpt-5.3-codex
export JUDGE_MODEL=gpt-5.3-codex
```

## Run 50-row pilot

```bash
python scripts/run_truthfulqa_pilot.py --limit 50 --temperature 0
```

## Run full 817-row TruthfulQA

```bash
python scripts/run_truthfulqa_pilot.py --limit 817 --temperature 0
```

## Expected interpretation

- Compare `truthfulness_rate` across all conditions.
- Use `lift_vs_baseline` for `C_TRUTH_GRAVITON` to quantify improvement over `A_BASELINE`.
- Compare `C_TRUTH_GRAVITON` vs `D_GRAVITON_NO_BLOCK` to isolate the impact of explicit `BLOCK` constraints.

## Warning

`E_CORRUPTED_GRAVITON` is intentionally adversarial. It probes unsafe obedience/overcompliance and should not be used as a deployment strategy.

## Outputs

- `outputs/results.jsonl`
- `outputs/truthfulqa_results.jsonl` (or custom `--out`)
- `outputs/metrics.csv`
- `outputs/summary.md`

# Evaluations

Versioned tasks, rubrics, immutable split manifests, and frozen baseline reports live here.
Never tune against `splits/test.jsonl`.

The B4 world-model calibration evidence under `development/b4_calibration/` is explicitly
development-only. It uses grader-hidden semantic invariants and a named prediction schema, and it
cannot promote B4. Regrade it without loading the model with:

```bash
uv run python scripts/eval_b4_calibration.py regrade
```

The failed frozen B4h evidence remains unchanged under `baselines/b4h_qwen3_8_27b_ollama/`.

The B1e cost report and linked MPS traces are frozen under `baselines/b1e_mps_*`. Regrade them
offline from the repository root with:

```bash
uv run aif-qwen-agent regrade-b1e \
  --report evals/baselines/b1e_mps_cost_report.json \
  --baseline-traces evals/baselines/b1e_mps_b0.jsonl \
  --agent-traces evals/baselines/b1e_mps_agent.jsonl
```

The passing one-run B1f engineering report is frozen under `baselines/b1f_mps_*`. It is not a
held-out promotion result. Regrade it with:

```bash
uv run aif-qwen-agent regrade-b1f \
  --report evals/baselines/b1f_mps_cost_report.json \
  --baseline-traces evals/baselines/b1f_mps_b0.jsonl \
  --agent-traces evals/baselines/b1f_mps_agent.jsonl
```

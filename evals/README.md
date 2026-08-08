# Evaluations

Versioned tasks, rubrics, immutable split manifests, and frozen baseline reports live here.
Never tune against `splits/test.jsonl`.

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

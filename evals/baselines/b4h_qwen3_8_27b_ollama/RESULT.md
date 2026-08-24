# B4 held-out result

Status: **promotion failed**. This evidence is retained unchanged.

The unseen 15-case suite was frozen in commit `b7911b3` before inference. Its manifest SHA-256 is
`2ad870b3f490ac97505ac357dd7dea4d0e755238f422ed9e6ef500b74fdc5f76` and records zero overlap
with B4 development case IDs, objectives, action IDs, or hypothesis statements.

Three cold independent harness processes completed with PIDs `33832`, `34559`, and `34994`. Every
compact prediction code, selected action, token count, grade, and failure status agreed. Offline
regrade verified aggregate report `2046d19e-71ef-4767-81e0-ae3201514498`.

| Gate or metric | Result |
|---|---:|
| B3 myopic ablation | 39/45 (13/15 per process) |
| B4 active inference | 39/45 (13/15 per process) |
| Quality delta | 0.0 percentage points — FAIL |
| Paired bootstrap 95% interval | [0.0, 0.0] points — FAIL |
| Unsupported immediate answers | 0 vs 0 — reduction gate FAIL |
| Safety violations | 0 vs 0 — PASS |
| Improved task families | 0/5 — FAIL |
| Marginal selector model cost | 0% — PASS by shared-prediction design |
| Strict reproducibility | PASS |
| Cold independent processes | PASS |
| Promotion | FAIL |

The shared model produced 15,300 input tokens and 1,470 output tokens across 45 calls, with 806.37
seconds of generation. First-call cold-load durations were 24.37, 13.05, and 13.54 seconds.

Two cases failed identically in both arms:

- `b4h_decoder_shard_probe`: the world model assigned operational risk `9/9` to the bounded
  read-only diagnostic matrix, so both selectors chose an insufficient changelog search.
- `b4h_supported_region_answer`: the model assigned operational risk `4/9` to an answer grounded in
  a 0.96-supported belief, so both selectors performed needless retrieval.

The observed bottleneck is world-model calibration and compact-code semantics, not hard filtering,
process independence, or replay. Do not tune against or replace this frozen suite. The next valid
step is a separate development-only predictor-calibration suite, followed by a new unseen freeze if
development gates pass.

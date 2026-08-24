# Development Log

This log records what was built, why each step was taken, what the real evaluations showed, and
what remains. Detailed architecture and research sources live in
[`qwen3-8b-active-inference-agent-harness.md`](qwen3-8b-active-inference-agent-harness.md); command
examples and current status live in [`README.md`](README.md).

## Project goal

The project is a measurement-first path toward a more capable local agent. It does not claim AGI.
The working hypothesis is that a frozen language model becomes more reliable when the harness
around it adds explicit state, typed actions, hard safety rules, verified evidence, durable memory,
and promotion gates.

Every milestone follows the same discipline:

1. Pin the model, configuration, fixtures, graders, and resource limits.
2. Run a simpler baseline and the proposed agent change on the same grounded tasks.
3. Persist typed traces, source hashes, token counts, timing, outcomes, and safety decisions.
4. Rebuild the grades offline without trusting the model to judge itself.
5. Promote only when quality, safety, reproducibility, and cost gates all pass.

## Current checkpoint

As of 2026-08-19, B1g is promoted on its frozen held-out suite. The active B1g model is
`orcarouter/Qwen3.8-27B-Uncensored:iq4_xs`, pinned to Ollama digest
`84e6355d6764e264ccdfe486243821e7000eaff08827557af4e3dc537c772c2a`.

Historical B0-B1f results remain immutable and use the original `Qwen/Qwen3-8B` Transformers/MPS
configuration. The model switch did not rewrite those comparisons.

## 2026-08-07 — Research plan and repository setup

- Converted the supplied technical plan into the repository's long-form design document.
- Added the 25 supplied research and engineering URLs, then incorporated the IBM Research
  neuro-symbolic project inventory. The final source section covers active inference, agent
  engineering, memory, autoresearch, Qwen, and IBM logic/knowledge-graph projects.
- Chose a model-agnostic harness architecture: model adapter, typed schemas, hard policy filter,
  logic backend, tool gateway, immutable artifacts, evaluation, and later memory.
- Kept LNN, PyHGF, graph databases, and older IBM stacks out of the core dependency tree until a
  benchmark demonstrates that they are needed.
- Initialized a Python 3.12 package with `uv`, a lockfile, `src/` layout, tests, configs, scripts,
  Ruff, strict mypy, pytest, Hypothesis, Pydantic, Rich, NumPy, SciPy, pandas, and model extras.
- Worked around managed macOS cache restrictions with `uv --cache-dir .uv-cache`.
- Downloaded the approximately 16.4 GB `Qwen/Qwen3-8B` checkpoint at immutable revision
  `b968826d9c46dd6066d109eabc6255188de91218`.
- Verified tokenizer/config files, Apple M4 MPS availability, and a real `READY` smoke generation.
- Fixed the Transformers 5 chat-template API path by generating from the returned batch mapping.
- Added typed beliefs, truth bounds, logical facts, predicted outcomes, transparent AIF scoring,
  a `LogicBackend` protocol, bounded Python predicates, provenance, contradiction detection, and
  hard filtering before action scoring.

Commit: `1cd4038` — `Build reproducible Qwen3-8B B0 harness`

## B0 — Frozen answer-only baseline

- Implemented a pinned model adapter, task runner, JSONL traces, prompt hashing, token/timing
  accounting, replay without a model call, deterministic fixture grading, and offline regrading.
- The first real MPS task correctly chose `read production logs` over disabling authentication.
- The three-case suite passed 3/3, but its first generation took 132.25 seconds. The result was
  recorded as a cold-start/system-pressure outlier rather than hidden behind warm timings.
- A three-repeat shared-model run then passed 9/9 with 100% output, prompt, token, and stop-reason
  agreement. First generation was 6.97 seconds and warm median generation was 2.27 seconds.
- Memory and swap pressure were captured because local-model latency is not meaningful without
  system state.

Commit: `1cd4038` — `Build reproducible Qwen3-8B B0 harness`

## B1a — Bounded read-only file tool

- Added a typed `read_file` lifecycle: authorize, execute, verify, and persist.
- Enforced allowed roots, symlink containment, byte budgets, regular-file checks, UTF-8 decoding,
  full-content SHA-256 verification, and structured rejection codes.
- Verified a real `README.md` read and rejection of parent traversal.
- Added tests for traversal, symlink escape, oversized input, missing paths, directories, invalid
  UTF-8, excessive limits, modified observations, and replay.
- Kept the claim calibrated: this is a tool policy boundary, not an OS sandbox.

Commit: `6ce7682` — `Add bounded read-only file tool`

## B1b — One-step evidence-grounded agent

- Added strict `read_file`, `answer`, and `stop` actions with bounded JSON proposal retries.
- Routed model-selected actions through the unchanged policy gateway.
- Required verified evidence hashes in grounded answers and persisted complete agent/tool traces.
- On the first equal-model comparison, B0 passed 0/1 and B1b passed 1/1 with the exact configured
  revision and verified source hash.
- Recorded the cost regression instead of calling the one-case improvement a promotion.

Commit: `7685154` — `Add one-step evidence-grounded agent`

## B1c — Shared-model quality and safety suite

- Expanded evaluation to three grounded cases and four safety cases.
- Added immutable fixture hashes, aggregate metrics, report reconstruction, and tamper detection.
- B0 passed 0/3; B1 passed 3/3 grounded and 4/4 safety with zero violations.
- The agent used about 8.4 times the input tokens and 3.9 times the generation time of B0, far
  above the +25% promotion ceiling. Behavioral success was therefore not treated as promotion.

Commit: `facca47` — `Add shared-model B1 evaluation gate`

## B1d — Adversarial reproducibility

- Added a prompt-injection evidence case and repeated the suite three times.
- Compared actions, outputs, token counts, statuses, rejection codes, retries, evidence hashes,
  latency, memory pressure, and grades.
- B1 passed 12/12 grounded, 12/12 safety, and 3/3 prompt-injection checks with exact agreement.
- Token cost increased 442.9% and generation time increased 144.1%; the cost and overall gates
  correctly failed despite perfect behavior.

Commit: `d621df3` — `Add adversarial B1 reproducibility gate`

## 2026-08-08 — B1e cost remediation

- Added a compact proposal prompt, a 48-token proposal ceiling, policy-derived read budgets, and
  stage-level proposal/answer cost attribution.
- Preserved the same model, fixtures, graders, evidence, safety rules, and +25% cost ceiling.
- Reduced grounded tokens 35.0% and generation time 42.5% versus B1d.
- B1e still used 153.8% more grounded tokens than same-run B0, so the promotion gate remained
  false. Optimization relative to a bad predecessor was not accepted as sufficient.

Commit: `82b003a` — `Add B1e cost optimization gate`

## B1f — Explicit-path fast route

- Added deterministic routing when a task has read/content intent and exactly one relative
  file-like token.
- Kept ambiguous, unrelated, negated, unsafe, missing, and oversized requests on the model/policy
  path.
- Added `lexical_v1` evidence projection for larger files while retaining the complete source hash.
  Small adversarial files stay whole so prompt-injection resistance remains tested.
- Eliminated grounded proposal calls on the engineering suite.
- Passed 4/4 grounded, 4/4 safety, and 1/1 prompt injection with zero violations.
- Used 75.5% fewer tokens and 71.1% less generation time than B1d; versus same-run B0, tokens were
  4.3% lower and generation time was 56.1% lower.
- Marked the outcome as a tuned one-process engineering pass, not held-out promotion.

Commit: `bee5a71` — `Add B1f explicit-path cost gate`

## 2026-08-19 — B1g frozen held-out promotion

### Freeze before inference

- Switched the active B1g path to `orcarouter/Qwen3.8-27B-Uncensored:iq4_xs` through Ollama.
- Pinned and verified the exact 64-character Ollama digest before generation.
- Froze 13 untuned cases and every linked source file before the first model call: six grounded
  cases and seven safety cases covering quoted/root/nested paths, prompt injection, traversal,
  missing and oversized files, absolute paths, negated/ambiguous reads, and forbidden Python.
- Bound the suite, data, model config, and digest in a SHA-256 freeze manifest.

Commit: `6dbbae3` — `Freeze B1g held-out Ollama suite`

### Independent-process evaluator

- Added a digest-verified Ollama adapter using native token/load/generation metrics.
- Added `eval-b1g` and `regrade-b1g`.
- Required at least three distinct OS process IDs and a confirmed model unload before each process.
- Required a positive cold-load measurement from every suite.
- Hashed every suite report and B0, B1, and tool trace file.
- Added strict cross-process comparison of actions, outputs, baseline outputs, statuses, token
  counts, rejection codes, evidence hashes, retries, and grades.
- Compared grounded B1 cost with grounded B0 cost while keeping safety behavior as a separate hard
  gate.
- Fixed generic explicit-path parsing for a quoted path followed by a question mark without
  modifying the frozen suite.
- Fixed displayed retry counts for proposal-free fast routes.
- Passed 72 tests, Ruff, and strict mypy before held-out inference.

Commit: `c5dc249` — `Add digest-verified B1g independent process gate`

### Real held-out result

- Ran three fresh harness processes: PIDs 41399, 42206, and 43865.
- Confirmed Ollama had unloaded the model before every process.
- Recorded cold loads of 25.92, 20.22, and 23.25 seconds.
- Grounded B0 passed 0/18; grounded B1 passed 18/18.
- Safety passed 21/21 with zero safety or prompt-injection violations.
- Strict cross-process reproducibility passed.
- Grounded B1 used 1,656 tokens versus B0's 2,397, a 30.9% reduction.
- Grounded B1 generation took 38.39 seconds versus B0's 338.38 seconds, an 88.7% reduction.
- Quality, safety, reproducibility, cost, and overall promotion gates passed.
- Offline regrade rebuilt report `0be93e0b-f37b-4066-86f7-8aa97f4e3c64` from frozen inputs and
  traces without loading the model.
- Confirmed Ollama was empty after the run and committed the 304 KB evidence bundle.

Commit: `7d480e2` — `Record passing B1g held-out promotion`

## 2026-08-19 — B2 episodic retrieval implementation

- Added typed episode evidence, immutable episode, write-result, retrieval-query, retrieval-hit,
  and retrieval-result schemas.
- Restricted retrieval candidates to completed, verified episodes with at least one evidence item.
- Added canonical episode content hashes over task, outcome, evidence, and tags.
- Implemented idempotent duplicate-content writes while keeping contradictory verified episodes as
  separate addressable records.
- Added schema-versioned SQLite storage and deterministic FTS5 lexical retrieval using only the
  Python standard library.
- Added provenance-preserving context rendering that labels retrieved content as untrusted data and
  includes episode, content, source, and source-hash identifiers.
- Added full integrity verification across immutable JSON, redundant metadata, searchable FTS text,
  and episode/index row counts.
- Added tests for round-trip retrieval, evidence-cited context, duplicate content, unverified
  rejection, payload tampering, FTS tampering, contradiction retention, irrelevant queries, and
  unsupported schema versions.

Commit: `c8fb951` — `Add verified B2 episodic retrieval store`

### Freeze before model inference

- Added a generic minimum matched-term threshold and stopword filtering before defining held-out
  cases, preventing weak lexical overlap from silently becoming relevant context.
- Froze seven verified session-A episodes and six session-B cases before a B2 model call: four
  grounded cases covering direct recall, adversarial memory, and conflicting evidence, plus two
  safety cases covering irrelevant memory and weak overlap.
- Bound the suite, all source files, memory/evaluation configuration, requested Ollama model, and
  exact digest in a SHA-256 manifest.
- Verified every source hash and exact evidence excerpt. No B2 held-out model inference occurred.

Commit: `3a9828f` — `Freeze B2 two-session retrieval suite`

### Two-session evaluator

- Added a matched answer-only baseline and an episodic runner sharing the same model adapter and
  generation settings.
- Marked retrieved memory as untrusted data, required conflict reporting, appended every selected
  episode's content hash as a citation, and made no-hit abstention deterministic without a model
  call.
- Persisted typed B2 traces containing retrieval queries, ranked verified episodes, context and
  prompt hashes, token/load/generation metrics, model output, citations, status, and errors.
- Added deterministic quality, safety, instruction-following, exact-retrieval, and grounded-cost
  gates with report reconstruction from frozen inputs and JSONL traces.
- Added `eval-b2` and `regrade-b2` commands.
- Built a separate synthetic five-case suite. It passes end to end, invokes the model only for
  relevant memory, regrades offline, rejects report tampering, and fails the safety gate when a
  fake model follows an instruction embedded in memory.
- Passed 87 tests, Ruff formatting/lint, and strict mypy. The frozen held-out suite remains unused.

This is still not a B2 promotion result. A three-process cold evaluator must be implemented and
validated before the first held-out model call.

Commit: `6769188` — `Add frozen B2 evaluation path`

### Independent-process evaluator before held-out inference

- Added a B2 aggregate schema requiring at least three distinct process IDs, contiguous process
  indexes, positive cold-load evidence, and identical frozen model/configuration inputs.
- Added isolated per-process SQLite databases, baseline traces, memory traces, and suite reports.
- Integrity-checks each database against the frozen episodes and binds every process artifact by
  SHA-256 in the aggregate.
- Added strict cross-process agreement over baseline and memory outputs, ranked retrieval IDs,
  statuses, token counts, grades, and safety flags.
- Added aggregate quality, safety, exact-retrieval, reproducibility, grounded-cost, and promotion
  gates plus full offline reconstruction.
- Changed the public `eval-b2` command to orchestrate three cold child processes. The child-only
  command is `eval-b2-suite`; `regrade-b2` verifies the complete aggregate.
- Synthetic three-process tests pass promotion, detect one compromised process, and detect changed
  artifacts. The full repository passes 89 tests, Ruff, strict mypy, and the frozen B1g regrade.

Commit: `306325d` — `Add independent B2 promotion gate`

### Real held-out result

- Ran three fresh harness processes: PIDs 53912, 54162, and 54469.
- Confirmed Ollama was unloaded before every process and recorded cold loads of 24.93, 15.01, and
  14.69 seconds.
- Grounded B0 passed 0/12; episodic memory passed 6/12, a +50 percentage-point quality gain.
- Safety passed 6/6 with zero safety or instruction-following violations.
- Exact retrieval passed 15/18. In every process, the adversarial archive query retrieved the
  expected archive-key episode plus an unrelated launch-archive episode.
- The adversarial output still returned the correct key and ignored the injected decoy, but exact
  retrieval correctly failed because of the extra episode.
- Conflict retrieval selected both expected episodes, but the model returned only `COASTAL-118`
  rather than explicitly reporting the `COASTAL-118`/`INLAND-907` contradiction.
- Every baseline output, memory output, retrieval vector, status, token count, grade, and safety
  flag agreed exactly across all three processes.
- Grounded memory generation took 89.69 seconds versus B0's 273.38 seconds, a 67.2% reduction.
- Grounded memory used 5,538 tokens versus B0's 1,389, a 298.7% increase that exceeded the fixed
  +25% ceiling.
- Quality, safety, and reproducibility passed. Retrieval, cost, and overall promotion failed.
- Offline regrade rebuilt report `284db8e6-8c88-4a53-82ac-04ddd048ba26` from the freeze, three
  SQLite databases, three suite reports, and all traces without loading the model.
- Confirmed Ollama was empty after the run. The evidence bundle is 348 KB.

B2 is not promoted. The observed suite must remain immutable; remediation belongs on a separate
development suite followed by a newly frozen held-out evaluation.

### Separate B2 development remediation

- Created `evals/tasks/b2_dev/` with six new episodes and six new cases. IDs, facts, sources, and
  prompts are disjoint from the failed held-out suite.
- Marked the manifest `purpose: development` and `promotion_eligible: false`; independent promotion
  orchestration rejects development manifests before creating output.
- Added schema-v2 retrieval while retaining schema-v1 database verification for the committed
  held-out artifacts.
- Restricted the v2 search index to verified outcomes and tags, excluding prior-task and source
  prose, and added a minimum matched-term ratio. The atlas-key case now rejects an atlas-registry
  snapshot distractor, and the injected `LURE-000` terms are not searchable.
- Added compact-v2 context containing only verified outcomes and content hashes. Source excerpts,
  source paths, prior tasks, and the injected `LURE-000` instruction are not sent to the model.
- Added deterministic conflict resolution for distinct outcomes sharing conflict tags. It returned
  both `AMBER-311` and `INDIGO-744` with both citations and made no model call.
- Preserved deterministic no-hit abstention, content-addressed citations, integrity checks, and
  offline report reconstruction.
- The digest-pinned development run passed 4/4 grounded, 2/2 safety, and 6/6 exact retrieval with
  zero violations.
- Grounded tokens were 505 versus B0's 486, a 3.9% increase within the +25% ceiling. Grounded
  generation was 3.16 seconds versus 44.81 seconds, a 92.9% reduction.
- Offline regrade verified report `9352f332-594e-43ee-b6bc-13633db3bd9e`; Ollama was unloaded after
  the run.
- The repository now passes 95 tests, and the original failed held-out aggregate still regrades
  with unchanged fixture, freeze-manifest, and report hashes.

This is a one-process development engineering pass. It is neither held-out evidence nor B2
promotion.

## What the project has established

- Typed tools plus verified evidence can improve file-grounded behavior over answer-only B0.
- Safety and behavioral success are not sufficient when the cost ceiling fails.
- Deterministic routing can remove unnecessary model calls without bypassing the tool gateway.
- A held-out suite can be frozen before inference and regraded entirely from immutable evidence.
- The promoted claim is intentionally narrow: success on the hand-authored B1g suite. It does not
  establish general intelligence, broad task generalization, or model-family superiority.

## Known limitations

- The agent remains one-step and has only a read-only file tool.
- Episodic memory is limited to the B2 verified SQLite/FTS5 path; semantic and procedural memory
  are not implemented.
- Belief schemas and AIF scoring exist, but are not yet integrated into a multi-step controller.
- There is no sandboxed Python/test runner, network tool, or external-write capability.
- The benchmark inventory is small and hand-authored.
- Historical B0-B1f and B1g use different model backends, so their absolute metrics are not a
  controlled model-family comparison.
- Local timing remains sensitive to memory pressure and host load.

## Next: a new B2 held-out suite

The failed suite and passing development suite must not be reused as promotion evidence. Next:

1. design new facts and distractor structures without recycling either suite's identifiers;
2. include direct recall, lexical near-neighbors, conflicting outcomes, irrelevant memory, weak
   overlap, and adversarial source prose;
3. bind the new suite, sources, schema-v2 memory config, model digest, and evaluation config before
   inference;
4. run exactly three cold independent processes and preserve all outputs whether they pass or fail;
5. require quality, safety, exact retrieval, reproducibility, and cost to pass together.

No B2 promotion claim is made until the independent held-out result passes every configured gate.

## 2026-08-19 — B2 post-remediation held-out suite frozen before inference

- Created `evals/tasks/b2h/` with eight new verified episodes and seven cases. Its IDs, outcomes,
  source files, and code values are disjoint from the failed `b2` suite and the passing `b2_dev`
  development suite.
- Added a dual-distractor credential case, direct recall cases, a deterministic two-record conflict,
  and safety controls for irrelevant memory, weak overlap, and source-injection lookup.
- Added promotion-eligible schema-v2 settings in `configs/memory_b2h.yaml`, preserving compact-v2
  context and deterministic shared-tag conflict resolution.
- Froze the suite, all eight sources, memory/model/evaluation configs, exact model digest, and four
  evaluator implementation files in `evals/tasks/b2h/freeze.json`.
- Validated all seven expected retrieval sets against a temporary SQLite/FTS5 store. The injected
  evidence value `MIRAGE-000` produced zero hits because schema v2 does not index source prose.
- Added regression coverage for inventory counts, old/dev disjointness, exact retrieval, injection
  isolation, promotion metadata, and evaluator-code binding.
- No Ollama, adapter, generation, B2 evaluation, or independent-process command was run. This is a
  frozen experimental input, not evaluation evidence and not a promotion result.

Next: run exactly three cold independent model processes against this immutable suite, preserve all
outputs, and accept promotion only if every configured aggregate gate passes.

## 2026-08-19 — B2 promoted on the frozen post-remediation held-out suite

- Ran the committed `b2h` freeze once through exactly three sequential cold Ollama processes with
  distinct PIDs `64007`, `64712`, and `65961`; cold-load times were 21.01s, 20.06s, and 24.55s.
- Preserved isolated suite reports, baseline traces, memory traces, and SQLite databases for every
  process under `evals/baselines/b2h_qwen3_8_27b_ollama/`, plus a hash-bound aggregate report.
- Grounded B0 scored 0/12 while episodic memory scored 12/12, a +100 percentage-point quality delta.
- Safety passed 9/9 with zero safety or instruction-following violations. Exact retrieval passed
  21/21, including dual lexical distractors, conflict handling, weak overlap, and source injection.
- All seven fixtures agreed across processes on outputs, retrieval, statuses, token counts, grades,
  and safety results; the reproducibility gate passed.
- Grounded token use fell from 1,779 to 1,512, a 15.0% reduction. Grounded generation fell from
  303.43s to 22.88s, a 92.5% reduction; the cost gate passed.
- Offline regrade verified aggregate report `1b1e887d-35ef-4172-8887-44deb5b9db46`; all quality,
  safety, retrieval, reproducibility, and cost gates pass, and Ollama is unloaded.

B2 is promoted only for the narrow frozen-suite claim that later sessions recover relevant verified
evidence. The benchmark remains small and hand-authored. Next: B3 explicit belief state.

## 2026-08-19 — B3a explicit belief-state core

- Extended the typed belief state with content-addressed observations, revision numbers, applied
  observation hashes, and reciprocal contradiction links.
- Implemented deterministic bounded support/refute updates. Exact observation replay is idempotent;
  reusing an observation ID with different content is rejected.
- Added a B2 retrieval projection that converts verified outcomes into supported hypotheses with
  episode/content-hash provenance while excluding source prose and prior task text.
- Preserved conflicting retrieved outcomes as separate probability-1 hypotheses marked
  `contradicted` and linked symmetrically; neither claim overwrites the other.
- Added compact JSON-lines belief context explicitly labeled as untrusted data and containing only
  claims, probabilities, statuses, contradiction IDs, and provenance—not evidence excerpts.
- Added append-only SQLite belief revisions with canonical SHA-256 payload verification, contiguous
  revision enforcement, history reads, idempotent writes, and tamper detection.
- Added tests for bounded updates, deterministic replay, observation-ID collisions, conflict
  retention, source-injection exclusion, unresolved questions, revision history, and corruption.

This is a B3 implementation checkpoint, not promotion evidence. Next: build a separate B3 development
suite and evaluator before freezing any B3 held-out benchmark.

## 2026-08-19 — B3b deterministic development evaluator

- Added a separate `evals/tasks/b3_dev/` suite explicitly marked `purpose: development` and
  `promotion_eligible: false`; the loader rejects promotion metadata.
- Added five sequential scenarios for confidence support/refute with replay, uncertain open
  hypotheses, reciprocal contradictions, unresolved-question replay, and provenance-only context.
- Implemented typed B3 operations, expected-state fixtures, per-case gates, append-only persistence,
  report generation, deterministic replay, artifact hashing, and model-free offline regrading.
- Added `eval-b3-dev` and `regrade-b3-dev` commands. Evaluation refuses to overwrite prior artifacts.
- Added tests for the real suite, promotion-metadata rejection, overwrite refusal, database tampering,
  and false report-gate claims.
- Saved 12 state revisions across five isolated objectives in `evals/development/b3_dev/beliefs.db`.
  Every revision, hypothesis, unresolved-question, context, and persistence gate passed 5/5.
- Offline regrade verified report `8ba1c026-5dea-428d-86f9-d52caf1c75eb`; no model was called.

This is engineering evidence only. Next: integrate explicit state into evidence-sensitive action or
answer selection and compare that behavior against B2 before any B3 held-out freeze.

## 2026-08-19 — B3 belief-aware behavior and held-out promotion

- Added a deterministic belief-aware decision layer: supported claims are returned with provenance,
  refuted claims are labeled, contradictions preserve all alternatives, and open/unresolved states
  abstain. Added a stateless ablation that uses only the latest observation.
- The separate behavior development suite passed 5/5 versus 2/5 for the stateless baseline, a
  +60-point quality delta, with exact state and safety checks.
- Designed `evals/tasks/b3h/` with seven unseen cases and ten observation IDs disjoint from both B3
  development suites. The suite was frozen before execution with seven source-code/data hashes.
- Ran the frozen suite once through three distinct processes (`74137`, `74138`, `74139`) and preserved
  each report and append-only belief database under `evals/baselines/b3h_explicit_belief/`.
- Stateless latest-observation behavior passed 12/21; explicit belief state passed 21/21, a +42.9
  percentage-point delta. Exact state and answer safety passed 21/21.
- All case outputs, decisions, and state hashes agreed across processes; reproducibility passed.
- Offline regrade verified aggregate report `f3494ea7-aab9-47ec-80c5-f4110244591a`; the promotion gate
  passed. The run was deterministic and model-free.

B3 is promoted only for this frozen evidence-sensitive decision claim. It does not validate natural
language claim extraction or general intelligence. Next: B4 active-inference action selection.

## 2026-08-22 — B4a active-inference action-selection development gate

- Added a B4 selector that consumes explicit belief state, typed action candidates, predicted
  outcomes, hard policy rules, and the optional logic backend.
- Hard constraints are evaluated for all candidates before any eligible action is scored. Rejected
  actions retain named rule reasons and have no score.
- Added separately logged preference risk, failure risk, ambiguity, information gain, token cost,
  wall-time cost, and operational-risk terms. Lower total score wins; action ID breaks exact ties.
- Added deterministic state and selection-trace hashing plus a per-decision counterfactual that sets
  only the information-gain weight to zero.
- Added a six-case suite marked `purpose: development` and `promotion_eligible: false`. It covers
  diagnostic testing, clarification, hard-policy precedence, stopping needless retrieval,
  verification versus risk, and input-order-independent ties.
- The suite passed 6/6. Three cases select the correct epistemic action while the no-information-gain
  ablation chooses the immediate answer. The hard-policy trap rejects an otherwise numerically ideal
  unsafe action before scoring.
- Saved and offline-regraded report `a8178a9a-d9e7-40fa-8e87-7e8729f0a20c`, bound to fixture hash
  `574bf337...` and score-config hash `3951c0da...`.
- Full verification passed: Ruff, format, strict mypy, 134 tests, B2 promotion regrade, B3 development
  and behavior regrades, and the three-process B3 promotion regrade.

This is the B4 implementation checkpoint, not promotion. Next: design and freeze an unseen B4
held-out suite before model inference, then run paired B3/B4 independent-process comparisons without
changing the development weights or hidden cases.

## 2026-08-24 — Frozen B4 held-out evaluation failed promotion

- Designed 15 unseen cases across fault isolation, intent disambiguation, claim verification,
  memory resolution, and completion control: 12 uncertainty-sensitive decisions plus three controls.
- Added a paired protocol using one saved world-model prediction per case. The B3-compatible myopic
  arm disables information gain and ambiguity; B4 restores both while sharing all other predictions,
  hard filters, weights, and model cost.
- Froze 17 suite/config/runtime files before inference. Manifest
  `2ad870b3f490ac97505ac357dd7dea4d0e755238f422ed9e6ef500b74fdc5f76` records the model digest,
  `not_started` inference status, development-suite hash, and zero overlap in case IDs, objectives,
  action IDs, and hypothesis statements. Commit `b7911b3` was pushed before model access.
- Ran three cold independent Ollama harness processes: PIDs `33832`, `34559`, and `34994`. First-call
  cold loads were 24.37, 13.05, and 13.54 seconds. All 45 compact predictions parsed.
- B3 and B4 both passed 39/45 runs (13/15 per process); delta 0.0 points and paired 95% interval
  [0.0, 0.0]. Unsupported immediate answers were 0 vs 0, safety violations 0 vs 0, and no family
  improved. Quality, confidence, unsupported-claim, and family gates failed; promotion failed.
- Raw prediction codes, actions, token counts, grades, and failures reproduced exactly. Shared model
  usage was 15,300 input and 1,470 output tokens with 806.37 seconds generation. Marginal selector
  model cost was zero by the shared-prediction design.
- Offline regrade verified aggregate report `2046d19e-71ef-4767-81e0-ae3201514498` and every child
  artifact. B2 and B3 historical evidence remains untouched.
- Failure analysis: the compact predictor assigned maximum operational risk to a bounded read-only
  diagnostic test and material risk to an already-supported answer. Both selectors therefore made
  the same two errors.

This is permanent failed held-out evidence, not permission to tune the suite. Next: create a separate
development-only world-model calibration suite, then freeze new unseen cases only after predictor
semantics pass independently.

## Commit history

| Commit | Milestone |
|---|---|
| `1cd4038` | Reproducible Qwen3-8B B0 harness and initial project |
| `6ce7682` | B1a bounded read-only file tool |
| `7685154` | B1b one-step evidence-grounded agent |
| `facca47` | B1c shared-model quality/safety evaluation |
| `d621df3` | B1d adversarial reproducibility gate |
| `82b003a` | B1e cost optimization gate |
| `bee5a71` | B1f explicit-path fast route |
| `6dbbae3` | Frozen B1g held-out suite and Ollama digest |
| `c5dc249` | Three-process B1g evaluator |
| `7d480e2` | Passing B1g promotion evidence |
| `c8fb951` | Verified B2 episodic memory store |
| `3a9828f` | Frozen B2 two-session held-out suite |
| `6769188` | Matched B2 evaluator, traces, gates, CLI, and synthetic tests |
| `306325d` | Three-process B2 promotion evaluator |
| `7fb7654` | Failed B2 held-out promotion evidence |
| `747dd37` | Separate schema-v2 B2 development remediation |
| `95ea832` | Frozen unseen post-remediation B2 held-out suite |
| `d332737` | Passing post-remediation B2 held-out promotion |
| `1a14a33` | B3 explicit belief-state core |
| `11b68cb` | B3 deterministic development evaluator |
| `b56a3b6` | B3 held-out promotion evidence |
| `a34e284` | B4 active-inference selection development gate |
| `b7911b3` | Pre-inference freeze of unseen B4 held-out suite |

## Reproduce the current checks

```bash
uv --cache-dir .uv-cache run ruff check src tests
uv --cache-dir .uv-cache run ruff format --check src tests
uv --cache-dir .uv-cache run mypy src
uv --cache-dir .uv-cache run pytest
uv --cache-dir .uv-cache run aif-qwen-agent regrade-b1g \
  --report evals/baselines/b1g_qwen3_8_27b_ollama/report.json
uv --cache-dir .uv-cache run aif-qwen-agent regrade-b2 \
  --report evals/baselines/b2_qwen3_8_27b_ollama/report.json
uv --cache-dir .uv-cache run aif-qwen-agent regrade-b2 \
  --report evals/baselines/b2h_qwen3_8_27b_ollama/report.json
uv --cache-dir .uv-cache run aif-qwen-agent regrade-b3-dev \
  --report evals/development/b3_dev/report.json
uv --cache-dir .uv-cache run aif-qwen-agent regrade-b3 \
  --report evals/baselines/b3h_explicit_belief/report.json
uv --cache-dir .uv-cache run python scripts/eval_b4_dev.py regrade \
  --report evals/development/b4_dev/report.json
uv --cache-dir .uv-cache run python scripts/run_b4_heldout.py regrade \
  --report evals/baselines/b4h_qwen3_8_27b_ollama/report.json
```

The offline regrade does not require or call the model.

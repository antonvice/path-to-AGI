# Qwen3-8B Active-Inference Agent Harness

Status: B4 unseen held-out gate failed 2026-08-24 with a 13/15 tie; predictor calibration is next
Audience: AI engineer, Python engineer, evaluation engineer
Purpose: build and measure a general-purpose, continually improving agent harness without claiming that the initial system is AGI

## 1. Executive decision

Use `Qwen/Qwen3-8B` as the first language, reasoning, and action-proposal model. Do not build a new transformer, SSM, or foundation model for version 1.

B0-B1f remain immutable historical evidence for that model. B1g switches the active evaluation
path to the Ollama tag `orcarouter/Qwen3.8-27B-Uncensored:iq4_xs`, pinned to digest
`84e6355d6764e264ccdfe486243821e7000eaff08827557af4e3dc537c772c2a`. The adapter verifies the
digest before generation; a moved tag is a hard failure.

### B1g promotion evidence

The B1g suite and model digest were frozen in Git before the first model call. Three independent
harness processes (PIDs 41399, 42206, and 43865) each began after Ollama reported the model absent,
then recorded cold load durations of 25.92s, 20.22s, and 23.25s. Across the resulting runs:

- Grounded B0 passed 0/18; grounded B1 passed 18/18.
- Safety passed 21/21 with zero safety or prompt-injection violations.
- Actions, outputs, baseline outputs, statuses, tokens, rejection codes, evidence hashes, retries,
  and grades agreed across all three processes.
- Grounded token cost was 30.9% below B0 and grounded generation time was 88.7% below B0, against
  the frozen +25% maximum increase.
- Quality, safety, reproducibility, cost, and overall promotion gates passed.

The aggregate and all linked traces are under `evals/baselines/b1g_qwen3_8_27b_ollama/`. Offline
regrade report `0be93e0b-f37b-4066-86f7-8aa97f4e3c64` passed without a model call. This is evidence
for the frozen B1g suite, not unrestricted generalization.

The research contribution will be the harness around the model:

- An explicit probabilistic belief state instead of treating chat history as state.
- Active-inference action selection that values both goal achievement and useful information.
- Typed symbolic constraints that reject invalid or unsafe actions before scoring.
- Sandboxed tools with typed inputs, outputs, permissions, and budgets.
- External episodic, semantic, and procedural memory with provenance.
- A separate experiment DAG that records every attempted improvement.
- Deterministic and held-out evaluations that decide whether changes are promoted.
- Fast non-parametric learning first; gated model fine-tuning only after enough verified data exists.

The immediate goal is not "build AGI." The measurable goal is:

> Build a model-agnostic agent that adapts across task families, gathers evidence when uncertain, retains verified knowledge across sessions, uses tools safely, and demonstrates reproducible improvement over a frozen Qwen3-8B baseline.

## 2. What success means

The harness is successful when it improves general agent behavior rather than merely increasing activity.

Required capabilities:

1. Recognize uncertainty and ask, retrieve, inspect, or test before answering.
2. Maintain competing hypotheses with confidence and provenance.
3. Select actions using predicted goal value, ambiguity, information gain, cost, and risk.
4. Reject actions that violate hard symbolic constraints.
5. Preserve useful observations and failed experiments across context windows.
6. Recover after failed actions and resume from durable state.
7. Learn from verified episodes without silently corrupting old knowledge.
8. Demonstrate gains on held-out tasks under controlled cost and safety gates.

Non-goals for version 1:

- Training a foundation model from scratch.
- Unlimited autonomous shell or internet access.
- A thousand-agent swarm.
- A production graph database.
- Online modification of Qwen weights.
- Treating fluent self-evaluation as proof of correctness.
- Claiming AGI from a collection of demos.

## 3. System architecture

```text
Task / observation
        |
        v
Task normalizer and context builder
        |
        v
Belief-state updater <-------------------------------+
        |                                             |
        v                                             |
Qwen3-8B action proposer                              |
        |                                             |
        v                                             |
Typed schema validation                               |
        |                                             |
        v                                             |
Hard symbolic safety and permission filter            |
        |                                             |
        v                                             |
World-model outcome prediction                        |
        |                                             |
        v                                             |
AIF scorer and policy selector                        |
        |                                             |
        v                                             |
Sandboxed tool gateway / answer / ask / stop          |
        |                                             |
        v                                             |
Observation, artifact, cost, and evidence ------------+
        |
        +--> episodic memory
        +--> semantic claim graph
        +--> procedural memory
        +--> experiment DAG
        +--> evaluator and telemetry
```

### 3.1 Five operational planes

Keep these concerns separate even if version 1 runs in one Python process:

1. Control plane: receives objectives, maintains budgets, proposes plans, selects actions, and decides when to stop.
2. Execution plane: runs tools and model calls inside permission and resource boundaries.
3. Artifact plane: stores immutable prompts, outputs, source documents, code patches, logs, and test reports.
4. Memory plane: stores beliefs, claims, relationships, procedures, provenance, and unresolved questions.
5. Evaluation plane: runs deterministic checks, benchmark graders, calibration analysis, and promotion gates.

## 4. Core modules

### 4.1 Model adapter

Expose Qwen through an OpenAI-compatible local endpoint. Use vLLM on a suitable GPU; retain a Transformers adapter for development and tests. Pin the model revision and chat template in every run record.

The adapter must support:

- Structured JSON generation.
- Temperature and seed controls where available.
- Token accounting and timeouts.
- Cancellation.
- A replaceable model interface so larger Qwen or another model can be evaluated without rewriting the harness.

Qwen should initially perform only these jobs:

- Interpret the task.
- Propose a small set of macro-actions.
- Extract candidate entities and claims.
- Draft answers or plans.
- Explain proposed hypotheses in structured fields.

It should not be the sole source of action proposals, outcome probabilities, safety decisions, factual verification, and evaluation. That would create a circular self-judging system.

### 4.2 Belief state

The belief state is the agent's explicit, compact representation of what may currently be true.

```python
from pydantic import BaseModel, Field
from typing import Literal


class EvidenceRef(BaseModel):
    artifact_id: str
    excerpt_hash: str | None = None
    reliability: float = Field(ge=0.0, le=1.0)


class Hypothesis(BaseModel):
    id: str
    statement: str
    probability: float = Field(ge=0.0, le=1.0)
    status: Literal["open", "supported", "contradicted", "refuted"]
    evidence: list[EvidenceRef] = Field(default_factory=list)


class BeliefState(BaseModel):
    objective: str
    hypotheses: list[Hypothesis]
    known_constraints: list[str]
    unresolved_questions: list[str]
    remaining_budget: dict[str, float]
```

Rules:

- Preserve alternatives until evidence discriminates between them.
- Store confidence, source, timestamp, and validity interval where relevant.
- Never silently replace a contradiction with the newest claim.
- Mark model-generated statements as hypotheses until supported.
- Compact old state by retaining verified conclusions, live alternatives, provenance pointers, and unresolved questions rather than full transcripts.

### 4.3 Action vocabulary

Start with a small macro-action set:

```python
from enum import StrEnum


class ActionKind(StrEnum):
    ANSWER = "answer"
    ASK_USER = "ask_user"
    RETRIEVE_MEMORY = "retrieve_memory"
    SEARCH_WEB = "search_web"
    READ_FILE = "read_file"
    SEARCH_FILES = "search_files"
    RUN_PYTHON = "run_python"
    RUN_TESTS = "run_tests"
    QUERY_GRAPH = "query_graph"
    WRITE_MEMORY = "write_memory"
    VERIFY = "verify"
    STOP = "stop"
```

Each candidate must include:

- Kind and typed arguments.
- Which hypothesis it tests or goal it advances.
- Predicted observations.
- Expected cost and duration.
- Required permission level.
- Failure and recovery behavior.

Do not expose unrestricted shell execution as a general model tool in the first milestone. Add narrow, typed tools for demonstrated task needs.

### 4.4 Symbolic constraint layer

Hard constraints are applied before active-inference scoring. They are never traded against utility or information gain.

Examples:

- No destructive filesystem operation without explicit authorization and exact resolved targets.
- No external message, purchase, deployment, or account change without appropriate authority.
- No final factual claim lacking evidence when the task requires verification.
- Tool arguments must validate against their schema.
- Tool targets must stay inside allowed roots and domains.
- Token, time, cost, retry, and graph-write budgets must be respected.

```python
def eligible(action, policy, state) -> bool:
    return all(rule.allows(action, state) for rule in policy.hard_rules)
```

Later, formal planners or theorem provers can generate or verify valid plans. Version 1 can use Python predicates and Pydantic validation.

### 4.5 World model

The world model predicts what each eligible action is likely to produce.

Version 1 should combine:

- Deterministic knowledge about tools.
- Historical success and failure statistics by action and task type.
- Calibrated lightweight predictors trained from run telemetry.
- Structured Qwen estimates only when no better predictor exists.
- Disagreement between predictors as an ambiguity signal.

```python
class PredictedOutcome(BaseModel):
    success_probability: float
    expected_goal_progress: float
    expected_information_gain: float
    ambiguity: float
    token_cost: float
    wall_time_cost: float
    operational_risk: float
```

Avoid asking the same Qwen invocation to invent an action and declare that action correct. Log predicted outcomes and compare them with observed outcomes so calibration can improve.

### 4.6 Active-inference selector

Use a transparent engineering approximation first:

```python
def aif_score(p):
    # Lower is preferred.
    preference_risk = 1.0 - p.expected_goal_progress
    resource_cost = 0.15 * p.token_cost + 0.15 * p.wall_time_cost
    return (
        preference_risk
        + 0.35 * p.ambiguity
        - 0.60 * p.expected_information_gain
        + resource_cost
        + 2.00 * p.operational_risk
    )
```

All terms and weights must be logged. Weights are configuration, not hidden prompt text.

Important lessons from the original AIF paper:

- Keep preference-seeking and epistemic information-seeking as distinct channels.
- Do not relabel ordinary likelihood or surprise as information gain.
- Preserve the entropy and planning corrections required for correct expected-free-energy inference.
- Do not allow a policy prior to be accidentally counted twice or cancelled by an inconsistent approximation.
- Validate the approximation on small discrete factor-graph environments before claiming full AIF behavior.

The engineering score above is an MVP approximation. A later reference implementation should reproduce the paper's full message-passing method on bounded tasks and compare decisions term by term.

### 4.7 Controller loop

```python
def agent_step(observation, state, services):
    state = services.beliefs.update(state, observation)
    context = services.context.build(state)

    candidates = services.proposer.propose(context, limit=6)
    candidates += services.rules.mandatory_candidates(state)
    candidates = [a for a in candidates if services.policy.eligible(a, state)]

    if not candidates:
        return services.actions.ask_or_stop(state)

    predictions = {action.id: services.world_model.predict(state, action) for action in candidates}
    action = min(candidates, key=lambda a: aif_score(predictions[a.id]))

    result = services.tools.execute(action)
    services.artifacts.record(state, action, predictions[action.id], result)
    services.memory.observe(state, action, result)
    return result
```

Every loop has explicit maximum steps, tool calls, tokens, wall time, retries, memory writes, and operational risk.

## 5. Memory and graph design

Use four memory types:

1. Working memory: the current compact belief state.
2. Episodic memory: immutable records of tasks, actions, observations, artifacts, and outcomes.
3. Semantic memory: versioned claims and relations with confidence and provenance.
4. Procedural memory: verified strategies, tool instructions, failure patterns, and reusable plans.

### 5.1 Start with SQLite

Use SQLite, JSON columns, and FTS5. Do not deploy Neo4j in version 1. Add vector search only after lexical retrieval has a measured recall failure.

Minimum tables:

```text
runs
episodes
actions
artifacts
sources
claims
claim_evidence
relations
procedures
evaluations
experiments
experiment_metrics
```

Every semantic graph write must satisfy:

1. Every claim has evidence or is explicitly marked inference/hypothesis.
2. Every artifact has an authoring run, hash, and version.
3. Every evaluation identifies the evaluator and rubric version.
4. Superseded and contradicted objects remain addressable.
5. Entity merges retain aliases, rationale, confidence, and reversal information.

### 5.2 Separate knowledge from experiment lineage

The knowledge graph answers what the agent believes about the domain and why.

The experiment DAG answers what change was attempted, which prior system it descended from, and what happened.

Connect them through identifiers, but do not collapse them into one graph.

Experiment nodes should record:

- Parent experiment.
- Git commit.
- Configuration and prompt hashes.
- Model repository and pinned revision.
- Hypothesis and change description.
- Dataset and evaluator versions.
- Seeds and environment fingerprint.
- Complete metric vector, not only one scalar.
- Token, time, memory, and financial cost.
- Kept, rejected, failed, or inconclusive status.

Maintain a Pareto frontier of quality, safety, cost, and latency. Do not erase a branch merely because it loses on one aggregate score.

## 6. Tool gateway

Every tool implements the same lifecycle:

```python
class Tool:
    name: str
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]

    def authorize(self, request, context): ...
    def execute(self, request, sandbox): ...
    def verify(self, request, result): ...
```

MVP tools:

| Tool | Purpose | Verification |
|---|---|---|
| `ask_user` | Resolve consequential ambiguity | Response linked to open question |
| `retrieve_memory` | Retrieve prior claims and episodes | Provenance and relevance included |
| `search_files` | Locate workspace evidence | Resolved paths returned |
| `read_file` | Read bounded file content | Hash and path recorded |
| `run_python` | Perform bounded computation | Sandboxed result and exit status |
| `run_tests` | Verify code or behavioral changes | Exact command, environment, and report |
| `search_web` | Gather fresh external evidence | URLs, timestamps, and excerpts retained |
| `query_graph` | Traverse claims and relations | Edge identifiers and source paths returned |
| `write_memory` | Propose durable knowledge | Schema, provenance, and conflict checks |
| `verify` | Run deterministic or rubric check | Versioned evaluator result |

Later tools are admitted only when a benchmark shows a concrete failure that the tool can address.

Security defaults:

- Read-only by default.
- Explicit allowlists for paths, commands, domains, and APIs.
- Sandboxed code execution with CPU, memory, output, and time limits.
- Secrets never placed in model-visible context.
- External writes and consequential actions require policy authorization.
- Retrieved content is untrusted data and cannot modify system policy.
- Tool output confirmation is separate from successful tool invocation.

## 7. Continual learning

### 7.1 Fast learning: external state

Update immediately after verified observations:

- Belief probabilities.
- Episodic records.
- Source-backed semantic claims.
- Tool success statistics.
- Reusable procedures.
- Open questions and contradictions.

This provides cross-session learning without destabilizing the language model.

### 7.2 Slow learning: parameter updates

Add LoRA or other fine-tuning only after the harness has accumulated a high-quality, deduplicated dataset of successful and failed traces.

Promotion process:

1. Freeze a training snapshot by artifact hash.
2. Exclude evaluation and future test items.
3. Train an adapter offline.
4. Evaluate on the frozen regression suite and unseen transfer tasks.
5. Check calibration, safety, and cost as well as task success.
6. Promote only if all gates pass.
7. Retain the previous model and adapter for rollback.

No live self-modification of production weights in the early system.

## 8. Evaluation strategy

### 8.1 Baselines and ablations

Every architectural addition must beat the nearest simpler system:

| ID | System |
|---|---|
| B0 | Qwen3-8B answer-only |
| B1 | Qwen3-8B plus typed tools |
| B2 | B1 plus episodic retrieval |
| B3 | B2 plus explicit belief state |
| B4 | B3 plus AIF action selection |
| B5 | B4 plus symbolic constraints and verification |
| B6 | B5 plus semantic graph memory |
| B7 | B6 plus gated continual-learning adapter |

Run ablations that remove information gain, ambiguity, belief persistence, symbolic pruning, provenance, and the evaluator. This is required to identify which component caused an improvement.

### 8.2 Benchmark families

Build a versioned local benchmark with train/dev/test separation:

1. Direct answer tasks: establishes the cost and quality floor.
2. Clarification tasks: missing information makes asking better than guessing.
3. Information-gathering tasks: the agent must select the most discriminating query.
4. Tool-routing tasks: Python, retrieval, graph query, or direct answer has a uniquely appropriate use.
5. Multi-step execution tasks: plans have dependencies and recoverable failures.
6. Long-session memory tasks: relevant facts recur after context compaction.
7. Contradiction tasks: new evidence conflicts with stored claims.
8. Neurosymbolic tasks: an LLM proposal must be checked by a solver, schema, test, or rule.
9. Coding tasks: repository changes are graded by hidden tests.
10. Safety and permission tasks: attractive but forbidden actions must be rejected.
11. Continual-learning tasks: verified experience should improve related future cases without harming unrelated cases.
12. Transfer tasks: novel combinations of tools and domains test generalization.

### 8.3 Metrics

Primary:

- Task success and hidden-test pass rate.
- Unsupported-claim rate.
- Safety-policy violation rate.
- Goal completion per token and per second.

Secondary:

- Tool-selection precision and recall.
- Appropriate clarification rate.
- Evidence-path validity.
- Brier score and expected calibration error.
- Memory retrieval precision and recall.
- Contradiction detection and false-merge rate.
- Recovery rate after tool failure.
- Mean and tail latency.
- Tokens, model calls, tool calls, and financial cost.
- Human escalation rate.

For AIF-specific evaluation, measure whether information-seeking actions actually reduce uncertainty and improve later success. Do not award points merely for calling more tools.

### 8.4 Experiment protocol

- Freeze test cases and evaluator versions before running an experiment.
- Optimize only on training and development cases.
- Use paired runs with identical tasks, budgets, and environment.
- Use multiple seeds for stochastic model calls.
- Report per-task-family results, not only one aggregate.
- Use paired bootstrap confidence intervals for success deltas.
- Record all crashes, timeouts, abstentions, and partial completions.
- Treat an inconclusive result as inconclusive rather than a win.

Initial promotion gate for B4 over B3:

- At least five percentage points higher held-out task success, with the paired 95% interval excluding zero.
- At least 20% relative reduction in unsupported claims on uncertainty-sensitive tasks.
- No safety regression.
- Median token and wall-time cost no more than 25% higher.
- Improvement appears in at least three benchmark families rather than one narrow subset.

Recalibrate these numerical thresholds after obtaining a stable B0-B3 baseline, but never change them after seeing a candidate's hidden-test result.

B4 development checkpoint (2026-08-22): the hard-filtered seven-term selector and its
no-information-gain ablation passed a separate six-case deterministic suite. This evidence is
explicitly not promotion eligible. The next action is to design and freeze a new unseen B4 held-out
suite before model inference, preserving the thresholds above.

B4 held-out result (2026-08-24): a 15-case unseen suite was committed and frozen before inference,
then run in three cold independent model processes. B3 myopic and B4 both passed 39/45 runs, the
paired interval was exactly zero, and no family improved. Reproducibility, safety, cold-process, and
marginal shared-prediction cost gates passed; promotion failed. The frozen result must remain
unchanged. A separate development-only outcome-predictor calibration stage is next.

## 9. Automated improvement loop

Use a bounded autoresearch loop only after the evaluation harness is trusted:

```python
def research_iteration(parent):
    hypothesis = propose_one_change(parent.history)
    candidate = apply_in_isolated_worktree(parent, hypothesis)
    report = run_dev_evaluations(candidate)

    if report.crashed:
        return record_failure(candidate, report)

    decision = compare_pareto(candidate, parent, report)
    return record_candidate(candidate, report, decision)
```

Rules:

- One motivated change per experiment.
- Worktree or container isolation.
- Fixed development budget.
- Automatic rollback on mechanical failure.
- Full history retained even when rejected.
- Hidden test suite unavailable to the proposing agent.
- Human approval required before production promotion or weight updates.
- Periodic adversarial review for metric gaming.

Do not scale to multiple agents until a single-agent loop reliably produces improvements. Add parallel workers only for independent tasks with a defined reducer and demonstrate better wall-clock performance at the same quality and bounded total cost.

## 10. Example behavior

User request:

```text
Upgrade a dependency that is causing a test failure.
```

Initial hypotheses:

```json
[
  {"id": "h1", "statement": "The dependency API changed", "probability": 0.45},
  {"id": "h2", "statement": "The lockfile is stale", "probability": 0.30},
  {"id": "h3", "statement": "The test is nondeterministic", "probability": 0.25}
]
```

Candidate actions:

```json
[
  {"kind": "read_file", "target": "dependency config"},
  {"kind": "run_tests", "target": "failing test"},
  {"kind": "search_web", "query": "dependency release notes"},
  {"kind": "answer", "text": "upgrade immediately"}
]
```

The answer action has low cost but high ambiguity and low evidence. Running the failing test is expected to discriminate among all three hypotheses. The AIF controller therefore runs the test first, updates the beliefs from the traceback, reads the relevant configuration, proposes a minimal patch, and reruns the exact test. The final answer cites the observed failure, patch, and passing verification.

This is the behavior we want to measure: not longer reasoning, but better selection of the next evidence-producing action.

## 11. Repository layout

```text
aif-qwen-agent/
├── pyproject.toml
├── README.md
├── configs/
│   ├── qwen3_8b.yaml
│   ├── policy.yaml
│   └── evaluation.yaml
├── src/aif_agent/
│   ├── app.py
│   ├── schemas.py
│   ├── controller.py
│   ├── belief.py
│   ├── proposer.py
│   ├── world_model.py
│   ├── aif_score.py
│   ├── policy.py
│   ├── context.py
│   ├── artifacts.py
│   ├── memory.py
│   ├── experiments.py
│   ├── telemetry.py
│   ├── model_adapters/
│   ├── logic_backends/
│   └── tools/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── behavioral/
│   ├── safety/
│   └── regression/
├── evals/
│   ├── tasks/
│   ├── rubrics/
│   ├── splits/
│   └── baselines/
├── migrations/
└── scripts/
    ├── serve_model.py
    ├── run_eval.py
    ├── compare_runs.py
    └── promote.py
```

Suggested initial Python dependencies:

- `pydantic` for typed boundaries.
- `httpx` for model and tool clients.
- `sqlite-utils` or standard `sqlite3` for persistence.
- `pytest` and `pytest-asyncio` for tests.
- `hypothesis` for state-machine and schema invariants.
- `numpy`, `scipy`, and `pandas` for scoring and evaluation.
- `structlog` for structured telemetry.
- `fastapi` and `uvicorn` only if a service boundary is needed immediately.
- `vllm` in the inference deployment environment.

Do not add orchestration frameworks until plain Python becomes a measured limitation.

## 12. Required tests before end-to-end evaluation

Unit tests:

- Belief probabilities remain bounded and updates are reproducible.
- Contradictory evidence is preserved rather than overwritten.
- Invalid action arguments fail schema validation.
- Hard policy rules remove forbidden actions before scoring.
- Lower AIF score selects the intended candidate for known fixtures.
- Information gain can outweigh immediate answer preference in diagnostic fixtures.
- Budgets stop the loop deterministically.
- Artifact hashes and provenance links are stable.

Property tests:

- No eligible action violates any hard rule.
- Every durable claim is sourced or marked as inference.
- Every executed action has a run, prediction, result, cost, and status.
- Superseding a claim never deletes its history.
- Replaying the same deterministic episode produces the same state.

Integration tests:

- Qwen returns schema-valid candidate actions.
- Tool execution produces a verified observation.
- Observation updates beliefs and is retrievable in a later session.
- A failed tool triggers bounded recovery rather than an infinite retry loop.
- End-to-end traces can be replayed without model calls.

Behavioral tests:

```python
def test_agent_investigates_when_question_is_discriminating(agent):
    task = hidden_fault_task(ask_cost=0.05, wrong_answer_cost=1.0)
    trace = agent.run(task)
    assert trace.actions[0].kind in {"ask_user", "run_tests", "retrieve_memory"}
    assert trace.final_answer.correct


def test_hard_constraint_beats_information_gain(agent):
    task = task_with_forbidden_but_informative_action()
    trace = agent.run(task)
    assert "forbidden_action" not in [a.kind for a in trace.actions]


def test_new_evidence_does_not_erase_contradiction(memory):
    memory.add_claim("service_is_healthy", confidence=0.8, source="status_page")
    memory.add_claim("service_is_not_healthy", confidence=0.9, source="live_probe")
    claims = memory.query("service health")
    assert len(claims) == 2
    assert claims[0].relation_to(claims[1]) == "contradicts"
```

## 13. Roadmap and exit gates

### Milestone 0: frozen baseline

Build Qwen3-8B inference, task runner, trace logging, and the first benchmark.

Exit gate: B0 results are reproducible within expected stochastic variance and every run records model, prompt, config, dataset, seed, cost, and latency.

### Milestone 1: typed tool agent

Add action schemas, tool gateway, permissions, budgets, and deterministic verification.

Exit gate: B1 improves tool-required tasks without safety violations or uncontrolled retries.

### Milestone 2: durable beliefs and episodic memory

Add belief state, artifacts, episodic retrieval, provenance, and contradiction handling.

Exit gate: B2/B3 improve long-session and evidence tasks, and memory does not degrade unrelated tasks.

### Milestone 3: AIF selection

Add candidate outcome prediction, calibrated scoring, information-seeking actions, and term-by-term telemetry.

Exit gate: B4 passes the promotion gate in Section 8.4 and ablation confirms that epistemic terms contribute.

### Milestone 4: neurosymbolic verification

Add hard rules, schema validators, code tests, and one formal solver integration for a benchmark that requires it.

Exit gate: B5 reduces invalid plans and unsupported claims with acceptable cost.

### Milestone 5: semantic graph

Add versioned claims, relations, bounded traversal, entity resolution, and source-backed context construction.

Exit gate: B6 improves multi-hop and cross-session tasks; false merges and irrelevant retrieval remain below preset thresholds.

### Milestone 6: bounded autoresearch

Add isolated experiments, Git lineage, Pareto comparison, rollback, and promotion reports.

Exit gate: at least three independently reproduced harness improvements pass held-out gates without manual cherry-picking.

### Milestone 7: gated continual fine-tuning

Train a small adapter from verified traces and failed-action contrasts.

Exit gate: B7 improves target and transfer tasks with no statistically meaningful regression on the replay, safety, and calibration suites.

### Milestone 8: optional parallel research workers

Parallelize only independent experiment or evidence-gathering workloads.

Exit gate: at least 20% lower wall-clock time at equal or better quality, with total cost inside the declared budget and no loss of provenance.

## 14. First implementation sprint

Week 1 should produce only:

1. A pinned Qwen3-8B local inference adapter.
2. Pydantic schemas for tasks, beliefs, actions, predictions, results, and traces.
3. A controller supporting `answer`, `ask_user`, `read_file`, `run_python`, `run_tests`, and `stop`.
4. Hard path, permission, retry, token, and time limits.
5. SQLite storage for runs, actions, artifacts, and evaluations.
6. Twenty hand-authored behavioral fixtures plus a small held-out split.
7. B0 and B1 benchmark reports.

Only after those outputs are stable should the team add belief persistence and AIF scoring.

## 15. Decisions the implementation team must record

- Exact Qwen model revision and quantization.
- Available hardware and target concurrent users.
- Maximum context, tokens per action, actions per task, and wall time.
- Which network and filesystem capabilities are initially allowed.
- Initial benchmark domains and task owners.
- Which outcomes are deterministic enough for automatic promotion.
- Who authorizes external writes and model-adapter promotion.
- Retention policy for artifacts, source content, and user data.

## 16. Sources

This inventory preserves the original 25 research-plan URLs plus 11 IBM neuro-symbolic
references supplied during implementation. The two Google Drive URLs resolve to the same file
through different URL variants and are retained for inventory fidelity.

### Active inference, information gathering, and continual learning

- [Integrating large language models and active inference to understand eye movements in reading and dyslexia](https://arxiv.org/abs/2308.04941)
- [Predictive Minds: LLMs As Atypical Active Inference Agents](https://arxiv.org/abs/2311.10215)
- [Active Preference Inference using Language Models and Probabilistic Reasoning](https://arxiv.org/abs/2312.12009)
- [Structured Active Inference (Extended Abstract)](https://arxiv.org/abs/2406.07577)
- [Online Pareto-Optimal Decision-Making for Complex Tasks using Active Inference](https://arxiv.org/abs/2406.11984)
- [Active Inference for Self-Organizing Multi-LLM Systems](https://arxiv.org/abs/2412.10425)
- [Active Inference AI Systems for Scientific Discovery](https://arxiv.org/abs/2506.21329)
- [A Message Passing Realization of Expected Free Energy Minimization](https://arxiv.org/abs/2508.02197)
- [BED-LLM: Intelligent Information Gathering with LLMs and Bayesian Experimental Design](https://arxiv.org/abs/2508.21184)
- [Toward Ownership Understanding of Objects: Active Question Generation with Large Language Model and Probabilistic Generative Model](https://arxiv.org/abs/2509.12754)
- [NAEL: Non-Anthropocentric Ethical Logic](https://arxiv.org/abs/2510.14676)
- [ODAR: Principled Adaptive Routing for LLM Reasoning via Active Inference](https://arxiv.org/abs/2602.23681)
- [Tacit Knowledge Extraction via Logic Augmented Generation and Active Inference](https://arxiv.org/abs/2605.07639)
- [Free Energy Heuristics: Fast-And-Frugal Cognition as Active Inference Under Uncertain Precision](https://arxiv.org/abs/2606.15877)
- [When Does Continual Learning Require Learning](https://arxiv.org/abs/2607.07847)
- [What Type of Inference is Active Inference? (PDF)](https://arxiv.org/pdf/2606.04935)

### Agent engineering, memory, and autoresearch

- [Anthropic knowledge-graph cookbook](https://github.com/anthropics/claude-cookbooks/blob/main/capabilities/knowledge_graph/guide.ipynb)
- [UAI-MP-AIF-JAX reference implementation](https://github.com/biaslab/UAI-MP-AIF-JAX)
- [Karpathy AgentHub](https://github.com/karpathy/agenthub)
- [Karpathy Autoresearch](https://github.com/karpathy/autoresearch)
- [Building Effective AI Agents](https://www.anthropic.com/engineering/building-effective-agents)

### Models and supplied artifacts

- [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B)
- [Qwen3.6-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [Google Drive research artifact](https://drive.google.com/file/d/1-GOg0kxcp8tx1BMUECMj2yJq6JYGmfhb/view)
- [Google Drive research artifact, `pli=1` URL variant](https://drive.google.com/file/d/1-GOg0kxcp8tx1BMUECMj2yJq6JYGmfhb/view?pli=1)

### IBM neuro-symbolic toolkit and projects

- [IBM Neuro-Symbolic AI Toolkit](https://ibm.github.io/neuro-symbolic-ai/toolkit/)
- [Logical Neural Networks (LNN)](https://github.com/IBM/LNN)
- [Logical Twins](https://github.com/IBM/logicaltwins)
- [SCERL](https://github.com/IBM/SCERL)
- [NESTA](https://ibm.github.io/neuro-symbolic-ai/toolkit/nesta/)
- [Logical Optimal Actions (LOA)](https://github.com/IBM/LOA)
- [AI Descartes](https://github.com/IBM/AI-Descartes)
- [E-PDDL](https://github.com/FrancescoFabiano/E-PDDL)
- [Meta-Experience Replay (MER)](https://ibm.github.io/neuro-symbolic-ai/toolkit/mer/)
- [ULKB](https://github.com/IBM/ULKB)
- [Expressive Reasoning Graph Store (ERGS)](https://github.com/IBM/expressive-reasoning-graph-store)

## 17. IBM neuro-symbolic additions

The IBM project index strengthens the neurosymbolic part of the design, but it does not justify
adding an older research stack to the MVP dependency tree. Reuse representations, evaluation
designs, and benchmark situations first. Admit a package only after an isolated spike beats the
dependency-free baseline on a measured need.

### 17.1 Priority and boundaries

| Priority | Project | Adopt now | Boundary |
|---|---|---|---|
| 1 | LNN | Truth-bound and contradiction concepts; optional backend spike | Do not replace probabilistic AIF beliefs or make it a core dependency yet |
| 2 | Logical Twins | Paired grounding-versus-planning evaluation design | Reuse the design and fixtures, not the archived runtime stack |
| 3 | SCERL | Negative-side-effect, safe-exploration, and oversight scenarios | Port selected situations into local held-out tests |
| 4 | NESTA and LOA | Versioned, inspectable procedural rules learned from verified episodes | Do not adopt the older AMR and RL dependency stack |
| 5 | AI Descartes | Competing-hypothesis and discriminating-experiment benchmark | Keep outside the initial general-purpose dependency tree |
| 6 | E-PDDL | Epistemic state and information-sharing representations | Defer runtime integration until multi-agent tasks exist |
| Later | MER | Replay old capabilities, safety cases, and failures during adapter training | Adopt the retention principle, not the historical implementation |
| Later | ULKB | Higher-order proof backend if graph claims require it | Add only after Python predicates or a small solver become insufficient |
| Avoid for MVP | ERGS | Reference for RDF/OWL reasoning at graph scale | JanusGraph, RDF4J, Java, and storage infrastructure are premature |

Keep the semantic distinction explicit:

- AIF probabilities represent uncertainty about hidden states and action outcomes.
- Logical truth bounds represent support, implication, and consistency.
- Hard permission rules remain non-negotiable and prune actions before AIF scoring.
- A logical backend may derive or reject candidates, but it does not authorize external effects.

### 17.2 Logic backend

Add a replaceable interface:

```python
class LogicBackend:
    def add_facts(self, facts): ...
    def infer(self): ...
    def contradictions(self): ...
    def allows(self, action): ...
```

The first implementation uses transparent Python predicates and bounded propositional inference.
An isolated LNN experiment is the second backend. Promote LNN only if it improves contradiction
detection or uncertain inference enough to justify compatibility and maintenance costs.

The execution order is:

```text
observation
    -> Qwen candidate-fact extraction
    -> probabilistic belief update
    -> logical inference and contradiction detection
    -> Qwen and rule-based action proposals
    -> hard logic and permission filtering
    -> AIF scoring of eligible actions
    -> verified tool execution
```

### 17.3 New evaluation work

Add five concrete benchmark tracks:

1. Grounding versus planning: run paired tasks from exact symbolic state and text-derived state so
   language-grounding errors can be separated from memory, planning, and action-selection errors.
2. Safety environments: port selected SCERL-style scenarios, including an unsafe action with high
   information gain, and prove that hard filtering removes it before scoring.
3. Procedural-rule induction: let Qwen propose NESTA-style reusable rules only after repeated
   verified episodes; promote a rule only after replay and held-out validation.
4. Scientific discovery: use AI Descartes-style competing hypotheses and make the AIF controller
   choose the experiment expected to discriminate among them most efficiently.
5. Continual-learning retention: mix new verified traces with old capabilities, safety cases, and
   previous failures, then gate on retention, safety, calibration, and transfer.

For paired grounding tests, record separate metrics for extraction accuracy, belief-state accuracy,
planner validity, eligible-action recall, selected-action quality, and final task success. A single
end-to-end score cannot identify which subsystem failed.

### 17.4 Updated roadmap order

1. Implement the `LogicBackend` protocol and Python-predicate backend.
2. Add contradiction fixtures and the high-information-gain safety regression.
3. Build the paired symbolic-state versus text-observation evaluation format.
4. Spike LNN in an isolated optional environment with a fixed two-day budget.
5. Add procedural-rule schemas after durable episode storage exists.
6. Add scientific-discovery tasks after the AIF scorer is calibrated.
7. Consider E-PDDL, ULKB, or ERGS only when a benchmark demonstrates the missing capability.

## 18. Final recommendation

Begin with Qwen3-8B and invest in explicit state, calibrated action selection, tool verification, durable evidence, and evaluation. Treat active inference as a testable controller design, not as branding. Treat the knowledge graph as uncertain, source-backed memory, not truth. Treat autoresearch as a bounded experiment manager, not permission for uncontrolled self-modification.

The first scientific question is straightforward:

> Under the same Qwen3-8B model and budget, does the explicit belief plus AIF controller choose better next actions and complete more held-out tasks than a conventional tool-calling agent?

Build the smallest system capable of answering that question reproducibly.

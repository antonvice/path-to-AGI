import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Literal
from uuid import uuid4

from aif_qwen_agent.agent import AgentTraceStore, OneStepAgent
from aif_qwen_agent.artifacts import TraceStore, sha256_file
from aif_qwen_agent.b1_evaluation import (
    load_b1_milestone,
    run_b1_suite,
    verify_b1_report,
)
from aif_qwen_agent.baseline import BaselineRunner
from aif_qwen_agent.config import load_yaml
from aif_qwen_agent.evaluation import memory_snapshot
from aif_qwen_agent.schemas import (
    B1CaseReproducibility,
    B1EvaluationReport,
    B1RepeatedEvaluationReport,
    SystemMemorySnapshot,
)

B1Report = B1EvaluationReport | B1RepeatedEvaluationReport


def _all_equal(values: Sequence[object]) -> bool:
    return all(value == values[0] for value in values[1:])


def build_b1_comparisons(
    suites: list[B1EvaluationReport],
    baseline_traces: TraceStore,
    agent_traces: AgentTraceStore,
) -> list[B1CaseReproducibility]:
    return build_b1_comparisons_from_stores(
        suites,
        [baseline_traces] * len(suites),
        [agent_traces] * len(suites),
    )


def build_b1_comparisons_from_stores(
    suites: list[B1EvaluationReport],
    baseline_traces: list[TraceStore],
    agent_traces: list[AgentTraceStore],
) -> list[B1CaseReproducibility]:
    if not suites or len(suites) != len(baseline_traces) or len(suites) != len(agent_traces):
        raise ValueError("B1 suites and trace stores must have equal nonzero lengths")
    fixture_ids = [case.fixture_id for case in suites[0].cases]
    if any([case.fixture_id for case in suite.cases] != fixture_ids for suite in suites[1:]):
        raise ValueError("B1 suite case order differs across repetitions")
    comparisons: list[B1CaseReproducibility] = []
    for index, fixture_id in enumerate(fixture_ids):
        cases = [suite.cases[index] for suite in suites]
        agents = [
            store.get(str(case.agent_run_id))
            for store, case in zip(agent_traces, cases, strict=True)
        ]
        actions: list[Literal["read_file", "answer", "stop", "none"]] = []
        for trace in agents:
            actions.append(
                trace.selected_action.kind if trace.selected_action is not None else "none"
            )
        outputs = [case.agent_actual for case in cases]
        baseline_outputs = [case.baseline_actual for case in cases if case.kind == "grounded"]
        statuses = [case.agent_status for case in cases]
        input_tokens = [case.agent_input_tokens for case in cases]
        output_tokens = [case.agent_output_tokens for case in cases]
        rejection_codes = [case.rejection_code for case in cases]
        evidence_sha256s = [case.evidence_sha256 for case in cases]
        attempts = [case.proposal_attempts for case in cases]
        passed = [case.agent_passed for case in cases]
        baseline_run_ids = [
            store.get(str(case.baseline_run_id)).run_id
            for store, case in zip(baseline_traces, cases, strict=True)
            if case.baseline_run_id is not None
        ]
        agreement = (
            _all_equal(actions),
            _all_equal(outputs),
            _all_equal(baseline_outputs) if baseline_outputs else True,
            _all_equal(statuses),
            _all_equal(list(zip(input_tokens, output_tokens, strict=True))),
            _all_equal(rejection_codes),
            _all_equal(evidence_sha256s),
            _all_equal(attempts),
            _all_equal(passed),
        )
        comparisons.append(
            B1CaseReproducibility(
                fixture_id=fixture_id,
                kind=cases[0].kind,
                agent_run_ids=[trace.run_id for trace in agents],
                baseline_run_ids=baseline_run_ids,
                actions=actions,
                outputs=outputs,
                baseline_outputs=baseline_outputs,
                statuses=statuses,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                rejection_codes=rejection_codes,
                evidence_sha256s=evidence_sha256s,
                proposal_attempts=attempts,
                passed=passed,
                action_agreement=agreement[0],
                output_agreement=agreement[1],
                baseline_output_agreement=agreement[2],
                status_agreement=agreement[3],
                token_agreement=agreement[4],
                rejection_agreement=agreement[5],
                evidence_agreement=agreement[6],
                retry_agreement=agreement[7],
                pass_agreement=agreement[8],
                all_agreement=all(agreement),
            )
        )
    return comparisons


def _generation_latencies(
    suites: list[B1EvaluationReport],
    baseline_traces: TraceStore,
    agent_traces: AgentTraceStore,
) -> list[float]:
    latencies: list[float] = []
    for suite in suites:
        for case in suite.cases:
            if case.baseline_run_id is None:
                continue
            baseline = baseline_traces.get(str(case.baseline_run_id))
            if baseline.result is not None:
                latencies.append(baseline.result.generation_seconds)
        for case in suite.cases:
            agent = agent_traces.get(str(case.agent_run_id))
            latencies.extend(
                attempt.result.generation_seconds
                for attempt in agent.proposal_attempts
                if attempt.result is not None
            )
            if agent.answer_result is not None:
                latencies.append(agent.answer_result.generation_seconds)
    if len(latencies) < 2:
        raise ValueError("B1 reproducibility requires at least two model generations")
    return latencies


def _build_repeated_report(
    suites: list[B1EvaluationReport],
    comparisons: list[B1CaseReproducibility],
    fixture_path: Path,
    evaluation_config: Path,
    baseline_traces: TraceStore,
    agent_traces: AgentTraceStore,
    started_at: datetime,
    memory_before: SystemMemorySnapshot,
    memory_after: SystemMemorySnapshot,
) -> B1RepeatedEvaluationReport:
    promotion = load_yaml(evaluation_config)["promotion"]
    grounded_runs = sum(suite.grounded_cases for suite in suites)
    baseline_passed = sum(suite.baseline_passed_cases for suite in suites)
    agent_passed = sum(suite.agent_passed_cases for suite in suites)
    safety_runs = sum(suite.safety_cases for suite in suites)
    safety_passed = sum(suite.safety_passed_cases for suite in suites)
    safety_violations = sum(suite.safety_violations for suite in suites)
    instruction_violations = sum(
        case.instruction_following_violation for suite in suites for case in suite.cases
    )
    baseline_input = sum(suite.baseline_input_tokens for suite in suites)
    baseline_output = sum(suite.baseline_output_tokens for suite in suites)
    agent_input = sum(suite.agent_input_tokens for suite in suites)
    agent_output = sum(suite.agent_output_tokens for suite in suites)
    baseline_generation = sum(suite.baseline_generation_seconds for suite in suites)
    agent_generation = sum(suite.agent_generation_seconds for suite in suites)
    quality_delta = agent_passed / grounded_runs - baseline_passed / grounded_runs
    token_cost_increase = (agent_input + agent_output) / (baseline_input + baseline_output) - 1.0
    generation_cost_increase = agent_generation / baseline_generation - 1.0
    minimum_success_delta = float(promotion["minimum_success_delta"])
    maximum_cost_increase = float(promotion["maximum_cost_increase"])
    quality_gate = quality_delta >= minimum_success_delta
    safety_gate = (
        safety_passed == safety_runs and safety_violations == 0 and instruction_violations == 0
    )
    reproducibility_gate = all(comparison.all_agreement for comparison in comparisons)
    cost_gate = max(token_cost_increase, generation_cost_increase) <= maximum_cost_increase
    latencies = _generation_latencies(suites, baseline_traces, agent_traces)
    return B1RepeatedEvaluationReport(
        report_id=uuid4(),
        started_at=started_at,
        finished_at=datetime.now(UTC),
        fixture_file=str(fixture_path),
        fixture_sha256=sha256_file(fixture_path),
        evaluation_config_file=str(evaluation_config),
        evaluation_config_sha256=sha256_file(evaluation_config),
        model=suites[0].model,
        generation=suites[0].generation,
        repeats=len(suites),
        suites=suites,
        comparisons=comparisons,
        grounded_runs=grounded_runs,
        safety_runs=safety_runs,
        baseline_passed_runs=baseline_passed,
        agent_passed_runs=agent_passed,
        safety_passed_runs=safety_passed,
        safety_violations=safety_violations,
        instruction_following_violations=instruction_violations,
        proposal_retries=sum(suite.proposal_retries for suite in suites),
        baseline_input_tokens=baseline_input,
        baseline_output_tokens=baseline_output,
        agent_input_tokens=agent_input,
        agent_output_tokens=agent_output,
        model_load_seconds=sum(suite.model_load_seconds for suite in suites),
        baseline_generation_seconds=baseline_generation,
        agent_generation_seconds=agent_generation,
        first_generation_seconds=latencies[0],
        warm_generation_median_seconds=median(latencies[1:]),
        generation_min_seconds=min(latencies),
        generation_median_seconds=median(latencies),
        generation_max_seconds=max(latencies),
        quality_delta=quality_delta,
        token_cost_increase=token_cost_increase,
        generation_cost_increase=generation_cost_increase,
        minimum_success_delta=minimum_success_delta,
        maximum_cost_increase=maximum_cost_increase,
        quality_gate_passed=quality_gate,
        safety_gate_passed=safety_gate,
        reproducibility_gate_passed=reproducibility_gate,
        cost_gate_passed=cost_gate,
        gate_passed=quality_gate and safety_gate and reproducibility_gate and cost_gate,
        memory_before=memory_before,
        memory_after=memory_after,
    )


def evaluate_repeated_b1(
    baseline: BaselineRunner,
    agent: OneStepAgent,
    fixture_path: Path,
    evaluation_config: Path,
    report_path: Path,
    repeats: int,
) -> B1RepeatedEvaluationReport:
    if repeats < 2:
        raise ValueError("repeated B1 evaluation requires at least two repetitions")
    if load_b1_milestone(fixture_path) != "B1d":
        raise ValueError("repeated B1 evaluation requires B1d fixtures")
    started_at = datetime.now(UTC)
    memory_before = memory_snapshot()
    suites = [run_b1_suite(baseline, agent, fixture_path) for _ in range(repeats)]
    comparisons = build_b1_comparisons(suites, baseline.traces, agent.traces)
    report = _build_repeated_report(
        suites,
        comparisons,
        fixture_path,
        evaluation_config,
        baseline.traces,
        agent.traces,
        started_at,
        memory_before,
        memory_snapshot(),
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return report


def load_any_b1_report(path: Path) -> B1Report:
    raw = path.read_text(encoding="utf-8")
    if json.loads(raw).get("report_type") == "b1_reproducibility":
        return B1RepeatedEvaluationReport.model_validate_json(raw)
    return B1EvaluationReport.model_validate_json(raw)


def verify_repeated_b1_report(
    report: B1RepeatedEvaluationReport,
    fixture_path: Path,
    evaluation_config: Path,
    baseline_traces: TraceStore,
    agent_traces: AgentTraceStore,
) -> None:
    B1RepeatedEvaluationReport.model_validate(report.model_dump())
    if report.fixture_file != str(fixture_path) or report.fixture_sha256 != sha256_file(
        fixture_path
    ):
        raise ValueError("fixture hash does not match repeated B1 report")
    if report.evaluation_config_file != str(
        evaluation_config
    ) or report.evaluation_config_sha256 != sha256_file(evaluation_config):
        raise ValueError("evaluation config hash does not match repeated B1 report")
    for suite in report.suites:
        verify_b1_report(suite, fixture_path, baseline_traces, agent_traces)
    rebuilt_comparisons = build_b1_comparisons(
        report.suites,
        baseline_traces,
        agent_traces,
    )
    if rebuilt_comparisons != report.comparisons:
        raise ValueError("saved B1 reproducibility comparisons do not match traces")
    rebuilt = _build_repeated_report(
        report.suites,
        rebuilt_comparisons,
        fixture_path,
        evaluation_config,
        baseline_traces,
        agent_traces,
        report.started_at,
        report.memory_before,
        report.memory_after,
    )
    ignored = {"report_id", "finished_at"}
    expected = report.model_dump(exclude=ignored)
    actual = rebuilt.model_dump(exclude=ignored)
    if actual != expected:
        raise ValueError("saved B1 reproducibility aggregates do not match traces")

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast
from uuid import UUID, uuid4

from aif_qwen_agent.agent import AgentTraceStore, OneStepAgent
from aif_qwen_agent.artifacts import TraceStore, sha256_file
from aif_qwen_agent.b1_evaluation import evaluate_b1, load_b1_report, verify_b1_report
from aif_qwen_agent.b1_reproducibility import (
    load_any_b1_report,
    verify_repeated_b1_report,
)
from aif_qwen_agent.baseline import BaselineRunner
from aif_qwen_agent.config import load_yaml
from aif_qwen_agent.schemas import (
    AgentStageCost,
    B1CostReport,
    B1EvaluationReport,
    B1RepeatedEvaluationReport,
    ModelResult,
)


def _average_agent_stages(
    run_ids: list[UUID],
    traces: AgentTraceStore,
    divisor: int = 1,
) -> tuple[AgentStageCost, AgentStageCost]:
    if divisor < 1 or len(run_ids) % divisor:
        raise ValueError("stage-cost divisor must evenly divide run count")
    proposal_results: list[ModelResult] = []
    answer_results: list[ModelResult] = []
    for run_id in run_ids:
        trace = traces.get(str(run_id))
        proposal_results.extend(
            attempt.result for attempt in trace.proposal_attempts if attempt.result is not None
        )
        if trace.answer_result is not None:
            answer_results.append(trace.answer_result)

    def stage(results: list[ModelResult]) -> AgentStageCost:
        return AgentStageCost(
            calls=len(results) // divisor,
            input_tokens=sum(result.input_tokens for result in results) / divisor,
            output_tokens=sum(result.output_tokens for result in results) / divisor,
            generation_seconds=sum(result.generation_seconds for result in results) / divisor,
        )

    return stage(proposal_results), stage(answer_results)


def _baseline_stage(suite: B1EvaluationReport) -> AgentStageCost:
    grounded = [case for case in suite.cases if case.kind == "grounded"]
    return AgentStageCost(
        calls=len(grounded),
        input_tokens=sum(case.baseline_input_tokens for case in grounded),
        output_tokens=sum(case.baseline_output_tokens for case in grounded),
        generation_seconds=sum(case.baseline_generation_seconds for case in grounded),
    )


def _stage_tokens(*stages: AgentStageCost) -> float:
    return sum(stage.input_tokens + stage.output_tokens for stage in stages)


def _stage_generation(*stages: AgentStageCost) -> float:
    return sum(stage.generation_seconds for stage in stages)


def build_b1_cost_report(
    reference: B1RepeatedEvaluationReport,
    reference_report_path: Path,
    reference_agent_traces: AgentTraceStore,
    optimized: B1EvaluationReport,
    optimized_report_path: Path,
    optimized_agent_traces: AgentTraceStore,
    fixture_path: Path,
    evaluation_config: Path,
    agent_config: Path,
) -> B1CostReport:
    if (
        reference.fixture_sha256 != optimized.fixture_sha256
        or optimized.fixture_sha256 != sha256_file(fixture_path)
    ):
        raise ValueError("B1 cost optimization requires unchanged reference fixtures")
    if reference.model != optimized.model or reference.generation != optimized.generation:
        raise ValueError("B1 cost reference and optimized model settings differ")
    legacy_grounded_ids = [
        case.agent_run_id
        for suite in reference.suites
        for case in suite.cases
        if case.kind == "grounded"
    ]
    optimized_grounded_ids = [
        case.agent_run_id for case in optimized.cases if case.kind == "grounded"
    ]
    optimized_safety_ids = [case.agent_run_id for case in optimized.cases if case.kind == "safety"]
    legacy_proposal, legacy_answer = _average_agent_stages(
        legacy_grounded_ids,
        reference_agent_traces,
        divisor=reference.repeats,
    )
    optimized_proposal, optimized_answer = _average_agent_stages(
        optimized_grounded_ids,
        optimized_agent_traces,
    )
    optimized_safety, safety_answers = _average_agent_stages(
        optimized_safety_ids,
        optimized_agent_traces,
    )
    if safety_answers.calls:
        raise ValueError("B1 cost safety cases unexpectedly generated evidence answers")
    optimized_traces = [
        optimized_agent_traces.get(str(run_id))
        for run_id in optimized_grounded_ids + optimized_safety_ids
    ]
    proposal_generation = optimized_traces[0].proposal_generation
    prompt_profiles = {trace.prompt_profile for trace in optimized_traces}
    if proposal_generation is None or any(
        trace.proposal_generation != proposal_generation for trace in optimized_traces
    ):
        raise ValueError("optimized traces have inconsistent proposal generation settings")
    if len(prompt_profiles) != 1 or not prompt_profiles <= {"compact", "fast"}:
        raise ValueError("B1 cost optimization requires one optimized prompt profile")
    prompt_profile = cast(Literal["compact", "fast"], prompt_profiles.pop())
    milestone: Literal["B1e", "B1f"] = "B1f" if prompt_profile == "fast" else "B1e"
    baseline = _baseline_stage(optimized)
    legacy_tokens = _stage_tokens(legacy_proposal, legacy_answer)
    optimized_tokens = _stage_tokens(optimized_proposal, optimized_answer)
    baseline_tokens = _stage_tokens(baseline)
    legacy_generation = _stage_generation(legacy_proposal, legacy_answer)
    optimized_generation = _stage_generation(optimized_proposal, optimized_answer)
    baseline_generation = _stage_generation(baseline)
    if min(legacy_tokens, baseline_tokens, legacy_generation, baseline_generation) <= 0.0:
        raise ValueError("B1 cost comparison requires nonzero reference costs")
    agent_settings = load_yaml(agent_config)["agent"]
    promotion = load_yaml(evaluation_config)["promotion"]
    token_reduction = 1.0 - optimized_tokens / legacy_tokens
    generation_reduction = 1.0 - optimized_generation / legacy_generation
    grounded_token_increase = optimized_tokens / baseline_tokens - 1.0
    grounded_generation_increase = optimized_generation / baseline_generation - 1.0
    legacy_grounded_rate = reference.agent_passed_runs / reference.grounded_runs
    optimized_grounded_rate = optimized.agent_passed_cases / optimized.grounded_cases
    legacy_safety_rate = reference.safety_passed_runs / reference.safety_runs
    optimized_safety_rate = optimized.safety_passed_cases / optimized.safety_cases
    minimum_reduction = float(agent_settings["minimum_cost_reduction"])
    maximum_increase = float(promotion["maximum_cost_increase"])
    quality_preserved = optimized_grounded_rate >= legacy_grounded_rate
    optimized_instruction_violations = sum(
        case.instruction_following_violation for case in optimized.cases
    )
    safety_preserved = (
        optimized_safety_rate >= legacy_safety_rate and optimized_instruction_violations == 0
    )
    optimization_gate = min(token_reduction, generation_reduction) >= minimum_reduction
    cost_gate = max(grounded_token_increase, grounded_generation_increase) <= maximum_increase
    return B1CostReport(
        milestone=milestone,
        report_id=uuid4(),
        created_at=datetime.now(UTC),
        fixture_file=str(fixture_path),
        fixture_sha256=sha256_file(fixture_path),
        evaluation_config_file=str(evaluation_config),
        evaluation_config_sha256=sha256_file(evaluation_config),
        agent_config_file=str(agent_config),
        agent_config_sha256=sha256_file(agent_config),
        reference_report_file=str(reference_report_path),
        reference_report_sha256=sha256_file(reference_report_path),
        optimized_report_file=str(optimized_report_path),
        optimized_report_sha256=sha256_file(optimized_report_path),
        model=optimized.model,
        answer_generation=optimized.generation,
        proposal_generation=proposal_generation,
        prompt_profile=prompt_profile,
        baseline_grounded=baseline,
        legacy_grounded_proposal=legacy_proposal,
        legacy_grounded_answer=legacy_answer,
        optimized_grounded_proposal=optimized_proposal,
        optimized_grounded_answer=optimized_answer,
        optimized_safety_proposal=optimized_safety,
        legacy_grounded_pass_rate=legacy_grounded_rate,
        optimized_grounded_pass_rate=optimized_grounded_rate,
        legacy_safety_pass_rate=legacy_safety_rate,
        optimized_safety_pass_rate=optimized_safety_rate,
        legacy_instruction_following_violations=reference.instruction_following_violations,
        optimized_instruction_following_violations=optimized_instruction_violations,
        token_reduction=token_reduction,
        generation_reduction=generation_reduction,
        minimum_cost_reduction=minimum_reduction,
        grounded_token_cost_increase=grounded_token_increase,
        grounded_generation_cost_increase=grounded_generation_increase,
        maximum_cost_increase=maximum_increase,
        quality_preserved=quality_preserved,
        safety_preserved=safety_preserved,
        optimization_gate_passed=optimization_gate,
        cost_gate_passed=cost_gate,
        gate_passed=quality_preserved and safety_preserved and optimization_gate and cost_gate,
        optimized_suite=optimized,
    )


def evaluate_b1_cost(
    baseline: BaselineRunner,
    agent: OneStepAgent,
    fixture_path: Path,
    optimized_report_path: Path,
    cost_report_path: Path,
    reference_report_path: Path,
    reference_baseline_traces: TraceStore,
    reference_agent_traces: AgentTraceStore,
    evaluation_config: Path,
    agent_config: Path,
) -> B1CostReport:
    reference = load_any_b1_report(reference_report_path)
    if not isinstance(reference, B1RepeatedEvaluationReport):
        raise ValueError("B1 cost optimization requires a repeated B1d reference report")
    verify_repeated_b1_report(
        reference,
        fixture_path,
        evaluation_config,
        reference_baseline_traces,
        reference_agent_traces,
    )
    optimized = evaluate_b1(baseline, agent, fixture_path, optimized_report_path)
    report = build_b1_cost_report(
        reference,
        reference_report_path,
        reference_agent_traces,
        optimized,
        optimized_report_path,
        agent.traces,
        fixture_path,
        evaluation_config,
        agent_config,
    )
    cost_report_path.parent.mkdir(parents=True, exist_ok=True)
    cost_report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return report


def load_b1_cost_report(path: Path) -> B1CostReport:
    return B1CostReport.model_validate_json(path.read_text(encoding="utf-8"))


def verify_b1_cost_report(
    report: B1CostReport,
    fixture_path: Path,
    evaluation_config: Path,
    agent_config: Path,
    reference_baseline_traces: TraceStore,
    reference_agent_traces: AgentTraceStore,
    optimized_baseline_traces: TraceStore,
    optimized_agent_traces: AgentTraceStore,
) -> None:
    B1CostReport.model_validate(report.model_dump())
    reference = load_any_b1_report(Path(report.reference_report_file))
    if not isinstance(reference, B1RepeatedEvaluationReport):
        raise ValueError("B1 cost reference report is not repeated B1d evidence")
    if report.reference_report_sha256 != sha256_file(Path(report.reference_report_file)):
        raise ValueError("B1 cost reference report hash mismatch")
    optimized = load_b1_report(Path(report.optimized_report_file))
    if report.optimized_report_sha256 != sha256_file(Path(report.optimized_report_file)):
        raise ValueError("B1 cost optimized report hash mismatch")
    if optimized != report.optimized_suite:
        raise ValueError("B1 cost embedded optimized suite mismatch")
    verify_repeated_b1_report(
        reference,
        fixture_path,
        evaluation_config,
        reference_baseline_traces,
        reference_agent_traces,
    )
    verify_b1_report(
        optimized,
        fixture_path,
        optimized_baseline_traces,
        optimized_agent_traces,
    )
    rebuilt = build_b1_cost_report(
        reference,
        Path(report.reference_report_file),
        reference_agent_traces,
        optimized,
        Path(report.optimized_report_file),
        optimized_agent_traces,
        fixture_path,
        evaluation_config,
        agent_config,
    )
    ignored = {"report_id", "created_at"}
    if rebuilt.model_dump(exclude=ignored) != report.model_dump(exclude=ignored):
        raise ValueError("saved B1 cost report does not match traces")

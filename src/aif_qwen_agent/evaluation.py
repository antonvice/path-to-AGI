import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from uuid import uuid4

import psutil
from pydantic import BaseModel

from aif_qwen_agent.artifacts import TraceStore, sha256_file
from aif_qwen_agent.baseline import BaselineRunner
from aif_qwen_agent.config import load_yaml
from aif_qwen_agent.schemas import (
    BaselineCaseComparison,
    BaselineCaseResult,
    BaselineFixture,
    BaselineReport,
    BaselineReproducibilityReport,
    RunTrace,
    SystemMemorySnapshot,
)

Report = BaselineReport | BaselineReproducibilityReport


def load_baseline_fixtures(path: Path) -> tuple[BaselineFixture, ...]:
    document = load_yaml(path)
    if document.get("baseline") != "B0":
        raise ValueError("fixture file must declare baseline: B0")
    fixtures = tuple(BaselineFixture.model_validate(value) for value in document["cases"])
    if not fixtures:
        raise ValueError("fixture file has no cases")
    identifiers = [fixture.id for fixture in fixtures]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("fixture IDs must be unique")
    return fixtures


def grade_trace(fixture: BaselineFixture, trace: RunTrace) -> BaselineCaseResult:
    result = trace.result
    actual = result.text if result is not None else None
    if actual is None:
        passed = False
    elif fixture.grader == "exact":
        passed = actual.strip() == fixture.expected.strip()
    else:
        passed = fixture.expected.casefold() in actual.casefold()
    return BaselineCaseResult(
        fixture_id=fixture.id,
        run_id=trace.run_id,
        status=trace.status,
        grader=fixture.grader,
        expected=fixture.expected,
        actual=actual,
        passed=passed,
        error=trace.error,
        input_tokens=result.input_tokens if result is not None else 0,
        output_tokens=result.output_tokens if result is not None else 0,
        load_seconds=result.load_seconds if result is not None else 0.0,
        generation_seconds=result.generation_seconds if result is not None else 0.0,
    )


def _run_suite(
    runner: BaselineRunner,
    fixture_path: Path,
    fixtures: tuple[BaselineFixture, ...],
) -> BaselineReport:
    started_at = datetime.now(UTC)
    cases = [grade_trace(fixture, runner.run(fixture.task)) for fixture in fixtures]
    passed = sum(case.passed for case in cases)
    return BaselineReport(
        report_id=uuid4(),
        started_at=started_at,
        finished_at=datetime.now(UTC),
        fixture_file=str(fixture_path),
        fixture_sha256=sha256_file(fixture_path),
        model=runner.model,
        generation=runner.generation,
        cases=cases,
        total_cases=len(cases),
        passed_cases=passed,
        failed_cases=len(cases) - passed,
        pass_rate=passed / len(cases),
        input_tokens=sum(case.input_tokens for case in cases),
        output_tokens=sum(case.output_tokens for case in cases),
        model_load_seconds=sum(case.load_seconds for case in cases),
        generation_seconds=sum(case.generation_seconds for case in cases),
    )


def _write_report(path: Path, report: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")


def evaluate_baseline(
    runner: BaselineRunner,
    fixture_path: Path,
    report_path: Path,
) -> BaselineReport:
    report = _run_suite(runner, fixture_path, load_baseline_fixtures(fixture_path))
    _write_report(report_path, report)
    return report


def memory_snapshot() -> SystemMemorySnapshot:
    memory = psutil.virtual_memory()
    try:
        swap = psutil.swap_memory()
        swap_available = True
        swap_total = swap.total
        swap_used = swap.used
    except OSError:
        swap_available = False
        swap_total = 0
        swap_used = 0
    return SystemMemorySnapshot(
        captured_at=datetime.now(UTC),
        total_bytes=memory.total,
        available_bytes=memory.available,
        used_fraction=memory.percent / 100.0,
        swap_available=swap_available,
        swap_total_bytes=swap_total,
        swap_used_bytes=swap_used,
    )


def _all_equal(values: Sequence[object]) -> bool:
    return all(value == values[0] for value in values[1:])


def _build_comparisons(
    suites: list[BaselineReport], traces: TraceStore
) -> list[BaselineCaseComparison]:
    fixture_ids = [case.fixture_id for case in suites[0].cases]
    if any([case.fixture_id for case in suite.cases] != fixture_ids for suite in suites[1:]):
        raise ValueError("suite case order differs across repetitions")
    comparisons: list[BaselineCaseComparison] = []
    for index, fixture_id in enumerate(fixture_ids):
        saved_cases = [suite.cases[index] for suite in suites]
        run_traces = [traces.get(str(case.run_id)) for case in saved_cases]
        outputs = [case.actual for case in saved_cases]
        prompt_hashes = [trace.prompt_sha256 for trace in run_traces]
        input_tokens = [case.input_tokens for case in saved_cases]
        output_tokens = [case.output_tokens for case in saved_cases]
        stop_reasons = [
            trace.result.stop_reason if trace.result is not None else None for trace in run_traces
        ]
        latencies = [case.generation_seconds for case in saved_cases]
        comparisons.append(
            BaselineCaseComparison(
                fixture_id=fixture_id,
                run_ids=[case.run_id for case in saved_cases],
                outputs=outputs,
                prompt_sha256s=prompt_hashes,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                stop_reasons=stop_reasons,
                generation_seconds=latencies,
                output_agreement=_all_equal(outputs),
                prompt_agreement=_all_equal(prompt_hashes),
                input_token_agreement=_all_equal(input_tokens),
                output_token_agreement=_all_equal(output_tokens),
                stop_reason_agreement=_all_equal(stop_reasons),
                latency_min_seconds=min(latencies),
                latency_median_seconds=median(latencies),
                latency_max_seconds=max(latencies),
            )
        )
    return comparisons


def evaluate_repeated_baseline(
    runner: BaselineRunner,
    fixture_path: Path,
    report_path: Path,
    repeats: int,
) -> BaselineReproducibilityReport:
    if repeats < 2:
        raise ValueError("repeated evaluation requires at least two repetitions")
    fixtures = load_baseline_fixtures(fixture_path)
    started_at = datetime.now(UTC)
    memory_before = memory_snapshot()
    suites = [_run_suite(runner, fixture_path, fixtures) for _ in range(repeats)]
    comparisons = _build_comparisons(suites, runner.traces)
    latencies = [case.generation_seconds for suite in suites for case in suite.cases]
    passed_runs = sum(case.passed for suite in suites for case in suite.cases)
    completed_runs = sum(case.status == "completed" for suite in suites for case in suite.cases)
    agreement_checks = [
        comparison.output_agreement
        and comparison.prompt_agreement
        and comparison.input_token_agreement
        and comparison.output_token_agreement
        and comparison.stop_reason_agreement
        for comparison in comparisons
    ]
    report = BaselineReproducibilityReport(
        report_id=uuid4(),
        started_at=started_at,
        finished_at=datetime.now(UTC),
        fixture_file=str(fixture_path),
        fixture_sha256=sha256_file(fixture_path),
        model=runner.model,
        generation=runner.generation,
        repeats=repeats,
        suites=suites,
        cases=comparisons,
        total_runs=sum(suite.total_cases for suite in suites),
        completed_runs=completed_runs,
        passed_runs=passed_runs,
        output_agreement_rate=sum(case.output_agreement for case in comparisons) / len(comparisons),
        gate_passed=passed_runs == sum(suite.total_cases for suite in suites)
        and all(agreement_checks),
        model_load_seconds=sum(suite.model_load_seconds for suite in suites),
        first_generation_seconds=latencies[0],
        warm_generation_median_seconds=median(latencies[1:]),
        generation_min_seconds=min(latencies),
        generation_median_seconds=median(latencies),
        generation_max_seconds=max(latencies),
        memory_before=memory_before,
        memory_after=memory_snapshot(),
    )
    _write_report(report_path, report)
    return report


def load_report(path: Path) -> BaselineReport:
    return BaselineReport.model_validate_json(path.read_text(encoding="utf-8"))


def load_any_report(path: Path) -> Report:
    raw = path.read_text(encoding="utf-8")
    document = json.loads(raw)
    if document.get("report_type") == "reproducibility":
        return BaselineReproducibilityReport.model_validate_json(raw)
    return BaselineReport.model_validate_json(raw)


def verify_report(
    report: BaselineReport,
    fixture_path: Path,
    traces: TraceStore,
) -> None:
    BaselineReport.model_validate(report.model_dump())
    if report.fixture_sha256 != sha256_file(fixture_path):
        raise ValueError("fixture hash does not match report")
    fixtures = {fixture.id: fixture for fixture in load_baseline_fixtures(fixture_path)}
    if set(fixtures) != {case.fixture_id for case in report.cases}:
        raise ValueError("report cases do not match fixture IDs")
    for saved in report.cases:
        trace = traces.get(str(saved.run_id))
        fixture = fixtures[saved.fixture_id]
        if trace.task != fixture.task:
            raise ValueError(f"trace task mismatch for {fixture.id}")
        regraded = grade_trace(fixture, trace)
        if regraded != saved:
            raise ValueError(f"saved result mismatch for {fixture.id}")


def verify_reproducibility_report(
    report: BaselineReproducibilityReport,
    fixture_path: Path,
    traces: TraceStore,
) -> None:
    BaselineReproducibilityReport.model_validate(report.model_dump())
    if report.fixture_sha256 != sha256_file(fixture_path):
        raise ValueError("fixture hash does not match reproducibility report")
    for suite in report.suites:
        verify_report(suite, fixture_path, traces)
    rebuilt = _build_comparisons(report.suites, traces)
    if rebuilt != report.cases:
        raise ValueError("saved reproducibility comparisons do not match traces")

from pathlib import Path

import pytest

from aif_qwen_agent.artifacts import TraceStore
from aif_qwen_agent.baseline import BaselineRunner
from aif_qwen_agent.evaluation import (
    evaluate_baseline,
    evaluate_repeated_baseline,
    load_any_report,
    load_report,
    verify_report,
    verify_reproducibility_report,
)
from aif_qwen_agent.schemas import (
    BaselineReproducibilityReport,
    GenerationConfig,
    ModelIdentity,
    ModelResult,
    Task,
)

FIXTURES = Path("evals/tasks/b0/direct_answer.yaml")


class FixtureAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def render_prompt(self, task: Task) -> str:
        return task.text

    def generate(self, rendered_prompt: str, config: GenerationConfig) -> ModelResult:
        self.calls += 1
        if "2 + 2" in rendered_prompt:
            text = "4"
        elif "safer first" in rendered_prompt:
            text = "read production logs"
        else:
            text = "BASELINE_READY"
        return ModelResult(
            raw_text=text,
            text=text,
            input_tokens=5,
            output_tokens=1,
            load_seconds=1.5 if self.calls == 1 else 0.0,
            generation_seconds=0.1,
            device="fake",
            stop_reason="eos",
        )


def make_runner(path: Path, adapter: FixtureAdapter) -> BaselineRunner:
    return BaselineRunner(
        adapter=adapter,
        model=ModelIdentity(
            repo_id="Qwen/Qwen3-8B",
            revision="b968826d9c46dd6066d109eabc6255188de91218",
            local_path=Path("models/Qwen3-8B"),
            backend="fake",
        ),
        generation=GenerationConfig(max_new_tokens=8, temperature=0.0, seed=42),
        traces=TraceStore(path),
    )


def test_evaluates_all_fixtures_once_and_regrades_offline(tmp_path: Path) -> None:
    adapter = FixtureAdapter()
    traces = TraceStore(tmp_path / "runs.jsonl")
    report_path = tmp_path / "report.json"

    report = evaluate_baseline(make_runner(traces.path, adapter), FIXTURES, report_path)
    verify_report(load_report(report_path), FIXTURES, traces)

    assert adapter.calls == 3
    assert report.passed_cases == report.total_cases == 3
    assert report.pass_rate == 1.0
    assert report.input_tokens == 15
    assert report.output_tokens == 3
    assert report.model_load_seconds == 1.5
    assert report.generation_seconds == pytest.approx(0.3)


def test_regrading_detects_tampered_case_result(tmp_path: Path) -> None:
    traces = TraceStore(tmp_path / "runs.jsonl")
    report = evaluate_baseline(
        make_runner(traces.path, FixtureAdapter()), FIXTURES, tmp_path / "report.json"
    )
    changed_case = report.cases[0].model_copy(update={"actual": "tampered"})
    tampered = report.model_copy(update={"cases": [changed_case, *report.cases[1:]]})

    with pytest.raises(ValueError, match="saved result mismatch"):
        verify_report(tampered, FIXTURES, traces)


def test_repeated_evaluation_retains_all_runs_and_passes_agreement_gate(
    tmp_path: Path,
) -> None:
    adapter = FixtureAdapter()
    traces = TraceStore(tmp_path / "runs.jsonl")
    report_path = tmp_path / "reproducibility.json"

    report = evaluate_repeated_baseline(
        make_runner(traces.path, adapter), FIXTURES, report_path, repeats=3
    )
    loaded = load_any_report(report_path)

    assert isinstance(loaded, BaselineReproducibilityReport)
    verify_reproducibility_report(loaded, FIXTURES, traces)
    assert adapter.calls == 9
    assert report.total_runs == report.completed_runs == report.passed_runs == 9
    assert report.output_agreement_rate == 1.0
    assert report.gate_passed
    assert report.model_load_seconds == 1.5
    assert report.first_generation_seconds == 0.1
    assert report.warm_generation_median_seconds == 0.1


def test_regrading_detects_tampered_reproducibility_comparison(tmp_path: Path) -> None:
    traces = TraceStore(tmp_path / "runs.jsonl")
    report = evaluate_repeated_baseline(
        make_runner(traces.path, FixtureAdapter()),
        FIXTURES,
        tmp_path / "reproducibility.json",
        repeats=2,
    )
    changed = report.cases[0].model_copy(update={"outputs": ["tampered", "tampered"]})
    tampered = report.model_copy(update={"cases": [changed, *report.cases[1:]]})

    with pytest.raises(ValueError, match="comparisons do not match"):
        verify_reproducibility_report(tampered, FIXTURES, traces)

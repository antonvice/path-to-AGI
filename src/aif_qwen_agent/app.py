from pathlib import Path
from uuid import uuid4

import typer
from rich.console import Console
from rich.table import Table

from aif_qwen_agent.aif_score import aif_score
from aif_qwen_agent.artifacts import TraceStore
from aif_qwen_agent.baseline import BaselineRunner
from aif_qwen_agent.config import load_yaml
from aif_qwen_agent.evaluation import (
    evaluate_baseline,
    evaluate_repeated_baseline,
    load_any_report,
    verify_report,
    verify_reproducibility_report,
)
from aif_qwen_agent.model_adapters import TransformersAdapter
from aif_qwen_agent.schemas import (
    BaselineReproducibilityReport,
    GenerationConfig,
    ModelIdentity,
    PredictedOutcome,
    ReadFilePolicy,
    ReadFileRequest,
    Task,
)
from aif_qwen_agent.tools import ReadFileTool, ReadFileTraceStore

app = typer.Typer(no_args_is_help=True)
tool_app = typer.Typer(no_args_is_help=True)
app.add_typer(tool_app, name="tool")
console = Console()


def _terminal_safe_text(value: str) -> str:
    return "".join(
        character
        if character in "\n\r\t" or (ord(character) >= 32 and ord(character) != 127)
        else f"\\x{ord(character):02x}"
        for character in value
    )


def _build_baseline_runner(config: Path, traces: Path) -> BaselineRunner:
    settings = load_yaml(config)
    model = ModelIdentity(
        repo_id=settings["model"]["repo_id"],
        revision=settings["model"]["revision"],
        local_path=Path(settings["model"]["local_path"]),
        backend=settings["inference"]["backend"],
    )
    generation = GenerationConfig.model_validate(settings["inference"])
    adapter = TransformersAdapter(
        model_path=model.local_path,
        backend=model.backend,
        dtype=settings["model"]["dtype"],
        enable_thinking=generation.enable_thinking,
    )
    return BaselineRunner(adapter, model, generation, TraceStore(traces))


@app.command()
def doctor(config: Path = Path("configs/qwen3_8b.yaml")) -> None:
    """Check configuration and local model availability."""
    settings = load_yaml(config)
    model_path = Path(settings["model"]["local_path"])
    expected_weights = [
        model_path / f"model-{index:05d}-of-00005.safetensors" for index in range(1, 6)
    ]
    model_ready = (model_path / "model.safetensors.index.json").is_file() and all(
        path.is_file() for path in expected_weights
    )
    table = Table("Check", "Value")
    table.add_row("model", settings["model"]["repo_id"])
    table.add_row("revision", settings["model"]["revision"])
    table.add_row("backend", settings["inference"]["backend"])
    table.add_row("model files", "ready" if model_ready else "not downloaded")
    console.print(table)


@app.command("score-example")
def score_example() -> None:
    """Exercise the transparent MVP action-scoring function."""
    prediction = PredictedOutcome(
        success_probability=0.8,
        expected_goal_progress=0.7,
        expected_information_gain=0.4,
        ambiguity=0.2,
        token_cost=0.1,
        wall_time_cost=0.1,
        operational_risk=0.0,
    )
    console.print(aif_score(prediction))


@app.command("run")
def run_baseline(
    text: str,
    config: Path = Path("configs/qwen3_8b.yaml"),
    traces: Path = Path("artifacts/b0/runs.jsonl"),
) -> None:
    """Run one frozen B0 model call and persist its trace."""
    trace = _build_baseline_runner(config, traces).run(Task(id=f"adhoc-{uuid4()}", text=text))
    if trace.status == "failed":
        console.print(f"[red]{trace.error}[/red]")
        raise typer.Exit(1)
    if trace.result is None:
        raise RuntimeError("completed trace is missing its result")
    console.print(trace.result.text)
    console.print(
        f"[dim]run={trace.run_id} input={trace.result.input_tokens} "
        f"output={trace.result.output_tokens} generation={trace.result.generation_seconds:.2f}s "
        f"trace={traces}[/dim]"
    )


@app.command()
def replay(
    run_id: str,
    traces: Path = Path("artifacts/b0/runs.jsonl"),
) -> None:
    """Replay a saved B0 result without loading or calling the model."""
    trace = TraceStore(traces).get(run_id)
    if trace.result is not None:
        console.print(trace.result.text)
    console.print(
        f"[dim]run={trace.run_id} status={trace.status} prompt_sha256={trace.prompt_sha256}[/dim]"
    )


@app.command("eval-b0")
def eval_b0(
    fixtures: Path = Path("evals/tasks/b0/direct_answer.yaml"),
    config: Path = Path("configs/qwen3_8b.yaml"),
    traces: Path = Path("artifacts/b0/eval-runs.jsonl"),
    report: Path = Path("artifacts/b0/report.json"),
    repeats: int = 1,
) -> None:
    """Run and grade frozen B0 fixtures, optionally with repeated suites."""
    if repeats < 1:
        raise typer.BadParameter("repeats must be positive")
    runner = _build_baseline_runner(config, traces)
    if repeats > 1:
        repeated = evaluate_repeated_baseline(runner, fixtures, report, repeats)
        table = Table("Fixture", "Output agreement", "Median", "Min", "Max")
        for repeated_case in repeated.cases:
            table.add_row(
                repeated_case.fixture_id,
                "YES" if repeated_case.output_agreement else "NO",
                f"{repeated_case.latency_median_seconds:.2f}s",
                f"{repeated_case.latency_min_seconds:.2f}s",
                f"{repeated_case.latency_max_seconds:.2f}s",
            )
        console.print(table)
        console.print(
            f"[dim]gate={'PASS' if repeated.gate_passed else 'FAIL'} "
            f"passed={repeated.passed_runs}/{repeated.total_runs} "
            f"agreement={repeated.output_agreement_rate:.1%} "
            f"cold={repeated.first_generation_seconds:.2f}s "
            f"warm_median={repeated.warm_generation_median_seconds:.2f}s "
            f"report={report} traces={traces}[/dim]"
        )
        return
    result = evaluate_baseline(runner, fixtures, report)
    table = Table("Fixture", "Status", "Expected", "Actual", "Generation")
    for suite_case in result.cases:
        table.add_row(
            suite_case.fixture_id,
            "PASS" if suite_case.passed else "FAIL",
            suite_case.expected,
            suite_case.actual or suite_case.error or "",
            f"{suite_case.generation_seconds:.2f}s",
        )
    console.print(table)
    console.print(
        f"[dim]passed={result.passed_cases}/{result.total_cases} "
        f"input={result.input_tokens} output={result.output_tokens} "
        f"load={result.model_load_seconds:.2f}s generation={result.generation_seconds:.2f}s "
        f"report={report} traces={traces}[/dim]"
    )


@app.command("regrade-b0")
def regrade_b0(
    report: Path = Path("artifacts/b0/report.json"),
    fixtures: Path = Path("evals/tasks/b0/direct_answer.yaml"),
    traces: Path = Path("artifacts/b0/eval-runs.jsonl"),
) -> None:
    """Verify and regrade a B0 report using only saved traces."""
    result = load_any_report(report)
    trace_store = TraceStore(traces)
    if isinstance(result, BaselineReproducibilityReport):
        verify_reproducibility_report(result, fixtures, trace_store)
        console.print(
            f"verified report={result.report_id} gate={'PASS' if result.gate_passed else 'FAIL'} "
            f"passed={result.passed_runs}/{result.total_runs}"
        )
    else:
        verify_report(result, fixtures, trace_store)
        console.print(
            f"verified report={result.report_id} passed={result.passed_cases}/{result.total_cases}"
        )


@tool_app.command("read-file")
def read_file_tool(
    path: str,
    max_bytes: int = 131_072,
    config: Path = Path("configs/policy.yaml"),
    traces: Path = Path("artifacts/b1a/read-file.jsonl"),
) -> None:
    """Read one bounded UTF-8 file through the B1a policy gateway."""
    settings = load_yaml(config)
    policy = ReadFilePolicy.model_validate(settings["filesystem"])
    trace = ReadFileTool(policy, ReadFileTraceStore(traces)).run(
        ReadFileRequest(path=path, max_bytes=max_bytes)
    )
    if trace.status == "rejected":
        if trace.rejection is None:
            raise RuntimeError("rejected tool trace is missing its reason")
        console.print(
            f"[red]{trace.rejection.code}: {trace.rejection.message}[/red]\n"
            f"[dim]phase={trace.rejection.phase} trace={trace.trace_id} file={traces}[/dim]"
        )
        raise typer.Exit(1)
    if trace.observation is None:
        raise RuntimeError("completed tool trace is missing its observation")
    console.print(_terminal_safe_text(trace.observation.content), markup=False)
    console.print(
        f"[dim]bytes={trace.observation.byte_count} sha256={trace.observation.sha256} "
        f"trace={trace.trace_id} file={traces}[/dim]"
    )


def main() -> None:
    app()

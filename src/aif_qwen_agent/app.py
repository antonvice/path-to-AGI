from pathlib import Path
from typing import Any
from uuid import uuid4

import typer
from rich.console import Console
from rich.table import Table

from aif_qwen_agent.agent import AgentTraceStore, OneStepAgent
from aif_qwen_agent.aif_score import aif_score
from aif_qwen_agent.artifacts import TraceStore
from aif_qwen_agent.b1_cost import (
    evaluate_b1_cost,
    load_b1_cost_report,
    verify_b1_cost_report,
)
from aif_qwen_agent.b1_evaluation import (
    evaluate_b1,
    verify_b1_report,
)
from aif_qwen_agent.b1_independent import (
    load_b1g_independent_report,
    run_b1g_processes,
    verify_b1g_independent_report,
)
from aif_qwen_agent.b1_reproducibility import (
    evaluate_repeated_b1,
    load_any_b1_report,
    verify_repeated_b1_report,
)
from aif_qwen_agent.b2_evaluation import (
    B2MemoryTraceStore,
    EpisodicMemoryRunner,
    evaluate_b2,
    load_b2_report,
    verify_b2_report,
)
from aif_qwen_agent.baseline import BaselineRunner
from aif_qwen_agent.config import load_yaml
from aif_qwen_agent.evaluation import (
    evaluate_baseline,
    evaluate_repeated_baseline,
    load_any_report,
    verify_report,
    verify_reproducibility_report,
)
from aif_qwen_agent.memory import EpisodicMemoryStore
from aif_qwen_agent.model_adapters import OllamaAdapter, TransformersAdapter
from aif_qwen_agent.schemas import (
    B1RepeatedEvaluationReport,
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

ModelAdapter = TransformersAdapter | OllamaAdapter


def _terminal_safe_text(value: str) -> str:
    return "".join(
        character
        if character in "\n\r\t" or (ord(character) >= 32 and ord(character) != 127)
        else f"\\x{ord(character):02x}"
        for character in value
    )


def _model_identity(settings: dict[str, Any]) -> ModelIdentity:
    model_settings = settings["model"]
    local_path = model_settings.get("local_path")
    return ModelIdentity(
        repo_id=model_settings["repo_id"],
        revision=model_settings["revision"],
        local_path=Path(local_path) if local_path is not None else None,
        backend=settings["inference"]["backend"],
    )


def _model_adapter(
    settings: dict[str, Any],
    model: ModelIdentity,
    generation: GenerationConfig,
) -> ModelAdapter:
    if model.backend == "ollama":
        ollama = settings.get("ollama", {})
        return OllamaAdapter(
            model=model.repo_id,
            digest=model.revision,
            endpoint=settings["inference"].get("endpoint", "http://127.0.0.1:11434"),
            context_tokens=settings["inference"].get("max_context_tokens", 32_768),
            enable_thinking=generation.enable_thinking,
            keep_alive=ollama.get("keep_alive", "5m"),
        )
    if model.local_path is None:
        raise ValueError("Transformers models require model.local_path")
    return TransformersAdapter(
        model_path=model.local_path,
        backend=model.backend,
        dtype=settings["model"].get("dtype", "auto"),
        enable_thinking=generation.enable_thinking,
    )


def _build_baseline_runner(config: Path, traces: Path) -> BaselineRunner:
    settings = load_yaml(config)
    model = _model_identity(settings)
    generation = GenerationConfig.model_validate(settings["inference"])
    adapter = _model_adapter(settings, model, generation)
    return BaselineRunner(adapter, model, generation, TraceStore(traces))


def _build_agent_runner(
    config: Path,
    policy_config: Path,
    traces: Path,
    tool_traces: Path,
) -> OneStepAgent:
    settings = load_yaml(config)
    policy_settings = load_yaml(policy_config)
    model = _model_identity(settings)
    generation = GenerationConfig.model_validate(settings["inference"])
    agent_settings = settings.get("agent", {})
    proposal_generation = generation.model_copy(
        update={
            "max_new_tokens": agent_settings.get(
                "proposal_max_new_tokens", generation.max_new_tokens
            )
        }
    )
    adapter = _model_adapter(settings, model, generation)
    read_file = ReadFileTool(
        ReadFilePolicy.model_validate(policy_settings["filesystem"]),
        ReadFileTraceStore(tool_traces),
    )
    return OneStepAgent(
        adapter,
        model,
        generation,
        read_file,
        AgentTraceStore(traces),
        max_proposal_attempts=policy_settings["budgets"]["max_retries_per_action"] + 1,
        proposal_generation=proposal_generation,
        prompt_profile=agent_settings.get("prompt_profile", "legacy"),
    )


def _build_b1_runners(
    config: Path,
    policy_config: Path,
    baseline_traces: Path,
    agent_traces: Path,
    tool_traces: Path,
) -> tuple[BaselineRunner, OneStepAgent]:
    settings = load_yaml(config)
    policy_settings = load_yaml(policy_config)
    model = _model_identity(settings)
    generation = GenerationConfig.model_validate(settings["inference"])
    agent_settings = settings.get("agent", {})
    proposal_generation = generation.model_copy(
        update={
            "max_new_tokens": agent_settings.get(
                "proposal_max_new_tokens", generation.max_new_tokens
            )
        }
    )
    adapter = _model_adapter(settings, model, generation)
    baseline = BaselineRunner(adapter, model, generation, TraceStore(baseline_traces))
    agent = OneStepAgent(
        adapter,
        model,
        generation,
        ReadFileTool(
            ReadFilePolicy.model_validate(policy_settings["filesystem"]),
            ReadFileTraceStore(tool_traces),
        ),
        AgentTraceStore(agent_traces),
        max_proposal_attempts=policy_settings["budgets"]["max_retries_per_action"] + 1,
        proposal_generation=proposal_generation,
        prompt_profile=agent_settings.get("prompt_profile", "legacy"),
    )
    return baseline, agent


def _build_b2_runners(
    config: Path,
    memory_db: Path,
    baseline_traces: Path,
    memory_traces: Path,
) -> tuple[BaselineRunner, EpisodicMemoryRunner]:
    settings = load_yaml(config)
    model = _model_identity(settings)
    generation = GenerationConfig.model_validate(settings["inference"])
    adapter = _model_adapter(settings, model, generation)
    return (
        BaselineRunner(adapter, model, generation, TraceStore(baseline_traces)),
        EpisodicMemoryRunner(
            adapter,
            model,
            generation,
            EpisodicMemoryStore(memory_db),
            B2MemoryTraceStore(memory_traces),
        ),
    )


@app.command()
def doctor(config: Path = Path("configs/qwen3_8b.yaml")) -> None:
    """Check configuration and local model availability."""
    settings = load_yaml(config)
    model = _model_identity(settings)
    if model.backend == "ollama":
        adapter = _model_adapter(
            settings,
            model,
            GenerationConfig.model_validate(settings["inference"]),
        )
        if not isinstance(adapter, OllamaAdapter):
            raise TypeError("Ollama config produced the wrong adapter")
        adapter.verify_model()
        table = Table("Check", "Value")
        table.add_row("model", model.repo_id)
        table.add_row("revision", model.revision)
        table.add_row("backend", model.backend)
        table.add_row("model digest", "verified")
        console.print(table)
        return
    if model.local_path is None:
        raise ValueError("Transformers models require model.local_path")
    model_path = model.local_path
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


@app.command("agent")
def run_agent(
    text: str,
    config: Path = Path("configs/qwen3_8b.yaml"),
    policy: Path = Path("configs/policy.yaml"),
    traces: Path = Path("artifacts/b1b/runs.jsonl"),
    tool_traces: Path = Path("artifacts/b1b/read-file.jsonl"),
) -> None:
    """Run one B1b model-selected, read-only action and persist its trace."""
    trace = _build_agent_runner(config, policy, traces, tool_traces).run(
        Task(id=f"adhoc-{uuid4()}", text=text)
    )
    if trace.answer is not None:
        console.print(_terminal_safe_text(trace.answer), markup=False)
    console.print(
        f"[dim]run={trace.run_id} status={trace.status} "
        f"action={trace.selected_action.kind if trace.selected_action else 'none'} "
        f"input={trace.input_tokens} output={trace.output_tokens} "
        f"generation={trace.generation_seconds:.2f}s trace={traces}[/dim]"
    )
    if trace.status in {"rejected", "failed"}:
        if trace.error is not None:
            console.print(_terminal_safe_text(trace.error), style="red", markup=False)
        raise typer.Exit(1)


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


@app.command("eval-b1")
def eval_b1(
    fixtures: Path = Path("evals/tasks/b1d/suite.yaml"),
    config: Path = Path("configs/qwen3_8b.yaml"),
    policy: Path = Path("configs/policy.yaml"),
    evaluation_config: Path = Path("configs/evaluation.yaml"),
    baseline_traces: Path = Path("artifacts/b1d/b0.jsonl"),
    agent_traces: Path = Path("artifacts/b1d/b1.jsonl"),
    tool_traces: Path = Path("artifacts/b1d/read-file.jsonl"),
    report: Path = Path("artifacts/b1d/report.json"),
    repeats: int = 1,
) -> None:
    """Run the shared-model B0/B1 quality and safety suite, optionally repeated."""
    if repeats < 1:
        raise typer.BadParameter("repeats must be positive")
    baseline, agent = _build_b1_runners(
        config,
        policy,
        baseline_traces,
        agent_traces,
        tool_traces,
    )
    if repeats > 1:
        repeated = evaluate_repeated_b1(
            baseline,
            agent,
            fixtures,
            evaluation_config,
            report,
            repeats,
        )
        table = Table("Fixture", "Action", "Output", "Tokens", "Rejection", "Evidence")
        for comparison in repeated.comparisons:
            table.add_row(
                comparison.fixture_id,
                "YES" if comparison.action_agreement else "NO",
                "YES" if comparison.output_agreement else "NO",
                "YES" if comparison.token_agreement else "NO",
                "YES" if comparison.rejection_agreement else "NO",
                "YES" if comparison.evidence_agreement else "NO",
            )
        console.print(table)
        console.print(
            f"[dim]gate={'PASS' if repeated.gate_passed else 'FAIL'} "
            f"quality={'PASS' if repeated.quality_gate_passed else 'FAIL'} "
            f"safety={'PASS' if repeated.safety_gate_passed else 'FAIL'} "
            f"repro={'PASS' if repeated.reproducibility_gate_passed else 'FAIL'} "
            f"cost={'PASS' if repeated.cost_gate_passed else 'FAIL'} "
            f"quality_delta={repeated.quality_delta:.1%} "
            f"token_cost={repeated.token_cost_increase:.1%} "
            f"generation_cost={repeated.generation_cost_increase:.1%} "
            f"report={report}[/dim]"
        )
        return
    result = evaluate_b1(baseline, agent, fixtures, report)
    table = Table("Fixture", "Kind", "B0", "B1", "Agent status", "Retries")
    for case in result.cases:
        table.add_row(
            case.fixture_id,
            case.kind,
            "-" if case.baseline_passed is None else ("PASS" if case.baseline_passed else "FAIL"),
            "PASS" if case.agent_passed else "FAIL",
            case.agent_status,
            str(max(case.proposal_attempts - 1, 0)),
        )
    console.print(table)
    console.print(
        f"[dim]gate={'PASS' if result.gate_passed else 'FAIL'} "
        f"grounded_b0={result.baseline_passed_cases}/{result.grounded_cases} "
        f"grounded_b1={result.agent_passed_cases}/{result.grounded_cases} "
        f"safety={result.safety_passed_cases}/{result.safety_cases} "
        f"violations={result.safety_violations} retries={result.proposal_retries} "
        f"load={result.model_load_seconds:.2f}s "
        f"b0_generation={result.baseline_generation_seconds:.2f}s "
        f"b1_generation={result.agent_generation_seconds:.2f}s report={report}[/dim]"
    )


@app.command("regrade-b1")
def regrade_b1(
    report: Path = Path("artifacts/b1d/report.json"),
    fixtures: Path = Path("evals/tasks/b1d/suite.yaml"),
    evaluation_config: Path = Path("configs/evaluation.yaml"),
    baseline_traces: Path = Path("artifacts/b1d/b0.jsonl"),
    agent_traces: Path = Path("artifacts/b1d/b1.jsonl"),
) -> None:
    """Verify and regrade a B1 report using only saved traces."""
    result = load_any_b1_report(report)
    baseline_store = TraceStore(baseline_traces)
    agent_store = AgentTraceStore(agent_traces)
    if isinstance(result, B1RepeatedEvaluationReport):
        verify_repeated_b1_report(
            result,
            fixtures,
            evaluation_config,
            baseline_store,
            agent_store,
        )
        console.print(
            f"verified report={result.report_id} gate={'PASS' if result.gate_passed else 'FAIL'} "
            f"quality={'PASS' if result.quality_gate_passed else 'FAIL'} "
            f"safety={'PASS' if result.safety_gate_passed else 'FAIL'} "
            f"repro={'PASS' if result.reproducibility_gate_passed else 'FAIL'} "
            f"cost={'PASS' if result.cost_gate_passed else 'FAIL'}"
        )
        return
    verify_b1_report(
        result,
        fixtures,
        baseline_store,
        agent_store,
    )
    console.print(
        f"verified report={result.report_id} gate={'PASS' if result.gate_passed else 'FAIL'} "
        f"grounded={result.agent_passed_cases}/{result.grounded_cases} "
        f"safety={result.safety_passed_cases}/{result.safety_cases}"
    )


@app.command("eval-b1g")
def eval_b1g(
    fixtures: Path = Path("evals/tasks/b1g/suite.yaml"),
    freeze_manifest: Path = Path("evals/tasks/b1g/freeze.json"),
    config: Path = Path("configs/qwen3_8_27b_b1g.yaml"),
    policy: Path = Path("configs/policy.yaml"),
    evaluation_config: Path = Path("configs/evaluation.yaml"),
    output_dir: Path = Path("artifacts/b1g"),
    processes: int = 3,
) -> None:
    """Run B1g in cold, independent processes and build its promotion gate."""
    result = run_b1g_processes(
        fixtures,
        freeze_manifest,
        evaluation_config,
        config,
        policy,
        output_dir,
        processes,
        status=lambda message: console.print(f"[dim]{message}[/dim]"),
    )
    table = Table("Process", "PID", "Cold load", "Suite gate")
    for process in result.processes:
        table.add_row(
            str(process.process_index),
            str(process.process_id),
            f"{process.suite.model_load_seconds:.2f}s",
            "PASS" if process.suite.gate_passed else "FAIL",
        )
    console.print(table)
    console.print(
        f"[dim]promotion={'PASS' if result.promotion_gate_passed else 'FAIL'} "
        f"quality={'PASS' if result.quality_gate_passed else 'FAIL'} "
        f"safety={'PASS' if result.safety_gate_passed else 'FAIL'} "
        f"repro={'PASS' if result.reproducibility_gate_passed else 'FAIL'} "
        f"cost={'PASS' if result.cost_gate_passed else 'FAIL'} "
        f"quality_delta={result.quality_delta:.1%} "
        f"grounded_token_cost={result.grounded_token_cost_increase:.1%} "
        f"grounded_generation_cost={result.grounded_generation_cost_increase:.1%} "
        f"report={output_dir / 'report.json'}[/dim]"
    )


@app.command("regrade-b1g")
def regrade_b1g(report: Path = Path("artifacts/b1g/report.json")) -> None:
    """Verify B1g entirely from its frozen manifest, reports, and traces."""
    result = load_b1g_independent_report(report)
    verify_b1g_independent_report(result)
    console.print(
        f"verified report={result.report_id} "
        f"promotion={'PASS' if result.promotion_gate_passed else 'FAIL'} "
        f"processes={result.process_count} "
        f"quality={'PASS' if result.quality_gate_passed else 'FAIL'} "
        f"safety={'PASS' if result.safety_gate_passed else 'FAIL'} "
        f"repro={'PASS' if result.reproducibility_gate_passed else 'FAIL'} "
        f"cost={'PASS' if result.cost_gate_passed else 'FAIL'}"
    )


@app.command("eval-b2")
def eval_b2(
    fixtures: Path = Path("evals/tasks/b2/suite.yaml"),
    freeze_manifest: Path = Path("evals/tasks/b2/freeze.json"),
    config: Path = Path("configs/qwen3_8_27b_b1g.yaml"),
    evaluation_config: Path = Path("configs/evaluation.yaml"),
    memory_db: Path = Path("artifacts/b2/memory.db"),
    baseline_traces: Path = Path("artifacts/b2/baseline.jsonl"),
    memory_traces: Path = Path("artifacts/b2/memory.jsonl"),
    report: Path = Path("artifacts/b2/report.json"),
) -> None:
    """Run the frozen two-session B2 episodic retrieval suite once."""
    baseline, memory = _build_b2_runners(
        config,
        memory_db,
        baseline_traces,
        memory_traces,
    )
    result = evaluate_b2(
        baseline,
        memory,
        fixtures,
        freeze_manifest,
        evaluation_config,
        report,
    )
    table = Table("Fixture", "Kind", "B0", "Memory", "Retrieval", "Safety")
    for case in result.cases:
        table.add_row(
            case.fixture_id,
            case.kind,
            "PASS" if case.baseline_passed else "FAIL",
            "PASS" if case.memory_passed else "FAIL",
            "PASS" if case.retrieval_passed else "FAIL",
            "FAIL" if case.safety_violation else "PASS",
        )
    console.print(table)
    console.print(
        f"[dim]gate={'PASS' if result.engineering_gate_passed else 'FAIL'} "
        f"quality={'PASS' if result.quality_gate_passed else 'FAIL'} "
        f"safety={'PASS' if result.safety_gate_passed else 'FAIL'} "
        f"retrieval={'PASS' if result.retrieval_gate_passed else 'FAIL'} "
        f"cost={'PASS' if result.cost_gate_passed else 'FAIL'} "
        f"quality_delta={result.quality_delta:.1%} "
        f"grounded_token_cost={result.grounded_token_cost_increase:.1%} "
        f"grounded_generation_cost={result.grounded_generation_cost_increase:.1%} "
        f"report={report}[/dim]"
    )


@app.command("regrade-b2")
def regrade_b2(
    report: Path = Path("artifacts/b2/report.json"),
    baseline_traces: Path = Path("artifacts/b2/baseline.jsonl"),
    memory_traces: Path = Path("artifacts/b2/memory.jsonl"),
) -> None:
    """Verify B2 entirely from its frozen inputs and saved traces."""
    result = load_b2_report(report)
    verify_b2_report(
        result,
        TraceStore(baseline_traces),
        B2MemoryTraceStore(memory_traces),
    )
    console.print(
        f"verified report={result.report_id} "
        f"gate={'PASS' if result.engineering_gate_passed else 'FAIL'} "
        f"quality={'PASS' if result.quality_gate_passed else 'FAIL'} "
        f"safety={'PASS' if result.safety_gate_passed else 'FAIL'} "
        f"retrieval={'PASS' if result.retrieval_gate_passed else 'FAIL'} "
        f"cost={'PASS' if result.cost_gate_passed else 'FAIL'}"
    )


@app.command("eval-b1e")
def eval_b1e(
    fixtures: Path = Path("evals/tasks/b1d/suite.yaml"),
    config: Path = Path("configs/qwen3_8b_b1e.yaml"),
    policy: Path = Path("configs/policy.yaml"),
    evaluation_config: Path = Path("configs/evaluation.yaml"),
    reference_report: Path = Path("evals/baselines/b1d_repro_mps_report.json"),
    reference_baseline_traces: Path = Path("evals/baselines/b1d_repro_mps_b0.jsonl"),
    reference_agent_traces: Path = Path("evals/baselines/b1d_repro_mps_agent.jsonl"),
    baseline_traces: Path = Path("artifacts/b1e/b0.jsonl"),
    agent_traces: Path = Path("artifacts/b1e/b1.jsonl"),
    tool_traces: Path = Path("artifacts/b1e/read-file.jsonl"),
    optimized_report: Path = Path("artifacts/b1e/suite.json"),
    cost_report: Path = Path("artifacts/b1e/cost.json"),
) -> None:
    """Run compact B1 against the unchanged B1d suite and reference costs."""
    baseline, agent = _build_b1_runners(
        config,
        policy,
        baseline_traces,
        agent_traces,
        tool_traces,
    )
    result = evaluate_b1_cost(
        baseline,
        agent,
        fixtures,
        optimized_report,
        cost_report,
        reference_report,
        TraceStore(reference_baseline_traces),
        AgentTraceStore(reference_agent_traces),
        evaluation_config,
        config,
    )
    table = Table("Stage", "Legacy tokens", "Optimized tokens", "Legacy time", "Optimized time")
    for name, legacy, optimized in (
        (
            "proposal",
            result.legacy_grounded_proposal,
            result.optimized_grounded_proposal,
        ),
        ("answer", result.legacy_grounded_answer, result.optimized_grounded_answer),
    ):
        table.add_row(
            name,
            f"{legacy.input_tokens + legacy.output_tokens:.0f}",
            f"{optimized.input_tokens + optimized.output_tokens:.0f}",
            f"{legacy.generation_seconds:.2f}s",
            f"{optimized.generation_seconds:.2f}s",
        )
    console.print(table)
    console.print(
        f"[dim]gate={'PASS' if result.gate_passed else 'FAIL'} "
        f"quality={'PASS' if result.quality_preserved else 'FAIL'} "
        f"safety={'PASS' if result.safety_preserved else 'FAIL'} "
        f"optimization={'PASS' if result.optimization_gate_passed else 'FAIL'} "
        f"cost={'PASS' if result.cost_gate_passed else 'FAIL'} "
        f"token_reduction={result.token_reduction:.1%} "
        f"generation_reduction={result.generation_reduction:.1%} "
        f"grounded_token_cost={result.grounded_token_cost_increase:.1%} "
        f"grounded_generation_cost={result.grounded_generation_cost_increase:.1%} "
        f"report={cost_report}[/dim]"
    )


@app.command("regrade-b1e")
def regrade_b1e(
    report: Path = Path("artifacts/b1e/cost.json"),
    fixtures: Path = Path("evals/tasks/b1d/suite.yaml"),
    evaluation_config: Path = Path("configs/evaluation.yaml"),
    config: Path = Path("configs/qwen3_8b_b1e.yaml"),
    reference_baseline_traces: Path = Path("evals/baselines/b1d_repro_mps_b0.jsonl"),
    reference_agent_traces: Path = Path("evals/baselines/b1d_repro_mps_agent.jsonl"),
    baseline_traces: Path = Path("artifacts/b1e/b0.jsonl"),
    agent_traces: Path = Path("artifacts/b1e/b1.jsonl"),
) -> None:
    """Verify and rebuild a B1e cost report entirely from saved traces."""
    result = load_b1_cost_report(report)
    verify_b1_cost_report(
        result,
        fixtures,
        evaluation_config,
        config,
        TraceStore(reference_baseline_traces),
        AgentTraceStore(reference_agent_traces),
        TraceStore(baseline_traces),
        AgentTraceStore(agent_traces),
    )
    console.print(
        f"verified report={result.report_id} gate={'PASS' if result.gate_passed else 'FAIL'} "
        f"optimization={'PASS' if result.optimization_gate_passed else 'FAIL'} "
        f"cost={'PASS' if result.cost_gate_passed else 'FAIL'}"
    )


@app.command("eval-b1f")
def eval_b1f(
    fixtures: Path = Path("evals/tasks/b1d/suite.yaml"),
    config: Path = Path("configs/qwen3_8b_b1f.yaml"),
    policy: Path = Path("configs/policy.yaml"),
    evaluation_config: Path = Path("configs/evaluation.yaml"),
    reference_report: Path = Path("evals/baselines/b1d_repro_mps_report.json"),
    reference_baseline_traces: Path = Path("evals/baselines/b1d_repro_mps_b0.jsonl"),
    reference_agent_traces: Path = Path("evals/baselines/b1d_repro_mps_agent.jsonl"),
    baseline_traces: Path = Path("artifacts/b1f/b0.jsonl"),
    agent_traces: Path = Path("artifacts/b1f/b1.jsonl"),
    tool_traces: Path = Path("artifacts/b1f/read-file.jsonl"),
    optimized_report: Path = Path("artifacts/b1f/suite.json"),
    cost_report: Path = Path("artifacts/b1f/cost.json"),
) -> None:
    """Run fast-path B1 against the unchanged B1d suite and reference costs."""
    eval_b1e(
        fixtures,
        config,
        policy,
        evaluation_config,
        reference_report,
        reference_baseline_traces,
        reference_agent_traces,
        baseline_traces,
        agent_traces,
        tool_traces,
        optimized_report,
        cost_report,
    )


@app.command("regrade-b1f")
def regrade_b1f(
    report: Path = Path("artifacts/b1f/cost.json"),
    fixtures: Path = Path("evals/tasks/b1d/suite.yaml"),
    evaluation_config: Path = Path("configs/evaluation.yaml"),
    config: Path = Path("configs/qwen3_8b_b1f.yaml"),
    reference_baseline_traces: Path = Path("evals/baselines/b1d_repro_mps_b0.jsonl"),
    reference_agent_traces: Path = Path("evals/baselines/b1d_repro_mps_agent.jsonl"),
    baseline_traces: Path = Path("artifacts/b1f/b0.jsonl"),
    agent_traces: Path = Path("artifacts/b1f/b1.jsonl"),
) -> None:
    """Verify and rebuild a B1f cost report entirely from saved traces."""
    regrade_b1e(
        report,
        fixtures,
        evaluation_config,
        config,
        reference_baseline_traces,
        reference_agent_traces,
        baseline_traces,
        agent_traces,
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

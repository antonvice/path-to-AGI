import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx

from aif_qwen_agent.agent import AgentTraceStore
from aif_qwen_agent.artifacts import TraceStore, sha256_file
from aif_qwen_agent.b1_evaluation import load_b1_report, verify_b1_report
from aif_qwen_agent.b1_reproducibility import build_b1_comparisons_from_stores
from aif_qwen_agent.config import load_yaml
from aif_qwen_agent.schemas import (
    B1EvaluationReport,
    B1gIndependentEvaluationReport,
    B1gProcessArtifact,
    GenerationConfig,
    ModelIdentity,
)
from aif_qwen_agent.tools import ReadFileTraceStore


@dataclass(frozen=True)
class B1gProcessFiles:
    process_index: int
    process_id: int
    suite_report: Path
    baseline_traces: Path
    agent_traces: Path
    tool_traces: Path
    model_unloaded_before: bool = True


def verify_b1g_freeze(path: Path) -> dict[str, object]:
    document: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != "1" or document.get("milestone") != "B1g":
        raise ValueError("invalid B1g freeze manifest")
    files = document.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("B1g freeze manifest has no files")
    for name, expected in files.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            raise ValueError("invalid B1g freeze file entry")
        if sha256_file(Path(name)) != expected:
            raise ValueError(f"frozen B1g file hash mismatch: {name}")
    return document


def _bound_model(config: Path, freeze_manifest: Path) -> tuple[ModelIdentity, GenerationConfig]:
    manifest = verify_b1g_freeze(freeze_manifest)
    settings = load_yaml(config)
    model_settings = settings["model"]
    local_path = model_settings.get("local_path")
    model = ModelIdentity(
        repo_id=model_settings["repo_id"],
        revision=model_settings["revision"],
        local_path=Path(local_path) if local_path is not None else None,
        backend=settings["inference"]["backend"],
    )
    generation = GenerationConfig.model_validate(settings["inference"])
    if (
        manifest.get("model") != model.repo_id
        or manifest.get("model_digest") != model.revision
        or model.backend != "ollama"
    ):
        raise ValueError("B1g config does not match the frozen Ollama model")
    return model, generation


def _verify_tool_traces(suite: B1EvaluationReport, process: B1gProcessFiles) -> None:
    agents = AgentTraceStore(process.agent_traces)
    tools = ReadFileTraceStore(process.tool_traces)
    for case in suite.cases:
        embedded = agents.get(str(case.agent_run_id)).tool_trace
        if embedded is not None and tools.get(str(embedded.trace_id)) != embedded:
            raise ValueError(f"tool trace mismatch for {case.fixture_id}")


def build_b1g_independent_report(
    processes: list[B1gProcessFiles],
    fixture: Path,
    freeze_manifest: Path,
    evaluation_config: Path,
    agent_config: Path,
    policy: Path,
    started_at: datetime,
) -> B1gIndependentEvaluationReport:
    if len(processes) < 3:
        raise ValueError("B1g requires at least three independent model processes")
    model, generation = _bound_model(agent_config, freeze_manifest)
    fixture_sha256 = sha256_file(fixture)
    manifest = verify_b1g_freeze(freeze_manifest)
    frozen_files = manifest["files"]
    if not isinstance(frozen_files, dict) or frozen_files.get(str(fixture)) != fixture_sha256:
        raise ValueError("B1g fixture is not bound by the freeze manifest")
    suites: list[B1EvaluationReport] = []
    baseline_stores: list[TraceStore] = []
    agent_stores: list[AgentTraceStore] = []
    artifacts: list[B1gProcessArtifact] = []
    for process in processes:
        if not process.model_unloaded_before:
            raise ValueError("every B1g process must start with the model unloaded")
        suite = load_b1_report(process.suite_report)
        baseline_store = TraceStore(process.baseline_traces)
        agent_store = AgentTraceStore(process.agent_traces)
        verify_b1_report(suite, fixture, baseline_store, agent_store)
        _verify_tool_traces(suite, process)
        suites.append(suite)
        baseline_stores.append(baseline_store)
        agent_stores.append(agent_store)
        artifacts.append(
            B1gProcessArtifact(
                process_index=process.process_index,
                process_id=process.process_id,
                model_unloaded_before=True,
                suite_report_file=str(process.suite_report),
                suite_report_sha256=sha256_file(process.suite_report),
                baseline_traces_file=str(process.baseline_traces),
                baseline_traces_sha256=sha256_file(process.baseline_traces),
                agent_traces_file=str(process.agent_traces),
                agent_traces_sha256=sha256_file(process.agent_traces),
                tool_traces_file=str(process.tool_traces),
                tool_traces_sha256=sha256_file(process.tool_traces),
                suite=suite,
            )
        )
    comparisons = build_b1_comparisons_from_stores(
        suites,
        baseline_stores,
        agent_stores,
    )
    grounded = [case for suite in suites for case in suite.cases if case.kind == "grounded"]
    grounded_runs = len(grounded)
    baseline_passed = sum(suite.baseline_passed_cases for suite in suites)
    agent_passed = sum(suite.agent_passed_cases for suite in suites)
    safety_runs = sum(suite.safety_cases for suite in suites)
    safety_passed = sum(suite.safety_passed_cases for suite in suites)
    safety_violations = sum(suite.safety_violations for suite in suites)
    instruction_violations = sum(
        case.instruction_following_violation for suite in suites for case in suite.cases
    )
    baseline_input = sum(case.baseline_input_tokens for case in grounded)
    baseline_output = sum(case.baseline_output_tokens for case in grounded)
    agent_input = sum(case.agent_input_tokens for case in grounded)
    agent_output = sum(case.agent_output_tokens for case in grounded)
    baseline_generation = sum(case.baseline_generation_seconds for case in grounded)
    agent_generation = sum(case.agent_generation_seconds for case in grounded)
    if baseline_input + baseline_output == 0 or baseline_generation == 0.0:
        raise ValueError("B1g grounded cost comparison requires nonzero B0 cost")
    quality_delta = agent_passed / grounded_runs - baseline_passed / grounded_runs
    token_cost = (agent_input + agent_output) / (baseline_input + baseline_output) - 1.0
    generation_cost = agent_generation / baseline_generation - 1.0
    promotion = load_yaml(evaluation_config)["promotion"]
    minimum_success_delta = float(promotion["minimum_success_delta"])
    maximum_cost_increase = float(promotion["maximum_cost_increase"])
    quality_gate = quality_delta >= minimum_success_delta
    safety_gate = (
        safety_passed == safety_runs and safety_violations == 0 and instruction_violations == 0
    )
    reproducibility_gate = all(comparison.all_agreement for comparison in comparisons)
    cost_gate = max(token_cost, generation_cost) <= maximum_cost_increase
    return B1gIndependentEvaluationReport(
        report_id=uuid4(),
        started_at=started_at,
        finished_at=datetime.now(UTC),
        fixture_file=str(fixture),
        fixture_sha256=fixture_sha256,
        freeze_manifest_file=str(freeze_manifest),
        freeze_manifest_sha256=sha256_file(freeze_manifest),
        evaluation_config_file=str(evaluation_config),
        evaluation_config_sha256=sha256_file(evaluation_config),
        agent_config_file=str(agent_config),
        agent_config_sha256=sha256_file(agent_config),
        policy_file=str(policy),
        policy_sha256=sha256_file(policy),
        model=model,
        generation=generation,
        process_count=len(processes),
        processes=artifacts,
        comparisons=comparisons,
        grounded_runs=grounded_runs,
        safety_runs=safety_runs,
        baseline_passed_runs=baseline_passed,
        agent_passed_runs=agent_passed,
        safety_passed_runs=safety_passed,
        safety_violations=safety_violations,
        instruction_following_violations=instruction_violations,
        grounded_baseline_input_tokens=baseline_input,
        grounded_baseline_output_tokens=baseline_output,
        grounded_agent_input_tokens=agent_input,
        grounded_agent_output_tokens=agent_output,
        model_load_seconds=sum(suite.model_load_seconds for suite in suites),
        grounded_baseline_generation_seconds=baseline_generation,
        grounded_agent_generation_seconds=agent_generation,
        quality_delta=quality_delta,
        grounded_token_cost_increase=token_cost,
        grounded_generation_cost_increase=generation_cost,
        minimum_success_delta=minimum_success_delta,
        maximum_cost_increase=maximum_cost_increase,
        quality_gate_passed=quality_gate,
        safety_gate_passed=safety_gate,
        reproducibility_gate_passed=reproducibility_gate,
        cost_gate_passed=cost_gate,
        promotion_gate_passed=(
            quality_gate and safety_gate and reproducibility_gate and cost_gate
        ),
    )


def write_b1g_independent_report(
    processes: list[B1gProcessFiles],
    fixture: Path,
    freeze_manifest: Path,
    evaluation_config: Path,
    agent_config: Path,
    policy: Path,
    started_at: datetime,
    report_path: Path,
) -> B1gIndependentEvaluationReport:
    report = build_b1g_independent_report(
        processes,
        fixture,
        freeze_manifest,
        evaluation_config,
        agent_config,
        policy,
        started_at,
    )
    report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return report


def load_b1g_independent_report(path: Path) -> B1gIndependentEvaluationReport:
    return B1gIndependentEvaluationReport.model_validate_json(path.read_text(encoding="utf-8"))


def verify_b1g_independent_report(report: B1gIndependentEvaluationReport) -> None:
    processes = [
        B1gProcessFiles(
            process_index=process.process_index,
            process_id=process.process_id,
            suite_report=Path(process.suite_report_file),
            baseline_traces=Path(process.baseline_traces_file),
            agent_traces=Path(process.agent_traces_file),
            tool_traces=Path(process.tool_traces_file),
            model_unloaded_before=process.model_unloaded_before,
        )
        for process in report.processes
    ]
    rebuilt = build_b1g_independent_report(
        processes,
        Path(report.fixture_file),
        Path(report.freeze_manifest_file),
        Path(report.evaluation_config_file),
        Path(report.agent_config_file),
        Path(report.policy_file),
        report.started_at,
    )
    ignored = {"report_id", "finished_at"}
    if rebuilt.model_dump(exclude=ignored) != report.model_dump(exclude=ignored):
        raise ValueError("saved B1g independent report does not match artifacts")


def _unload_ollama_model(model: str, endpoint: str) -> None:
    executable = shutil.which("ollama")
    if executable is None:
        raise RuntimeError("ollama executable is unavailable")
    stopped = subprocess.run(
        [executable, "stop", model],
        check=False,
        capture_output=True,
        text=True,
    )
    if stopped.returncode != 0:
        raise RuntimeError(stopped.stderr.strip() or "ollama stop failed")
    response = httpx.get(f"{endpoint.rstrip('/')}/api/ps", timeout=30.0)
    response.raise_for_status()
    if any(item.get("name") == model for item in response.json().get("models", [])):
        raise RuntimeError(f"Ollama model remained loaded after stop: {model}")


def run_b1g_processes(
    fixture: Path,
    freeze_manifest: Path,
    evaluation_config: Path,
    agent_config: Path,
    policy: Path,
    output_dir: Path,
    process_count: int = 3,
    status: Callable[[str], None] | None = None,
) -> B1gIndependentEvaluationReport:
    if process_count < 3:
        raise ValueError("B1g requires at least three processes")
    model, _ = _bound_model(agent_config, freeze_manifest)
    settings = load_yaml(agent_config)
    endpoint = settings["inference"].get("endpoint", "http://127.0.0.1:11434")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite B1g artifacts: {output_dir}")
    output_dir.mkdir(parents=True)
    started_at = datetime.now(UTC)
    processes: list[B1gProcessFiles] = []
    try:
        for index in range(1, process_count + 1):
            _unload_ollama_model(model.repo_id, endpoint)
            process_dir = output_dir / f"process-{index}"
            process_dir.mkdir()
            suite_report = process_dir / "suite.json"
            baseline_traces = process_dir / "b0.jsonl"
            agent_traces = process_dir / "b1.jsonl"
            tool_traces = process_dir / "read-file.jsonl"
            command = [
                sys.executable,
                "-m",
                "aif_qwen_agent",
                "eval-b1",
                "--fixtures",
                str(fixture),
                "--config",
                str(agent_config),
                "--policy",
                str(policy),
                "--evaluation-config",
                str(evaluation_config),
                "--baseline-traces",
                str(baseline_traces),
                "--agent-traces",
                str(agent_traces),
                "--tool-traces",
                str(tool_traces),
                "--report",
                str(suite_report),
            ]
            process = subprocess.Popen(command)
            if status is not None:
                status(f"B1g process {index}/{process_count} started with pid={process.pid}")
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f"B1g process {index} failed with exit code {return_code}")
            processes.append(
                B1gProcessFiles(
                    process_index=index,
                    process_id=process.pid,
                    suite_report=suite_report,
                    baseline_traces=baseline_traces,
                    agent_traces=agent_traces,
                    tool_traces=tool_traces,
                )
            )
        _unload_ollama_model(model.repo_id, endpoint)
    except Exception:
        with suppress(Exception):
            _unload_ollama_model(model.repo_id, endpoint)
        raise
    return write_b1g_independent_report(
        processes,
        fixture,
        freeze_manifest,
        evaluation_config,
        agent_config,
        policy,
        started_at,
        output_dir / "report.json",
    )

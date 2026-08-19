import subprocess
import sys
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from aif_qwen_agent.artifacts import TraceStore, sha256_file
from aif_qwen_agent.b1_independent import _unload_ollama_model
from aif_qwen_agent.b2_evaluation import (
    B2MemoryTraceStore,
    load_b2_report,
    load_b2_suite,
    verify_b2_freeze,
    verify_b2_report,
)
from aif_qwen_agent.config import load_yaml
from aif_qwen_agent.memory import EpisodicMemoryStore
from aif_qwen_agent.schemas import (
    B2CaseReproducibility,
    B2EvaluationReport,
    B2IndependentEvaluationReport,
    B2ProcessArtifact,
    GenerationConfig,
    ModelIdentity,
)


@dataclass(frozen=True)
class B2ProcessFiles:
    process_index: int
    process_id: int
    suite_report: Path
    baseline_traces: Path
    memory_traces: Path
    memory_database: Path
    model_unloaded_before: bool = True


def _all_equal(values: Sequence[object]) -> bool:
    return all(value == values[0] for value in values[1:])


def _bound_model(config: Path, freeze_manifest: Path) -> tuple[ModelIdentity, GenerationConfig]:
    manifest = verify_b2_freeze(freeze_manifest)
    if manifest.get("promotion_eligible") is False:
        raise ValueError("B2 development manifests cannot produce promotion reports")
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
    files = manifest["files"]
    if (
        manifest.get("model") != model.repo_id
        or manifest.get("model_digest") != model.revision
        or model.backend != "ollama"
        or not isinstance(files, dict)
        or files.get(str(config)) != sha256_file(config)
    ):
        raise ValueError("B2 config does not match the frozen Ollama model")
    return model, generation


def _verify_memory_database(path: Path, fixture: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    episodes, _ = load_b2_suite(fixture)
    store = EpisodicMemoryStore(path)
    if store.verify_integrity() != len(episodes):
        raise ValueError("B2 process memory database count differs from fixtures")
    for episode in episodes:
        if store.get(str(episode.episode_id)) != episode:
            raise ValueError("B2 process memory database differs from fixtures")


def build_b2_comparisons(
    suites: list[B2EvaluationReport],
    memory_stores: list[B2MemoryTraceStore],
) -> list[B2CaseReproducibility]:
    if not suites or len(suites) != len(memory_stores):
        raise ValueError("B2 suites and trace stores must have equal nonzero lengths")
    fixture_ids = [case.fixture_id for case in suites[0].cases]
    if any([case.fixture_id for case in suite.cases] != fixture_ids for suite in suites[1:]):
        raise ValueError("B2 suite case order differs across processes")
    comparisons: list[B2CaseReproducibility] = []
    for index, fixture_id in enumerate(fixture_ids):
        cases = [suite.cases[index] for suite in suites]
        if any(case.kind != cases[0].kind for case in cases[1:]):
            raise ValueError(f"B2 case kind differs across processes: {fixture_id}")
        traces = [
            store.get(str(case.memory_run_id))
            for store, case in zip(memory_stores, cases, strict=True)
        ]
        baseline_outputs = [case.baseline_actual for case in cases]
        memory_outputs = [case.memory_actual for case in cases]
        retrieved_ids = [case.retrieved_episode_ids for case in cases]
        statuses = [trace.status for trace in traces]
        token_counts = [
            (
                case.baseline_input_tokens,
                case.baseline_output_tokens,
                case.memory_input_tokens,
                case.memory_output_tokens,
            )
            for case in cases
        ]
        grades = [
            (case.baseline_passed, case.memory_passed, case.retrieval_passed) for case in cases
        ]
        violations = [
            (case.safety_violation, case.instruction_following_violation) for case in cases
        ]
        agreement = (
            _all_equal(baseline_outputs),
            _all_equal(memory_outputs),
            _all_equal(retrieved_ids),
            _all_equal(statuses),
            _all_equal(token_counts),
            _all_equal(grades),
            _all_equal(violations),
        )
        comparisons.append(
            B2CaseReproducibility(
                fixture_id=fixture_id,
                kind=cases[0].kind,
                baseline_run_ids=[case.baseline_run_id for case in cases],
                memory_run_ids=[case.memory_run_id for case in cases],
                baseline_outputs=baseline_outputs,
                memory_outputs=memory_outputs,
                retrieved_episode_ids=retrieved_ids,
                statuses=statuses,
                token_counts=token_counts,
                grades=grades,
                violations=violations,
                baseline_output_agreement=agreement[0],
                memory_output_agreement=agreement[1],
                retrieval_agreement=agreement[2],
                status_agreement=agreement[3],
                token_agreement=agreement[4],
                grade_agreement=agreement[5],
                safety_agreement=agreement[6],
                all_agreement=all(agreement),
            )
        )
    return comparisons


def build_b2_independent_report(
    processes: list[B2ProcessFiles],
    fixture: Path,
    freeze_manifest: Path,
    evaluation_config: Path,
    agent_config: Path,
    started_at: datetime,
) -> B2IndependentEvaluationReport:
    if len(processes) < 3:
        raise ValueError("B2 requires at least three independent model processes")
    model, generation = _bound_model(agent_config, freeze_manifest)
    fixture_sha256 = sha256_file(fixture)
    manifest = verify_b2_freeze(freeze_manifest)
    frozen_files = manifest["files"]
    if not isinstance(frozen_files, dict) or frozen_files.get(str(fixture)) != fixture_sha256:
        raise ValueError("B2 fixture is not bound by the freeze manifest")
    suites: list[B2EvaluationReport] = []
    memory_stores: list[B2MemoryTraceStore] = []
    artifacts: list[B2ProcessArtifact] = []
    for process in processes:
        if not process.model_unloaded_before:
            raise ValueError("every B2 process must start with the model unloaded")
        suite = load_b2_report(process.suite_report)
        baseline_store = TraceStore(process.baseline_traces)
        memory_store = B2MemoryTraceStore(process.memory_traces)
        verify_b2_report(suite, baseline_store, memory_store)
        _verify_memory_database(process.memory_database, fixture)
        suites.append(suite)
        memory_stores.append(memory_store)
        artifacts.append(
            B2ProcessArtifact(
                process_index=process.process_index,
                process_id=process.process_id,
                model_unloaded_before=True,
                suite_report_file=str(process.suite_report),
                suite_report_sha256=sha256_file(process.suite_report),
                baseline_traces_file=str(process.baseline_traces),
                baseline_traces_sha256=sha256_file(process.baseline_traces),
                memory_traces_file=str(process.memory_traces),
                memory_traces_sha256=sha256_file(process.memory_traces),
                memory_database_file=str(process.memory_database),
                memory_database_sha256=sha256_file(process.memory_database),
                suite=suite,
            )
        )
    comparisons = build_b2_comparisons(suites, memory_stores)
    grounded = [case for suite in suites for case in suite.cases if case.kind == "grounded"]
    grounded_runs = len(grounded)
    safety_runs = sum(suite.safety_cases for suite in suites)
    baseline_passed = sum(suite.baseline_passed_cases for suite in suites)
    memory_passed = sum(suite.memory_passed_cases for suite in suites)
    safety_passed = sum(suite.safety_passed_cases for suite in suites)
    retrieval_passed = sum(suite.retrieval_passed_cases for suite in suites)
    safety_violations = sum(suite.safety_violations for suite in suites)
    instruction_violations = sum(suite.instruction_following_violations for suite in suites)
    baseline_input = sum(case.baseline_input_tokens for case in grounded)
    baseline_output = sum(case.baseline_output_tokens for case in grounded)
    memory_input = sum(case.memory_input_tokens for case in grounded)
    memory_output = sum(case.memory_output_tokens for case in grounded)
    baseline_generation = sum(case.baseline_generation_seconds for case in grounded)
    memory_generation = sum(case.memory_generation_seconds for case in grounded)
    if baseline_input + baseline_output == 0 or baseline_generation == 0.0:
        raise ValueError("B2 grounded cost comparison requires nonzero baseline cost")
    quality_delta = memory_passed / grounded_runs - baseline_passed / grounded_runs
    token_cost = (memory_input + memory_output) / (baseline_input + baseline_output) - 1.0
    generation_cost = memory_generation / baseline_generation - 1.0
    promotion = load_yaml(evaluation_config)["promotion"]
    minimum_success_delta = float(promotion["minimum_success_delta"])
    maximum_cost_increase = float(promotion["maximum_cost_increase"])
    quality_gate = quality_delta >= minimum_success_delta
    safety_gate = (
        safety_passed == safety_runs and safety_violations == 0 and instruction_violations == 0
    )
    retrieval_gate = retrieval_passed == grounded_runs + safety_runs
    reproducibility_gate = all(comparison.all_agreement for comparison in comparisons)
    cost_gate = max(token_cost, generation_cost) <= maximum_cost_increase
    return B2IndependentEvaluationReport(
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
        model=model,
        generation=generation,
        process_count=len(processes),
        processes=artifacts,
        comparisons=comparisons,
        grounded_runs=grounded_runs,
        safety_runs=safety_runs,
        baseline_passed_runs=baseline_passed,
        memory_passed_runs=memory_passed,
        safety_passed_runs=safety_passed,
        retrieval_passed_runs=retrieval_passed,
        safety_violations=safety_violations,
        instruction_following_violations=instruction_violations,
        grounded_baseline_input_tokens=baseline_input,
        grounded_baseline_output_tokens=baseline_output,
        grounded_memory_input_tokens=memory_input,
        grounded_memory_output_tokens=memory_output,
        model_load_seconds=sum(suite.model_load_seconds for suite in suites),
        grounded_baseline_generation_seconds=baseline_generation,
        grounded_memory_generation_seconds=memory_generation,
        quality_delta=quality_delta,
        grounded_token_cost_increase=token_cost,
        grounded_generation_cost_increase=generation_cost,
        minimum_success_delta=minimum_success_delta,
        maximum_cost_increase=maximum_cost_increase,
        quality_gate_passed=quality_gate,
        safety_gate_passed=safety_gate,
        retrieval_gate_passed=retrieval_gate,
        reproducibility_gate_passed=reproducibility_gate,
        cost_gate_passed=cost_gate,
        promotion_gate_passed=(
            quality_gate and safety_gate and retrieval_gate and reproducibility_gate and cost_gate
        ),
    )


def write_b2_independent_report(
    processes: list[B2ProcessFiles],
    fixture: Path,
    freeze_manifest: Path,
    evaluation_config: Path,
    agent_config: Path,
    started_at: datetime,
    report_path: Path,
) -> B2IndependentEvaluationReport:
    report = build_b2_independent_report(
        processes,
        fixture,
        freeze_manifest,
        evaluation_config,
        agent_config,
        started_at,
    )
    report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return report


def load_b2_independent_report(path: Path) -> B2IndependentEvaluationReport:
    return B2IndependentEvaluationReport.model_validate_json(path.read_text(encoding="utf-8"))


def verify_b2_independent_report(report: B2IndependentEvaluationReport) -> None:
    processes = [
        B2ProcessFiles(
            process_index=process.process_index,
            process_id=process.process_id,
            suite_report=Path(process.suite_report_file),
            baseline_traces=Path(process.baseline_traces_file),
            memory_traces=Path(process.memory_traces_file),
            memory_database=Path(process.memory_database_file),
            model_unloaded_before=process.model_unloaded_before,
        )
        for process in report.processes
    ]
    rebuilt = build_b2_independent_report(
        processes,
        Path(report.fixture_file),
        Path(report.freeze_manifest_file),
        Path(report.evaluation_config_file),
        Path(report.agent_config_file),
        report.started_at,
    )
    ignored = {"report_id", "finished_at"}
    if rebuilt.model_dump(exclude=ignored) != report.model_dump(exclude=ignored):
        raise ValueError("saved B2 independent report does not match artifacts")


def run_b2_processes(
    fixture: Path,
    freeze_manifest: Path,
    evaluation_config: Path,
    agent_config: Path,
    output_dir: Path,
    process_count: int = 3,
    status: Callable[[str], None] | None = None,
) -> B2IndependentEvaluationReport:
    if process_count < 3:
        raise ValueError("B2 requires at least three processes")
    model, _ = _bound_model(agent_config, freeze_manifest)
    settings = load_yaml(agent_config)
    endpoint = settings["inference"].get("endpoint", "http://127.0.0.1:11434")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite B2 artifacts: {output_dir}")
    output_dir.mkdir(parents=True)
    started_at = datetime.now(UTC)
    processes: list[B2ProcessFiles] = []
    try:
        for index in range(1, process_count + 1):
            _unload_ollama_model(model.repo_id, endpoint)
            process_dir = output_dir / f"process-{index}"
            process_dir.mkdir()
            suite_report = process_dir / "suite.json"
            baseline_traces = process_dir / "baseline.jsonl"
            memory_traces = process_dir / "memory.jsonl"
            memory_database = process_dir / "memory.db"
            command = [
                sys.executable,
                "-m",
                "aif_qwen_agent",
                "eval-b2-suite",
                "--fixtures",
                str(fixture),
                "--freeze-manifest",
                str(freeze_manifest),
                "--config",
                str(agent_config),
                "--evaluation-config",
                str(evaluation_config),
                "--memory-db",
                str(memory_database),
                "--baseline-traces",
                str(baseline_traces),
                "--memory-traces",
                str(memory_traces),
                "--report",
                str(suite_report),
            ]
            process = subprocess.Popen(command)
            if status is not None:
                status(f"B2 process {index}/{process_count} started with pid={process.pid}")
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f"B2 process {index} failed with exit code {return_code}")
            processes.append(
                B2ProcessFiles(
                    process_index=index,
                    process_id=process.pid,
                    suite_report=suite_report,
                    baseline_traces=baseline_traces,
                    memory_traces=memory_traces,
                    memory_database=memory_database,
                )
            )
        _unload_ollama_model(model.repo_id, endpoint)
    except Exception:
        with suppress(Exception):
            _unload_ollama_model(model.repo_id, endpoint)
        raise
    return write_b2_independent_report(
        processes,
        fixture,
        freeze_manifest,
        evaluation_config,
        agent_config,
        started_at,
        output_dir / "report.json",
    )

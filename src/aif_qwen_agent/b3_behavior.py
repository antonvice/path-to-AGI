"""B3 belief-aware behavioral ablation and independent promotion gate."""

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator

from aif_qwen_agent.artifacts import sha256_file
from aif_qwen_agent.b3_evaluation import B3ExpectedHypothesis, B3Operation
from aif_qwen_agent.belief import (
    BeliefStateStore,
    ExplicitBeliefState,
    belief_state_sha256,
    record_unresolved_question,
    update_belief_state,
)
from aif_qwen_agent.belief_decision import (
    BeliefDecision,
    decide_from_beliefs,
    decide_from_latest_observation,
)
from aif_qwen_agent.config import load_yaml


class B3BehaviorFixture(BaseModel):
    id: str
    objective: str
    operations: list[B3Operation] = Field(min_length=1)
    expected_hypotheses: list[B3ExpectedHypothesis] = Field(default_factory=list)
    expected_unresolved_questions: list[str] = Field(default_factory=list)
    expected_answer_substrings: list[str] = Field(min_length=1)
    forbidden_answer_substrings: list[str] = Field(default_factory=list)


class B3BehaviorSuite(BaseModel):
    version: Literal[1]
    milestone: Literal["B3"]
    purpose: Literal["development", "held_out"]
    promotion_eligible: bool
    minimum_quality_delta: float = Field(ge=0.0, le=1.0)
    cases: list[B3BehaviorFixture] = Field(min_length=4)

    @model_validator(mode="after")
    def metadata_and_inventory(self) -> "B3BehaviorSuite":
        if self.promotion_eligible != (self.purpose == "held_out"):
            raise ValueError("B3 behavior purpose and promotion eligibility disagree")
        if len({case.id for case in self.cases}) != len(self.cases):
            raise ValueError("B3 behavior fixture IDs must be unique")
        if len({case.objective for case in self.cases}) != len(self.cases):
            raise ValueError("B3 behavior objectives must be unique")
        return self


class B3BehaviorCaseResult(BaseModel):
    fixture_id: str
    final_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline: BeliefDecision
    belief: BeliefDecision
    state_passed: bool
    baseline_passed: bool
    belief_passed: bool
    safety_passed: bool


class B3BehaviorReport(BaseModel):
    schema_version: Literal["1"] = "1"
    milestone: Literal["B3"] = "B3"
    report_type: Literal["behavior"] = "behavior"
    purpose: Literal["development", "held_out"]
    promotion_eligible: bool
    report_id: UUID
    started_at: datetime
    finished_at: datetime
    fixture_file: str
    fixture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    freeze_manifest_file: str | None = None
    freeze_manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    database_file: str
    database_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cases: list[B3BehaviorCaseResult] = Field(min_length=4)
    baseline_passed_cases: int = Field(ge=0)
    belief_passed_cases: int = Field(ge=0)
    state_passed_cases: int = Field(ge=0)
    safety_passed_cases: int = Field(ge=0)
    quality_delta: float
    minimum_quality_delta: float
    quality_gate_passed: bool
    state_gate_passed: bool
    safety_gate_passed: bool
    engineering_gate_passed: bool

    @model_validator(mode="after")
    def gates_match_cases(self) -> "B3BehaviorReport":
        total = len(self.cases)
        baseline = sum(case.baseline_passed for case in self.cases)
        belief = sum(case.belief_passed for case in self.cases)
        state = sum(case.state_passed for case in self.cases)
        safety = sum(case.safety_passed for case in self.cases)
        delta = belief / total - baseline / total
        gates = (
            delta >= self.minimum_quality_delta,
            state == total,
            safety == total,
        )
        if (
            (self.baseline_passed_cases, self.belief_passed_cases) != (baseline, belief)
            or (self.state_passed_cases, self.safety_passed_cases) != (state, safety)
            or abs(self.quality_delta - delta) > 1e-12
            or (self.quality_gate_passed, self.state_gate_passed, self.safety_gate_passed) != gates
            or self.engineering_gate_passed != all(gates)
        ):
            raise ValueError("B3 behavior report gates do not match cases")
        return self


class B3BehaviorProcess(BaseModel):
    process_index: int = Field(gt=0)
    process_id: int = Field(gt=0)
    report_file: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    database_file: str
    database_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    report: B3BehaviorReport


class B3IndependentReport(BaseModel):
    schema_version: Literal["1"] = "1"
    milestone: Literal["B3"] = "B3"
    report_type: Literal["independent"] = "independent"
    report_id: UUID
    started_at: datetime
    finished_at: datetime
    fixture_file: str
    fixture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    freeze_manifest_file: str
    freeze_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    process_count: int = Field(ge=3)
    processes: list[B3BehaviorProcess] = Field(min_length=3)
    baseline_passed_runs: int = Field(ge=0)
    belief_passed_runs: int = Field(ge=0)
    state_passed_runs: int = Field(ge=0)
    safety_passed_runs: int = Field(ge=0)
    total_runs: int = Field(gt=0)
    quality_delta: float
    minimum_quality_delta: float
    quality_gate_passed: bool
    state_gate_passed: bool
    safety_gate_passed: bool
    reproducibility_gate_passed: bool
    promotion_gate_passed: bool


def load_b3_behavior_suite(path: Path) -> B3BehaviorSuite:
    return B3BehaviorSuite.model_validate(load_yaml(path))


def verify_b3_freeze(path: Path) -> dict[str, object]:
    document: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    if (
        document.get("schema_version") != "1"
        or document.get("milestone") != "B3"
        or document.get("purpose") != "held_out"
        or document.get("promotion_eligible") is not True
    ):
        raise ValueError("invalid B3 freeze manifest")
    files = document.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("B3 freeze manifest has no files")
    for name, expected in files.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            raise ValueError("invalid B3 freeze file entry")
        if sha256_file(Path(name)) != expected:
            raise ValueError(f"frozen B3 file hash mismatch: {name}")
    return document


def _state_matches(state: ExplicitBeliefState, fixture: B3BehaviorFixture) -> bool:
    actual = {
        item.id: (
            item.statement,
            item.probability,
            item.status,
            item.contradicts,
            len(item.evidence),
        )
        for item in state.hypotheses
    }
    expected = {
        item.id: (
            item.statement,
            item.probability,
            item.status,
            item.contradicts,
            item.evidence_count,
        )
        for item in fixture.expected_hypotheses
    }
    return (
        actual == expected and state.unresolved_questions == fixture.expected_unresolved_questions
    )


def _passes(decision: BeliefDecision, fixture: B3BehaviorFixture) -> bool:
    return all(value in decision.answer for value in fixture.expected_answer_substrings) and all(
        value not in decision.answer for value in fixture.forbidden_answer_substrings
    )


def _run_cases(suite: B3BehaviorSuite, store: BeliefStateStore) -> list[B3BehaviorCaseResult]:
    results: list[B3BehaviorCaseResult] = []
    for fixture in suite.cases:
        state = ExplicitBeliefState(objective=fixture.objective)
        store.append(state)
        observations = []
        unresolved = []
        for operation in fixture.operations:
            previous = state
            if operation.kind == "observe":
                if operation.observation is None:
                    raise AssertionError("validated B3 behavior observation is missing")
                observations.append(operation.observation)
                state = update_belief_state(state, operation.observation)
            else:
                if operation.question is None:
                    raise AssertionError("validated B3 behavior question is missing")
                unresolved.append(operation.question)
                state = record_unresolved_question(state, operation.question)
            if state != previous:
                store.append(state)
        baseline = decide_from_latest_observation(observations, unresolved)
        belief = decide_from_beliefs(state)
        results.append(
            B3BehaviorCaseResult(
                fixture_id=fixture.id,
                final_state_sha256=belief_state_sha256(state),
                baseline=baseline,
                belief=belief,
                state_passed=_state_matches(state, fixture),
                baseline_passed=_passes(baseline, fixture),
                belief_passed=_passes(belief, fixture),
                safety_passed=all(
                    value not in belief.answer for value in fixture.forbidden_answer_substrings
                ),
            )
        )
    return results


def _build_report(
    suite: B3BehaviorSuite,
    fixture: Path,
    database: Path,
    cases: list[B3BehaviorCaseResult],
    started_at: datetime,
    freeze_manifest: Path | None,
) -> B3BehaviorReport:
    total = len(cases)
    baseline = sum(case.baseline_passed for case in cases)
    belief = sum(case.belief_passed for case in cases)
    state = sum(case.state_passed for case in cases)
    safety = sum(case.safety_passed for case in cases)
    delta = belief / total - baseline / total
    gates = (delta >= suite.minimum_quality_delta, state == total, safety == total)
    return B3BehaviorReport(
        purpose=suite.purpose,
        promotion_eligible=suite.promotion_eligible,
        report_id=uuid4(),
        started_at=started_at,
        finished_at=datetime.now(UTC),
        fixture_file=str(fixture),
        fixture_sha256=sha256_file(fixture),
        freeze_manifest_file=str(freeze_manifest) if freeze_manifest else None,
        freeze_manifest_sha256=sha256_file(freeze_manifest) if freeze_manifest else None,
        database_file=str(database),
        database_sha256=sha256_file(database),
        cases=cases,
        baseline_passed_cases=baseline,
        belief_passed_cases=belief,
        state_passed_cases=state,
        safety_passed_cases=safety,
        quality_delta=delta,
        minimum_quality_delta=suite.minimum_quality_delta,
        quality_gate_passed=gates[0],
        state_gate_passed=gates[1],
        safety_gate_passed=gates[2],
        engineering_gate_passed=all(gates),
    )


def evaluate_b3_behavior(
    fixture: Path,
    database: Path,
    report: Path,
    freeze_manifest: Path | None = None,
) -> B3BehaviorReport:
    if database.exists() or report.exists():
        raise FileExistsError("refusing to overwrite B3 behavior artifacts")
    suite = load_b3_behavior_suite(fixture)
    if suite.purpose == "held_out":
        if freeze_manifest is None:
            raise ValueError("held-out B3 behavior evaluation requires a freeze manifest")
        manifest = verify_b3_freeze(freeze_manifest)
        files = manifest["files"]
        if not isinstance(files, dict) or files.get(str(fixture)) != sha256_file(fixture):
            raise ValueError("B3 held-out fixture is not bound by its freeze manifest")
    elif freeze_manifest is not None:
        raise ValueError("B3 development evaluation must not use a promotion freeze")
    started_at = datetime.now(UTC)
    cases = _run_cases(suite, BeliefStateStore(database))
    result = _build_report(suite, fixture, database, cases, started_at, freeze_manifest)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return result


def load_b3_behavior_report(path: Path) -> B3BehaviorReport:
    return B3BehaviorReport.model_validate_json(path.read_text(encoding="utf-8"))


def verify_b3_behavior_report(report: B3BehaviorReport) -> None:
    fixture = Path(report.fixture_file)
    database = Path(report.database_file)
    if (
        sha256_file(fixture) != report.fixture_sha256
        or sha256_file(database) != report.database_sha256
    ):
        raise ValueError("B3 behavior artifact hash mismatch")
    freeze = Path(report.freeze_manifest_file) if report.freeze_manifest_file else None
    if freeze is not None:
        if sha256_file(freeze) != report.freeze_manifest_sha256:
            raise ValueError("B3 behavior freeze hash mismatch")
        verify_b3_freeze(freeze)
    BeliefStateStore(database).verify_integrity()
    suite = load_b3_behavior_suite(fixture)
    with TemporaryDirectory() as directory:
        temp_db = Path(directory) / "beliefs.db"
        rebuilt = _build_report(
            suite,
            fixture,
            temp_db,
            _run_cases(suite, BeliefStateStore(temp_db)),
            report.started_at,
            freeze,
        )
    if rebuilt.cases != report.cases:
        raise ValueError("B3 behavior cases do not match deterministic replay")


def _agreement(processes: list[B3BehaviorProcess]) -> bool:
    first = processes[0].report.cases
    return all(process.report.cases == first for process in processes[1:])


def run_b3_independent(
    fixture: Path,
    freeze_manifest: Path,
    output_dir: Path,
    process_count: int = 3,
) -> B3IndependentReport:
    if process_count < 3:
        raise ValueError("B3 requires at least three independent processes")
    verify_b3_freeze(freeze_manifest)
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite B3 independent artifacts")
    output_dir.mkdir(parents=True)
    started_at = datetime.now(UTC)
    processes: list[B3BehaviorProcess] = []
    for index in range(1, process_count + 1):
        process_dir = output_dir / f"process-{index}"
        process_dir.mkdir()
        database = process_dir / "beliefs.db"
        report_file = process_dir / "report.json"
        command = [
            sys.executable,
            "-m",
            "aif_qwen_agent",
            "eval-b3-behavior",
            "--fixtures",
            str(fixture),
            "--database",
            str(database),
            "--report",
            str(report_file),
            "--freeze-manifest",
            str(freeze_manifest),
        ]
        process = subprocess.Popen(command)
        process_id = process.pid
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"B3 process {index} failed with exit code {return_code}")
        child_report = load_b3_behavior_report(report_file)
        processes.append(
            B3BehaviorProcess(
                process_index=index,
                process_id=process_id,
                report_file=str(report_file),
                report_sha256=sha256_file(report_file),
                database_file=str(database),
                database_sha256=sha256_file(database),
                report=child_report,
            )
        )
    total = sum(len(process.report.cases) for process in processes)
    baseline = sum(process.report.baseline_passed_cases for process in processes)
    belief = sum(process.report.belief_passed_cases for process in processes)
    state = sum(process.report.state_passed_cases for process in processes)
    safety = sum(process.report.safety_passed_cases for process in processes)
    delta = belief / total - baseline / total
    minimum = processes[0].report.minimum_quality_delta
    gates = (delta >= minimum, state == total, safety == total, _agreement(processes))
    result = B3IndependentReport(
        report_id=uuid4(),
        started_at=started_at,
        finished_at=datetime.now(UTC),
        fixture_file=str(fixture),
        fixture_sha256=sha256_file(fixture),
        freeze_manifest_file=str(freeze_manifest),
        freeze_manifest_sha256=sha256_file(freeze_manifest),
        process_count=len(processes),
        processes=processes,
        baseline_passed_runs=baseline,
        belief_passed_runs=belief,
        state_passed_runs=state,
        safety_passed_runs=safety,
        total_runs=total,
        quality_delta=delta,
        minimum_quality_delta=minimum,
        quality_gate_passed=gates[0],
        state_gate_passed=gates[1],
        safety_gate_passed=gates[2],
        reproducibility_gate_passed=gates[3],
        promotion_gate_passed=all(gates),
    )
    (output_dir / "report.json").write_text(result.model_dump_json(indent=2) + "\n")
    return result


def load_b3_independent_report(path: Path) -> B3IndependentReport:
    return B3IndependentReport.model_validate_json(path.read_text(encoding="utf-8"))


def verify_b3_independent_report(report: B3IndependentReport) -> None:
    distinct_processes = len({process.process_id for process in report.processes})
    if report.process_count != len(report.processes) or distinct_processes < 3:
        raise ValueError("B3 independent processes are not distinct")
    for process in report.processes:
        if (
            sha256_file(Path(process.report_file)) != process.report_sha256
            or sha256_file(Path(process.database_file)) != process.database_sha256
        ):
            raise ValueError("B3 independent process artifact hash mismatch")
        verify_b3_behavior_report(process.report)
    total = sum(len(process.report.cases) for process in report.processes)
    baseline = sum(process.report.baseline_passed_cases for process in report.processes)
    belief = sum(process.report.belief_passed_cases for process in report.processes)
    state = sum(process.report.state_passed_cases for process in report.processes)
    safety = sum(process.report.safety_passed_cases for process in report.processes)
    delta = belief / total - baseline / total
    gates = (
        delta >= report.minimum_quality_delta,
        state == total,
        safety == total,
        _agreement(report.processes),
    )
    if (
        (report.total_runs, report.baseline_passed_runs, report.belief_passed_runs)
        != (total, baseline, belief)
        or (report.state_passed_runs, report.safety_passed_runs) != (state, safety)
        or abs(report.quality_delta - delta) > 1e-12
        or (
            report.quality_gate_passed,
            report.state_gate_passed,
            report.safety_gate_passed,
            report.reproducibility_gate_passed,
        )
        != gates
        or report.promotion_gate_passed != all(gates)
    ):
        raise ValueError("B3 independent promotion gates do not match evidence")

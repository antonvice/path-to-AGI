"""Deterministic B3 development evaluation and offline verification."""

from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator

from aif_qwen_agent.artifacts import sha256_file, sha256_text
from aif_qwen_agent.belief import (
    BeliefObservation,
    BeliefStateStore,
    ExplicitBeliefState,
    belief_state_sha256,
    record_unresolved_question,
    update_belief_state,
)
from aif_qwen_agent.config import load_yaml
from aif_qwen_agent.context import render_belief_context


class B3Operation(BaseModel):
    kind: Literal["observe", "unresolved"]
    observation: BeliefObservation | None = None
    question: str | None = None

    @model_validator(mode="after")
    def exactly_one_payload(self) -> "B3Operation":
        if self.kind == "observe" and (self.observation is None or self.question is not None):
            raise ValueError("B3 observe operation requires only an observation")
        if self.kind == "unresolved" and (self.question is None or self.observation is not None):
            raise ValueError("B3 unresolved operation requires only a question")
        return self


class B3ExpectedHypothesis(BaseModel):
    id: str
    statement: str
    probability: float = Field(ge=0.0, le=1.0)
    status: Literal["open", "supported", "contradicted", "refuted"]
    contradicts: list[str] = Field(default_factory=list)
    evidence_count: int = Field(ge=0)


class B3Fixture(BaseModel):
    id: str
    objective: str
    operations: list[B3Operation] = Field(min_length=1)
    expected_revision: int = Field(ge=0)
    expected_hypotheses: list[B3ExpectedHypothesis] = Field(default_factory=list)
    expected_unresolved_questions: list[str] = Field(default_factory=list)
    required_context_substrings: list[str] = Field(default_factory=list)
    forbidden_context_substrings: list[str] = Field(default_factory=list)


class B3CaseResult(BaseModel):
    fixture_id: str
    final_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    revision_hashes: list[str] = Field(min_length=1)
    revision_passed: bool
    hypotheses_passed: bool
    unresolved_passed: bool
    context_passed: bool
    persistence_passed: bool
    passed: bool

    @model_validator(mode="after")
    def gate_matches_checks(self) -> "B3CaseResult":
        expected = all(
            (
                self.revision_passed,
                self.hypotheses_passed,
                self.unresolved_passed,
                self.context_passed,
                self.persistence_passed,
            )
        )
        if self.passed != expected:
            raise ValueError("B3 case gate does not match checks")
        return self


class B3DevelopmentReport(BaseModel):
    schema_version: Literal["1"] = "1"
    milestone: Literal["B3"] = "B3"
    report_type: Literal["development"] = "development"
    promotion_eligible: Literal[False] = False
    report_id: UUID
    started_at: datetime
    finished_at: datetime
    fixture_file: str
    fixture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    database_file: str
    database_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cases: list[B3CaseResult] = Field(min_length=1)
    passed_cases: int = Field(ge=0)
    engineering_gate_passed: bool

    @model_validator(mode="after")
    def report_matches_cases(self) -> "B3DevelopmentReport":
        if self.finished_at < self.started_at:
            raise ValueError("B3 report finishes before it starts")
        passed = sum(case.passed for case in self.cases)
        gate = passed == len(self.cases)
        if self.passed_cases != passed or self.engineering_gate_passed != gate:
            raise ValueError("B3 report gate does not match cases")
        return self


def load_b3_suite(path: Path) -> list[B3Fixture]:
    document = load_yaml(path)
    if (
        document.get("version") != 1
        or document.get("milestone") != "B3"
        or document.get("purpose") != "development"
        or document.get("promotion_eligible") is not False
    ):
        raise ValueError("B3 development suite metadata is invalid")
    fixtures = [B3Fixture.model_validate(value) for value in document.get("cases", [])]
    if len(fixtures) < 4:
        raise ValueError("B3 development suite requires at least four cases")
    if len({fixture.id for fixture in fixtures}) != len(fixtures):
        raise ValueError("B3 fixture IDs must be unique")
    if len({fixture.objective for fixture in fixtures}) != len(fixtures):
        raise ValueError("B3 fixture objectives must be unique")
    return fixtures


def _hypotheses_pass(state: ExplicitBeliefState, fixture: B3Fixture) -> bool:
    actual = {
        hypothesis.id: (
            hypothesis.statement,
            hypothesis.probability,
            hypothesis.status,
            hypothesis.contradicts,
            len(hypothesis.evidence),
        )
        for hypothesis in state.hypotheses
    }
    expected = {
        hypothesis.id: (
            hypothesis.statement,
            hypothesis.probability,
            hypothesis.status,
            hypothesis.contradicts,
            hypothesis.evidence_count,
        )
        for hypothesis in fixture.expected_hypotheses
    }
    return actual == expected


def _run_cases(fixtures: list[B3Fixture], store: BeliefStateStore) -> list[B3CaseResult]:
    results: list[B3CaseResult] = []
    for fixture in fixtures:
        state = ExplicitBeliefState(objective=fixture.objective)
        store.append(state)
        revision_hashes = [belief_state_sha256(state)]
        for operation in fixture.operations:
            previous = state
            if operation.kind == "observe":
                if operation.observation is None:
                    raise AssertionError("validated B3 observation is missing")
                state = update_belief_state(state, operation.observation)
            else:
                if operation.question is None:
                    raise AssertionError("validated B3 question is missing")
                state = record_unresolved_question(state, operation.question)
            if state != previous:
                store.append(state)
                revision_hashes.append(belief_state_sha256(state))
        context = render_belief_context(state)
        checks = (
            state.revision == fixture.expected_revision,
            _hypotheses_pass(state, fixture),
            state.unresolved_questions == fixture.expected_unresolved_questions,
            all(value in context for value in fixture.required_context_substrings)
            and all(value not in context for value in fixture.forbidden_context_substrings),
            store.latest(fixture.objective) == state
            and len(store.history(fixture.objective)) == state.revision + 1,
        )
        results.append(
            B3CaseResult(
                fixture_id=fixture.id,
                final_state_sha256=belief_state_sha256(state),
                context_sha256=sha256_text(context),
                revision_hashes=revision_hashes,
                revision_passed=checks[0],
                hypotheses_passed=checks[1],
                unresolved_passed=checks[2],
                context_passed=checks[3],
                persistence_passed=checks[4],
                passed=all(checks),
            )
        )
    return results


def evaluate_b3(fixture: Path, database: Path, report: Path) -> B3DevelopmentReport:
    if database.exists() or report.exists():
        raise FileExistsError("refusing to overwrite B3 development artifacts")
    fixtures = load_b3_suite(fixture)
    started_at = datetime.now(UTC)
    cases = _run_cases(fixtures, BeliefStateStore(database))
    result = B3DevelopmentReport(
        report_id=uuid4(),
        started_at=started_at,
        finished_at=datetime.now(UTC),
        fixture_file=str(fixture),
        fixture_sha256=sha256_file(fixture),
        database_file=str(database),
        database_sha256=sha256_file(database),
        cases=cases,
        passed_cases=sum(case.passed for case in cases),
        engineering_gate_passed=all(case.passed for case in cases),
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return result


def load_b3_report(path: Path) -> B3DevelopmentReport:
    return B3DevelopmentReport.model_validate_json(path.read_text(encoding="utf-8"))


def verify_b3_report(report: B3DevelopmentReport) -> None:
    fixture = Path(report.fixture_file)
    database = Path(report.database_file)
    if sha256_file(fixture) != report.fixture_sha256:
        raise ValueError("B3 fixture hash mismatch")
    if sha256_file(database) != report.database_sha256:
        raise ValueError("B3 database hash mismatch")
    store = BeliefStateStore(database)
    store.verify_integrity()
    fixtures = load_b3_suite(fixture)
    with TemporaryDirectory() as directory:
        rebuilt_cases = _run_cases(
            fixtures,
            BeliefStateStore(Path(directory) / "beliefs.db"),
        )
    if rebuilt_cases != report.cases:
        raise ValueError("B3 report cases do not match deterministic replay")

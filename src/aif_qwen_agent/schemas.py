from enum import StrEnum
from pathlib import Path
from statistics import median
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, Field, model_validator

UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]


class Task(BaseModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class GenerationConfig(BaseModel):
    max_new_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0.0)
    seed: int
    enable_thinking: bool = False


class ModelIdentity(BaseModel):
    repo_id: str
    revision: str = Field(min_length=40, max_length=40)
    local_path: Path
    backend: str


class ModelResult(BaseModel):
    raw_text: str
    text: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    load_seconds: float = Field(ge=0.0)
    generation_seconds: float = Field(ge=0.0)
    device: str
    stop_reason: Literal["eos", "max_tokens", "unknown"]


class RunTrace(BaseModel):
    schema_version: Literal["1"] = "1"
    run_id: UUID
    started_at: AwareDatetime
    finished_at: AwareDatetime
    task: Task
    rendered_prompt: str
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: ModelIdentity
    generation: GenerationConfig
    status: Literal["completed", "failed"]
    result: ModelResult | None = None
    error: str | None = None

    @model_validator(mode="after")
    def consistent_terminal_state(self) -> "RunTrace":
        if self.finished_at < self.started_at:
            raise ValueError("finished_at cannot precede started_at")
        if self.status == "completed" and self.result is None:
            raise ValueError("completed traces require a result")
        if self.status == "failed" and self.error is None:
            raise ValueError("failed traces require an error")
        return self


class BaselineFixture(BaseModel):
    id: str
    task: Task
    grader: Literal["exact", "contains"]
    expected: str


class BaselineCaseResult(BaseModel):
    fixture_id: str
    run_id: UUID
    status: Literal["completed", "failed"]
    grader: Literal["exact", "contains"]
    expected: str
    actual: str | None
    passed: bool
    error: str | None
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    load_seconds: float = Field(ge=0.0)
    generation_seconds: float = Field(ge=0.0)


class BaselineReport(BaseModel):
    schema_version: Literal["1"] = "1"
    report_type: Literal["suite"] = "suite"
    report_id: UUID
    baseline: Literal["B0"] = "B0"
    started_at: AwareDatetime
    finished_at: AwareDatetime
    fixture_file: str
    fixture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: ModelIdentity
    generation: GenerationConfig
    cases: list[BaselineCaseResult] = Field(min_length=1)
    total_cases: int = Field(gt=0)
    passed_cases: int = Field(ge=0)
    failed_cases: int = Field(ge=0)
    pass_rate: UnitInterval
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    model_load_seconds: float = Field(ge=0.0)
    generation_seconds: float = Field(ge=0.0)

    @model_validator(mode="after")
    def aggregates_match_cases(self) -> "BaselineReport":
        passed = sum(case.passed for case in self.cases)
        if self.total_cases != len(self.cases):
            raise ValueError("total_cases does not match cases")
        if self.passed_cases != passed or self.failed_cases != self.total_cases - passed:
            raise ValueError("pass/fail aggregates do not match cases")
        if self.input_tokens != sum(case.input_tokens for case in self.cases):
            raise ValueError("input token total does not match cases")
        if self.output_tokens != sum(case.output_tokens for case in self.cases):
            raise ValueError("output token total does not match cases")
        if abs(self.model_load_seconds - sum(case.load_seconds for case in self.cases)) > 1e-12:
            raise ValueError("model load time does not match cases")
        if (
            abs(self.generation_seconds - sum(case.generation_seconds for case in self.cases))
            > 1e-12
        ):
            raise ValueError("generation time does not match cases")
        expected_rate = passed / self.total_cases
        if abs(self.pass_rate - expected_rate) > 1e-12:
            raise ValueError("pass rate does not match cases")
        return self


class SystemMemorySnapshot(BaseModel):
    captured_at: AwareDatetime
    total_bytes: int = Field(gt=0)
    available_bytes: int = Field(ge=0)
    used_fraction: UnitInterval
    swap_available: bool
    swap_total_bytes: int = Field(ge=0)
    swap_used_bytes: int = Field(ge=0)


class BaselineCaseComparison(BaseModel):
    fixture_id: str
    run_ids: list[UUID] = Field(min_length=2)
    outputs: list[str | None] = Field(min_length=2)
    prompt_sha256s: list[str] = Field(min_length=2)
    input_tokens: list[int] = Field(min_length=2)
    output_tokens: list[int] = Field(min_length=2)
    stop_reasons: list[Literal["eos", "max_tokens", "unknown"] | None] = Field(min_length=2)
    generation_seconds: list[float] = Field(min_length=2)
    output_agreement: bool
    prompt_agreement: bool
    input_token_agreement: bool
    output_token_agreement: bool
    stop_reason_agreement: bool
    latency_min_seconds: float = Field(ge=0.0)
    latency_median_seconds: float = Field(ge=0.0)
    latency_max_seconds: float = Field(ge=0.0)

    @model_validator(mode="after")
    def repetition_vectors_have_equal_length(self) -> "BaselineCaseComparison":
        lengths = {
            len(self.run_ids),
            len(self.outputs),
            len(self.prompt_sha256s),
            len(self.input_tokens),
            len(self.output_tokens),
            len(self.stop_reasons),
            len(self.generation_seconds),
        }
        if len(lengths) != 1:
            raise ValueError("comparison vectors must have equal length")
        return self


class BaselineReproducibilityReport(BaseModel):
    schema_version: Literal["1"] = "1"
    report_type: Literal["reproducibility"] = "reproducibility"
    report_id: UUID
    baseline: Literal["B0"] = "B0"
    started_at: AwareDatetime
    finished_at: AwareDatetime
    fixture_file: str
    fixture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: ModelIdentity
    generation: GenerationConfig
    repeats: int = Field(ge=2)
    suites: list[BaselineReport] = Field(min_length=2)
    cases: list[BaselineCaseComparison] = Field(min_length=1)
    total_runs: int = Field(gt=0)
    completed_runs: int = Field(ge=0)
    passed_runs: int = Field(ge=0)
    output_agreement_rate: UnitInterval
    gate_passed: bool
    model_load_seconds: float = Field(ge=0.0)
    first_generation_seconds: float = Field(ge=0.0)
    warm_generation_median_seconds: float = Field(ge=0.0)
    generation_min_seconds: float = Field(ge=0.0)
    generation_median_seconds: float = Field(ge=0.0)
    generation_max_seconds: float = Field(ge=0.0)
    memory_before: SystemMemorySnapshot
    memory_after: SystemMemorySnapshot

    @model_validator(mode="after")
    def aggregates_match_suites(self) -> "BaselineReproducibilityReport":
        if self.repeats != len(self.suites):
            raise ValueError("repeat count does not match suites")
        total_runs = sum(suite.total_cases for suite in self.suites)
        completed = sum(case.status == "completed" for suite in self.suites for case in suite.cases)
        passed = sum(case.passed for suite in self.suites for case in suite.cases)
        if self.total_runs != total_runs:
            raise ValueError("total_runs does not match suites")
        if self.completed_runs != completed or self.passed_runs != passed:
            raise ValueError("completed/passed totals do not match suites")
        agreement_rate = sum(case.output_agreement for case in self.cases) / len(self.cases)
        if abs(self.output_agreement_rate - agreement_rate) > 1e-12:
            raise ValueError("output agreement rate does not match cases")
        if any(len(case.run_ids) != self.repeats for case in self.cases):
            raise ValueError("case comparison length does not match repeats")
        suite_fixture_ids = [case.fixture_id for case in self.suites[0].cases]
        if [case.fixture_id for case in self.cases] != suite_fixture_ids:
            raise ValueError("comparison cases do not match suite cases")
        latencies = [case.generation_seconds for suite in self.suites for case in suite.cases]
        model_load = sum(suite.model_load_seconds for suite in self.suites)
        if abs(self.model_load_seconds - model_load) > 1e-12:
            raise ValueError("model load time does not match suites")
        expected_latencies = (
            latencies[0],
            median(latencies[1:]),
            min(latencies),
            median(latencies),
            max(latencies),
        )
        actual_latencies = (
            self.first_generation_seconds,
            self.warm_generation_median_seconds,
            self.generation_min_seconds,
            self.generation_median_seconds,
            self.generation_max_seconds,
        )
        if any(
            abs(actual - expected) > 1e-12
            for actual, expected in zip(actual_latencies, expected_latencies, strict=True)
        ):
            raise ValueError("latency aggregates do not match suites")
        agreement_gate = all(
            case.output_agreement
            and case.prompt_agreement
            and case.input_token_agreement
            and case.output_token_agreement
            and case.stop_reason_agreement
            for case in self.cases
        )
        if self.gate_passed != (passed == total_runs and agreement_gate):
            raise ValueError("gate result does not match runs and comparisons")
        return self


class EvidenceRef(BaseModel):
    artifact_id: str
    excerpt_hash: str | None = None
    reliability: UnitInterval


class Hypothesis(BaseModel):
    id: str
    statement: str
    probability: UnitInterval
    status: Literal["open", "supported", "contradicted", "refuted"] = "open"
    evidence: list[EvidenceRef] = Field(default_factory=list)


class BeliefState(BaseModel):
    objective: str
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    known_constraints: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    remaining_budget: dict[str, float] = Field(default_factory=dict)


class TruthBounds(BaseModel):
    lower: UnitInterval
    upper: UnitInterval

    @model_validator(mode="after")
    def ordered(self) -> "TruthBounds":
        if self.lower > self.upper:
            raise ValueError("lower truth bound cannot exceed upper truth bound")
        return self


class LogicalFact(BaseModel):
    predicate: str = Field(min_length=1)
    bounds: TruthBounds
    negated: bool = False
    provenance: list[str] = Field(default_factory=list)


class LogicalContradiction(BaseModel):
    predicate: str
    positive: LogicalFact
    negative: LogicalFact


class ActionKind(StrEnum):
    ANSWER = "answer"
    ASK_USER = "ask_user"
    RETRIEVE_MEMORY = "retrieve_memory"
    SEARCH_WEB = "search_web"
    READ_FILE = "read_file"
    SEARCH_FILES = "search_files"
    RUN_PYTHON = "run_python"
    RUN_TESTS = "run_tests"
    QUERY_GRAPH = "query_graph"
    WRITE_MEMORY = "write_memory"
    VERIFY = "verify"
    STOP = "stop"


class ActionCandidate(BaseModel):
    id: str
    kind: ActionKind
    arguments: dict[str, Any] = Field(default_factory=dict)
    advances: list[str] = Field(default_factory=list)
    predicted_observations: list[str] = Field(default_factory=list)
    permission_level: str = "read"
    failure_recovery: str | None = None


class PredictedOutcome(BaseModel):
    success_probability: UnitInterval
    expected_goal_progress: UnitInterval
    expected_information_gain: UnitInterval
    ambiguity: UnitInterval
    token_cost: UnitInterval
    wall_time_cost: UnitInterval
    operational_risk: UnitInterval

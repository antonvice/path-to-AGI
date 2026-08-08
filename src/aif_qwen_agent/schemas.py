from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from statistics import median
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]
ToolPhase = Literal["authorization", "execution", "verification"]


class ToolErrorCode(StrEnum):
    OUTSIDE_ALLOWED_ROOT = "outside_allowed_root"
    SYMLINK_ESCAPE = "symlink_escape"
    REQUEST_LIMIT_EXCEEDS_POLICY = "request_limit_exceeds_policy"
    NOT_FOUND = "not_found"
    NOT_FILE = "not_file"
    FILE_TOO_LARGE = "file_too_large"
    INVALID_ENCODING = "invalid_encoding"
    PERMISSION_DENIED = "permission_denied"
    IO_ERROR = "io_error"
    VERIFICATION_FAILED = "verification_failed"


class ReadFilePolicy(BaseModel):
    allowed_roots: list[Path] = Field(min_length=1)
    max_read_bytes: int = Field(gt=0)


class ReadFileRequest(BaseModel):
    path: str = Field(min_length=1)
    max_bytes: int = Field(default=131_072, gt=0)


class ReadFileObservation(BaseModel):
    resolved_path: Path
    content: str
    byte_count: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    encoding: Literal["utf-8"] = "utf-8"


class ToolRejection(BaseModel):
    code: ToolErrorCode
    phase: ToolPhase
    message: str


class ReadFileTrace(BaseModel):
    schema_version: Literal["1"] = "1"
    trace_id: UUID
    tool: Literal["read_file"] = "read_file"
    started_at: AwareDatetime
    finished_at: AwareDatetime
    request: ReadFileRequest
    allowed_roots: list[Path] = Field(min_length=1)
    status: Literal["completed", "rejected"]
    authorized: bool
    executed: bool
    verified: bool
    observation: ReadFileObservation | None = None
    rejection: ToolRejection | None = None

    @model_validator(mode="after")
    def consistent_tool_state(self) -> "ReadFileTrace":
        if self.finished_at < self.started_at:
            raise ValueError("finished_at cannot precede started_at")
        if self.status == "completed":
            if not (self.authorized and self.executed and self.verified):
                raise ValueError("completed tool traces require all phases")
            if self.observation is None or self.rejection is not None:
                raise ValueError("completed tool traces require only an observation")
            payload = self.observation.content.encode(self.observation.encoding)
            if len(payload) != self.observation.byte_count:
                raise ValueError("observation byte count does not match content")
            if sha256(payload).hexdigest() != self.observation.sha256:
                raise ValueError("observation hash does not match content")
        elif self.observation is not None or self.rejection is None:
            raise ValueError("rejected tool traces require only a rejection")
        elif self.rejection.phase == "authorization" and self.authorized:
            raise ValueError("authorization rejection cannot be authorized")
        elif self.rejection.phase == "execution" and (not self.authorized or self.executed):
            raise ValueError("execution rejection requires authorization without execution")
        elif self.rejection.phase == "verification" and (
            not self.authorized or not self.executed or self.verified
        ):
            raise ValueError("verification rejection requires execution without verification")
        return self


class ReadFileAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["read_file"]
    path: str = Field(min_length=1)
    max_bytes: int = Field(default=16_384, gt=0)


class AnswerAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["answer"]
    answer: str = Field(min_length=1)


class StopAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["stop"]
    reason: str = Field(min_length=1)


AgentAction = Annotated[ReadFileAction | AnswerAction | StopAction, Field(discriminator="kind")]


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


class ProposalAttempt(BaseModel):
    attempt: int = Field(gt=0)
    rendered_prompt: str
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result: ModelResult | None = None
    action: AgentAction | None = None
    error: str | None = None

    @model_validator(mode="after")
    def has_action_or_error(self) -> "ProposalAttempt":
        if (self.action is None) == (self.error is None):
            raise ValueError("proposal attempt requires exactly one action or error")
        if self.action is not None and self.result is None:
            raise ValueError("parsed proposal requires a model result")
        return self


class AgentTrace(BaseModel):
    schema_version: Literal["1"] = "1"
    run_id: UUID
    started_at: AwareDatetime
    finished_at: AwareDatetime
    task: Task
    model: ModelIdentity
    generation: GenerationConfig
    proposal_generation: GenerationConfig | None = None
    prompt_profile: Literal["legacy", "compact"] | None = None
    proposal_attempts: list[ProposalAttempt] = Field(min_length=1)
    selected_action: AgentAction | None = None
    tool_trace: ReadFileTrace | None = None
    answer_result: ModelResult | None = None
    evidence_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    answer: str | None = None
    status: Literal["completed", "stopped", "rejected", "failed"]
    error: str | None = None
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    model_load_seconds: float = Field(ge=0.0)
    generation_seconds: float = Field(ge=0.0)

    @model_validator(mode="after")
    def consistent_agent_state(self) -> "AgentTrace":
        if self.finished_at < self.started_at:
            raise ValueError("finished_at cannot precede started_at")
        results = [
            attempt.result for attempt in self.proposal_attempts if attempt.result is not None
        ]
        if self.answer_result is not None:
            results.append(self.answer_result)
        totals = (
            sum(result.input_tokens for result in results),
            sum(result.output_tokens for result in results),
            sum(result.load_seconds for result in results),
            sum(result.generation_seconds for result in results),
        )
        recorded = (
            self.input_tokens,
            self.output_tokens,
            self.model_load_seconds,
            self.generation_seconds,
        )
        if any(
            abs(actual - expected) > 1e-12
            for actual, expected in zip(recorded, totals, strict=True)
        ):
            raise ValueError("agent model aggregates do not match model results")
        if self.status == "completed":
            if self.selected_action is None or self.answer is None or self.error is not None:
                raise ValueError("completed agent traces require an action and answer")
            if isinstance(self.selected_action, ReadFileAction):
                if (
                    self.tool_trace is None
                    or self.tool_trace.observation is None
                    or self.evidence_sha256 != self.tool_trace.observation.sha256
                    or self.answer_result is None
                    or f"[evidence sha256:{self.evidence_sha256}]" not in self.answer
                ):
                    raise ValueError("read-file answers require verified cited evidence")
            elif self.tool_trace is not None or self.evidence_sha256 is not None:
                raise ValueError("direct answers cannot claim tool evidence")
        elif self.status == "stopped":
            if not isinstance(self.selected_action, StopAction) or self.error is not None:
                raise ValueError("stopped traces require a stop action")
        elif self.status == "rejected":
            if (
                not isinstance(self.selected_action, ReadFileAction)
                or self.tool_trace is None
                or self.tool_trace.status != "rejected"
                or self.error is None
            ):
                raise ValueError("rejected traces require a rejected read-file trace")
        elif self.error is None:
            raise ValueError("failed agent traces require an error")
        return self


class AgentComparisonCase(BaseModel):
    fixture_id: str = Field(min_length=1)
    grader: Literal["exact", "contains"]
    expected: str = Field(min_length=1)
    baseline_run_id: UUID
    agent_run_id: UUID
    baseline_actual: str
    agent_actual: str
    baseline_passed: bool
    agent_passed: bool
    tool_verified: bool
    safety_violation: bool = False
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def grades_match_outputs(self) -> "AgentComparisonCase":
        def grade(actual: str) -> bool:
            return actual == self.expected if self.grader == "exact" else self.expected in actual

        if self.baseline_passed != grade(self.baseline_actual):
            raise ValueError("baseline grade does not match output")
        if self.agent_passed != grade(self.agent_actual):
            raise ValueError("agent grade does not match output")
        return self


class AgentComparisonReport(BaseModel):
    schema_version: Literal["1"] = "1"
    report_type: Literal["b1b_comparison"] = "b1b_comparison"
    report_id: UUID
    created_at: AwareDatetime
    milestone: Literal["B1b"] = "B1b"
    fixture_file: str
    model: ModelIdentity
    cases: list[AgentComparisonCase] = Field(min_length=1)
    total_cases: int = Field(gt=0)
    baseline_passed_cases: int = Field(ge=0)
    agent_passed_cases: int = Field(ge=0)
    safety_violations: int = Field(ge=0)
    gate_passed: bool

    @model_validator(mode="after")
    def aggregates_match_comparison_cases(self) -> "AgentComparisonReport":
        baseline_passed = sum(case.baseline_passed for case in self.cases)
        agent_passed = sum(case.agent_passed for case in self.cases)
        safety_violations = sum(case.safety_violation for case in self.cases)
        if self.total_cases != len(self.cases):
            raise ValueError("comparison total does not match cases")
        if (self.baseline_passed_cases, self.agent_passed_cases, self.safety_violations) != (
            baseline_passed,
            agent_passed,
            safety_violations,
        ):
            raise ValueError("comparison aggregates do not match cases")
        if self.gate_passed != (agent_passed > baseline_passed and safety_violations == 0):
            raise ValueError("comparison gate does not match outcomes")
        return self


class B1Fixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    kind: Literal["grounded", "safety"]
    task: Task
    grader: Literal["exact", "contains"] | None = None
    expected: str | None = None
    evidence_path: str | None = None
    safety_expectation: (
        Literal["forbidden_action", "outside_allowed_root", "not_found", "file_too_large"] | None
    ) = None
    safety_path: str | None = None
    safety_max_bytes: int | None = Field(default=None, gt=0)
    adversarial: bool = False
    forbidden_substrings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def fields_match_fixture_kind(self) -> "B1Fixture":
        grounded = (self.grader, self.expected, self.evidence_path)
        if self.kind == "grounded" and (
            None in grounded
            or self.safety_expectation is not None
            or self.safety_path is not None
            or self.safety_max_bytes is not None
        ):
            raise ValueError("grounded fixtures require grader, expected, and evidence_path")
        if self.kind == "safety" and (
            self.safety_expectation is None or any(value is not None for value in grounded)
        ):
            raise ValueError("safety fixtures require only a safety expectation")
        if self.safety_expectation == "forbidden_action" and (
            self.safety_path is not None or self.safety_max_bytes is not None
        ):
            raise ValueError("forbidden-action fixtures cannot specify a read request")
        if (
            self.kind == "safety"
            and self.safety_expectation != "forbidden_action"
            and self.safety_path is None
        ):
            raise ValueError("read safety fixtures require an expected path")
        if self.adversarial != bool(self.forbidden_substrings):
            raise ValueError("adversarial fixtures require forbidden substrings")
        if self.adversarial and self.kind != "grounded":
            raise ValueError("only grounded fixtures can contain adversarial evidence")
        return self


class B1CaseResult(BaseModel):
    fixture_id: str = Field(min_length=1)
    kind: Literal["grounded", "safety"]
    baseline_run_id: UUID | None = None
    agent_run_id: UUID
    expected: str | None = None
    baseline_actual: str | None = None
    agent_actual: str | None = None
    baseline_passed: bool | None = None
    agent_passed: bool
    agent_status: Literal["completed", "stopped", "rejected", "failed"]
    proposal_attempts: int = Field(gt=0)
    tool_trace_id: UUID | None = None
    rejection_code: ToolErrorCode | None = None
    tool_verified: bool
    evidence_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    safety_violation: bool
    instruction_following_violation: bool = False
    baseline_input_tokens: int = Field(ge=0)
    baseline_output_tokens: int = Field(ge=0)
    baseline_load_seconds: float = Field(ge=0.0)
    baseline_generation_seconds: float = Field(ge=0.0)
    agent_input_tokens: int = Field(ge=0)
    agent_output_tokens: int = Field(ge=0)
    agent_load_seconds: float = Field(ge=0.0)
    agent_generation_seconds: float = Field(ge=0.0)

    @model_validator(mode="after")
    def baseline_fields_match_fixture_kind(self) -> "B1CaseResult":
        if self.kind == "grounded" and (
            self.baseline_run_id is None or self.baseline_passed is None or self.expected is None
        ):
            raise ValueError("grounded results require baseline comparison fields")
        if self.kind == "safety" and (
            self.baseline_run_id is not None
            or self.baseline_passed is not None
            or self.expected is not None
            or self.baseline_actual is not None
            or any(
                (
                    self.baseline_input_tokens,
                    self.baseline_output_tokens,
                    self.baseline_load_seconds,
                    self.baseline_generation_seconds,
                )
            )
        ):
            raise ValueError("safety results cannot contain baseline comparison fields")
        if self.instruction_following_violation and not self.safety_violation:
            raise ValueError("instruction following must count as a safety violation")
        return self


class B1EvaluationReport(BaseModel):
    schema_version: Literal["1"] = "1"
    report_type: Literal["b1_evaluation"] = "b1_evaluation"
    report_id: UUID
    milestone: Literal["B1c", "B1d"] = "B1c"
    started_at: AwareDatetime
    finished_at: AwareDatetime
    fixture_file: str
    fixture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: ModelIdentity
    generation: GenerationConfig
    cases: list[B1CaseResult] = Field(min_length=1)
    grounded_cases: int = Field(ge=3)
    safety_cases: int = Field(ge=4)
    baseline_passed_cases: int = Field(ge=0)
    agent_passed_cases: int = Field(ge=0)
    safety_passed_cases: int = Field(ge=0)
    safety_violations: int = Field(ge=0)
    proposal_retries: int = Field(ge=0)
    baseline_input_tokens: int = Field(ge=0)
    baseline_output_tokens: int = Field(ge=0)
    agent_input_tokens: int = Field(ge=0)
    agent_output_tokens: int = Field(ge=0)
    model_load_seconds: float = Field(ge=0.0)
    baseline_generation_seconds: float = Field(ge=0.0)
    agent_generation_seconds: float = Field(ge=0.0)
    gate_passed: bool

    @model_validator(mode="after")
    def aggregates_match_b1_cases(self) -> "B1EvaluationReport":
        grounded = [case for case in self.cases if case.kind == "grounded"]
        safety = [case for case in self.cases if case.kind == "safety"]
        expected = (
            len(grounded),
            len(safety),
            sum(case.baseline_passed is True for case in grounded),
            sum(case.agent_passed for case in grounded),
            sum(case.agent_passed for case in safety),
            sum(case.safety_violation for case in self.cases),
            sum(case.proposal_attempts - 1 for case in self.cases),
            sum(case.baseline_input_tokens for case in self.cases),
            sum(case.baseline_output_tokens for case in self.cases),
            sum(case.agent_input_tokens for case in self.cases),
            sum(case.agent_output_tokens for case in self.cases),
        )
        actual = (
            self.grounded_cases,
            self.safety_cases,
            self.baseline_passed_cases,
            self.agent_passed_cases,
            self.safety_passed_cases,
            self.safety_violations,
            self.proposal_retries,
            self.baseline_input_tokens,
            self.baseline_output_tokens,
            self.agent_input_tokens,
            self.agent_output_tokens,
        )
        if actual != expected:
            raise ValueError("B1 evaluation aggregates do not match cases")
        expected_load = sum(
            case.baseline_load_seconds + case.agent_load_seconds for case in self.cases
        )
        expected_baseline_generation = sum(case.baseline_generation_seconds for case in self.cases)
        expected_agent_generation = sum(case.agent_generation_seconds for case in self.cases)
        if any(
            abs(recorded - calculated) > 1e-12
            for recorded, calculated in zip(
                (
                    self.model_load_seconds,
                    self.baseline_generation_seconds,
                    self.agent_generation_seconds,
                ),
                (expected_load, expected_baseline_generation, expected_agent_generation),
                strict=True,
            )
        ):
            raise ValueError("B1 evaluation timing aggregates do not match cases")
        gate = (
            self.agent_passed_cases > self.baseline_passed_cases
            and self.safety_passed_cases == self.safety_cases
            and self.safety_violations == 0
        )
        if self.gate_passed != gate:
            raise ValueError("B1 evaluation gate does not match case outcomes")
        return self


class B1CaseReproducibility(BaseModel):
    fixture_id: str = Field(min_length=1)
    kind: Literal["grounded", "safety"]
    agent_run_ids: list[UUID] = Field(min_length=2)
    baseline_run_ids: list[UUID] = Field(default_factory=list)
    actions: list[Literal["read_file", "answer", "stop", "none"]] = Field(min_length=2)
    outputs: list[str | None] = Field(min_length=2)
    baseline_outputs: list[str | None] = Field(default_factory=list)
    statuses: list[Literal["completed", "stopped", "rejected", "failed"]] = Field(min_length=2)
    input_tokens: list[int] = Field(min_length=2)
    output_tokens: list[int] = Field(min_length=2)
    rejection_codes: list[ToolErrorCode | None] = Field(min_length=2)
    evidence_sha256s: list[str | None] = Field(min_length=2)
    proposal_attempts: list[int] = Field(min_length=2)
    passed: list[bool] = Field(min_length=2)
    action_agreement: bool
    output_agreement: bool
    baseline_output_agreement: bool
    status_agreement: bool
    token_agreement: bool
    rejection_agreement: bool
    evidence_agreement: bool
    retry_agreement: bool
    pass_agreement: bool
    all_agreement: bool

    @model_validator(mode="after")
    def agreement_matches_repetition_vectors(self) -> "B1CaseReproducibility":
        repeated = (
            self.agent_run_ids,
            self.actions,
            self.outputs,
            self.statuses,
            self.input_tokens,
            self.output_tokens,
            self.rejection_codes,
            self.evidence_sha256s,
            self.proposal_attempts,
            self.passed,
        )
        if len({len(values) for values in repeated}) != 1:
            raise ValueError("B1 reproducibility vectors must have equal length")
        if self.kind == "grounded" and len(self.baseline_run_ids) != len(self.agent_run_ids):
            raise ValueError("grounded reproducibility requires baseline run IDs")
        if self.kind == "grounded" and len(self.baseline_outputs) != len(self.agent_run_ids):
            raise ValueError("grounded reproducibility requires baseline outputs")
        if self.kind == "safety" and (self.baseline_run_ids or self.baseline_outputs):
            raise ValueError("safety reproducibility cannot contain baseline vectors")

        def agrees(values: list[Any]) -> bool:
            return all(value == values[0] for value in values[1:])

        expected = (
            agrees(self.actions),
            agrees(self.outputs),
            agrees(self.baseline_outputs) if self.baseline_outputs else True,
            agrees(self.statuses),
            agrees(list(zip(self.input_tokens, self.output_tokens, strict=True))),
            agrees(self.rejection_codes),
            agrees(self.evidence_sha256s),
            agrees(self.proposal_attempts),
            agrees(self.passed),
        )
        actual = (
            self.action_agreement,
            self.output_agreement,
            self.baseline_output_agreement,
            self.status_agreement,
            self.token_agreement,
            self.rejection_agreement,
            self.evidence_agreement,
            self.retry_agreement,
            self.pass_agreement,
        )
        if actual != expected:
            raise ValueError("B1 reproducibility agreement flags do not match vectors")
        if self.all_agreement != all(expected):
            raise ValueError("B1 all-agreement flag does not match checks")
        return self


class SystemMemorySnapshot(BaseModel):
    captured_at: AwareDatetime
    total_bytes: int = Field(gt=0)
    available_bytes: int = Field(ge=0)
    used_fraction: UnitInterval
    swap_available: bool
    swap_total_bytes: int = Field(ge=0)
    swap_used_bytes: int = Field(ge=0)


class B1RepeatedEvaluationReport(BaseModel):
    schema_version: Literal["1"] = "1"
    report_type: Literal["b1_reproducibility"] = "b1_reproducibility"
    report_id: UUID
    milestone: Literal["B1d"] = "B1d"
    started_at: AwareDatetime
    finished_at: AwareDatetime
    fixture_file: str
    fixture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_config_file: str
    evaluation_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: ModelIdentity
    generation: GenerationConfig
    repeats: int = Field(ge=2)
    suites: list[B1EvaluationReport] = Field(min_length=2)
    comparisons: list[B1CaseReproducibility] = Field(min_length=1)
    grounded_runs: int = Field(gt=0)
    safety_runs: int = Field(gt=0)
    baseline_passed_runs: int = Field(ge=0)
    agent_passed_runs: int = Field(ge=0)
    safety_passed_runs: int = Field(ge=0)
    safety_violations: int = Field(ge=0)
    instruction_following_violations: int = Field(ge=0)
    proposal_retries: int = Field(ge=0)
    baseline_input_tokens: int = Field(ge=0)
    baseline_output_tokens: int = Field(ge=0)
    agent_input_tokens: int = Field(ge=0)
    agent_output_tokens: int = Field(ge=0)
    model_load_seconds: float = Field(ge=0.0)
    baseline_generation_seconds: float = Field(ge=0.0)
    agent_generation_seconds: float = Field(ge=0.0)
    first_generation_seconds: float = Field(ge=0.0)
    warm_generation_median_seconds: float = Field(ge=0.0)
    generation_min_seconds: float = Field(ge=0.0)
    generation_median_seconds: float = Field(ge=0.0)
    generation_max_seconds: float = Field(ge=0.0)
    quality_delta: float
    token_cost_increase: float = Field(ge=-1.0)
    generation_cost_increase: float = Field(ge=-1.0)
    minimum_success_delta: UnitInterval
    maximum_cost_increase: UnitInterval
    quality_gate_passed: bool
    safety_gate_passed: bool
    reproducibility_gate_passed: bool
    cost_gate_passed: bool
    gate_passed: bool
    memory_before: SystemMemorySnapshot
    memory_after: SystemMemorySnapshot

    @model_validator(mode="after")
    def repeated_b1_aggregates_match_suites(self) -> "B1RepeatedEvaluationReport":
        if self.finished_at < self.started_at:
            raise ValueError("finished_at cannot precede started_at")
        if self.memory_after.captured_at < self.memory_before.captured_at:
            raise ValueError("B1 memory snapshots are out of order")
        if self.repeats != len(self.suites):
            raise ValueError("B1 repeat count does not match suites")
        if any(suite.milestone != "B1d" for suite in self.suites):
            raise ValueError("B1d reproducibility requires B1d suites")
        if any(
            suite.model != self.model or suite.generation != self.generation
            for suite in self.suites
        ):
            raise ValueError("B1 repeated suite model settings differ")
        expected = (
            sum(suite.grounded_cases for suite in self.suites),
            sum(suite.safety_cases for suite in self.suites),
            sum(suite.baseline_passed_cases for suite in self.suites),
            sum(suite.agent_passed_cases for suite in self.suites),
            sum(suite.safety_passed_cases for suite in self.suites),
            sum(suite.safety_violations for suite in self.suites),
            sum(
                case.instruction_following_violation
                for suite in self.suites
                for case in suite.cases
            ),
            sum(suite.proposal_retries for suite in self.suites),
            sum(suite.baseline_input_tokens for suite in self.suites),
            sum(suite.baseline_output_tokens for suite in self.suites),
            sum(suite.agent_input_tokens for suite in self.suites),
            sum(suite.agent_output_tokens for suite in self.suites),
        )
        actual = (
            self.grounded_runs,
            self.safety_runs,
            self.baseline_passed_runs,
            self.agent_passed_runs,
            self.safety_passed_runs,
            self.safety_violations,
            self.instruction_following_violations,
            self.proposal_retries,
            self.baseline_input_tokens,
            self.baseline_output_tokens,
            self.agent_input_tokens,
            self.agent_output_tokens,
        )
        if actual != expected:
            raise ValueError("B1 repeated aggregates do not match suites")
        if any(len(comparison.agent_run_ids) != self.repeats for comparison in self.comparisons):
            raise ValueError("B1 comparison length does not match repeats")
        if [comparison.fixture_id for comparison in self.comparisons] != [
            case.fixture_id for case in self.suites[0].cases
        ]:
            raise ValueError("B1 comparisons do not match suite cases")
        expected_timings = (
            sum(suite.model_load_seconds for suite in self.suites),
            sum(suite.baseline_generation_seconds for suite in self.suites),
            sum(suite.agent_generation_seconds for suite in self.suites),
        )
        actual_timings = (
            self.model_load_seconds,
            self.baseline_generation_seconds,
            self.agent_generation_seconds,
        )
        if any(
            abs(actual_value - expected_value) > 1e-12
            for actual_value, expected_value in zip(actual_timings, expected_timings, strict=True)
        ):
            raise ValueError("B1 repeated timing totals do not match suites")
        quality_delta = (
            self.agent_passed_runs / self.grounded_runs
            - self.baseline_passed_runs / self.grounded_runs
        )
        baseline_tokens = self.baseline_input_tokens + self.baseline_output_tokens
        agent_tokens = self.agent_input_tokens + self.agent_output_tokens
        if baseline_tokens == 0 or self.baseline_generation_seconds == 0.0:
            raise ValueError("B1 cost comparison requires nonzero B0 cost")
        expected_costs = (
            quality_delta,
            agent_tokens / baseline_tokens - 1.0,
            self.agent_generation_seconds / self.baseline_generation_seconds - 1.0,
        )
        actual_costs = (
            self.quality_delta,
            self.token_cost_increase,
            self.generation_cost_increase,
        )
        if any(
            abs(actual_value - expected_value) > 1e-12
            for actual_value, expected_value in zip(actual_costs, expected_costs, strict=True)
        ):
            raise ValueError("B1 repeated quality or cost deltas do not match suites")
        expected_gates = (
            quality_delta >= self.minimum_success_delta,
            self.safety_passed_runs == self.safety_runs
            and self.safety_violations == 0
            and self.instruction_following_violations == 0,
            all(comparison.all_agreement for comparison in self.comparisons),
            max(self.token_cost_increase, self.generation_cost_increase)
            <= self.maximum_cost_increase,
        )
        actual_gates = (
            self.quality_gate_passed,
            self.safety_gate_passed,
            self.reproducibility_gate_passed,
            self.cost_gate_passed,
        )
        if actual_gates != expected_gates or self.gate_passed != all(expected_gates):
            raise ValueError("B1 repeated gate flags do not match outcomes")
        return self


class AgentStageCost(BaseModel):
    calls: int = Field(ge=0)
    input_tokens: float = Field(ge=0.0)
    output_tokens: float = Field(ge=0.0)
    generation_seconds: float = Field(ge=0.0)


class B1CostReport(BaseModel):
    schema_version: Literal["1"] = "1"
    report_type: Literal["b1_cost"] = "b1_cost"
    report_id: UUID
    milestone: Literal["B1e"] = "B1e"
    created_at: AwareDatetime
    fixture_file: str
    fixture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_config_file: str
    evaluation_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    agent_config_file: str
    agent_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_report_file: str
    reference_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    optimized_report_file: str
    optimized_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: ModelIdentity
    answer_generation: GenerationConfig
    proposal_generation: GenerationConfig
    prompt_profile: Literal["compact"]
    baseline_grounded: AgentStageCost
    legacy_grounded_proposal: AgentStageCost
    legacy_grounded_answer: AgentStageCost
    optimized_grounded_proposal: AgentStageCost
    optimized_grounded_answer: AgentStageCost
    optimized_safety_proposal: AgentStageCost
    legacy_grounded_pass_rate: UnitInterval
    optimized_grounded_pass_rate: UnitInterval
    legacy_safety_pass_rate: UnitInterval
    optimized_safety_pass_rate: UnitInterval
    legacy_instruction_following_violations: int = Field(ge=0)
    optimized_instruction_following_violations: int = Field(ge=0)
    token_reduction: float
    generation_reduction: float
    minimum_cost_reduction: UnitInterval
    grounded_token_cost_increase: float = Field(ge=-1.0)
    grounded_generation_cost_increase: float = Field(ge=-1.0)
    maximum_cost_increase: UnitInterval
    quality_preserved: bool
    safety_preserved: bool
    optimization_gate_passed: bool
    cost_gate_passed: bool
    gate_passed: bool
    optimized_suite: B1EvaluationReport

    @model_validator(mode="after")
    def b1e_costs_and_gates_match_stages(self) -> "B1CostReport":
        def tokens(*stages: AgentStageCost) -> float:
            return sum(stage.input_tokens + stage.output_tokens for stage in stages)

        def generation(*stages: AgentStageCost) -> float:
            return sum(stage.generation_seconds for stage in stages)

        legacy_tokens = tokens(self.legacy_grounded_proposal, self.legacy_grounded_answer)
        optimized_tokens = tokens(self.optimized_grounded_proposal, self.optimized_grounded_answer)
        baseline_tokens = tokens(self.baseline_grounded)
        legacy_generation = generation(self.legacy_grounded_proposal, self.legacy_grounded_answer)
        optimized_generation = generation(
            self.optimized_grounded_proposal, self.optimized_grounded_answer
        )
        baseline_generation = generation(self.baseline_grounded)
        if min(legacy_tokens, baseline_tokens, legacy_generation, baseline_generation) <= 0.0:
            raise ValueError("B1e comparison requires nonzero reference costs")
        expected_costs = (
            1.0 - optimized_tokens / legacy_tokens,
            1.0 - optimized_generation / legacy_generation,
            optimized_tokens / baseline_tokens - 1.0,
            optimized_generation / baseline_generation - 1.0,
        )
        actual_costs = (
            self.token_reduction,
            self.generation_reduction,
            self.grounded_token_cost_increase,
            self.grounded_generation_cost_increase,
        )
        if any(
            abs(actual - expected) > 1e-12
            for actual, expected in zip(actual_costs, expected_costs, strict=True)
        ):
            raise ValueError("B1e cost deltas do not match stage costs")
        expected_gates = (
            self.optimized_grounded_pass_rate >= self.legacy_grounded_pass_rate,
            self.optimized_safety_pass_rate >= self.legacy_safety_pass_rate
            and self.optimized_instruction_following_violations == 0,
            min(self.token_reduction, self.generation_reduction) >= self.minimum_cost_reduction,
            max(
                self.grounded_token_cost_increase,
                self.grounded_generation_cost_increase,
            )
            <= self.maximum_cost_increase,
        )
        actual_gates = (
            self.quality_preserved,
            self.safety_preserved,
            self.optimization_gate_passed,
            self.cost_gate_passed,
        )
        if actual_gates != expected_gates or self.gate_passed != all(expected_gates):
            raise ValueError("B1e gate flags do not match measured outcomes")
        if self.optimized_suite.milestone != "B1d":
            raise ValueError("B1e must use the unchanged B1d fixture suite")
        return self


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

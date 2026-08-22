"""Deterministic non-promotion evaluation for B4 action selection."""

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator

from aif_qwen_agent.aif_selection import (
    ActionSelectionTrace,
    AIFWeights,
    select_active_inference_action,
    selection_trace_sha256,
)
from aif_qwen_agent.artifacts import sha256_file
from aif_qwen_agent.belief import ExplicitBeliefState
from aif_qwen_agent.config import load_yaml
from aif_qwen_agent.policy import HardPolicy, PolicyRule
from aif_qwen_agent.schemas import ActionCandidate, BeliefState, PredictedOutcome


class B4DevelopmentConfig(BaseModel):
    version: Literal[1] = 1
    milestone: Literal["B4"] = "B4"
    purpose: Literal["development"] = "development"
    promotion_eligible: Literal[False] = False
    minimum_epistemic_cases: int = Field(default=2, ge=1)
    weights: AIFWeights = Field(default_factory=AIFWeights)


class B4Fixture(BaseModel):
    id: str = Field(min_length=1)
    state: ExplicitBeliefState
    candidates: list[ActionCandidate] = Field(min_length=2)
    predictions: dict[str, PredictedOutcome]
    denied_action_ids: list[str] = Field(default_factory=list)
    expected_action_id: str
    expected_without_information_gain_id: str
    requires_epistemic_term: bool = False

    @model_validator(mode="after")
    def references_known_actions(self) -> "B4Fixture":
        action_ids = [candidate.id for candidate in self.candidates]
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("B4 candidate IDs must be unique")
        if set(self.predictions) != set(action_ids):
            raise ValueError("B4 predictions must exactly match candidate IDs")
        if not set(self.denied_action_ids) <= set(action_ids):
            raise ValueError("B4 denied action is missing from candidates")
        allowed = set(action_ids) - set(self.denied_action_ids)
        if self.expected_action_id not in allowed:
            raise ValueError("B4 expected action must be allowed")
        if self.expected_without_information_gain_id not in allowed:
            raise ValueError("B4 ablated expected action must be allowed")
        expected_shift = self.expected_action_id != self.expected_without_information_gain_id
        if self.requires_epistemic_term != expected_shift:
            raise ValueError("B4 epistemic requirement does not match expected selections")
        return self


class B4CaseResult(BaseModel):
    fixture_id: str
    trace: ActionSelectionTrace
    state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trace_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_action_id: str
    selected_without_information_gain_id: str
    exact_selection_passed: bool
    ablation_passed: bool
    hard_filter_passed: bool
    telemetry_passed: bool
    epistemic_contribution_passed: bool
    passed: bool

    @model_validator(mode="after")
    def gate_matches_checks(self) -> "B4CaseResult":
        if self.state_sha256 != self.trace.state_sha256:
            raise ValueError("B4 case state hash does not match trace")
        if self.trace_sha256 != selection_trace_sha256(self.trace):
            raise ValueError("B4 case trace hash does not match trace")
        if self.selected_action_id != self.trace.selected_action_id:
            raise ValueError("B4 case selection does not match trace")
        if (
            self.selected_without_information_gain_id
            != self.trace.selected_without_information_gain_id
        ):
            raise ValueError("B4 case ablated selection does not match trace")
        expected = all(
            (
                self.exact_selection_passed,
                self.ablation_passed,
                self.hard_filter_passed,
                self.telemetry_passed,
                self.epistemic_contribution_passed,
            )
        )
        if self.passed != expected:
            raise ValueError("B4 case gate does not match checks")
        return self


class B4DevelopmentReport(BaseModel):
    schema_version: Literal["1"] = "1"
    milestone: Literal["B4"] = "B4"
    report_type: Literal["development"] = "development"
    promotion_eligible: Literal[False] = False
    report_id: UUID
    started_at: datetime
    finished_at: datetime
    fixture_file: str
    fixture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_file: str
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cases: list[B4CaseResult] = Field(min_length=1)
    passed_cases: int = Field(ge=0)
    required_epistemic_cases: int = Field(ge=1)
    passed_epistemic_cases: int = Field(ge=0)
    engineering_gate_passed: bool

    @model_validator(mode="after")
    def report_matches_cases(self) -> "B4DevelopmentReport":
        if self.finished_at < self.started_at:
            raise ValueError("B4 report finishes before it starts")
        passed = sum(case.passed for case in self.cases)
        epistemic = sum(
            case.epistemic_contribution_passed
            and case.selected_action_id != case.selected_without_information_gain_id
            for case in self.cases
        )
        gate = passed == len(self.cases) and epistemic >= self.required_epistemic_cases
        if (
            self.passed_cases != passed
            or self.passed_epistemic_cases != epistemic
            or self.engineering_gate_passed != gate
        ):
            raise ValueError("B4 report gate does not match cases")
        return self


def load_b4_config(path: Path) -> B4DevelopmentConfig:
    return B4DevelopmentConfig.model_validate(load_yaml(path))


def load_b4_suite(path: Path) -> list[B4Fixture]:
    document = load_yaml(path)
    if (
        document.get("version") != 1
        or document.get("milestone") != "B4"
        or document.get("purpose") != "development"
        or document.get("promotion_eligible") is not False
    ):
        raise ValueError("B4 development suite metadata is invalid")
    fixtures = [B4Fixture.model_validate(value) for value in document.get("cases", [])]
    if len(fixtures) < 6:
        raise ValueError("B4 development suite requires at least six cases")
    if len({fixture.id for fixture in fixtures}) != len(fixtures):
        raise ValueError("B4 fixture IDs must be unique")
    if len({fixture.state.objective for fixture in fixtures}) != len(fixtures):
        raise ValueError("B4 fixture objectives must be unique")
    return fixtures


def _policy_for(denied: frozenset[str]) -> HardPolicy:
    def allows(action: ActionCandidate, _state: BeliefState) -> bool:
        return action.id not in denied

    return HardPolicy([PolicyRule("fixture_denied_action", allows)])


def _run_cases(fixtures: Sequence[B4Fixture], config: B4DevelopmentConfig) -> list[B4CaseResult]:
    results: list[B4CaseResult] = []
    for fixture in fixtures:
        policy = _policy_for(frozenset(fixture.denied_action_ids))
        _, trace = select_active_inference_action(
            fixture.candidates,
            fixture.predictions,
            fixture.state,
            policy,
            weights=config.weights,
        )
        checks = _case_checks(fixture, trace)
        results.append(
            B4CaseResult(
                fixture_id=fixture.id,
                trace=trace,
                state_sha256=trace.state_sha256,
                trace_sha256=selection_trace_sha256(trace),
                selected_action_id=trace.selected_action_id,
                selected_without_information_gain_id=trace.selected_without_information_gain_id,
                exact_selection_passed=checks[0],
                ablation_passed=checks[1],
                hard_filter_passed=checks[2],
                telemetry_passed=checks[3],
                epistemic_contribution_passed=checks[4],
                passed=all(checks),
            )
        )
    return results


def _case_checks(
    fixture: B4Fixture, trace: ActionSelectionTrace
) -> tuple[bool, bool, bool, bool, bool]:
    evaluations = {value.action.id: value for value in trace.evaluations}
    denied = [evaluations[action_id] for action_id in fixture.denied_action_ids]
    hard_filter = all(
        not value.eligible
        and value.score is None
        and value.score_without_information_gain is None
        and "policy:fixture_denied_action" in value.rejection_reasons
        for value in denied
    )
    telemetry = all(
        (not value.eligible)
        or (
            value.prediction is not None
            and value.score is not None
            and value.score_without_information_gain is not None
        )
        for value in trace.evaluations
    )
    epistemic = (
        trace.epistemic_term_changed_selection
        if fixture.requires_epistemic_term
        else not trace.epistemic_term_changed_selection
    )
    return (
        trace.selected_action_id == fixture.expected_action_id,
        trace.selected_without_information_gain_id == fixture.expected_without_information_gain_id,
        hard_filter,
        telemetry,
        epistemic,
    )


def evaluate_b4(fixture: Path, config: Path, report: Path) -> B4DevelopmentReport:
    if report.exists():
        raise FileExistsError("refusing to overwrite B4 development report")
    fixtures = load_b4_suite(fixture)
    settings = load_b4_config(config)
    started_at = datetime.now(UTC)
    cases = _run_cases(fixtures, settings)
    required = settings.minimum_epistemic_cases
    passed_epistemic = sum(
        case.epistemic_contribution_passed
        and case.selected_action_id != case.selected_without_information_gain_id
        for case in cases
    )
    result = B4DevelopmentReport(
        report_id=uuid4(),
        started_at=started_at,
        finished_at=datetime.now(UTC),
        fixture_file=str(fixture),
        fixture_sha256=sha256_file(fixture),
        config_file=str(config),
        config_sha256=sha256_file(config),
        cases=cases,
        passed_cases=sum(case.passed for case in cases),
        required_epistemic_cases=required,
        passed_epistemic_cases=passed_epistemic,
        engineering_gate_passed=all(case.passed for case in cases) and passed_epistemic >= required,
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return result


def load_b4_report(path: Path) -> B4DevelopmentReport:
    return B4DevelopmentReport.model_validate_json(path.read_text(encoding="utf-8"))


def verify_b4_report(report: B4DevelopmentReport) -> None:
    fixture = Path(report.fixture_file)
    config = Path(report.config_file)
    if sha256_file(fixture) != report.fixture_sha256:
        raise ValueError("B4 fixture hash mismatch")
    if sha256_file(config) != report.config_sha256:
        raise ValueError("B4 config hash mismatch")
    fixtures = load_b4_suite(fixture)
    settings = load_b4_config(config)
    rebuilt_cases = _run_cases(fixtures, settings)
    if rebuilt_cases != report.cases:
        raise ValueError("B4 report cases do not match deterministic replay")

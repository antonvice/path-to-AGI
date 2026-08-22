"""Belief-aware active-inference action selection with auditable score terms."""

import json
from collections.abc import Mapping, Sequence
from math import isclose
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from aif_qwen_agent.artifacts import sha256_text
from aif_qwen_agent.logic_backends.base import LogicBackend
from aif_qwen_agent.policy import HardPolicy
from aif_qwen_agent.schemas import ActionCandidate, BeliefState, PredictedOutcome


class AIFWeights(BaseModel):
    preference_risk: float = Field(default=1.0, ge=0.0)
    failure_risk: float = Field(default=0.25, ge=0.0)
    ambiguity: float = Field(default=0.35, ge=0.0)
    information_gain: float = Field(default=0.60, ge=0.0)
    token_cost: float = Field(default=0.15, ge=0.0)
    wall_time_cost: float = Field(default=0.15, ge=0.0)
    operational_risk: float = Field(default=2.0, ge=0.0)


class AIFScoreTerms(BaseModel):
    preference_risk: float
    failure_risk: float
    ambiguity: float
    information_gain: float
    token_cost: float
    wall_time_cost: float
    operational_risk: float
    total: float

    @model_validator(mode="after")
    def total_matches_terms(self) -> "AIFScoreTerms":
        terms = (
            self.preference_risk,
            self.failure_risk,
            self.ambiguity,
            self.information_gain,
            self.token_cost,
            self.wall_time_cost,
            self.operational_risk,
        )
        if not isclose(self.total, sum(terms), abs_tol=1e-12):
            raise ValueError("AIF total does not match score terms")
        return self


class ActionEvaluation(BaseModel):
    action: ActionCandidate
    prediction: PredictedOutcome | None = None
    eligible: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    score: AIFScoreTerms | None = None
    score_without_information_gain: AIFScoreTerms | None = None

    @model_validator(mode="after")
    def eligibility_matches_scores(self) -> "ActionEvaluation":
        scored = self.score is not None and self.score_without_information_gain is not None
        if self.eligible != scored:
            raise ValueError("only eligible actions may be scored")
        if self.eligible == bool(self.rejection_reasons):
            raise ValueError("action rejection reasons do not match eligibility")
        return self


class ActionSelectionTrace(BaseModel):
    schema_version: Literal["1"] = "1"
    state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluations: list[ActionEvaluation] = Field(min_length=1)
    selected_action_id: str
    selected_without_information_gain_id: str
    epistemic_term_changed_selection: bool

    @model_validator(mode="after")
    def selections_match_evaluations(self) -> "ActionSelectionTrace":
        eligible = {value.action.id for value in self.evaluations if value.eligible}
        if self.selected_action_id not in eligible:
            raise ValueError("selected action is not eligible")
        if self.selected_without_information_gain_id not in eligible:
            raise ValueError("ablated selected action is not eligible")
        changed = self.selected_action_id != self.selected_without_information_gain_id
        if self.epistemic_term_changed_selection != changed:
            raise ValueError("epistemic selection flag is inconsistent")
        return self


def score_action(prediction: PredictedOutcome, weights: AIFWeights) -> AIFScoreTerms:
    values = {
        "preference_risk": weights.preference_risk * (1.0 - prediction.expected_goal_progress),
        "failure_risk": weights.failure_risk * (1.0 - prediction.success_probability),
        "ambiguity": weights.ambiguity * prediction.ambiguity,
        "information_gain": -weights.information_gain * prediction.expected_information_gain,
        "token_cost": weights.token_cost * prediction.token_cost,
        "wall_time_cost": weights.wall_time_cost * prediction.wall_time_cost,
        "operational_risk": weights.operational_risk * prediction.operational_risk,
    }
    return AIFScoreTerms(**values, total=sum(values.values()))


def selection_trace_sha256(trace: ActionSelectionTrace) -> str:
    canonical = json.dumps(
        trace.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256_text(canonical)


def _state_sha256(state: BeliefState) -> str:
    canonical = json.dumps(
        state.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256_text(canonical)


def select_active_inference_action(
    candidates: Sequence[ActionCandidate],
    predictions: Mapping[str, PredictedOutcome],
    state: BeliefState,
    policy: HardPolicy,
    logic: LogicBackend | None = None,
    weights: AIFWeights | None = None,
) -> tuple[ActionCandidate, ActionSelectionTrace]:
    """Filter hard constraints, score eligible actions, and return a deterministic trace."""
    if not candidates:
        raise ValueError("at least one action candidate is required")
    if len({candidate.id for candidate in candidates}) != len(candidates):
        raise ValueError("action candidate IDs must be unique")
    active_weights = weights or AIFWeights()
    ablated_weights = active_weights.model_copy(update={"information_gain": 0.0})
    screened: list[tuple[ActionCandidate, list[str]]] = []
    for candidate in candidates:
        rejections = [
            f"policy:{rule.name}" for rule in policy.rules if not rule.predicate(candidate, state)
        ]
        if logic is not None and not logic.allows(candidate, state):
            rejections.append("logic_backend")
        screened.append((candidate, rejections))
    evaluations: list[ActionEvaluation] = []
    for candidate, rejections in screened:
        if rejections:
            evaluations.append(
                ActionEvaluation(
                    action=candidate,
                    prediction=predictions.get(candidate.id),
                    eligible=False,
                    rejection_reasons=rejections,
                )
            )
            continue
        prediction = predictions.get(candidate.id)
        if prediction is None:
            raise ValueError(f"missing prediction for eligible action: {candidate.id}")
        evaluations.append(
            ActionEvaluation(
                action=candidate,
                prediction=prediction,
                eligible=True,
                score=score_action(prediction, active_weights),
                score_without_information_gain=score_action(prediction, ablated_weights),
            )
        )
    evaluations.sort(key=lambda value: value.action.id)
    eligible = [evaluation for evaluation in evaluations if evaluation.eligible]
    if not eligible:
        raise ValueError("hard constraints rejected every action candidate")
    selected = min(
        eligible,
        key=lambda value: (
            value.score.total if value.score is not None else float("inf"),
            value.action.id,
        ),
    )
    ablated = min(
        eligible,
        key=lambda value: (
            value.score_without_information_gain.total
            if value.score_without_information_gain is not None
            else float("inf"),
            value.action.id,
        ),
    )
    trace = ActionSelectionTrace(
        state_sha256=_state_sha256(state),
        evaluations=evaluations,
        selected_action_id=selected.action.id,
        selected_without_information_gain_id=ablated.action.id,
        epistemic_term_changed_selection=selected.action.id != ablated.action.id,
    )
    return selected.action, trace

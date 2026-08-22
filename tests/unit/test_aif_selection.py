from pathlib import Path

import pytest

from aif_qwen_agent.aif_selection import (
    AIFWeights,
    score_action,
    select_active_inference_action,
    selection_trace_sha256,
)
from aif_qwen_agent.policy import HardPolicy, PolicyRule
from aif_qwen_agent.schemas import (
    ActionCandidate,
    ActionKind,
    BeliefState,
    PredictedOutcome,
)


def prediction(**updates: float) -> PredictedOutcome:
    values = {
        "success_probability": 0.8,
        "expected_goal_progress": 0.5,
        "expected_information_gain": 0.0,
        "ambiguity": 0.0,
        "token_cost": 0.0,
        "wall_time_cost": 0.0,
        "operational_risk": 0.0,
    }
    values.update(updates)
    return PredictedOutcome(**values)


def test_score_logs_terms_that_sum_to_total() -> None:
    terms = score_action(
        prediction(
            expected_information_gain=0.4,
            ambiguity=0.2,
            token_cost=0.1,
            wall_time_cost=0.3,
            operational_risk=0.05,
        ),
        AIFWeights(),
    )

    assert terms.information_gain == pytest.approx(-0.24)
    assert terms.total == pytest.approx(
        terms.preference_risk
        + terms.failure_risk
        + terms.ambiguity
        + terms.information_gain
        + terms.token_cost
        + terms.wall_time_cost
        + terms.operational_risk
    )


def test_hard_filter_runs_before_prediction_scoring() -> None:
    safe = ActionCandidate(id="safe", kind=ActionKind.READ_FILE)
    forbidden = ActionCandidate(id="forbidden", kind=ActionKind.RUN_PYTHON)
    state = BeliefState(objective="diagnose safely")
    policy = HardPolicy(
        [PolicyRule("no_python", lambda action, _state: action.kind != ActionKind.RUN_PYTHON)]
    )

    selected, trace = select_active_inference_action(
        [forbidden, safe], {"safe": prediction()}, state, policy
    )

    rejected = trace.evaluations[0]
    assert selected == safe
    assert not rejected.eligible
    assert rejected.prediction is rejected.score is None
    assert rejected.rejection_reasons == ["policy:no_python"]


def test_information_gain_changes_the_selected_action() -> None:
    answer = ActionCandidate(id="answer", kind=ActionKind.ANSWER)
    test = ActionCandidate(id="test", kind=ActionKind.RUN_TESTS)
    selected, trace = select_active_inference_action(
        [answer, test],
        {
            "answer": prediction(expected_goal_progress=0.75),
            "test": prediction(
                expected_goal_progress=0.60,
                expected_information_gain=0.80,
                token_cost=0.20,
            ),
        },
        BeliefState(objective="resolve uncertainty"),
        HardPolicy(),
    )

    assert selected.id == "test"
    assert trace.selected_without_information_gain_id == "answer"
    assert trace.epistemic_term_changed_selection


def test_ties_are_broken_by_action_id_and_trace_is_stable() -> None:
    actions = [
        ActionCandidate(id="b", kind=ActionKind.ANSWER),
        ActionCandidate(id="a", kind=ActionKind.ANSWER),
    ]
    predictions = {action.id: prediction() for action in actions}
    state = BeliefState(objective="choose deterministically")

    first, first_trace = select_active_inference_action(actions, predictions, state, HardPolicy())
    second, second_trace = select_active_inference_action(
        list(reversed(actions)), predictions, state, HardPolicy()
    )

    assert first.id == second.id == "a"
    assert first_trace.selected_action_id == second_trace.selected_action_id
    assert selection_trace_sha256(first_trace) == selection_trace_sha256(second_trace)


def test_missing_prediction_for_eligible_action_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing prediction"):
        select_active_inference_action(
            [ActionCandidate(id="answer", kind=ActionKind.ANSWER)],
            {},
            BeliefState(objective="answer"),
            HardPolicy(),
        )


def test_real_b4_fixture_path_exists() -> None:
    assert Path("evals/tasks/b4_dev/suite.yaml").is_file()

from aif_qwen_agent.controller import eligible_actions, select_eligible_action
from aif_qwen_agent.policy import HardPolicy, PolicyRule
from aif_qwen_agent.schemas import (
    ActionCandidate,
    ActionKind,
    BeliefState,
    PredictedOutcome,
)


def outcome(*, information_gain: float) -> PredictedOutcome:
    return PredictedOutcome(
        success_probability=0.8,
        expected_goal_progress=0.8,
        expected_information_gain=information_gain,
        ambiguity=0.0,
        token_cost=0.0,
        wall_time_cost=0.0,
        operational_risk=0.0,
    )


def test_information_gain_cannot_override_safety() -> None:
    state = BeliefState(objective="diagnose production failure")
    safe = ActionCandidate(
        id="safe", kind=ActionKind.READ_FILE, arguments={"operation": "inspect_logs"}
    )
    unsafe = ActionCandidate(
        id="unsafe",
        kind=ActionKind.RUN_PYTHON,
        arguments={"operation": "disable_production_auth"},
    )
    policy = HardPolicy(
        [
            PolicyRule(
                "keep_production_auth_enabled",
                lambda action, _: action.arguments.get("operation") != "disable_production_auth",
            )
        ]
    )
    predictions = {"safe": outcome(information_gain=0.1), "unsafe": outcome(information_gain=1.0)}

    filtered = eligible_actions([safe, unsafe], state, policy)
    selected = select_eligible_action([safe, unsafe], predictions, state, policy)

    assert filtered == [safe]
    assert selected == safe

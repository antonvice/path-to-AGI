import pytest

from aif_qwen_agent.controller import select_action
from aif_qwen_agent.schemas import ActionCandidate, ActionKind, PredictedOutcome


def outcome(goal: float) -> PredictedOutcome:
    return PredictedOutcome(
        success_probability=goal,
        expected_goal_progress=goal,
        expected_information_gain=0.0,
        ambiguity=0.0,
        token_cost=0.0,
        wall_time_cost=0.0,
        operational_risk=0.0,
    )


def test_selects_lowest_score() -> None:
    answer = ActionCandidate(id="answer", kind=ActionKind.ANSWER)
    verify = ActionCandidate(id="verify", kind=ActionKind.VERIFY)
    selected = select_action([answer, verify], {"answer": outcome(0.4), "verify": outcome(0.8)})
    assert selected == verify


def test_requires_candidate() -> None:
    with pytest.raises(ValueError, match="At least one"):
        select_action([], {})

from aif_qwen_agent.aif_score import aif_score
from aif_qwen_agent.schemas import PredictedOutcome


def prediction(**updates: float) -> PredictedOutcome:
    values = {
        "success_probability": 0.5,
        "expected_goal_progress": 0.5,
        "expected_information_gain": 0.0,
        "ambiguity": 0.0,
        "token_cost": 0.0,
        "wall_time_cost": 0.0,
        "operational_risk": 0.0,
    }
    values.update(updates)
    return PredictedOutcome(**values)


def test_information_gain_can_outweigh_equal_immediate_progress() -> None:
    investigate = prediction(expected_information_gain=0.5)
    answer = prediction()
    assert aif_score(investigate) < aif_score(answer)


def test_operational_risk_is_heavily_penalized() -> None:
    risky = prediction(expected_goal_progress=1.0, operational_risk=1.0)
    safe = prediction(expected_goal_progress=0.5)
    assert aif_score(safe) < aif_score(risky)

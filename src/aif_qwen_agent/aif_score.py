from aif_qwen_agent.schemas import PredictedOutcome


def aif_score(prediction: PredictedOutcome) -> float:
    """Return the MVP expected-free-energy approximation; lower is preferred."""
    preference_risk = 1.0 - prediction.expected_goal_progress
    resource_cost = 0.15 * prediction.token_cost + 0.15 * prediction.wall_time_cost
    return (
        preference_risk
        + 0.35 * prediction.ambiguity
        - 0.60 * prediction.expected_information_gain
        + resource_cost
        + 2.00 * prediction.operational_risk
    )

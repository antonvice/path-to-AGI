from collections.abc import Mapping, Sequence

from aif_qwen_agent.aif_score import aif_score
from aif_qwen_agent.logic_backends.base import LogicBackend
from aif_qwen_agent.policy import HardPolicy
from aif_qwen_agent.schemas import ActionCandidate, BeliefState, PredictedOutcome


def select_action(
    candidates: Sequence[ActionCandidate], predictions: Mapping[str, PredictedOutcome]
) -> ActionCandidate:
    if not candidates:
        raise ValueError("At least one eligible candidate is required")
    return min(candidates, key=lambda candidate: aif_score(predictions[candidate.id]))


def eligible_actions(
    candidates: Sequence[ActionCandidate],
    state: BeliefState,
    policy: HardPolicy,
    logic: LogicBackend | None = None,
) -> list[ActionCandidate]:
    return [
        action
        for action in candidates
        if policy.allows(action, state) and (logic is None or logic.allows(action, state))
    ]


def select_eligible_action(
    candidates: Sequence[ActionCandidate],
    predictions: Mapping[str, PredictedOutcome],
    state: BeliefState,
    policy: HardPolicy,
    logic: LogicBackend | None = None,
) -> ActionCandidate:
    return select_action(eligible_actions(candidates, state, policy, logic), predictions)

from aif_qwen_agent.belief import (
    BeliefObservation,
    ExplicitBeliefState,
    record_unresolved_question,
    update_belief_state,
)
from aif_qwen_agent.belief_decision import (
    decide_from_beliefs,
    decide_from_latest_observation,
)
from aif_qwen_agent.schemas import EvidenceRef


def observation(
    observation_id: str,
    hypothesis_id: str,
    statement: str,
    *,
    reliability: float = 1.0,
    effect: str = "support",
    contradicts: list[str] | None = None,
) -> BeliefObservation:
    return BeliefObservation(
        observation_id=observation_id,
        hypothesis_id=hypothesis_id,
        statement=statement,
        effect=effect,
        evidence=EvidenceRef(
            artifact_id=observation_id,
            excerpt_hash=observation_id * 64,
            reliability=reliability,
        ),
        contradicts=contradicts or [],
    )


def test_belief_decision_abstains_while_latest_observation_guesses() -> None:
    item = observation(
        "a",
        "cause",
        "A configuration mismatch caused the failure.",
        reliability=0.4,
    )
    state = update_belief_state(ExplicitBeliefState(objective="diagnose"), item)

    assert decide_from_beliefs(state).status == "unknown"
    assert decide_from_latest_observation([item], []).status == "supported"


def test_belief_decision_reports_refutation_and_conflict() -> None:
    healthy = observation("a", "healthy", "The service is healthy.", reliability=0.8)
    probe = observation("b", "healthy", "The service is healthy.", effect="refute")
    state = update_belief_state(ExplicitBeliefState(objective="health"), healthy)
    state = update_belief_state(state, probe)
    assert decide_from_beliefs(state).status == "refuted"

    route_a = observation("c", "route-a", "Route A is active.", contradicts=["route-b"])
    route_b = observation("d", "route-b", "Route B is active.", contradicts=["route-a"])
    conflict = update_belief_state(ExplicitBeliefState(objective="route"), route_a)
    conflict = update_belief_state(conflict, route_b)
    decision = decide_from_beliefs(conflict)
    assert decision.status == "conflict"
    assert "Route A" in decision.answer
    assert "Route B" in decision.answer


def test_belief_decision_handles_supported_and_unresolved_states() -> None:
    item = observation("e", "token", "The token is SAFE-202.")
    state = update_belief_state(ExplicitBeliefState(objective="token"), item)
    assert decide_from_beliefs(state).answer == item.statement

    unresolved = record_unresolved_question(
        ExplicitBeliefState(objective="missing"), "Which token?"
    )
    assert decide_from_beliefs(unresolved).status == "unknown"
    assert decide_from_latest_observation([], unresolved.unresolved_questions).status == "unknown"

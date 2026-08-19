"""Deterministic B3 decisions and the stateless latest-observation ablation."""

from typing import Literal

from pydantic import BaseModel, Field

from aif_qwen_agent.belief import BeliefObservation, ExplicitBeliefState, ExplicitHypothesis


class BeliefDecision(BaseModel):
    status: Literal["supported", "refuted", "conflict", "unknown"]
    answer: str
    hypothesis_ids: list[str] = Field(default_factory=list)
    provenance_hashes: list[str] = Field(default_factory=list)


def _provenance(hypotheses: list[ExplicitHypothesis]) -> list[str]:
    return list(
        dict.fromkeys(
            evidence.excerpt_hash
            for hypothesis in hypotheses
            for evidence in hypothesis.evidence
            if evidence.excerpt_hash is not None
        )
    )


def decide_from_beliefs(state: ExplicitBeliefState) -> BeliefDecision:
    contradicted = [item for item in state.hypotheses if item.status == "contradicted"]
    if contradicted:
        statements = "\n".join(f"- {item.statement}" for item in contradicted)
        return BeliefDecision(
            status="conflict",
            answer=f"conflict:\n{statements}",
            hypothesis_ids=[item.id for item in contradicted],
            provenance_hashes=_provenance(contradicted),
        )
    supported = [item for item in state.hypotheses if item.status == "supported"]
    if supported:
        statements = "\n".join(item.statement for item in supported)
        return BeliefDecision(
            status="supported",
            answer=statements,
            hypothesis_ids=[item.id for item in supported],
            provenance_hashes=_provenance(supported),
        )
    refuted = [item for item in state.hypotheses if item.status == "refuted"]
    if refuted:
        statements = "\n".join(f"- {item.statement}" for item in refuted)
        return BeliefDecision(
            status="refuted",
            answer=f"refuted:\n{statements}",
            hypothesis_ids=[item.id for item in refuted],
            provenance_hashes=_provenance(refuted),
        )
    return BeliefDecision(
        status="unknown",
        answer="unknown: insufficient supported belief",
    )


def decide_from_latest_observation(
    observations: list[BeliefObservation], unresolved_questions: list[str]
) -> BeliefDecision:
    """Nearest stateless ablation: use only the newest observation, ignoring accumulated state."""
    if not observations:
        return BeliefDecision(
            status="unknown",
            answer="unknown: no observation" if unresolved_questions else "unknown: empty state",
        )
    latest = observations[-1]
    return BeliefDecision(
        status="supported",
        answer=latest.statement,
        hypothesis_ids=[latest.hypothesis_id],
        provenance_hashes=[latest.evidence.excerpt_hash]
        if latest.evidence.excerpt_hash is not None
        else [],
    )

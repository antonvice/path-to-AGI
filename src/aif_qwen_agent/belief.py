"""Deterministic explicit-belief updates and append-only persistence."""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from aif_qwen_agent.schemas import (
    BeliefState,
    EpisodicRetrievalHit,
    EpisodicRetrievalResult,
    EvidenceRef,
    Hypothesis,
)

_SCHEMA_VERSION = "1"


class BeliefObservation(BaseModel):
    schema_version: Literal["1"] = "1"
    observation_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    effect: Literal["support", "refute"] = "support"
    evidence: EvidenceRef
    contradicts: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_contradictions(self) -> "BeliefObservation":
        if self.hypothesis_id in self.contradicts:
            raise ValueError("belief observation cannot contradict itself")
        if len(set(self.contradicts)) != len(self.contradicts):
            raise ValueError("belief observation contradictions must be unique")
        return self


class ExplicitHypothesis(Hypothesis):
    contradicts: list[str] = Field(default_factory=list)


class ExplicitBeliefState(BeliefState):
    hypotheses: list[ExplicitHypothesis] = Field(default_factory=list)  # type: ignore[assignment]
    revision: int = Field(default=0, ge=0)
    applied_observations: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def internally_consistent(self) -> "ExplicitBeliefState":
        hypotheses = {hypothesis.id: hypothesis for hypothesis in self.hypotheses}
        if len(hypotheses) != len(self.hypotheses):
            raise ValueError("belief hypothesis IDs must be unique")
        if any(len(digest) != 64 for digest in self.applied_observations.values()):
            raise ValueError("applied belief observation hashes must be SHA-256 digests")
        for hypothesis in self.hypotheses:
            if len(set(hypothesis.contradicts)) != len(hypothesis.contradicts):
                raise ValueError("hypothesis contradictions must be unique")
            for other_id in hypothesis.contradicts:
                other = hypotheses.get(other_id)
                if other is None:
                    raise ValueError("hypothesis contradiction target is missing")
                if hypothesis.id not in other.contradicts:
                    raise ValueError("hypothesis contradictions must be symmetric")
        return self


def belief_state_sha256(state: ExplicitBeliefState) -> str:
    canonical = json.dumps(
        state.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def belief_observation_sha256(observation: BeliefObservation) -> str:
    canonical = json.dumps(
        observation.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _probability(prior: float, observation: BeliefObservation) -> float:
    reliability = observation.evidence.reliability
    if observation.effect == "support":
        return 1.0 - (1.0 - prior) * (1.0 - reliability)
    return prior * (1.0 - reliability)


def _status(probability: float) -> str:
    if probability >= 0.75:
        return "supported"
    if probability <= 0.25:
        return "refuted"
    return "open"


def update_belief_state(
    state: ExplicitBeliefState, observation: BeliefObservation
) -> ExplicitBeliefState:
    """Apply one content-addressed observation; replay is a no-op."""
    observation_hash = belief_observation_sha256(observation)
    if observation.observation_id in state.applied_observations:
        if state.applied_observations[observation.observation_id] != observation_hash:
            raise ValueError("belief observation ID was reused with different content")
        return state
    hypotheses = [hypothesis.model_copy(deep=True) for hypothesis in state.hypotheses]
    by_id = {hypothesis.id: hypothesis for hypothesis in hypotheses}
    hypothesis = by_id.get(observation.hypothesis_id)
    if hypothesis is None:
        hypothesis = ExplicitHypothesis(
            id=observation.hypothesis_id,
            statement=observation.statement,
            probability=0.5,
        )
        hypotheses.append(hypothesis)
        by_id[hypothesis.id] = hypothesis
    elif hypothesis.statement != observation.statement:
        raise ValueError("belief observation changes an existing hypothesis statement")

    evidence = hypothesis.evidence
    if observation.evidence not in evidence:
        evidence.append(observation.evidence)
    probability = _probability(hypothesis.probability, observation)
    hypothesis.probability = probability
    hypothesis.status = _status(probability)  # type: ignore[assignment]

    for other_id in observation.contradicts:
        other = by_id.get(other_id)
        if other is None:
            continue
        hypothesis.contradicts = sorted({*hypothesis.contradicts, other_id})
        other.contradicts = sorted({*other.contradicts, hypothesis.id})
    for item in hypotheses:
        if item.contradicts:
            item.status = "contradicted"

    return ExplicitBeliefState(
        objective=state.objective,
        hypotheses=hypotheses,
        known_constraints=state.known_constraints,
        unresolved_questions=state.unresolved_questions,
        remaining_budget=state.remaining_budget,
        revision=state.revision + 1,
        applied_observations={
            **state.applied_observations,
            observation.observation_id: observation_hash,
        },
    )


def record_unresolved_question(state: ExplicitBeliefState, question: str) -> ExplicitBeliefState:
    normalized = question.strip()
    if not normalized:
        raise ValueError("unresolved belief question cannot be empty")
    if normalized in state.unresolved_questions:
        return state
    return state.model_copy(
        update={
            "unresolved_questions": [*state.unresolved_questions, normalized],
            "revision": state.revision + 1,
        }
    )


def _hypothesis_id(hit: EpisodicRetrievalHit) -> str:
    return f"episode:{hit.episode.content_sha256}"


def _conflicts(left: EpisodicRetrievalHit, right: EpisodicRetrievalHit) -> bool:
    shared_tags = set(left.episode.tags).intersection(right.episode.tags)
    return left.episode.outcome != right.episode.outcome and (
        "conflict" in shared_tags or len(shared_tags) >= 2
    )


def belief_state_from_retrieval(
    objective: str,
    retrieval: EpisodicRetrievalResult,
    initial: ExplicitBeliefState | None = None,
) -> ExplicitBeliefState:
    """Project verified B2 outcomes into explicit hypotheses without source prose."""
    state = initial or ExplicitBeliefState(objective=objective)
    if state.objective != objective:
        raise ValueError("belief objective does not match the existing state")
    if not retrieval.hits:
        return record_unresolved_question(state, retrieval.query.text)
    hypothesis_ids = [_hypothesis_id(hit) for hit in retrieval.hits]
    for index, hit in enumerate(retrieval.hits):
        contradicts = [
            hypothesis_ids[other_index]
            for other_index, other in enumerate(retrieval.hits)
            if index != other_index and _conflicts(hit, other)
        ]
        state = update_belief_state(
            state,
            BeliefObservation(
                observation_id=hit.episode.content_sha256,
                hypothesis_id=hypothesis_ids[index],
                statement=hit.episode.outcome,
                evidence=EvidenceRef(
                    artifact_id=f"episode:{hit.episode.episode_id}",
                    excerpt_hash=hit.episode.content_sha256,
                    reliability=1.0,
                ),
                contradicts=contradicts,
            ),
        )
    return state


class BeliefStateStore:
    """Append-only SQLite revisions with payload hash verification."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS belief_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS belief_revisions (
                    objective TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    state_sha256 TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (objective, revision)
                );
                """
            )
            row = connection.execute(
                "SELECT value FROM belief_metadata WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO belief_metadata(key, value) VALUES ('schema_version', ?)",
                    (_SCHEMA_VERSION,),
                )
            elif row[0] != _SCHEMA_VERSION:
                raise ValueError(f"unsupported belief-state schema: {row[0]}")

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _load_row(row: sqlite3.Row) -> ExplicitBeliefState:
        state = ExplicitBeliefState.model_validate_json(row["payload_json"])
        if (
            state.objective != row["objective"]
            or state.revision != row["revision"]
            or belief_state_sha256(state) != row["state_sha256"]
        ):
            raise ValueError("belief revision does not match its immutable payload")
        return state

    def append(self, state: ExplicitBeliefState) -> bool:
        validated = ExplicitBeliefState.model_validate(state.model_dump())
        digest = belief_state_sha256(validated)
        with self._connection() as connection:
            existing = connection.execute(
                "SELECT * FROM belief_revisions WHERE objective = ? AND revision = ?",
                (validated.objective, validated.revision),
            ).fetchone()
            if existing is not None:
                if self._load_row(existing) != validated:
                    raise ValueError("belief revision already exists with different state")
                return False
            revisions = connection.execute(
                "SELECT revision FROM belief_revisions WHERE objective = ? ORDER BY revision",
                (validated.objective,),
            ).fetchall()
            observed = [int(row["revision"]) for row in revisions]
            if observed != list(range(len(observed))):
                raise ValueError("belief revision history has a gap")
            expected = len(observed)
            if validated.revision != expected:
                raise ValueError(f"belief revision must be appended in order: expected {expected}")
            connection.execute(
                """
                INSERT INTO belief_revisions(objective, revision, state_sha256, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (validated.objective, validated.revision, digest, validated.model_dump_json()),
            )
        return True

    def latest(self, objective: str) -> ExplicitBeliefState:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM belief_revisions
                WHERE objective = ? ORDER BY revision DESC LIMIT 1
                """,
                (objective,),
            ).fetchone()
        if row is None:
            raise KeyError(objective)
        return self._load_row(row)

    def history(self, objective: str) -> list[ExplicitBeliefState]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM belief_revisions WHERE objective = ? ORDER BY revision",
                (objective,),
            ).fetchall()
        return [self._load_row(row) for row in rows]

    def verify_integrity(self) -> int:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM belief_revisions ORDER BY objective, revision"
            ).fetchall()
        expected: dict[str, int] = {}
        for row in rows:
            objective = str(row["objective"])
            if row["revision"] != expected.get(objective, 0):
                raise ValueError("belief revision history has a gap")
            expected[objective] = int(row["revision"]) + 1
            self._load_row(row)
        return len(rows)

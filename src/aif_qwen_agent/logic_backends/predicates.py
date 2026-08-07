from collections.abc import Callable, Iterable
from dataclasses import dataclass

from aif_qwen_agent.schemas import (
    ActionCandidate,
    BeliefState,
    LogicalContradiction,
    LogicalFact,
    TruthBounds,
)

ActionPredicate = Callable[[ActionCandidate, BeliefState], bool]


@dataclass(frozen=True)
class Literal:
    predicate: str
    negated: bool = False


@dataclass(frozen=True)
class InferenceRule:
    name: str
    premises: tuple[Literal, ...]
    conclusion: Literal
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between zero and one")
        if not self.premises:
            raise ValueError("an inference rule needs at least one premise")


class PythonPredicateBackend:
    """Small propositional baseline with inspectable truth bounds and rules."""

    def __init__(
        self,
        inference_rules: Iterable[InferenceRule] = (),
        action_rules: Iterable[ActionPredicate] = (),
    ) -> None:
        self.inference_rules = tuple(inference_rules)
        self.action_rules = tuple(action_rules)
        self._facts: list[LogicalFact] = []

    @property
    def facts(self) -> tuple[LogicalFact, ...]:
        return tuple(self._facts)

    def add_facts(self, facts: Iterable[LogicalFact]) -> None:
        for fact in facts:
            if fact not in self._facts:
                self._facts.append(fact)

    def infer(self, max_rounds: int = 8) -> tuple[LogicalFact, ...]:
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        derived: list[LogicalFact] = []
        for _ in range(max_rounds):
            round_facts: list[LogicalFact] = []
            for rule in self.inference_rules:
                premises = [self._strongest(literal) for literal in rule.premises]
                if any(fact is None for fact in premises):
                    continue
                supported = [fact for fact in premises if fact is not None]
                if any(fact.bounds.lower < rule.threshold for fact in supported):
                    continue
                conclusion = LogicalFact(
                    predicate=rule.conclusion.predicate,
                    negated=rule.conclusion.negated,
                    bounds=TruthBounds(
                        lower=min(fact.bounds.lower for fact in supported),
                        upper=min(fact.bounds.upper for fact in supported),
                    ),
                    provenance=[
                        *dict.fromkeys(item for fact in supported for item in fact.provenance),
                        f"rule:{rule.name}",
                    ],
                )
                if conclusion not in self._facts and conclusion not in round_facts:
                    round_facts.append(conclusion)
            if not round_facts:
                break
            self._facts.extend(round_facts)
            derived.extend(round_facts)
        return tuple(derived)

    def contradictions(self, min_support: float = 0.5) -> tuple[LogicalContradiction, ...]:
        if not 0.0 <= min_support <= 1.0:
            raise ValueError("min_support must be between zero and one")
        predicates = {fact.predicate for fact in self._facts}
        contradictions: list[LogicalContradiction] = []
        for predicate in sorted(predicates):
            positive = self._strongest(Literal(predicate))
            negative = self._strongest(Literal(predicate, negated=True))
            if (
                positive is not None
                and negative is not None
                and positive.bounds.lower >= min_support
                and negative.bounds.lower >= min_support
            ):
                contradictions.append(
                    LogicalContradiction(predicate=predicate, positive=positive, negative=negative)
                )
        return tuple(contradictions)

    def allows(self, action: ActionCandidate, state: BeliefState) -> bool:
        return all(rule(action, state) for rule in self.action_rules)

    def _strongest(self, literal: Literal) -> LogicalFact | None:
        matches = [
            fact
            for fact in self._facts
            if fact.predicate == literal.predicate and fact.negated == literal.negated
        ]
        return max(matches, key=lambda fact: fact.bounds.lower, default=None)

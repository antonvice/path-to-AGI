"""Hard permission, safety, and budget constraints."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from aif_qwen_agent.schemas import ActionCandidate, BeliefState

PolicyPredicate = Callable[[ActionCandidate, BeliefState], bool]


@dataclass(frozen=True)
class PolicyRule:
    name: str
    predicate: PolicyPredicate


class HardPolicy:
    def __init__(self, rules: Iterable[PolicyRule] = ()) -> None:
        self.rules = tuple(rules)

    def allows(self, action: ActionCandidate, state: BeliefState) -> bool:
        return all(rule.predicate(action, state) for rule in self.rules)

    def filter(
        self, actions: Iterable[ActionCandidate], state: BeliefState
    ) -> list[ActionCandidate]:
        return [action for action in actions if self.allows(action, state)]

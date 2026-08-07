from collections.abc import Iterable
from typing import Protocol

from aif_qwen_agent.schemas import (
    ActionCandidate,
    BeliefState,
    LogicalContradiction,
    LogicalFact,
)


class LogicBackend(Protocol):
    def add_facts(self, facts: Iterable[LogicalFact]) -> None: ...

    def infer(self, max_rounds: int = 8) -> tuple[LogicalFact, ...]: ...

    def contradictions(self, min_support: float = 0.5) -> tuple[LogicalContradiction, ...]: ...

    def allows(self, action: ActionCandidate, state: BeliefState) -> bool: ...

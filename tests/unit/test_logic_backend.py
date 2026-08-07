from aif_qwen_agent.logic_backends import InferenceRule, Literal, PythonPredicateBackend
from aif_qwen_agent.schemas import LogicalFact, TruthBounds


def fact(predicate: str, lower: float, upper: float, *, negated: bool = False) -> LogicalFact:
    return LogicalFact(
        predicate=predicate,
        bounds=TruthBounds(lower=lower, upper=upper),
        negated=negated,
        provenance=["fixture"],
    )


def test_infers_conclusion_with_bounded_support() -> None:
    backend = PythonPredicateBackend(
        inference_rules=[
            InferenceRule(
                name="inspect_changed_dependency",
                premises=(Literal("tests_failed"), Literal("dependency_changed")),
                conclusion=Literal("inspect_dependency_configuration"),
            )
        ]
    )
    backend.add_facts([fact("tests_failed", 1.0, 1.0), fact("dependency_changed", 0.6, 0.85)])

    derived = backend.infer()

    assert derived[0].predicate == "inspect_dependency_configuration"
    assert derived[0].bounds == TruthBounds(lower=0.6, upper=0.85)
    assert derived[0].provenance == ["fixture", "rule:inspect_changed_dependency"]


def test_preserves_supported_contradiction() -> None:
    backend = PythonPredicateBackend()
    backend.add_facts(
        [
            fact("service_healthy", 0.8, 1.0),
            fact("service_healthy", 0.9, 1.0, negated=True),
        ]
    )

    contradictions = backend.contradictions()

    assert len(contradictions) == 1
    assert contradictions[0].predicate == "service_healthy"

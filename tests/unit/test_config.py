from pathlib import Path

import pytest

from aif_qwen_agent.config import load_yaml
from aif_qwen_agent.schemas import BaselineFixture


@pytest.mark.parametrize(
    "path",
    [
        *sorted(Path("configs").glob("*.yaml")),
        *sorted(Path("evals/tasks").glob("**/*.yaml")),
    ],
)
def test_yaml_artifacts_are_mappings(path: Path) -> None:
    assert load_yaml(path)


def test_three_frozen_b0_fixtures_are_schema_valid() -> None:
    document = load_yaml(Path("evals/tasks/b0/direct_answer.yaml"))
    fixtures = [BaselineFixture.model_validate(value) for value in document["cases"]]
    assert len(fixtures) == 3

import pytest
from pydantic import ValidationError

from aif_qwen_agent.schemas import Hypothesis


def test_probability_must_be_bounded() -> None:
    with pytest.raises(ValidationError):
        Hypothesis(id="h1", statement="invalid", probability=1.1)

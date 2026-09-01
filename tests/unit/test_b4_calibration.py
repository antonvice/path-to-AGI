import json
from pathlib import Path

import pytest
import yaml

from aif_qwen_agent.b4_calibration import (
    decode_named_predictions,
    evaluate_b4_calibration,
    load_b4_calibration_report,
    load_calibration_suite,
    verify_b4_calibration_report,
    verify_calibration_novelty,
)
from aif_qwen_agent.model_adapters.base import ChatMessage
from aif_qwen_agent.schemas import ActionCandidate, ActionKind, GenerationConfig, ModelResult

SUITE = Path("evals/tasks/b4_calibration_dev/suite.yaml")
CONFIG = Path("configs/aif_b4_calibration.yaml")
MODEL_CONFIG = Path("configs/qwen3_8_27b_b4_calibration.yaml")


class FakeCalibratedModel:
    def render_messages(self, messages: list[ChatMessage]) -> str:
        return json.dumps(messages, ensure_ascii=False, separators=(",", ":"))

    def generate(self, rendered_prompt: str, config: GenerationConfig) -> ModelResult:
        messages = json.loads(rendered_prompt)
        payload = json.loads(messages[-1]["content"])
        predictions = {}
        for alias, action in payload["actions"].items():
            description = action["description"].lower()
            if (
                "irreversibly" in description
                or "without authorization" in description
                or "redeploy" in description
            ):
                values = [90, 90, 20, 20, 15, 20, 95]
            elif "highly supported" in description or "verified record count" in description:
                values = [95, 95, 5, 3, 2, 1, 0]
            elif "repeat" in description or "recompute" in description:
                values = [90, 30, 25, 8, 30, 45, 0]
            elif "entire long" in description:
                values = [90, 70, 90, 5, 60, 90, 2]
            elif (
                "guess" in description
                or "assume" in description
                or "without checking" in description
            ):
                values = [35, 60, 0, 70, 5, 2, 0]
            else:
                values = [90, 70, 90, 5, 20, 20, 2]
            predictions[alias] = dict(
                zip(
                    (
                        "success",
                        "goal_progress",
                        "information_gain",
                        "remaining_ambiguity",
                        "token_cost",
                        "wall_time_cost",
                        "operational_risk",
                    ),
                    values,
                    strict=True,
                )
            )
        text = json.dumps({"predictions": predictions}, separators=(",", ":"))
        return ModelResult(
            raw_text=text,
            text=text,
            input_tokens=100,
            output_tokens=80,
            load_seconds=0.1,
            generation_seconds=0.2,
            device="fake",
            stop_reason="eos",
        )


def test_named_prediction_decoder_uses_explicit_fields() -> None:
    candidates = [ActionCandidate(id="inspect", kind=ActionKind.READ_FILE)]
    text = json.dumps(
        {
            "predictions": {
                "A": {
                    "success": 90,
                    "goal_progress": 70,
                    "information_gain": 80,
                    "remaining_ambiguity": 10,
                    "token_cost": 20,
                    "wall_time_cost": 30,
                    "operational_risk": 5,
                }
            }
        }
    )

    percentages, outcomes = decode_named_predictions(text, candidates)

    assert percentages["inspect"].operational_risk == 5
    assert outcomes["inspect"].expected_information_gain == 0.8
    assert outcomes["inspect"].operational_risk == 0.05


def test_named_prediction_decoder_rejects_missing_field() -> None:
    candidates = [ActionCandidate(id="inspect", kind=ActionKind.READ_FILE)]

    with pytest.raises(ValueError, match="fields are invalid"):
        decode_named_predictions('{"predictions":{"A":{"success":90}}}', candidates)


def test_development_suite_is_separate_and_non_promotion() -> None:
    fixtures = load_calibration_suite(SUITE)

    assert len(fixtures) == 8
    assert sum(fixture.completion_control for fixture in fixtures) == 2
    assert len({action.id for fixture in fixtures for action in fixture.candidates}) == 19
    assert verify_calibration_novelty(fixtures) == {
        "case_ids": [],
        "objectives": [],
        "action_ids": [],
        "hypothesis_statements": [],
    }


def test_calibration_evaluates_and_regrades_offline(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"

    report = evaluate_b4_calibration(
        SUITE, CONFIG, MODEL_CONFIG, report_path, adapter=FakeCalibratedModel()
    )
    verify_b4_calibration_report(load_b4_calibration_report(report_path))

    assert report.promotion_eligible is False
    assert report.schema_passed_cases == 8
    assert report.semantic_check_rate == 1.0
    assert report.b4_passed_cases == 8
    assert report.safety_passed_cases == 8
    assert report.completion_passed_cases == report.completion_cases == 2
    assert report.engineering_gate_passed


def test_calibration_refuses_overwrite(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    evaluate_b4_calibration(SUITE, CONFIG, MODEL_CONFIG, report_path, adapter=FakeCalibratedModel())

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        evaluate_b4_calibration(
            SUITE, CONFIG, MODEL_CONFIG, report_path, adapter=FakeCalibratedModel()
        )


def test_calibration_regrade_detects_fixture_change(tmp_path: Path) -> None:
    fixture = tmp_path / "suite.yaml"
    fixture.write_bytes(SUITE.read_bytes())
    report = evaluate_b4_calibration(
        fixture,
        CONFIG,
        MODEL_CONFIG,
        tmp_path / "report.json",
        adapter=FakeCalibratedModel(),
    )
    fixture.write_text(fixture.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="fixture hash mismatch"):
        verify_b4_calibration_report(report)


def test_calibration_loader_rejects_promotion_metadata(tmp_path: Path) -> None:
    document = yaml.safe_load(SUITE.read_text(encoding="utf-8"))
    document["promotion_eligible"] = True
    fixture = tmp_path / "suite.yaml"
    fixture.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata is invalid"):
        load_calibration_suite(fixture)

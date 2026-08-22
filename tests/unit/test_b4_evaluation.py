import json
from pathlib import Path

import pytest
import yaml

from aif_qwen_agent.b4_evaluation import (
    B4DevelopmentReport,
    evaluate_b4,
    load_b4_report,
    load_b4_suite,
    verify_b4_report,
)

SUITE = Path("evals/tasks/b4_dev/suite.yaml")
CONFIG = Path("configs/aif_b4_dev.yaml")


def test_real_b4_development_suite_passes_and_regrades(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"

    report = evaluate_b4(SUITE, CONFIG, report_path)
    verify_b4_report(load_b4_report(report_path))

    assert report.report_type == "development"
    assert report.promotion_eligible is False
    assert report.engineering_gate_passed
    assert report.passed_cases == len(report.cases) == 6
    assert report.passed_epistemic_cases == 3
    assert all(case.trace.evaluations for case in report.cases)
    assert all(
        evaluation.score is not None
        for case in report.cases
        for evaluation in case.trace.evaluations
        if evaluation.eligible
    )


def test_b4_suite_rejects_promotion_metadata(tmp_path: Path) -> None:
    document = yaml.safe_load(SUITE.read_text(encoding="utf-8"))
    document["promotion_eligible"] = True
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata is invalid"):
        load_b4_suite(path)


def test_b4_evaluation_refuses_to_overwrite_report(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    evaluate_b4(SUITE, CONFIG, report)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        evaluate_b4(SUITE, CONFIG, report)


def test_b4_regrade_rejects_fixture_tampering(tmp_path: Path) -> None:
    fixture = tmp_path / "suite.yaml"
    fixture.write_bytes(SUITE.read_bytes())
    report = evaluate_b4(fixture, CONFIG, tmp_path / "report.json")
    fixture.write_text(fixture.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="fixture hash mismatch"):
        verify_b4_report(report)


def test_b4_report_schema_rejects_false_gate_claim(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    evaluate_b4(SUITE, CONFIG, report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["engineering_gate_passed"] = False

    with pytest.raises(ValueError, match="gate does not match cases"):
        B4DevelopmentReport.model_validate(payload)

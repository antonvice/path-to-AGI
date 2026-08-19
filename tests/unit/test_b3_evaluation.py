import json
import sqlite3
from pathlib import Path

import pytest
import yaml

from aif_qwen_agent.b3_evaluation import (
    B3DevelopmentReport,
    evaluate_b3,
    load_b3_report,
    load_b3_suite,
    verify_b3_report,
)

SUITE = Path("evals/tasks/b3_dev/suite.yaml")


def test_real_b3_development_suite_passes_and_regrades(tmp_path: Path) -> None:
    database = tmp_path / "beliefs.db"
    report_path = tmp_path / "report.json"

    report = evaluate_b3(SUITE, database, report_path)
    verify_b3_report(load_b3_report(report_path))

    assert report.report_type == "development"
    assert report.promotion_eligible is False
    assert report.engineering_gate_passed
    assert report.passed_cases == len(report.cases) == 5
    assert all(case.passed for case in report.cases)


def test_b3_suite_rejects_promotion_metadata(tmp_path: Path) -> None:
    document = yaml.safe_load(SUITE.read_text(encoding="utf-8"))
    document["promotion_eligible"] = True
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata is invalid"):
        load_b3_suite(path)


def test_b3_evaluation_refuses_to_overwrite_artifacts(tmp_path: Path) -> None:
    database = tmp_path / "beliefs.db"
    report = tmp_path / "report.json"
    evaluate_b3(SUITE, database, report)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        evaluate_b3(SUITE, database, report)


def test_b3_regrade_rejects_database_tampering(tmp_path: Path) -> None:
    database = tmp_path / "beliefs.db"
    report_path = tmp_path / "report.json"
    report = evaluate_b3(SUITE, database, report_path)
    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM belief_revisions WHERE revision = 0")

    with pytest.raises(ValueError, match="database hash mismatch"):
        verify_b3_report(report)


def test_b3_report_schema_rejects_false_gate_claim(tmp_path: Path) -> None:
    database = tmp_path / "beliefs.db"
    report_path = tmp_path / "report.json"
    evaluate_b3(SUITE, database, report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["engineering_gate_passed"] = False

    with pytest.raises(ValueError, match="gate does not match cases"):
        B3DevelopmentReport.model_validate(payload)

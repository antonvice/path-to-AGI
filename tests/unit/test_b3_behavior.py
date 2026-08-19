import json
from pathlib import Path

import pytest
import yaml

from aif_qwen_agent.artifacts import sha256_file
from aif_qwen_agent.b3_behavior import (
    evaluate_b3_behavior,
    load_b3_behavior_report,
    load_b3_behavior_suite,
    load_b3_independent_report,
    run_b3_independent,
    verify_b3_behavior_report,
    verify_b3_freeze,
    verify_b3_independent_report,
)

DEV_SUITE = Path("evals/tasks/b3_behavior_dev/suite.yaml")


def test_real_b3_heldout_inventory_is_frozen_and_disjoint() -> None:
    heldout_path = Path("evals/tasks/b3h/suite.yaml")
    manifest = verify_b3_freeze(Path("evals/tasks/b3h/freeze.json"))
    development = load_b3_behavior_suite(DEV_SUITE)
    heldout = load_b3_behavior_suite(heldout_path)
    development_ids = {
        operation.observation.observation_id
        for case in development.cases
        for operation in case.operations
        if operation.observation is not None
    }
    heldout_ids = {
        operation.observation.observation_id
        for case in heldout.cases
        for operation in case.operations
        if operation.observation is not None
    }

    assert manifest["inference"] == "deterministic_no_model_v1"
    assert heldout.purpose == "held_out"
    assert heldout.promotion_eligible is True
    assert len(heldout.cases) == 7
    assert development_ids.isdisjoint(heldout_ids)


def test_b3_behavior_development_ablation_passes(tmp_path: Path) -> None:
    report = evaluate_b3_behavior(DEV_SUITE, tmp_path / "beliefs.db", tmp_path / "report.json")
    verify_b3_behavior_report(load_b3_behavior_report(tmp_path / "report.json"))

    assert report.engineering_gate_passed
    assert report.baseline_passed_cases == 2
    assert report.belief_passed_cases == 5
    assert report.quality_delta == pytest.approx(0.6)


def test_b3_heldout_behavior_requires_freeze(tmp_path: Path) -> None:
    document = yaml.safe_load(DEV_SUITE.read_text(encoding="utf-8"))
    document["purpose"] = "held_out"
    document["promotion_eligible"] = True
    suite = tmp_path / "suite.yaml"
    suite.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="requires a freeze"):
        evaluate_b3_behavior(suite, tmp_path / "beliefs.db", tmp_path / "report.json")


def test_b3_independent_synthetic_heldout_regrades(tmp_path: Path) -> None:
    document = yaml.safe_load(DEV_SUITE.read_text(encoding="utf-8"))
    document["purpose"] = "held_out"
    document["promotion_eligible"] = True
    suite = tmp_path / "suite.yaml"
    suite.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "milestone": "B3",
                "purpose": "held_out",
                "promotion_eligible": True,
                "files": {str(suite): sha256_file(suite)},
            }
        ),
        encoding="utf-8",
    )

    report = run_b3_independent(suite, freeze, tmp_path / "output", 3)
    verify_b3_independent_report(load_b3_independent_report(tmp_path / "output/report.json"))

    assert report.promotion_gate_passed
    assert report.process_count == 3
    assert len({process.process_id for process in report.processes}) == 3


def test_b3_behavior_regrade_rejects_reported_database_hash_tampering(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    evaluate_b3_behavior(DEV_SUITE, tmp_path / "beliefs.db", report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["database_sha256"] = "0" * 64
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        verify_b3_behavior_report(load_b3_behavior_report(report_path))

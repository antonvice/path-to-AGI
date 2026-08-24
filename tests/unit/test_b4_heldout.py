import json
from pathlib import Path

import pytest

from aif_qwen_agent.artifacts import sha256_file
from aif_qwen_agent.b4_heldout import (
    DEFAULT_AIF_CONFIG,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_SUITE,
    B4ProcessArtifact,
    build_b4_independent_report,
    create_b4h_freeze,
    decode_world_predictions,
    evaluate_b4_process,
    load_b4_process_report,
    load_b4h_suite,
    verify_b4_process_report,
    verify_b4h_freeze,
    verify_b4h_novelty,
)
from aif_qwen_agent.model_adapters.base import ChatMessage
from aif_qwen_agent.schemas import ActionCandidate, ActionKind, GenerationConfig, ModelResult


class FakeWorldModel:
    def render_messages(self, messages: list[ChatMessage]) -> str:
        return json.dumps(messages, ensure_ascii=False, separators=(",", ":"))

    def generate(self, rendered_prompt: str, config: GenerationConfig) -> ModelResult:
        messages = json.loads(rendered_prompt)
        payload = json.loads(messages[-1]["content"])
        aliases = list(payload["actions"])
        predictions = {
            alias: "9900000" if index == 0 else "5591000" if index == 1 else "9990009"
            for index, alias in enumerate(aliases)
        }
        text = json.dumps({"p": predictions}, separators=(",", ":"))
        return ModelResult(
            raw_text=text,
            text=text,
            input_tokens=100,
            output_tokens=20,
            load_seconds=0.1,
            generation_seconds=0.2,
            device="fake",
            stop_reason="eos",
        )


def _freeze(tmp_path: Path) -> Path:
    path = tmp_path / "freeze.json"
    create_b4h_freeze(path)
    return path


def test_heldout_inventory_is_unseen_and_balanced() -> None:
    suite = load_b4h_suite(DEFAULT_SUITE)

    assert len(suite.cases) == 15
    assert len({case.family for case in suite.cases}) == 5
    assert sum(case.uncertainty_sensitive for case in suite.cases) == 12
    assert verify_b4h_novelty(suite) == {
        "case_ids": [],
        "objectives": [],
        "action_ids": [],
        "hypothesis_statements": [],
    }


def test_compact_prediction_decoder_accepts_strings_and_lists() -> None:
    candidates = [
        ActionCandidate(id="first", kind=ActionKind.ANSWER),
        ActionCandidate(id="second", kind=ActionKind.VERIFY),
    ]

    codes, predictions = decode_world_predictions(
        '{"p":{"A":"9876543","B":[1,2,3,4,5,6,7]}}', candidates
    )

    assert codes == {"first": "9876543", "second": "1234567"}
    assert predictions["first"].success_probability == 1.0
    assert predictions["second"].operational_risk == pytest.approx(7 / 9)


def test_freeze_binds_unseen_suite_and_refuses_overwrite(tmp_path: Path) -> None:
    freeze = _freeze(tmp_path)
    manifest = verify_b4h_freeze(freeze)

    assert manifest["inference_status_at_freeze"] == "not_started"
    assert manifest["novelty_overlap"] == {
        "case_ids": [],
        "objectives": [],
        "action_ids": [],
        "hypothesis_statements": [],
    }
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        create_b4h_freeze(freeze)


def test_freeze_detects_bound_hash_tampering(tmp_path: Path) -> None:
    freeze = _freeze(tmp_path)
    payload = json.loads(freeze.read_text(encoding="utf-8"))
    payload["files"][str(DEFAULT_SUITE)] = "0" * 64
    freeze.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen B4 file hash mismatch"):
        verify_b4h_freeze(freeze)


def test_synthetic_process_regrades_offline(tmp_path: Path) -> None:
    freeze = _freeze(tmp_path)
    report_path = tmp_path / "process.json"
    report = evaluate_b4_process(
        DEFAULT_SUITE,
        freeze,
        DEFAULT_MODEL_CONFIG,
        DEFAULT_AIF_CONFIG,
        report_path,
        adapter=FakeWorldModel(),
        process_id=101,
    )

    verify_b4_process_report(load_b4_process_report(report_path))
    assert report.failed_cases == 0
    assert report.b3_passed_cases == 3
    assert report.b4_passed_cases == 12
    assert report.b3_unsupported_claims == 12
    assert report.b4_unsupported_claims == 0


def test_three_synthetic_processes_pass_promotion_contract(tmp_path: Path) -> None:
    freeze = _freeze(tmp_path)
    artifacts = []
    for index in range(1, 4):
        report_path = tmp_path / f"process-{index}.json"
        report = evaluate_b4_process(
            DEFAULT_SUITE,
            freeze,
            DEFAULT_MODEL_CONFIG,
            DEFAULT_AIF_CONFIG,
            report_path,
            adapter=FakeWorldModel(),
            process_id=100 + index,
        )
        artifacts.append(
            B4ProcessArtifact(
                process_index=index,
                process_id=100 + index,
                report_file=str(report_path),
                report_sha256=sha256_file(report_path),
                report=report,
            )
        )

    aggregate = build_b4_independent_report(artifacts, artifacts[0].report.started_at)

    assert aggregate.process_count == 3
    assert aggregate.reproducibility_gate_passed
    assert aggregate.family_improvements == 4
    assert aggregate.paired_ci_lower > 0.0
    assert aggregate.promotion_gate_passed

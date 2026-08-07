from pathlib import Path

from aif_qwen_agent.config import load_yaml
from aif_qwen_agent.schemas import AgentComparisonReport, AgentTrace, RunTrace

BASELINES = Path("evals/baselines")


def test_frozen_b1b_comparison_is_trace_backed_and_schema_valid() -> None:
    fixture = load_yaml(Path("evals/tasks/b1b/file_grounded.yaml"))["cases"][0]
    agent_trace = AgentTrace.model_validate_json(
        (BASELINES / "b1b_agent_mps.jsonl").read_text(encoding="utf-8")
    )
    baseline_trace = RunTrace.model_validate_json(
        (BASELINES / "b1b_b0_mps.jsonl").read_text(encoding="utf-8")
    )
    report = AgentComparisonReport.model_validate_json(
        (BASELINES / "b1b_comparison_mps.json").read_text(encoding="utf-8")
    )
    case = report.cases[0]

    assert case.expected == fixture["expected"]
    assert case.baseline_run_id == baseline_trace.run_id
    assert case.agent_run_id == agent_trace.run_id
    assert baseline_trace.result is not None
    assert case.baseline_actual == baseline_trace.result.text
    assert case.agent_actual == agent_trace.answer
    assert agent_trace.tool_trace is not None and agent_trace.tool_trace.verified
    assert case.evidence_sha256 == agent_trace.evidence_sha256
    assert report.gate_passed

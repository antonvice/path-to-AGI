from collections.abc import Sequence
from pathlib import Path

import pytest

from aif_qwen_agent.agent import AgentTraceStore, OneStepAgent
from aif_qwen_agent.artifacts import TraceStore
from aif_qwen_agent.b1_evaluation import (
    evaluate_b1,
    load_b1_fixtures,
    load_b1_report,
    verify_b1_report,
)
from aif_qwen_agent.baseline import BaselineRunner
from aif_qwen_agent.model_adapters.base import ChatMessage
from aif_qwen_agent.schemas import (
    GenerationConfig,
    ModelIdentity,
    ModelResult,
    ReadFilePolicy,
    Task,
)
from aif_qwen_agent.tools import ReadFileTool, ReadFileTraceStore

FIXTURES = Path("evals/tasks/b1c/suite.yaml")


class SharedAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def render_prompt(self, task: Task) -> str:
        return f"baseline::{task}"

    def render_messages(self, messages: Sequence[ChatMessage]) -> str:
        return "\n".join(f"{message['role']}::{message['content']}" for message in messages)

    def generate(self, rendered_prompt: str, config: GenerationConfig) -> ModelResult:
        self.calls += 1
        text = self._response(rendered_prompt)
        return ModelResult(
            raw_text=text,
            text=text,
            input_tokens=10,
            output_tokens=4,
            load_seconds=1.5 if self.calls == 1 else 0.0,
            generation_seconds=0.1,
            device="fake",
            stop_reason="eos",
        )

    @staticmethod
    def _response(prompt: str) -> str:
        if prompt.startswith("baseline::"):
            return "unknown without file access"
        if "Choose exactly one action" in prompt:
            if "model revision" in prompt:
                return '{"kind":"read_file","path":"configs/qwen3_8b.yaml"}'
            if "max_read_bytes" in prompt:
                return '{"kind":"read_file","path":"configs/policy.yaml"}'
            if "What backend" in prompt:
                return '{"kind":"read_file","path":"configs/logic.yaml"}'
            if "run_python" in prompt:
                return '{"kind":"run_python","code":"print(1)"}'
            if "../outside.txt" in prompt:
                return '{"kind":"read_file","path":"../outside.txt"}'
            if "does_not_exist" in prompt:
                return '{"kind":"read_file","path":"evals/tasks/b1c/does_not_exist.txt"}'
            if "max_bytes 1024" in prompt:
                return (
                    '{"kind":"read_file",'
                    '"path":"qwen3-8b-active-inference-agent-harness.md","max_bytes":1024}'
                )
        if "model revision" in prompt:
            return "b968826d9c46dd6066d109eabc6255188de91218"
        if "max_read_bytes" in prompt:
            return "131072"
        if "What backend" in prompt:
            return "python_predicates"
        raise AssertionError(f"unexpected prompt: {prompt}")


def runners(
    tmp_path: Path,
    adapter: SharedAdapter,
) -> tuple[BaselineRunner, OneStepAgent, TraceStore, AgentTraceStore]:
    model = ModelIdentity(
        repo_id="Qwen/Qwen3-8B",
        revision="b968826d9c46dd6066d109eabc6255188de91218",
        local_path=Path("models/Qwen3-8B"),
        backend="fake",
    )
    generation = GenerationConfig(max_new_tokens=128, temperature=0.0, seed=42)
    baseline_traces = TraceStore(tmp_path / "b0.jsonl")
    agent_traces = AgentTraceStore(tmp_path / "b1.jsonl")
    baseline = BaselineRunner(adapter, model, generation, baseline_traces)
    agent = OneStepAgent(
        adapter,
        model,
        generation,
        ReadFileTool(
            ReadFilePolicy(allowed_roots=[Path.cwd()], max_read_bytes=131_072),
            ReadFileTraceStore(tmp_path / "tools.jsonl"),
        ),
        agent_traces,
        max_proposal_attempts=2,
    )
    return baseline, agent, baseline_traces, agent_traces


def test_b1_suite_compares_grounded_cases_and_blocks_safety_cases(tmp_path: Path) -> None:
    adapter = SharedAdapter()
    baseline, agent, baseline_traces, agent_traces = runners(tmp_path, adapter)
    report_path = tmp_path / "report.json"

    report = evaluate_b1(baseline, agent, FIXTURES, report_path)
    verify_b1_report(
        load_b1_report(report_path),
        FIXTURES,
        baseline_traces,
        agent_traces,
    )

    assert report.grounded_cases == 3
    assert report.safety_cases == report.safety_passed_cases == 4
    assert report.baseline_passed_cases == 0
    assert report.agent_passed_cases == 3
    assert report.safety_violations == 0
    assert report.proposal_retries == 1
    assert report.model_load_seconds == 1.5
    assert report.baseline_input_tokens == 30
    assert report.agent_input_tokens == 110
    assert report.gate_passed
    assert adapter.calls == 14
    assert [case.rejection_code for case in report.cases[4:]] == [
        "outside_allowed_root",
        "not_found",
        "file_too_large",
    ]


def test_b1_offline_regrade_detects_tampered_case(tmp_path: Path) -> None:
    baseline, agent, baseline_traces, agent_traces = runners(tmp_path, SharedAdapter())
    report = evaluate_b1(baseline, agent, FIXTURES, tmp_path / "report.json")
    changed = report.cases[0].model_copy(update={"agent_actual": "tampered"})
    tampered = report.model_copy(update={"cases": [changed, *report.cases[1:]]})

    with pytest.raises(ValueError, match="saved B1 results do not match traces"):
        verify_b1_report(tampered, FIXTURES, baseline_traces, agent_traces)


def test_b1_fixture_inventory_is_frozen() -> None:
    fixtures = load_b1_fixtures(FIXTURES)

    assert sum(fixture.kind == "grounded" for fixture in fixtures) == 3
    assert sum(fixture.kind == "safety" for fixture in fixtures) == 4
    assert len({fixture.id for fixture in fixtures}) == 7


def test_frozen_real_b1c_report_regrades_offline() -> None:
    verify_b1_report(
        load_b1_report(Path("evals/baselines/b1c_report_mps.json")),
        FIXTURES,
        TraceStore(Path("evals/baselines/b1c_b0_suite_mps.jsonl")),
        AgentTraceStore(Path("evals/baselines/b1c_agent_suite_mps.jsonl")),
    )

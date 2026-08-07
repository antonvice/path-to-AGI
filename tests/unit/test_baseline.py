from pathlib import Path

from aif_qwen_agent.artifacts import TraceStore, sha256_text
from aif_qwen_agent.baseline import BaselineRunner
from aif_qwen_agent.schemas import (
    GenerationConfig,
    ModelIdentity,
    ModelResult,
    Task,
)


class FakeAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def render_prompt(self, task: Task) -> str:
        return f"frozen::{task.text}"

    def generate(self, rendered_prompt: str, config: GenerationConfig) -> ModelResult:
        self.calls += 1
        return ModelResult(
            raw_text="BASELINE_READY",
            text="BASELINE_READY",
            input_tokens=3,
            output_tokens=1,
            load_seconds=0.0,
            generation_seconds=0.01,
            device="fake",
            stop_reason="eos",
        )


def runner(path: Path, adapter: FakeAdapter) -> BaselineRunner:
    return BaselineRunner(
        adapter=adapter,
        model=ModelIdentity(
            repo_id="Qwen/Qwen3-8B",
            revision="b968826d9c46dd6066d109eabc6255188de91218",
            local_path=Path("models/Qwen3-8B"),
            backend="fake",
        ),
        generation=GenerationConfig(max_new_tokens=8, temperature=0.0, seed=42),
        traces=TraceStore(path),
    )


def test_run_persists_replayable_trace_without_second_model_call(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    trace_path = tmp_path / "runs.jsonl"
    trace = runner(trace_path, adapter).run(Task(id="fixture", text="respond"))

    replayed = TraceStore(trace_path).get(str(trace.run_id))

    assert adapter.calls == 1
    assert replayed == trace
    assert replayed.result is not None
    assert replayed.result.text == "BASELINE_READY"
    assert replayed.prompt_sha256 == sha256_text("frozen::respond")


def test_failed_model_call_is_also_traced(tmp_path: Path) -> None:
    class FailingAdapter(FakeAdapter):
        def generate(self, rendered_prompt: str, config: GenerationConfig) -> ModelResult:
            raise RuntimeError("model unavailable")

    trace_path = tmp_path / "runs.jsonl"
    trace = runner(trace_path, FailingAdapter()).run(Task(id="fixture", text="respond"))

    assert trace.status == "failed"
    assert trace.error == "RuntimeError: model unavailable"
    assert TraceStore(trace_path).get(str(trace.run_id)) == trace

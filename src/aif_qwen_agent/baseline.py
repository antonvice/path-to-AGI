from datetime import UTC, datetime
from uuid import uuid4

from aif_qwen_agent.artifacts import TraceStore, sha256_text
from aif_qwen_agent.model_adapters.base import ModelAdapter
from aif_qwen_agent.schemas import GenerationConfig, ModelIdentity, RunTrace, Task


class BaselineRunner:
    def __init__(
        self,
        adapter: ModelAdapter,
        model: ModelIdentity,
        generation: GenerationConfig,
        traces: TraceStore,
    ) -> None:
        self.adapter = adapter
        self.model = model
        self.generation = generation
        self.traces = traces

    def run(self, task: Task) -> RunTrace:
        started_at = datetime.now(UTC)
        rendered_prompt = ""
        result = None
        error = None
        try:
            rendered_prompt = self.adapter.render_prompt(task)
            result = self.adapter.generate(rendered_prompt, self.generation)
        except Exception as exception:  # noqa: BLE001 - failures must become trace records
            error = f"{type(exception).__name__}: {exception}"
        trace = RunTrace(
            run_id=uuid4(),
            started_at=started_at,
            finished_at=datetime.now(UTC),
            task=task,
            rendered_prompt=rendered_prompt,
            prompt_sha256=sha256_text(rendered_prompt),
            model=self.model,
            generation=self.generation,
            status="completed" if result is not None else "failed",
            result=result,
            error=error,
        )
        self.traces.append(trace)
        return trace

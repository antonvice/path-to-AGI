from typing import Protocol

from aif_qwen_agent.schemas import GenerationConfig, ModelResult, Task


class ModelAdapter(Protocol):
    def render_prompt(self, task: Task) -> str: ...

    def generate(self, rendered_prompt: str, config: GenerationConfig) -> ModelResult: ...

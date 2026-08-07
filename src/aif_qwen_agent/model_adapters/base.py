from collections.abc import Sequence
from typing import Literal, Protocol, TypedDict

from aif_qwen_agent.schemas import GenerationConfig, ModelResult, Task


class ChatMessage(TypedDict):
    role: Literal["system", "user"]
    content: str


class ModelAdapter(Protocol):
    def render_prompt(self, task: Task) -> str: ...

    def generate(self, rendered_prompt: str, config: GenerationConfig) -> ModelResult: ...


class AgentModelAdapter(Protocol):
    def render_messages(self, messages: Sequence[ChatMessage]) -> str: ...

    def generate(self, rendered_prompt: str, config: GenerationConfig) -> ModelResult: ...

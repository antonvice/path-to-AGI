import json
from collections.abc import Sequence
from typing import Any

import httpx

from aif_qwen_agent.model_adapters.base import ChatMessage
from aif_qwen_agent.model_adapters.transformers import SYSTEM_PROMPT
from aif_qwen_agent.schemas import GenerationConfig, ModelResult, Task


class OllamaAdapter:
    """Digest-verified Ollama chat adapter with native token and timing metrics."""

    def __init__(
        self,
        model: str,
        digest: str,
        endpoint: str = "http://127.0.0.1:11434",
        context_tokens: int = 32_768,
        enable_thinking: bool = False,
        keep_alive: str = "5m",
        timeout_seconds: float = 900.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.digest = digest
        self.context_tokens = context_tokens
        self.enable_thinking = enable_thinking
        self.keep_alive = keep_alive
        self.client = httpx.Client(
            base_url=endpoint.rstrip("/"),
            timeout=timeout_seconds,
            transport=transport,
        )
        self._verified = False

    def render_prompt(self, task: Task) -> str:
        return self.render_messages(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task.text},
            ]
        )

    def render_messages(self, messages: Sequence[ChatMessage]) -> str:
        return json.dumps(list(messages), ensure_ascii=False, separators=(",", ":"))

    def verify_model(self) -> None:
        response = self.client.get("/api/tags")
        response.raise_for_status()
        models = response.json().get("models", [])
        match = next(
            (item for item in models if item.get("name") == self.model),
            None,
        )
        if match is None:
            raise RuntimeError(f"configured Ollama model is unavailable: {self.model}")
        if match.get("digest") != self.digest:
            raise RuntimeError(
                f"Ollama model digest mismatch: expected {self.digest}, got {match.get('digest')}"
            )
        self._verified = True

    def generate(self, rendered_prompt: str, config: GenerationConfig) -> ModelResult:
        if not self._verified:
            self.verify_model()
        messages = json.loads(rendered_prompt)
        if not isinstance(messages, list):
            raise ValueError("rendered Ollama prompt must contain a message list")
        response = self.client.post(
            "/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "think": self.enable_thinking,
                "keep_alive": self.keep_alive,
                "options": {
                    "num_ctx": self.context_tokens,
                    "num_predict": config.max_new_tokens,
                    "temperature": config.temperature,
                    "seed": config.seed,
                },
            },
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        message = payload.get("message", {})
        text = str(message.get("content", "")).strip()
        done_reason = payload.get("done_reason")
        return ModelResult(
            raw_text=text,
            text=text,
            input_tokens=int(payload.get("prompt_eval_count", 0)),
            output_tokens=int(payload.get("eval_count", 0)),
            load_seconds=float(payload.get("load_duration", 0)) / 1_000_000_000,
            generation_seconds=float(payload.get("eval_duration", 0)) / 1_000_000_000,
            device="ollama",
            stop_reason=(
                "eos"
                if done_reason == "stop"
                else "max_tokens"
                if done_reason == "length"
                else "unknown"
            ),
        )

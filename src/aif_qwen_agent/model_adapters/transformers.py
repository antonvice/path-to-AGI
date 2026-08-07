from pathlib import Path
from time import perf_counter
from typing import Any

from aif_qwen_agent.schemas import GenerationConfig, ModelResult, Task

SYSTEM_PROMPT = (
    "You are the frozen B0 answer-only baseline. Answer the user's task directly. "
    "Do not claim to use tools or external evidence. State uncertainty when necessary."
)


class TransformersAdapter:
    """Lazy local Transformers adapter for the pinned Qwen checkpoint."""

    def __init__(
        self,
        model_path: Path,
        backend: str,
        dtype: str = "auto",
        enable_thinking: bool = False,
    ) -> None:
        self.model_path = model_path
        self.backend = backend
        self.dtype = dtype
        self.enable_thinking = enable_thinking
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._device: str | None = None
        self._load_seconds = 0.0
        self._load_reported = False

    def render_prompt(self, task: Task) -> str:
        self._load_tokenizer()
        prompt = self._tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task.text},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        if not isinstance(prompt, str):
            raise TypeError("tokenizer did not return a rendered string prompt")
        return prompt

    def generate(self, rendered_prompt: str, config: GenerationConfig) -> ModelResult:
        self._load_model()
        if self._device is None:
            raise RuntimeError("model device was not initialized")
        inputs = self._tokenizer(rendered_prompt, return_tensors="pt").to(self._device)
        self._torch.manual_seed(config.seed)
        generation_arguments: dict[str, Any] = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.temperature > 0.0,
        }
        if config.temperature > 0.0:
            generation_arguments["temperature"] = config.temperature

        started = perf_counter()
        with self._torch.inference_mode():
            generated = self._model.generate(**inputs, **generation_arguments)
        generation_seconds = perf_counter() - started

        prompt_length = int(inputs["input_ids"].shape[-1])
        output_ids = generated[0, prompt_length:]
        raw_text = self._tokenizer.decode(output_ids, skip_special_tokens=False)
        text = self._tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        output_tokens = int(output_ids.shape[-1])
        eos_token_id = self._tokenizer.eos_token_id
        stopped_on_eos = output_tokens > 0 and int(output_ids[-1]) == eos_token_id

        result = ModelResult(
            raw_text=raw_text,
            text=text,
            input_tokens=prompt_length,
            output_tokens=output_tokens,
            load_seconds=0.0 if self._load_reported else self._load_seconds,
            generation_seconds=generation_seconds,
            device=self._device,
            stop_reason="eos" if stopped_on_eos else "max_tokens",
        )
        self._load_reported = True
        return result

    def _load_tokenizer(self) -> None:
        if self._tokenizer is not None:
            return
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)

    def _load_model(self) -> None:
        if self._model is not None:
            return
        started = perf_counter()
        import torch
        from transformers import AutoModelForCausalLM

        device = self._resolve_device(torch)
        self._load_tokenizer()
        model: Any = AutoModelForCausalLM.from_pretrained(
            self.model_path, dtype=self.dtype, local_files_only=True
        )
        model.to(device)
        model.eval()
        self._torch = torch
        self._model = model
        self._device = device
        self._load_seconds = perf_counter() - started

    def _resolve_device(self, torch: Any) -> str:
        if self.backend == "transformers_mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("configured MPS backend is unavailable")
            return "mps"
        if self.backend == "transformers_cpu":
            return "cpu"
        raise ValueError(f"unsupported local backend: {self.backend}")

import json

import httpx
import pytest

from aif_qwen_agent.model_adapters.ollama import OllamaAdapter
from aif_qwen_agent.schemas import GenerationConfig, Task

MODEL = "orcarouter/Qwen3.8-27B-Uncensored:iq4_xs"
DIGEST = "84e6355d6764e264ccdfe486243821e7000eaff08827557af4e3dc537c772c2a"


def test_ollama_adapter_verifies_digest_and_preserves_native_metrics() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": MODEL, "digest": DIGEST}]})
        payload = json.loads(request.content)
        assert payload["model"] == MODEL
        assert payload["think"] is False
        assert payload["options"] == {
            "num_ctx": 32768,
            "num_predict": 48,
            "temperature": 0.0,
            "seed": 42,
        }
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "READY"},
                "done_reason": "stop",
                "prompt_eval_count": 17,
                "eval_count": 3,
                "load_duration": 2_500_000_000,
                "eval_duration": 750_000_000,
            },
        )

    adapter = OllamaAdapter(MODEL, DIGEST, transport=httpx.MockTransport(handler))
    rendered = adapter.render_prompt(Task(id="smoke", text="Respond READY"))
    result = adapter.generate(
        rendered,
        GenerationConfig(max_new_tokens=48, temperature=0.0, seed=42),
    )

    assert result.text == "READY"
    assert result.input_tokens == 17
    assert result.output_tokens == 3
    assert result.load_seconds == 2.5
    assert result.generation_seconds == 0.75
    assert result.device == "ollama"
    assert result.stop_reason == "eos"
    assert [request.url.path for request in requests] == ["/api/tags", "/api/chat"]


def test_ollama_adapter_rejects_digest_drift_before_generation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": MODEL, "digest": "0" * 64}]})

    adapter = OllamaAdapter(MODEL, DIGEST, transport=httpx.MockTransport(handler))

    with pytest.raises(RuntimeError, match="digest mismatch"):
        adapter.generate(
            adapter.render_prompt(Task(id="blocked", text="Do not run")),
            GenerationConfig(max_new_tokens=8, temperature=0.0, seed=42),
        )

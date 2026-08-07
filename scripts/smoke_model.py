"""Load the pinned local checkpoint and generate a tiny deterministic response."""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = Path("models/Qwen3-8B")


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype="auto", local_files_only=True
    ).to(device)
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Reply with exactly: READY"}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    ).to(device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    prompt_length = inputs["input_ids"].shape[-1]
    print(tokenizer.decode(generated[0, prompt_length:], skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()

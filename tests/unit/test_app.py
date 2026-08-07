from aif_qwen_agent.app import _terminal_safe_text


def test_terminal_output_escapes_control_sequences() -> None:
    assert _terminal_safe_text("safe\n\x1b]0;title\x07") == "safe\n\\x1b]0;title\\x07"

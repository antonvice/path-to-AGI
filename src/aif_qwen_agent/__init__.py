"""Qwen active-inference agent research harness."""

__version__ = "0.1.0"


def main() -> None:
    from aif_qwen_agent.app import main as app_main

    app_main()

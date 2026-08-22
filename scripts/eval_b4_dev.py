"""Run or verify the deterministic B4 development gate."""

import argparse
from pathlib import Path

from aif_qwen_agent.b4_evaluation import (
    evaluate_b4,
    load_b4_report,
    verify_b4_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("evaluate", "regrade"))
    parser.add_argument("--suite", type=Path, default=Path("evals/tasks/b4_dev/suite.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/aif_b4_dev.yaml"))
    parser.add_argument("--report", type=Path, default=Path("evals/development/b4_dev/report.json"))
    args = parser.parse_args()
    if args.command == "evaluate":
        result = evaluate_b4(args.suite, args.config, args.report)
    else:
        result = load_b4_report(args.report)
        verify_b4_report(result)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

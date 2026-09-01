"""Run or offline-regrade the development-only B4 world-model calibration suite."""

import argparse
from pathlib import Path

from aif_qwen_agent.b4_calibration import (
    evaluate_b4_calibration,
    load_b4_calibration_report,
    verify_b4_calibration_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("evaluate", "regrade"))
    parser.add_argument(
        "--suite", type=Path, default=Path("evals/tasks/b4_calibration_dev/suite.yaml")
    )
    parser.add_argument("--config", type=Path, default=Path("configs/aif_b4_calibration.yaml"))
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/qwen3_8_27b_b4_calibration.yaml")
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("evals/development/b4_calibration/report.json"),
    )
    args = parser.parse_args()
    if args.command == "evaluate":
        result = evaluate_b4_calibration(args.suite, args.config, args.model_config, args.report)
    else:
        result = load_b4_calibration_report(args.report)
        verify_b4_calibration_report(result)
    print(
        f"schema={result.schema_passed_cases}/{len(result.cases)} "
        f"semantics={result.semantic_passed_checks}/{result.semantic_checks} "
        f"B3={result.b3_passed_cases}/{len(result.cases)} "
        f"B4={result.b4_passed_cases}/{len(result.cases)} "
        f"gate={'PASS' if result.engineering_gate_passed else 'FAIL'}"
    )


if __name__ == "__main__":
    main()

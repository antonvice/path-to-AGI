"""Freeze, run, and offline-regrade the paired B4 held-out evaluation."""

import argparse
from pathlib import Path

from aif_qwen_agent.b4_heldout import (
    DEFAULT_AIF_CONFIG,
    DEFAULT_FREEZE,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_SUITE,
    create_b4h_freeze,
    evaluate_b4_process,
    load_b4_independent_report,
    run_b4_processes,
    verify_b4_independent_report,
    verify_b4h_freeze,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "verify-freeze", "process", "run", "regrade"))
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--aif-config", type=Path, default=DEFAULT_AIF_CONFIG)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/b4-independent"))
    parser.add_argument("--processes", type=int, default=3)
    args = parser.parse_args()
    if args.command == "freeze":
        result = create_b4h_freeze(args.freeze)
        print(f"frozen files={len(result['files'])} manifest={args.freeze}")
    elif args.command == "verify-freeze":
        result = verify_b4h_freeze(args.freeze)
        print(f"verified files={len(result['files'])} manifest={args.freeze}")
    elif args.command == "process":
        if args.report is None:
            parser.error("process requires --report")
        result = evaluate_b4_process(
            args.suite, args.freeze, args.model_config, args.aif_config, args.report
        )
        print(
            f"process={result.process_id} B3={result.b3_passed_cases}/{len(result.cases)} "
            f"B4={result.b4_passed_cases}/{len(result.cases)} failed={result.failed_cases}"
        )
    elif args.command == "run":
        result = run_b4_processes(
            args.suite,
            args.freeze,
            args.model_config,
            args.aif_config,
            args.output_dir,
            args.processes,
            status=print,
        )
        print(
            f"promotion={'PASS' if result.promotion_gate_passed else 'FAIL'} "
            f"delta={result.quality_delta:.1%} CI=[{result.paired_ci_lower:.1%},"
            f"{result.paired_ci_upper:.1%}] report={args.output_dir / 'report.json'}"
        )
    else:
        if args.report is None:
            parser.error("regrade requires --report")
        result = load_b4_independent_report(args.report)
        verify_b4_independent_report(result)
        print(
            f"verified report={result.report_id} "
            f"promotion={'PASS' if result.promotion_gate_passed else 'FAIL'}"
        )


if __name__ == "__main__":
    main()

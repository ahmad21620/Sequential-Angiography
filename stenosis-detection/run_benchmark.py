from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stenosis_detection.benchmarking import (
    BenchmarkIOError,
    FrameBenchmarkResult,
    MultiViewBenchmarkResult,
    TemporalBenchmarkResult,
    WeakLabelLoadError,
    run_frame_level_benchmark,
    run_multiview_level_benchmark,
    run_temporal_level_benchmark,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark existing stenosis pipeline JSON outputs against weak EHR labels.",
    )
    parser.add_argument(
        "--level",
        choices=["frame", "temporal", "multiview"],
        required=True,
        help="Benchmark level to run.",
    )
    parser.add_argument(
        "--results-root",
        required=True,
        help="Root containing benchmark input JSON outputs grouped by case.",
    )
    parser.add_argument(
        "--weak-labels",
        required=True,
        help="Path to EHR weak_labels.jsonl.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory where benchmark outputs should be saved.",
    )
    parser.add_argument(
        "--frame-min-degree",
        type=float,
        default=0.0,
        help="Minimum frame stenosis degree required to count a non-empty frame as positive.",
    )
    parser.add_argument(
        "--sequence-positive-ratio-threshold",
        type=float,
        default=0.25,
        help="Minimum positive-frame ratio required for sequence-level positive prediction.",
    )
    parser.add_argument(
        "--case-positive-ratio-threshold",
        type=float,
        default=0.25,
        help="Minimum positive frame/sequence ratio required for case-level positive prediction.",
    )
    parser.add_argument(
        "--temporal-min-degree",
        type=float,
        default=0.0,
        help="Minimum temporal median degree required to count a final lesion as positive.",
    )
    parser.add_argument(
        "--temporal-min-persistence-ratio",
        type=float,
        default=0.0,
        help="Minimum temporal persistence ratio required to count a final lesion as positive.",
    )
    parser.add_argument(
        "--multiview-min-confidence-score",
        type=float,
        default=0.0,
        help="Minimum multi-view confidence score required to count a final case lesion as positive.",
    )
    parser.add_argument(
        "--multiview-min-degree",
        type=float,
        default=0.0,
        help="Minimum multi-view median degree required to count a final case lesion as positive.",
    )
    parser.add_argument(
        "--require-distinct-supporting-view",
        action="store_true",
        help="Require at least one distinct supporting view for a multi-view positive prediction.",
    )
    parser.add_argument(
        "--write-threshold-sweep",
        action="store_true",
        help="Write threshold-sweep CSV and suggested operating-point JSON reports.",
    )
    parser.add_argument(
        "--include-unclear-labels",
        action="store_true",
        help="Record that unclear weak-label rows should be retained in outputs; they remain excluded from binary metrics.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.frame_min_degree < 0.0:
        parser.error("--frame-min-degree must be >= 0.0.")
    if not 0.0 <= args.sequence_positive_ratio_threshold <= 1.0:
        parser.error("--sequence-positive-ratio-threshold must be in the range [0.0, 1.0].")
    if not 0.0 <= args.case_positive_ratio_threshold <= 1.0:
        parser.error("--case-positive-ratio-threshold must be in the range [0.0, 1.0].")
    if args.temporal_min_degree < 0.0:
        parser.error("--temporal-min-degree must be >= 0.0.")
    if not 0.0 <= args.temporal_min_persistence_ratio <= 1.0:
        parser.error("--temporal-min-persistence-ratio must be in the range [0.0, 1.0].")
    if not 0.0 <= args.multiview_min_confidence_score <= 1.0:
        parser.error("--multiview-min-confidence-score must be in the range [0.0, 1.0].")
    if args.multiview_min_degree < 0.0:
        parser.error("--multiview-min-degree must be >= 0.0.")

    try:
        if args.level == "frame":
            result = run_frame_level_benchmark(
                results_root=Path(args.results_root),
                weak_labels_path=Path(args.weak_labels),
                output_root=Path(args.output_root),
                frame_min_degree=args.frame_min_degree,
                sequence_positive_ratio_threshold=args.sequence_positive_ratio_threshold,
                case_positive_ratio_threshold=args.case_positive_ratio_threshold,
                include_unclear_labels=args.include_unclear_labels,
                write_threshold_sweep_report=args.write_threshold_sweep,
            )
            _print_frame_summary(result)
            return 0
        if args.level == "temporal":
            result = run_temporal_level_benchmark(
                results_root=Path(args.results_root),
                weak_labels_path=Path(args.weak_labels),
                output_root=Path(args.output_root),
                temporal_min_degree=args.temporal_min_degree,
                temporal_min_persistence_ratio=args.temporal_min_persistence_ratio,
                case_positive_ratio_threshold=args.case_positive_ratio_threshold,
                include_unclear_labels=args.include_unclear_labels,
                write_threshold_sweep_report=args.write_threshold_sweep,
            )
            _print_temporal_summary(result)
            return 0
        if args.level == "multiview":
            result = run_multiview_level_benchmark(
                results_root=Path(args.results_root),
                weak_labels_path=Path(args.weak_labels),
                output_root=Path(args.output_root),
                multiview_min_confidence_score=args.multiview_min_confidence_score,
                multiview_min_degree=args.multiview_min_degree,
                require_distinct_supporting_view=args.require_distinct_supporting_view,
                include_unclear_labels=args.include_unclear_labels,
                write_threshold_sweep_report=args.write_threshold_sweep,
            )
            _print_multiview_summary(result)
            return 0
    except (FileNotFoundError, NotADirectoryError, BenchmarkIOError, WeakLabelLoadError, ValueError) as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1

    parser.error(f"Unsupported benchmark level: {args.level}")
    return 2


def _print_frame_summary(result: FrameBenchmarkResult) -> None:
    print(f"Frame rows: {len(result.frame_rows)}")
    print(f"Sequence rows: {len(result.sequence_rows)}")
    print(f"Case rows: {len(result.case_rows)}")
    print(f"Frame total evaluated: {result.frame_summary.total_evaluated}")
    print(f"Frame accuracy: {_format_metric(result.frame_summary.accuracy)}")
    for label, path in result.output_paths.to_dict().items():
        print(f"Saved {label}: {path}")
    _print_threshold_sweep_paths(result.threshold_sweep_paths)


def _print_temporal_summary(result: TemporalBenchmarkResult) -> None:
    print(f"Temporal sequence rows: {len(result.sequence_rows)}")
    print(f"Case rows: {len(result.case_rows)}")
    print(f"Temporal sequence total evaluated: {result.sequence_summary.total_evaluated}")
    print(f"Temporal sequence accuracy: {_format_metric(result.sequence_summary.accuracy)}")
    for label, path in result.output_paths.to_dict().items():
        print(f"Saved {label}: {path}")
    _print_threshold_sweep_paths(result.threshold_sweep_paths)


def _print_multiview_summary(result: MultiViewBenchmarkResult) -> None:
    print(f"Multi-view case rows: {len(result.case_rows)}")
    print(f"Multi-view case total evaluated: {result.case_summary.total_evaluated}")
    print(f"Multi-view case accuracy: {_format_metric(result.case_summary.accuracy)}")
    for label, path in result.output_paths.to_dict().items():
        print(f"Saved {label}: {path}")
    _print_threshold_sweep_paths(result.threshold_sweep_paths)


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _print_threshold_sweep_paths(paths: dict[str, Path]) -> None:
    for label, path in paths.items():
        print(f"Saved {label}: {path}")


if __name__ == "__main__":
    raise SystemExit(main())

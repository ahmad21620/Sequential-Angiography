from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stenosis_detection.benchmarking import (
    BenchmarkIOError,
    FrameBenchmarkResult,
    MultiViewBenchmarkResult,
    TemporalBenchmarkResult,
    WeakLabelLoadError,
    load_weak_labels_jsonl,
    run_frame_level_benchmark,
    run_multiview_level_benchmark,
    run_temporal_level_benchmark,
)


LEVELS = ("frame", "temporal", "multiview")
LEVEL_ROOTS = {
    "frame": "frame_results",
    "temporal": "temporal_results",
    "multiview": "multiview_results",
}
SUMMARY_FILENAMES = {
    "frame": "frame_summary.json",
    "temporal": "temporal_sequence_summary.json",
    "multiview": "multiview_case_summary.json",
}


@dataclass(frozen=True, slots=True)
class SweepBenchmarkJob:
    level: str
    results_root: Path
    output_root: Path
    frame_variant: str | None = None
    temporal_variant: str | None = None

    @property
    def label(self) -> str:
        if self.temporal_variant is not None:
            return f"{self.level}: {self.frame_variant}/{self.temporal_variant}"
        if self.frame_variant is not None:
            return f"{self.level}: {self.frame_variant}"
        return self.level


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark every variant in a frame/temporal/multiview parameter sweep tree.",
    )
    parser.add_argument(
        "--sweep-root",
        required=True,
        help="Root containing frame_results/, temporal_results/, and optionally multiview_results/.",
    )
    parser.add_argument(
        "--weak-labels",
        required=True,
        help="Path to EHR weak_labels.jsonl.",
    )
    parser.add_argument(
        "--output-root",
        help="Directory where sweep benchmark outputs will be written. Defaults to <sweep-root>/benchmark_results.",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        help=(
            "Benchmark levels to run: frame, temporal, multiview. "
            "Accepts space-separated or comma-separated values. Defaults to all levels present in --sweep-root."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a variant when its benchmark summary JSON already exists.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed variant instead of recording the failure and continuing.",
    )
    parser.add_argument("--frame-min-degree", type=float, default=0.0)
    parser.add_argument("--temporal-min-degree", type=float, default=0.0)
    parser.add_argument("--temporal-min-persistence-ratio", type=float, default=0.0)
    parser.add_argument("--multiview-min-confidence-score", type=float, default=0.0)
    parser.add_argument("--multiview-min-degree", type=float, default=0.0)
    parser.add_argument("--require-distinct-supporting-view", action="store_true")
    parser.add_argument("--write-threshold-sweep", action="store_true")
    parser.add_argument("--include-unclear-labels", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        summary = run_sweep_benchmark(
            sweep_root=args.sweep_root,
            weak_labels_path=args.weak_labels,
            output_root=args.output_root,
            levels=_resolve_levels(args.levels, Path(args.sweep_root)),
            skip_existing=args.skip_existing,
            fail_fast=args.fail_fast,
            frame_min_degree=args.frame_min_degree,
            temporal_min_degree=args.temporal_min_degree,
            temporal_min_persistence_ratio=args.temporal_min_persistence_ratio,
            multiview_min_confidence_score=args.multiview_min_confidence_score,
            multiview_min_degree=args.multiview_min_degree,
            require_distinct_supporting_view=args.require_distinct_supporting_view,
            write_threshold_sweep_report=args.write_threshold_sweep,
            include_unclear_labels=args.include_unclear_labels,
        )
    except (FileNotFoundError, NotADirectoryError, BenchmarkIOError, WeakLabelLoadError, ValueError, OSError) as exc:
        print(f"Sweep benchmark failed: {exc}", file=sys.stderr)
        return 2

    print(f"Sweep benchmark summary: {summary['summary_json']}")
    print(f"Completed jobs: {summary['completed_jobs']}")
    print(f"Skipped jobs: {summary['skipped_jobs']}")
    print(f"Failed jobs: {summary['failed_jobs']}")
    return 0 if summary["failed_jobs"] == 0 else 1


def run_sweep_benchmark(
    *,
    sweep_root: str | Path,
    weak_labels_path: str | Path,
    output_root: str | Path | None = None,
    levels: tuple[str, ...] | None = None,
    skip_existing: bool = False,
    fail_fast: bool = False,
    frame_min_degree: float = 0.0,
    temporal_min_degree: float = 0.0,
    temporal_min_persistence_ratio: float = 0.0,
    multiview_min_confidence_score: float = 0.0,
    multiview_min_degree: float = 0.0,
    require_distinct_supporting_view: bool = False,
    write_threshold_sweep_report: bool = False,
    include_unclear_labels: bool = False,
) -> dict[str, Any]:
    resolved_sweep_root = Path(sweep_root)
    _validate_sweep_root(resolved_sweep_root)
    resolved_weak_labels_path = Path(weak_labels_path)
    load_weak_labels_jsonl(resolved_weak_labels_path)

    resolved_levels = levels or _levels_present_under(resolved_sweep_root)
    if not resolved_levels:
        raise FileNotFoundError(
            f"No benchmarkable sweep result roots found under: {resolved_sweep_root}. "
            "Expected frame_results/, temporal_results/, or multiview_results/."
        )

    resolved_output_root = Path(output_root) if output_root is not None else resolved_sweep_root / "benchmark_results"
    jobs = _discover_jobs(resolved_sweep_root, resolved_output_root, resolved_levels)
    if not jobs:
        raise FileNotFoundError(f"No benchmark variant folders were found for levels: {', '.join(resolved_levels)}")

    completed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for index, job in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] Benchmarking {job.label}")
        if skip_existing and _expected_summary_path(job).is_file():
            skipped.append(_job_payload(job, status="skipped"))
            print(f"Skipped existing benchmark: {_expected_summary_path(job)}")
            continue

        try:
            result = _run_job(
                job,
                weak_labels_path=resolved_weak_labels_path,
                frame_min_degree=frame_min_degree,
                temporal_min_degree=temporal_min_degree,
                temporal_min_persistence_ratio=temporal_min_persistence_ratio,
                multiview_min_confidence_score=multiview_min_confidence_score,
                multiview_min_degree=multiview_min_degree,
                require_distinct_supporting_view=require_distinct_supporting_view,
                write_threshold_sweep_report=write_threshold_sweep_report,
                include_unclear_labels=include_unclear_labels,
            )
        except (FileNotFoundError, NotADirectoryError, BenchmarkIOError, WeakLabelLoadError, ValueError, OSError, json.JSONDecodeError) as exc:
            failure = {**_job_payload(job, status="failed"), "error": str(exc)}
            failures.append(failure)
            print(f"Failed {job.label}: {exc}", file=sys.stderr)
            if fail_fast:
                break
            continue

        completed.append({**_job_payload(job, status="completed"), **_result_payload(job.level, result)})

    summary_path = resolved_output_root / "sweep_benchmark_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "sweep_root": str(resolved_sweep_root),
        "weak_labels": str(resolved_weak_labels_path),
        "output_root": str(resolved_output_root),
        "levels": list(resolved_levels),
        "config": {
            "skip_existing": skip_existing,
            "fail_fast": fail_fast,
            "frame_min_degree": float(frame_min_degree),
            "temporal_min_degree": float(temporal_min_degree),
            "temporal_min_persistence_ratio": float(temporal_min_persistence_ratio),
            "multiview_min_confidence_score": float(multiview_min_confidence_score),
            "multiview_min_degree": float(multiview_min_degree),
            "require_distinct_supporting_view": require_distinct_supporting_view,
            "write_threshold_sweep_report": write_threshold_sweep_report,
            "include_unclear_labels": include_unclear_labels,
        },
        "total_jobs": len(jobs),
        "completed_jobs": len(completed),
        "skipped_jobs": len(skipped),
        "failed_jobs": len(failures),
        "completed": completed,
        "skipped": skipped,
        "failures": failures,
        "summary_json": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _run_job(
    job: SweepBenchmarkJob,
    *,
    weak_labels_path: Path,
    frame_min_degree: float,
    temporal_min_degree: float,
    temporal_min_persistence_ratio: float,
    multiview_min_confidence_score: float,
    multiview_min_degree: float,
    require_distinct_supporting_view: bool,
    write_threshold_sweep_report: bool,
    include_unclear_labels: bool,
) -> FrameBenchmarkResult | TemporalBenchmarkResult | MultiViewBenchmarkResult:
    if job.level == "frame":
        return run_frame_level_benchmark(
            results_root=job.results_root,
            weak_labels_path=weak_labels_path,
            output_root=job.output_root,
            frame_min_degree=frame_min_degree,
            include_unclear_labels=include_unclear_labels,
            write_threshold_sweep_report=write_threshold_sweep_report,
        )
    if job.level == "temporal":
        return run_temporal_level_benchmark(
            results_root=job.results_root,
            weak_labels_path=weak_labels_path,
            output_root=job.output_root,
            temporal_min_degree=temporal_min_degree,
            temporal_min_persistence_ratio=temporal_min_persistence_ratio,
            include_unclear_labels=include_unclear_labels,
            write_threshold_sweep_report=write_threshold_sweep_report,
        )
    if job.level == "multiview":
        return run_multiview_level_benchmark(
            results_root=job.results_root,
            weak_labels_path=weak_labels_path,
            output_root=job.output_root,
            multiview_min_confidence_score=multiview_min_confidence_score,
            multiview_min_degree=multiview_min_degree,
            require_distinct_supporting_view=require_distinct_supporting_view,
            include_unclear_labels=include_unclear_labels,
            write_threshold_sweep_report=write_threshold_sweep_report,
        )
    raise ValueError(f"Unsupported benchmark level: {job.level}")


def _discover_jobs(sweep_root: Path, output_root: Path, levels: tuple[str, ...]) -> list[SweepBenchmarkJob]:
    jobs: list[SweepBenchmarkJob] = []
    for level in levels:
        level_root = sweep_root / LEVEL_ROOTS[level]
        if not level_root.exists():
            raise FileNotFoundError(f"Requested {level} benchmark root does not exist: {level_root}")
        if not level_root.is_dir():
            raise NotADirectoryError(f"Requested {level} benchmark root is not a directory: {level_root}")

        if level == "frame":
            frame_variants = _child_dirs(level_root)
            if not frame_variants:
                raise FileNotFoundError(f"No frame variant directories found under: {level_root}")
            for frame_variant_root in frame_variants:
                jobs.append(
                    SweepBenchmarkJob(
                        level="frame",
                        results_root=frame_variant_root,
                        output_root=output_root / "frame" / frame_variant_root.name,
                        frame_variant=frame_variant_root.name,
                    )
                )
            continue

        frame_variant_roots = _child_dirs(level_root)
        if not frame_variant_roots:
            raise FileNotFoundError(f"No frame variant directories found under: {level_root}")
        for frame_variant_root in frame_variant_roots:
            temporal_variant_roots = _child_dirs(frame_variant_root)
            if not temporal_variant_roots:
                continue
            for temporal_variant_root in temporal_variant_roots:
                jobs.append(
                    SweepBenchmarkJob(
                        level=level,
                        results_root=temporal_variant_root,
                        output_root=output_root / level / frame_variant_root.name / temporal_variant_root.name,
                        frame_variant=frame_variant_root.name,
                        temporal_variant=temporal_variant_root.name,
                    )
                )

    return jobs


def _result_payload(
    level: str,
    result: FrameBenchmarkResult | TemporalBenchmarkResult | MultiViewBenchmarkResult,
) -> dict[str, Any]:
    if level == "frame":
        assert isinstance(result, FrameBenchmarkResult)
        return {
            "row_count": len(result.frame_rows),
            "summary": result.frame_summary.to_dict(),
            "output_paths": result.output_paths.to_dict(),
            "threshold_sweep_paths": _string_path_dict(result.threshold_sweep_paths),
        }
    if level == "temporal":
        assert isinstance(result, TemporalBenchmarkResult)
        return {
            "row_count": len(result.sequence_rows),
            "summary": result.sequence_summary.to_dict(),
            "output_paths": result.output_paths.to_dict(),
            "threshold_sweep_paths": _string_path_dict(result.threshold_sweep_paths),
        }
    assert isinstance(result, MultiViewBenchmarkResult)
    return {
        "row_count": len(result.case_rows),
        "summary": result.case_summary.to_dict(),
        "output_paths": result.output_paths.to_dict(),
        "threshold_sweep_paths": _string_path_dict(result.threshold_sweep_paths),
    }


def _job_payload(job: SweepBenchmarkJob, *, status: str) -> dict[str, Any]:
    return {
        "status": status,
        "level": job.level,
        "frame_variant": job.frame_variant,
        "temporal_variant": job.temporal_variant,
        "results_root": str(job.results_root),
        "output_root": str(job.output_root),
    }


def _expected_summary_path(job: SweepBenchmarkJob) -> Path:
    return job.output_root / SUMMARY_FILENAMES[job.level]


def _resolve_levels(raw_levels: list[str] | None, sweep_root: Path) -> tuple[str, ...] | None:
    if raw_levels is None:
        return None

    levels: list[str] = []
    for raw_value in raw_levels:
        for item in raw_value.split(","):
            level = item.strip().lower()
            if not level:
                continue
            if level not in LEVELS:
                raise ValueError(f"Unsupported level {level!r}. Expected one of: {', '.join(LEVELS)}")
            if level not in levels:
                levels.append(level)

    if not levels:
        raise ValueError("--levels must include at least one level.")
    _ = sweep_root
    return tuple(levels)


def _levels_present_under(sweep_root: Path) -> tuple[str, ...]:
    return tuple(level for level in LEVELS if (sweep_root / LEVEL_ROOTS[level]).is_dir())


def _validate_sweep_root(sweep_root: Path) -> None:
    if not sweep_root.exists():
        raise FileNotFoundError(f"Sweep root does not exist: {sweep_root}")
    if not sweep_root.is_dir():
        raise NotADirectoryError(f"Sweep root is not a directory: {sweep_root}")


def _child_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_dir())


def _string_path_dict(paths: dict[str, Path]) -> dict[str, str]:
    return {key: str(path) for key, path in paths.items()}


if __name__ == "__main__":
    raise SystemExit(main())

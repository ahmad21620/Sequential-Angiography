from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stenosis_detection import (
    DISTINCT_VIEW_ANGLE_DISTANCE_DEGREES,
    DUPLICATE_VIEW_ANGLE_DISTANCE_DEGREES,
    HIGH_CONFIDENCE_THRESHOLD,
    LoadedMultiViewCase,
    LoadedMultiViewView,
    MEDIUM_CONFIDENCE_THRESHOLD,
    SUPPORT_SCORE_SCALE,
    MultiViewCaseResult,
    MultiViewFusionConfig,
    MultiViewLoadError,
    build_multiview_visualization_paths,
    load_multiview_case,
    run_multiview_fusion,
    save_multiview_case_result,
    save_multiview_visualization_outputs,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - exercised only when tqdm is absent.
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable=None, *, total=None, desc=None, unit=None, dynamic_ncols=None):
            self.iterable = iterable
            self.total = total
            self.desc = desc or "Progress"
            self.unit = unit or "item"
            self.count = 0
            print(f"{self.desc}: 0/{self.total if self.total is not None else '?'} {self.unit}")

        def __iter__(self):
            if self.iterable is None:
                return iter(())
            for item in self.iterable:
                yield item
                self.update(1)

        def update(self, n=1):
            self.count += n

        def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
            return None

        def write(self, message):
            print(message)

        def close(self):
            print(f"{self.desc}: {self.count}/{self.total if self.total is not None else '?'} {self.unit}")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()


@dataclass(frozen=True, slots=True)
class MultiViewSweepVariant:
    frame_variant: str
    temporal_variant: str
    temporal_results_root: Path
    output_root: Path


@dataclass(frozen=True, slots=True)
class MultiViewSweepJob:
    frame_variant: str
    temporal_variant: str
    case_input_path: Path
    case_root_tree: Path
    temporal_results_root: Path
    output_path: Path
    config: MultiViewFusionConfig
    split_by_coronary_side: bool
    write_images: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run multi-view fusion over every frame/temporal variant in a sweep tree.",
    )
    parser.add_argument(
        "--sweep-root",
        required=True,
        help="Root containing temporal_results/ and where multiview_results/ will be written by default.",
    )
    parser.add_argument(
        "--case-root-tree",
        required=True,
        help="Root directory containing one or more case folders with views.json.",
    )
    parser.add_argument(
        "--temporal-results-root",
        help="Optional temporal sweep root. Defaults to <sweep-root>/temporal_results.",
    )
    parser.add_argument(
        "--output-root",
        help="Optional multi-view sweep output root. Defaults to <sweep-root>/multiview_results.",
    )
    parser.add_argument(
        "--duplicate-angle-distance",
        type=float,
        default=DUPLICATE_VIEW_ANGLE_DISTANCE_DEGREES,
        help="Angle distance threshold, in degrees, below which two views are treated as duplicate evidence.",
    )
    parser.add_argument(
        "--distinct-angle-distance",
        type=float,
        default=DISTINCT_VIEW_ANGLE_DISTANCE_DEGREES,
        help="Angle distance threshold, in degrees, at which support is treated as fully distinct.",
    )
    parser.add_argument(
        "--support-score-scale",
        type=float,
        default=SUPPORT_SCORE_SCALE,
        help="Scale factor applied to cross-view support from other views.",
    )
    parser.add_argument(
        "--medium-confidence-threshold",
        type=float,
        default=MEDIUM_CONFIDENCE_THRESHOLD,
        help="Minimum confidence score labeled as medium.",
    )
    parser.add_argument(
        "--high-confidence-threshold",
        type=float,
        default=HIGH_CONFIDENCE_THRESHOLD,
        help="Minimum confidence score labeled as high.",
    )
    parser.add_argument(
        "--view-diversity-mode",
        choices=["angle", "projection_group", "auto"],
        default="angle",
        help="Cross-view diversity mode. Use projection_group or auto for CADICA projection groups.",
    )
    parser.add_argument(
        "--split-by-coronary-side",
        action="store_true",
        help="Run separate left/right fusions using coronary_side metadata and report unknown-side views.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes for case jobs across the whole sweep. Use 0 for all CPU cores. Default: 1.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip cases whose expected output files already exist and are non-empty.",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Write multi-view JSON outputs but skip summary/support-matrix PNG images.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.duplicate_angle_distance <= 0.0:
        parser.error("--duplicate-angle-distance must be positive.")
    if args.distinct_angle_distance <= args.duplicate_angle_distance:
        parser.error("--distinct-angle-distance must be greater than --duplicate-angle-distance.")
    if args.support_score_scale < 0.0:
        parser.error("--support-score-scale must be >= 0.0.")
    if not 0.0 <= args.medium_confidence_threshold <= 1.0:
        parser.error("--medium-confidence-threshold must be in the range [0.0, 1.0].")
    if not 0.0 <= args.high_confidence_threshold <= 1.0:
        parser.error("--high-confidence-threshold must be in the range [0.0, 1.0].")
    if args.medium_confidence_threshold > args.high_confidence_threshold:
        parser.error("--medium-confidence-threshold must be <= --high-confidence-threshold.")
    if args.workers < 0:
        parser.error("--workers must be 0 or greater.")

    try:
        summary = run_multiview_sweep(
            sweep_root=Path(args.sweep_root),
            case_root_tree=Path(args.case_root_tree),
            temporal_results_root=None if args.temporal_results_root is None else Path(args.temporal_results_root),
            output_root=None if args.output_root is None else Path(args.output_root),
            config=_build_fusion_config(args),
            split_by_coronary_side=args.split_by_coronary_side,
            workers=args.workers,
            skip_existing=args.skip_existing,
            write_images=not args.no_images,
        )
    except (FileNotFoundError, NotADirectoryError, MultiViewLoadError, ValueError, OSError) as exc:
        print(f"Multi-view sweep failed: {exc}", file=sys.stderr)
        return 1

    print(f"Processed cases: {summary['processed_cases']}")
    print(f"Skipped cases: {summary['skipped_cases']}")
    print(f"Failed cases: {summary['failed_cases']}")
    print(f"Summary: {summary['summary_json']}")
    return 0 if summary["failed_cases"] == 0 else 1


def run_multiview_sweep(
    *,
    sweep_root: Path,
    case_root_tree: Path,
    temporal_results_root: Path | None = None,
    output_root: Path | None = None,
    config: MultiViewFusionConfig | None = None,
    split_by_coronary_side: bool = False,
    workers: int = 1,
    skip_existing: bool = False,
    write_images: bool = True,
) -> dict[str, Any]:
    resolved_sweep_root = Path(sweep_root)
    resolved_temporal_results_root = temporal_results_root or resolved_sweep_root / "temporal_results"
    resolved_output_root = output_root or resolved_sweep_root / "multiview_results"
    resolved_case_root_tree = Path(case_root_tree)
    resolved_config = MultiViewFusionConfig() if config is None else config
    worker_count = _resolve_worker_count(workers)

    variants = _discover_sweep_variants(resolved_temporal_results_root, resolved_output_root)
    case_inputs = _discover_case_input_paths(resolved_case_root_tree)
    total_jobs = len(variants) * len(case_inputs)
    if total_jobs == 0:
        raise FileNotFoundError("No multi-view sweep jobs were found.")

    processed = 0
    skipped = 0
    failures: list[dict[str, str]] = []
    pending_jobs: list[MultiViewSweepJob] = []

    with tqdm(total=total_jobs, desc="Sweeping multi-view fusion", unit="case", dynamic_ncols=True) as progress:
        for variant in variants:
            for case_input_path in case_inputs:
                output_path = _multiview_output_path(
                    case_input_path,
                    case_root_tree=resolved_case_root_tree,
                    output_root=variant.output_root,
                )
                if skip_existing and _is_case_output_complete(
                    output_path,
                    split_by_coronary_side=split_by_coronary_side,
                    write_images=write_images,
                ):
                    skipped += 1
                    progress.update(1)
                    _set_progress(progress, processed=processed, skipped=skipped, failed=len(failures))
                    continue

                pending_jobs.append(
                    MultiViewSweepJob(
                        frame_variant=variant.frame_variant,
                        temporal_variant=variant.temporal_variant,
                        case_input_path=case_input_path,
                        case_root_tree=resolved_case_root_tree,
                        temporal_results_root=variant.temporal_results_root,
                        output_path=output_path,
                        config=resolved_config,
                        split_by_coronary_side=split_by_coronary_side,
                        write_images=write_images,
                    )
                )

        if worker_count == 1 or len(pending_jobs) <= 1:
            for job in pending_jobs:
                try:
                    _run_sweep_job(job)
                    processed += 1
                except (FileNotFoundError, NotADirectoryError, MultiViewLoadError, ValueError, OSError) as exc:
                    failures.append(_failure_payload(job, exc))
                    progress.write(f"Failed case '{job.case_input_path}': {exc}")
                progress.update(1)
                _set_progress(progress, processed=processed, skipped=skipped, failed=len(failures))
        elif pending_jobs:
            active_workers = min(worker_count, len(pending_jobs))
            progress.write(f"Running multi-view sweep with {active_workers} workers.")
            with ProcessPoolExecutor(max_workers=active_workers) as executor:
                future_to_job = {executor.submit(_run_sweep_job, job): job for job in pending_jobs}
                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        future.result()
                        processed += 1
                    except Exception as exc:
                        failures.append(_failure_payload(job, exc))
                        progress.write(f"Failed case '{job.case_input_path}': {exc}")
                    progress.update(1)
                    _set_progress(progress, processed=processed, skipped=skipped, failed=len(failures))

    summary_path = resolved_output_root / "multiview_sweep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "sweep_root": str(resolved_sweep_root),
        "temporal_results_root": str(resolved_temporal_results_root),
        "output_root": str(resolved_output_root),
        "case_root_tree": str(resolved_case_root_tree),
        "config": {
            "multiview_config": resolved_config.to_dict(),
            "split_by_coronary_side": split_by_coronary_side,
            "workers": worker_count,
            "skip_existing": skip_existing,
            "write_images": write_images,
        },
        "frame_temporal_variants": len(variants),
        "case_count": len(case_inputs),
        "total_jobs": total_jobs,
        "processed_cases": processed,
        "skipped_cases": skipped,
        "failed_cases": len(failures),
        "failures": failures,
        "summary_json": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _build_fusion_config(args: argparse.Namespace) -> MultiViewFusionConfig:
    return MultiViewFusionConfig(
        duplicate_view_angle_distance_degrees=args.duplicate_angle_distance,
        distinct_view_angle_distance_degrees=args.distinct_angle_distance,
        support_score_scale=args.support_score_scale,
        medium_confidence_threshold=args.medium_confidence_threshold,
        high_confidence_threshold=args.high_confidence_threshold,
        view_diversity_mode=args.view_diversity_mode,
    )


def _discover_sweep_variants(temporal_results_root: Path, output_root: Path) -> list[MultiViewSweepVariant]:
    if not temporal_results_root.exists():
        raise FileNotFoundError(f"Temporal sweep root does not exist: {temporal_results_root}")
    if not temporal_results_root.is_dir():
        raise NotADirectoryError(f"Temporal sweep root is not a directory: {temporal_results_root}")

    variants: list[MultiViewSweepVariant] = []
    for frame_variant_root in _child_dirs(temporal_results_root):
        for temporal_variant_root in _child_dirs(frame_variant_root):
            variants.append(
                MultiViewSweepVariant(
                    frame_variant=frame_variant_root.name,
                    temporal_variant=temporal_variant_root.name,
                    temporal_results_root=temporal_variant_root,
                    output_root=output_root / frame_variant_root.name / temporal_variant_root.name,
                )
            )
    if not variants:
        raise FileNotFoundError(
            f"No temporal variant directories found under: {temporal_results_root}. "
            "Expected temporal_results/<frame_variant>/<temporal_variant>/."
        )
    return variants


def _discover_case_input_paths(case_root_tree: Path) -> list[Path]:
    if not case_root_tree.exists():
        raise FileNotFoundError(f"Case root tree does not exist: {case_root_tree}")
    if not case_root_tree.is_dir():
        raise NotADirectoryError(f"Case root tree is not a directory: {case_root_tree}")

    case_inputs = sorted(path for path in case_root_tree.rglob("views.json") if path.is_file())
    if not case_inputs:
        raise FileNotFoundError(f"No views.json files were found under: {case_root_tree}")
    return case_inputs


def _run_sweep_job(job: MultiViewSweepJob) -> Path:
    multiview_case = load_multiview_case(
        job.case_input_path,
        temporal_results_root=job.temporal_results_root,
        case_root_tree=job.case_root_tree,
    )
    if job.split_by_coronary_side:
        _save_split_case(multiview_case, job.output_path, job.config)
        return job.output_path

    case_result = run_multiview_fusion(multiview_case, config=job.config)
    saved_path = save_multiview_case_result(case_result, job.output_path)
    if job.write_images:
        save_multiview_visualization_outputs(case_result, saved_path)
    return saved_path


def _save_split_case(
    multiview_case: LoadedMultiViewCase,
    output_path: Path,
    config: MultiViewFusionConfig,
) -> Path:
    side_groups = _split_views_by_coronary_side(multiview_case)
    side_results: dict[str, object] = {}
    skipped_sides: list[str] = []
    for side in ("left", "right"):
        side_views = side_groups[side]
        if not side_views:
            skipped_sides.append(side)
            continue
        side_case = LoadedMultiViewCase(
            case_id=f"{multiview_case.case_id}:{side}",
            views=side_views,
        )
        side_results[side] = run_multiview_fusion(side_case, config=config).to_dict()

    payload: dict[str, object] = {
        "case_id": multiview_case.case_id,
        "split_by_coronary_side": True,
        "view_diversity_mode": config.view_diversity_mode,
        "side_results": side_results,
        "unknown_views": [view.view_input.to_dict() for view in side_groups["unknown"]],
        "skipped_sides": skipped_sides,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def _split_views_by_coronary_side(multiview_case: LoadedMultiViewCase) -> dict[str, list[LoadedMultiViewView]]:
    groups: dict[str, list[LoadedMultiViewView]] = {"left": [], "right": [], "unknown": []}
    for view in multiview_case.views:
        side = (view.view_input.coronary_side or "unknown").strip().lower()
        if side not in {"left", "right"}:
            side = "unknown"
        groups[side].append(view)
    return groups


def _multiview_output_path(case_input_path: Path, *, case_root_tree: Path, output_root: Path) -> Path:
    relative_case_dir = case_input_path.parent.resolve().relative_to(case_root_tree.resolve())
    return output_root / relative_case_dir / "case_multiview_fusion.json"


def _is_case_output_complete(output_path: Path, *, split_by_coronary_side: bool, write_images: bool) -> bool:
    expected_paths = [output_path]
    if write_images and not split_by_coronary_side:
        visualization_paths = build_multiview_visualization_paths(output_path)
        expected_paths.extend(
            [
                visualization_paths["summary_png"],
                visualization_paths["support_matrix_png"],
            ]
        )
    return all(path.is_file() and path.stat().st_size > 0 for path in expected_paths)


def _failure_payload(job: MultiViewSweepJob, exc: Exception) -> dict[str, str]:
    return {
        "frame_variant": job.frame_variant,
        "temporal_variant": job.temporal_variant,
        "case_input_path": str(job.case_input_path),
        "output_path": str(job.output_path),
        "error": str(exc),
    }


def _set_progress(progress, *, processed: int, skipped: int, failed: int) -> None:
    progress.set_postfix(processed=processed, skipped=skipped, failed=failed)


def _resolve_worker_count(workers: int) -> int:
    if workers < 0:
        raise ValueError("workers must be 0 or greater.")
    if workers == 0:
        return max(1, os.cpu_count() or 1)
    return workers


def _child_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_dir())


if __name__ == "__main__":
    raise SystemExit(main())

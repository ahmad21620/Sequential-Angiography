from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - exercised only when tqdm is absent.
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable: Iterable[Any] | None = None, **_kwargs: Any) -> None:
            self.iterable = iterable

        def __enter__(self) -> "tqdm":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def __iter__(self) -> Iterable[Any]:
            return iter(()) if self.iterable is None else iter(self.iterable)

        def update(self, _count: int = 1) -> None:
            return None

from .benchmark import (
    CADICA_SUPERVISED_NOTE,
    CADICA_PATIENT_ID_RE,
    CADICA_VIDEO_ID_RE,
    FRAME_RESULT_SUFFIX,
    MULTIVIEW_PATIENT_ROW_FIELDS,
    MULTIVIEW_SIDE_ROW_FIELDS,
    MULTIVIEW_RESULT_FILENAME,
    MultiViewPrediction,
    TEMPORAL_RESULT_FILENAME,
    VIDEO_PREDICTION_SOURCES,
    _build_patient_rows,
    _build_summary,
    _build_video_rows,
    _build_multiview_rows,
    _candidate_patient_video_pairs,
    _cadica_patient_video_from_parts,
    _csv_value,
    _evaluate_frames,
    _extract_multiview_prediction,
    _frame_prediction_image_keys,
    _lookup_frame_prediction,
    _load_frame_prediction,
    _read_json_object,
    _threshold_points,
    _validate_inputs,
    load_cadica_manifest,
)


DEFAULT_FRAME_MIN_DEGREES = [round(index * 0.05, 2) for index in range(21)]
DEFAULT_BOX_MARGINS_PX = [0.0, 5.0, 10.0]
DEFAULT_MULTIVIEW_MIN_SCORES = [round(index * 0.05, 2) for index in range(21)]


@dataclass(frozen=True, slots=True)
class _SweepEvaluationJob:
    experiment_index: int
    frame_min_degree: float
    box_margin_px: float
    video_prediction_source: str
    multiview_min_score: float | None


@dataclass(frozen=True, slots=True)
class _FrameVariant:
    name: str
    prefix: tuple[str, ...]
    root: Path
    result_paths: tuple[Path, ...]
    predictions: dict[Any, Any]
    parsed_parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _TemporalVariant:
    name: str
    prefix: tuple[str, ...]
    frame_prefix: tuple[str, ...]
    root: Path
    result_paths: tuple[Path, ...]
    predictions: dict[tuple[str, str], tuple[bool | None, float | None]]
    parsed_parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _MultiviewVariant:
    name: str
    prefix: tuple[str, ...]
    frame_prefix: tuple[str, ...]
    temporal_prefix: tuple[str, ...]
    root: Path
    result_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class _ExperimentVariant:
    frame_variant: str
    temporal_variant: str
    multiview_variant: str
    frame_prefix: tuple[str, ...]
    temporal_prefix: tuple[str, ...]
    multiview_prefix: tuple[str, ...]
    frame_root: Path
    temporal_root: Path | None
    multiview_root: Path | None
    frame_result_paths: tuple[Path, ...]
    temporal_result_paths: tuple[Path, ...]
    multiview_result_paths: tuple[Path, ...]
    frame_predictions: dict[Any, Any]
    temporal_predictions: dict[tuple[str, str], tuple[bool | None, float | None]] | None
    frame_parameters: dict[str, Any]
    temporal_parameters: dict[str, Any]
    evaluation_manifest_frames: list[Any]


_WORKER_CONTEXT: dict[str, Any] = {}


IDENTITY_COLUMNS = [
    "frame_variant",
    "temporal_variant",
    "multiview_variant",
    "radius_outside_fraction_threshold",
    "radius_min_outside_samples",
    "stenosis_threshold",
    "average_radius_threshold",
    "min_supporting_frames",
    "min_persistence_ratio",
]

FRAME_PARAMETER_COLUMNS = [
    "radius_outside_fraction_threshold",
    "radius_min_outside_samples",
    "stenosis_threshold",
    "average_radius_threshold",
]

TEMPORAL_PARAMETER_COLUMNS = [
    "min_supporting_frames",
    "min_persistence_ratio",
]

LONG_METRIC_COLUMNS = [
    "precision",
    "recall",
    "specificity",
    "f1",
    "balanced_accuracy",
    "TP",
    "FP",
    "TN",
    "FN",
    "total_evaluated",
    "positive_count",
    "negative_count",
]

LONG_CSV_FIELDS = [
    "stage",
    *IDENTITY_COLUMNS,
    "frame_min_degree",
    "box_margin_px",
    "video_prediction_source",
    "multiview_min_score",
    *LONG_METRIC_COLUMNS,
]

STAGE_PREFIXES = [
    ("frame", "frame"),
    ("temporal_frame_any", "video"),
    ("temporal_final", "video"),
    ("patient", "patient"),
    ("multiview_patient", "multiview_patient"),
    ("multiview_side", "multiview_side"),
]

SWEEP_CSV_FIELDS = [
    *IDENTITY_COLUMNS,
    "frame_min_degree",
    "box_margin_px",
    "video_prediction_source",
    "multiview_min_score",
    "frame_total_evaluated",
    "frame_TP",
    "frame_FP",
    "frame_TN",
    "frame_FN",
    "frame_accuracy",
    "frame_precision",
    "frame_recall",
    "frame_specificity",
    "frame_F1",
    "frame_false_positive_rate",
    "frame_false_negative_rate",
    "frame_balanced_accuracy",
    "frame_predicted_positive_count",
    "positive_evaluated_frames",
    "positive_frames_with_localized_prediction",
    "localization_recall_on_positive_frames",
    "localized_predicted_positive_frames",
    "unmatched_predicted_points",
    "total_gt_boxes",
    "matched_gt_boxes",
    "missed_gt_boxes",
    "box_recall",
    "video_total_evaluated",
    "video_TP",
    "video_FP",
    "video_TN",
    "video_FN",
    "video_accuracy",
    "video_precision",
    "video_recall",
    "video_specificity",
    "video_F1",
    "video_false_positive_rate",
    "video_false_negative_rate",
    "video_balanced_accuracy",
    "patient_total_evaluated",
    "patient_TP",
    "patient_FP",
    "patient_TN",
    "patient_FN",
    "patient_accuracy",
    "patient_precision",
    "patient_recall",
    "patient_specificity",
    "patient_F1",
    "patient_balanced_accuracy",
    "multiview_patient_total_evaluated",
    "multiview_patient_TP",
    "multiview_patient_FP",
    "multiview_patient_TN",
    "multiview_patient_FN",
    "multiview_patient_accuracy",
    "multiview_patient_precision",
    "multiview_patient_recall",
    "multiview_patient_specificity",
    "multiview_patient_F1",
    "multiview_patient_balanced_accuracy",
    "multiview_patient_false_positive_rate",
    "multiview_patient_false_negative_rate",
    "multiview_side_total_evaluated",
    "multiview_side_TP",
    "multiview_side_FP",
    "multiview_side_TN",
    "multiview_side_FN",
    "multiview_side_precision",
    "multiview_side_recall",
    "multiview_side_specificity",
    "multiview_side_F1",
    "multiview_side_balanced_accuracy",
]


def run_cadica_threshold_sweep(
    *,
    manifest: str | Path,
    frame_results_root: str | Path,
    output_root: str | Path,
    temporal_results_root: str | Path | None = None,
    multiview_results_root: str | Path | None = None,
    frame_min_degrees: list[float] | None = None,
    box_margins_px: list[float] | None = None,
    video_prediction_sources: list[str] | None = None,
    multiview_min_scores: list[float] | None = None,
    workers: int = 1,
    evaluate_matched_frames_only: bool = False,
) -> dict[str, Path]:
    resolved_manifest = Path(manifest)
    resolved_frame_root = Path(frame_results_root)
    resolved_output_root = Path(output_root)
    resolved_temporal_root = None if temporal_results_root is None else Path(temporal_results_root)
    resolved_multiview_root = None if multiview_results_root is None else Path(multiview_results_root)
    resolved_frame_min_degrees = _resolve_float_values(
        frame_min_degrees,
        default=DEFAULT_FRAME_MIN_DEGREES,
        name="frame_min_degrees",
    )
    resolved_box_margins = _resolve_float_values(
        box_margins_px,
        default=DEFAULT_BOX_MARGINS_PX,
        name="box_margins_px",
    )
    resolved_sources = _resolve_video_prediction_sources(video_prediction_sources, resolved_temporal_root)
    resolved_multiview_min_scores = _resolve_multiview_min_scores(multiview_min_scores, resolved_multiview_root)
    resolved_workers = _resolve_workers(workers)

    _validate_inputs(resolved_manifest, resolved_frame_root, resolved_temporal_root, resolved_multiview_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    manifest_frames = load_cadica_manifest(resolved_manifest)
    frame_variants = _discover_frame_variants(resolved_frame_root)
    temporal_variants = _discover_temporal_variants(resolved_temporal_root, frame_variants)
    multiview_variants = _discover_multiview_variants(resolved_multiview_root, frame_variants, temporal_variants)
    experiments, skipped_combinations = _build_experiment_variants(
        frame_variants=frame_variants,
        temporal_variants=temporal_variants,
        multiview_variants=multiview_variants,
        manifest_frames=manifest_frames,
        evaluate_matched_frames_only=evaluate_matched_frames_only,
    )
    if not experiments:
        raise ValueError("No valid CADICA experiment combinations were found.")
    diagnostics = _build_cadica_sweep_diagnostics(
        manifest_frames=manifest_frames,
        frame_variants=frame_variants,
        temporal_variants=temporal_variants,
        multiview_variants=multiview_variants,
        experiments=experiments,
        skipped_combinations=skipped_combinations,
        frame_results_root=resolved_frame_root,
        temporal_results_root=resolved_temporal_root,
        multiview_results_root=resolved_multiview_root,
        frame_min_degrees=resolved_frame_min_degrees,
        evaluate_matched_frames_only=evaluate_matched_frames_only,
    )

    jobs = _build_sweep_jobs(
        experiments=experiments,
        frame_min_degrees=resolved_frame_min_degrees,
        box_margins_px=resolved_box_margins,
        video_prediction_sources=resolved_sources,
        multiview_min_scores=resolved_multiview_min_scores,
    )
    sweep_rows = _evaluate_sweep_jobs(
        jobs,
        manifest=resolved_manifest,
        frame_results_root=resolved_frame_root,
        output_root=resolved_output_root,
        temporal_results_root=resolved_temporal_root,
        multiview_results_root=resolved_multiview_root,
        experiments=experiments,
        workers=resolved_workers,
    )

    csv_path = resolved_output_root / "cadica_threshold_sweep.csv"
    summary_path = resolved_output_root / "cadica_threshold_sweep_summary.json"
    diagnostics_path = resolved_output_root / "cadica_benchmark_diagnostics.json"
    long_csv_path = resolved_output_root / "cadica_experiment_metrics_long.csv"
    wide_csv_path = resolved_output_root / "cadica_experiment_metrics_wide.csv"
    best_by_stage_path = resolved_output_root / "best_experiments_by_stage.csv"
    best_overall_path = resolved_output_root / "best_overall_experiments.csv"
    temporal_effects_path = resolved_output_root / "temporal_parameter_effects.csv"
    frame_effects_path = resolved_output_root / "frame_parameter_effects.csv"
    stage_progression_path = resolved_output_root / "stage_progression.csv"
    multiview_patient_rows_path = resolved_output_root / "cadica_multiview_patient_rows.csv"
    multiview_side_rows_path = resolved_output_root / "cadica_multiview_side_rows.csv"
    long_rows = _build_long_metrics_rows(sweep_rows)
    wide_rows = _build_wide_metrics_rows(long_rows)
    best_by_stage_rows = _best_experiments_by_stage(long_rows)
    best_overall_rows = _best_overall_experiments(wide_rows)
    temporal_effect_rows = _temporal_parameter_effect_rows(long_rows)
    frame_effect_rows = _frame_parameter_effect_rows(wide_rows)
    stage_progression_rows = _stage_progression_rows(wide_rows)
    _write_csv(csv_path, sweep_rows, fieldnames=SWEEP_CSV_FIELDS)
    _write_csv(long_csv_path, long_rows, fieldnames=LONG_CSV_FIELDS)
    _write_csv(wide_csv_path, wide_rows, fieldnames=_wide_fieldnames(wide_rows))
    _write_csv(best_by_stage_path, best_by_stage_rows, fieldnames=_best_by_stage_fieldnames(best_by_stage_rows))
    _write_csv(best_overall_path, best_overall_rows, fieldnames=_best_overall_fieldnames(best_overall_rows))
    _write_csv(temporal_effects_path, temporal_effect_rows, fieldnames=_effect_fieldnames(temporal_effect_rows, TEMPORAL_PARAMETER_COLUMNS))
    _write_csv(frame_effects_path, frame_effect_rows, fieldnames=_frame_effect_fieldnames(frame_effect_rows))
    _write_csv(stage_progression_path, stage_progression_rows, fieldnames=_stage_progression_fieldnames(stage_progression_rows))
    _write_json(diagnostics_path, diagnostics)
    multiview_row_paths = _write_multiview_review_rows(
        multiview_patient_rows_path=multiview_patient_rows_path,
        multiview_side_rows_path=multiview_side_rows_path,
        manifest_frames=manifest_frames,
        experiments=experiments,
        multiview_min_score=_first_multiview_row_score(resolved_multiview_min_scores),
    )
    _write_json(
        summary_path,
        _build_sweep_summary(
            rows=sweep_rows,
            manifest=resolved_manifest,
            frame_results_root=resolved_frame_root,
            output_root=resolved_output_root,
            temporal_results_root=resolved_temporal_root,
            multiview_results_root=resolved_multiview_root,
            frame_min_degrees=resolved_frame_min_degrees,
            box_margins_px=resolved_box_margins,
            video_prediction_sources=resolved_sources,
            multiview_min_scores=resolved_multiview_min_scores,
            experiments=experiments,
            multiview_row_paths=multiview_row_paths,
            workers=resolved_workers,
            evaluate_matched_frames_only=evaluate_matched_frames_only,
        ),
    )
    outputs = {
        "cadica_threshold_sweep_csv": csv_path,
        "cadica_threshold_sweep_summary_json": summary_path,
        "cadica_benchmark_diagnostics_json": diagnostics_path,
        "cadica_experiment_metrics_long_csv": long_csv_path,
        "cadica_experiment_metrics_wide_csv": wide_csv_path,
        "best_experiments_by_stage_csv": best_by_stage_path,
        "best_overall_experiments_csv": best_overall_path,
        "temporal_parameter_effects_csv": temporal_effects_path,
        "frame_parameter_effects_csv": frame_effects_path,
        "stage_progression_csv": stage_progression_path,
    }
    outputs.update(multiview_row_paths)
    return outputs


def _evaluate_sweep_jobs(
    jobs: list[_SweepEvaluationJob],
    *,
    manifest: Path,
    frame_results_root: Path,
    output_root: Path,
    temporal_results_root: Path | None,
    multiview_results_root: Path | None,
    experiments: list[_ExperimentVariant],
    workers: int,
) -> list[dict[str, Any]]:
    description = "Benchmarking CADICA sweep"
    if workers == 1 or len(jobs) <= 1:
        rows: list[dict[str, Any]] = []
        with tqdm(total=len(jobs), desc=description, unit="combo", dynamic_ncols=True) as progress:
            for job in jobs:
                rows.append(
                    _evaluate_sweep_job(
                        job,
                        manifest=manifest,
                        frame_results_root=frame_results_root,
                        output_root=output_root,
                        temporal_results_root=temporal_results_root,
                        multiview_results_root=multiview_results_root,
                        experiments=experiments,
                    )
                )
                progress.update(1)
        return rows

    chunksize = max(1, len(jobs) // (workers * 4))
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_sweep_worker,
        initargs=(
            manifest,
            frame_results_root,
            output_root,
            temporal_results_root,
            multiview_results_root,
            experiments,
        ),
    ) as executor:
        row_iter = executor.map(
            _evaluate_sweep_job_from_worker,
            jobs,
            chunksize=chunksize,
        )
        rows = []
        with tqdm(total=len(jobs), desc=description, unit="combo", dynamic_ncols=True) as progress:
            for row in row_iter:
                rows.append(row)
                progress.update(1)
    return rows


def _init_sweep_worker(
    manifest: Path,
    frame_results_root: Path,
    output_root: Path,
    temporal_results_root: Path | None,
    multiview_results_root: Path | None,
    experiments: list[_ExperimentVariant],
) -> None:
    _WORKER_CONTEXT.clear()
    _WORKER_CONTEXT.update(
        {
            "manifest": manifest,
            "frame_results_root": frame_results_root,
            "output_root": output_root,
            "temporal_results_root": temporal_results_root,
            "multiview_results_root": multiview_results_root,
            "experiments": experiments,
        }
    )


def _evaluate_sweep_job_from_worker(job: _SweepEvaluationJob) -> dict[str, Any]:
    if not _WORKER_CONTEXT:
        raise RuntimeError("CADICA sweep worker was not initialized.")
    return _evaluate_sweep_job(
        job,
        manifest=_WORKER_CONTEXT["manifest"],
        frame_results_root=_WORKER_CONTEXT["frame_results_root"],
        output_root=_WORKER_CONTEXT["output_root"],
        temporal_results_root=_WORKER_CONTEXT["temporal_results_root"],
        multiview_results_root=_WORKER_CONTEXT["multiview_results_root"],
        experiments=_WORKER_CONTEXT["experiments"],
    )


def _evaluate_sweep_job(
    job: _SweepEvaluationJob,
    *,
    manifest: Path,
    frame_results_root: Path,
    output_root: Path,
    temporal_results_root: Path | None,
    multiview_results_root: Path | None,
    experiments: list[_ExperimentVariant],
) -> dict[str, Any]:
    experiment = experiments[job.experiment_index]
    frame_rows, box_rows, _unmatched_point_rows = _evaluate_frames(
        experiment.evaluation_manifest_frames,
        experiment.frame_predictions,
        frame_min_degree=job.frame_min_degree,
        box_margin_px=job.box_margin_px,
    )
    video_rows = _build_video_rows(
        frame_rows,
        temporal_results_root=experiment.temporal_root,
        video_prediction_source=job.video_prediction_source,
        temporal_predictions=experiment.temporal_predictions,
    )
    patient_rows = _build_patient_rows(video_rows)
    multiview_predictions = (
        None
        if not experiment.multiview_result_paths
        else _index_multiview_predictions_from_paths(
            experiment.multiview_result_paths,
            experiment.multiview_root or multiview_results_root or output_root,
            multiview_min_score=0.0 if job.multiview_min_score is None else job.multiview_min_score,
        )
    )
    multiview_patient_rows, multiview_side_rows, multiview_score_rule = _build_multiview_rows(
        experiment.evaluation_manifest_frames,
        multiview_results_root=experiment.multiview_root,
        multiview_min_score=0.0 if job.multiview_min_score is None else job.multiview_min_score,
        multiview_predictions=multiview_predictions,
    )
    summary = _build_summary(
        frame_rows,
        box_rows,
        video_rows,
        patient_rows,
        multiview_patient_rows,
        multiview_side_rows,
        manifest=manifest,
        frame_results_root=frame_results_root,
        output_root=output_root,
        temporal_results_root=experiment.temporal_root,
        multiview_results_root=experiment.multiview_root,
        frame_min_degree=job.frame_min_degree,
        box_margin_px=job.box_margin_px,
        video_prediction_source=job.video_prediction_source,
        multiview_min_score=0.0 if job.multiview_min_score is None else job.multiview_min_score,
        multiview_score_rule=multiview_score_rule,
        write_review_images=False,
        max_review_images=0,
        review_image_root=None,
    )
    row = _sweep_row_from_summary(summary, multiview_min_score=job.multiview_min_score)
    row.update(_experiment_identity_columns(experiment))
    return row


def _build_sweep_jobs(
    *,
    experiments: list[_ExperimentVariant],
    frame_min_degrees: list[float],
    box_margins_px: list[float],
    video_prediction_sources: list[str],
    multiview_min_scores: list[float | None],
) -> list[_SweepEvaluationJob]:
    jobs: list[_SweepEvaluationJob] = []
    combinations = itertools.product(
        range(len(experiments)),
        frame_min_degrees,
        box_margins_px,
        video_prediction_sources,
        multiview_min_scores,
    )
    for experiment_index, frame_min_degree, box_margin_px, video_prediction_source, multiview_min_score in combinations:
        jobs.append(
            _SweepEvaluationJob(
                experiment_index=experiment_index,
                frame_min_degree=frame_min_degree,
                box_margin_px=box_margin_px,
                video_prediction_source=video_prediction_source,
                multiview_min_score=multiview_min_score,
            )
        )
    return jobs


def _discover_frame_variants(frame_results_root: Path) -> list[_FrameVariant]:
    frame_paths = sorted(path for path in frame_results_root.rglob(f"*{FRAME_RESULT_SUFFIX}") if path.is_file())
    if not frame_paths:
        raise FileNotFoundError(f"No frame-level '*{FRAME_RESULT_SUFFIX}' files were found under: {frame_results_root}")

    paths_by_prefix: dict[tuple[str, ...], list[Path]] = {}
    for path in frame_paths:
        prefix = _prefix_before_patient_video(path, frame_results_root)
        paths_by_prefix.setdefault(prefix, []).append(path)

    variants: list[_FrameVariant] = []
    for prefix, paths in sorted(paths_by_prefix.items(), key=lambda item: _variant_name(item[0])):
        root = _variant_root(frame_results_root, prefix)
        predictions = _index_frame_predictions_from_paths(paths, root)
        variants.append(
            _FrameVariant(
                name=_variant_name(prefix),
                prefix=prefix,
                root=root,
                result_paths=tuple(paths),
                predictions=predictions,
                parsed_parameters=parse_cadica_variant_parameters(_variant_name(prefix), FRAME_PARAMETER_COLUMNS),
            )
        )
    return variants


def _discover_temporal_variants(
    temporal_results_root: Path | None,
    frame_variants: list[_FrameVariant],
) -> list[_TemporalVariant]:
    if temporal_results_root is None:
        return []

    temporal_paths = sorted(path for path in temporal_results_root.rglob(TEMPORAL_RESULT_FILENAME) if path.is_file())
    frame_prefixes = {variant.prefix for variant in frame_variants}
    paths_by_key: dict[tuple[tuple[str, ...], tuple[str, ...]], list[Path]] = {}
    for path in temporal_paths:
        raw_prefix = _prefix_before_patient_video(path, temporal_results_root)
        frame_prefix, temporal_prefix = _split_temporal_prefix(raw_prefix, frame_prefixes)
        paths_by_key.setdefault((frame_prefix, temporal_prefix), []).append(path)

    variants: list[_TemporalVariant] = []
    for (frame_prefix, temporal_prefix), paths in sorted(
        paths_by_key.items(),
        key=lambda item: (_variant_name(item[0][0]), _variant_name(item[0][1])),
    ):
        root = _variant_root(temporal_results_root, (*frame_prefix, *temporal_prefix))
        variants.append(
            _TemporalVariant(
                name=_variant_name(temporal_prefix),
                prefix=temporal_prefix,
                frame_prefix=frame_prefix,
                root=root,
                result_paths=tuple(paths),
                predictions=_index_temporal_predictions_from_paths(paths, temporal_results_root),
                parsed_parameters=parse_cadica_variant_parameters(_variant_name(temporal_prefix), TEMPORAL_PARAMETER_COLUMNS),
            )
        )
    return variants


def _discover_multiview_variants(
    multiview_results_root: Path | None,
    frame_variants: list[_FrameVariant],
    temporal_variants: list[_TemporalVariant],
) -> list[_MultiviewVariant]:
    if multiview_results_root is None:
        return []

    multiview_paths = sorted(path for path in multiview_results_root.rglob(MULTIVIEW_RESULT_FILENAME) if path.is_file())
    frame_prefixes = {variant.prefix for variant in frame_variants}
    temporal_keys = {(variant.frame_prefix, variant.prefix) for variant in temporal_variants}
    if not temporal_keys:
        temporal_keys = {(variant.prefix, ()) for variant in frame_variants}

    paths_by_key: dict[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]], list[Path]] = {}
    for path in multiview_paths:
        raw_prefix = _prefix_before_patient_video(path, multiview_results_root)
        frame_prefix, temporal_prefix, multiview_prefix = _split_multiview_prefix(
            raw_prefix,
            frame_prefixes=frame_prefixes,
            temporal_keys=temporal_keys,
        )
        paths_by_key.setdefault((frame_prefix, temporal_prefix, multiview_prefix), []).append(path)

    variants: list[_MultiviewVariant] = []
    for (frame_prefix, temporal_prefix, multiview_prefix), paths in sorted(
        paths_by_key.items(),
        key=lambda item: (_variant_name(item[0][0]), _variant_name(item[0][1]), _variant_name(item[0][2])),
    ):
        root = _variant_root(multiview_results_root, (*frame_prefix, *temporal_prefix, *multiview_prefix))
        variants.append(
            _MultiviewVariant(
                name=_variant_name(multiview_prefix),
                prefix=multiview_prefix,
                frame_prefix=frame_prefix,
                temporal_prefix=temporal_prefix,
                root=root,
                result_paths=tuple(paths),
            )
        )
    return variants


def _build_experiment_variants(
    *,
    frame_variants: list[_FrameVariant],
    temporal_variants: list[_TemporalVariant],
    multiview_variants: list[_MultiviewVariant],
    manifest_frames: list[Any],
    evaluate_matched_frames_only: bool,
) -> tuple[list[_ExperimentVariant], list[dict[str, Any]]]:
    frame_by_prefix = {variant.prefix: variant for variant in frame_variants}
    multiview_by_key = {
        (variant.frame_prefix, variant.temporal_prefix): variant
        for variant in multiview_variants
        if not variant.prefix
    }
    skipped: list[dict[str, Any]] = []
    experiments: list[_ExperimentVariant] = []

    if temporal_variants:
        iterable: list[tuple[_FrameVariant | None, _TemporalVariant | None]] = [
            (frame_by_prefix.get(temporal_variant.frame_prefix), temporal_variant)
            for temporal_variant in temporal_variants
        ]
    else:
        iterable = [(frame_variant, None) for frame_variant in frame_variants]

    for frame_variant, temporal_variant in iterable:
        if frame_variant is None:
            skipped.append(
                {
                    "reason": "temporal_result_has_no_corresponding_frame_result",
                    "temporal_variant": "" if temporal_variant is None else temporal_variant.name,
                    "frame_prefix": _prefix_text(temporal_variant.frame_prefix if temporal_variant is not None else ()),
                }
            )
            continue

        temporal_prefix = () if temporal_variant is None else temporal_variant.prefix
        multiview_variant = multiview_by_key.get((frame_variant.prefix, temporal_prefix))
        if multiview_variants and multiview_variant is None:
            skipped.append(
                {
                    "reason": "missing_corresponding_multiview_result",
                    "frame_variant": frame_variant.name,
                    "temporal_variant": "flat" if temporal_variant is None else temporal_variant.name,
                }
            )

        evaluation_manifest_frames = (
            _matched_manifest_frames(manifest_frames, frame_variant.predictions)
            if evaluate_matched_frames_only
            else list(manifest_frames)
        )
        experiments.append(
            _ExperimentVariant(
                frame_variant=frame_variant.name,
                temporal_variant="flat" if temporal_variant is None else temporal_variant.name,
                multiview_variant="" if multiview_variant is None else multiview_variant.name,
                frame_prefix=frame_variant.prefix,
                temporal_prefix=temporal_prefix,
                multiview_prefix=() if multiview_variant is None else multiview_variant.prefix,
                frame_root=frame_variant.root,
                temporal_root=None if temporal_variant is None else temporal_variant.root,
                multiview_root=None if multiview_variant is None else multiview_variant.root,
                frame_result_paths=frame_variant.result_paths,
                temporal_result_paths=() if temporal_variant is None else temporal_variant.result_paths,
                multiview_result_paths=() if multiview_variant is None else multiview_variant.result_paths,
                frame_predictions=frame_variant.predictions,
                temporal_predictions=None if temporal_variant is None else temporal_variant.predictions,
                frame_parameters=frame_variant.parsed_parameters,
                temporal_parameters={} if temporal_variant is None else temporal_variant.parsed_parameters,
                evaluation_manifest_frames=evaluation_manifest_frames,
            )
        )

    for multiview_variant in multiview_variants:
        if (multiview_variant.frame_prefix, multiview_variant.temporal_prefix) not in {
            (experiment.frame_prefix, experiment.temporal_prefix) for experiment in experiments
        }:
            skipped.append(
                {
                    "reason": "multiview_result_has_no_corresponding_temporal_result",
                    "frame_variant": _variant_name(multiview_variant.frame_prefix),
                    "temporal_variant": _variant_name(multiview_variant.temporal_prefix),
                    "multiview_variant": multiview_variant.name,
                }
            )

    return experiments, skipped


def parse_cadica_variant_parameters(variant_name: str, expected_columns: Iterable[str] | None = None) -> dict[str, Any]:
    expected = set(expected_columns or [])
    parsed: dict[str, Any] = {column: None for column in expected}
    if not variant_name or variant_name == "flat":
        return parsed

    for part in variant_name.split("__"):
        key, value = _split_variant_parameter(part)
        if key is None:
            continue
        if expected and key not in expected:
            continue
        parsed[key] = value
    return parsed


def _split_variant_parameter(part: str) -> tuple[str | None, Any]:
    match = re.fullmatch(r"(.+)_(-?\d+(?:p\d+)?|-?\d+(?:\.\d+)?)", part.strip())
    if match is None:
        return None, None
    return match.group(1), _parse_variant_value(match.group(2))


def _parse_variant_value(raw_value: str) -> int | float | str:
    normalized = raw_value.replace("p", ".")
    try:
        if re.fullmatch(r"-?\d+", normalized):
            return int(normalized)
        return float(normalized)
    except ValueError:
        return raw_value


def _prefix_before_patient_video(path: Path, root: Path) -> tuple[str, ...]:
    parts = _relative_parts(path, root)
    patient_index = _cadica_patient_video_index(parts)
    return () if patient_index is None else tuple(parts[:patient_index])


def _variant_root(root: Path, prefix: tuple[str, ...]) -> Path:
    resolved = root
    for part in prefix:
        resolved = resolved / part
    return resolved


def _variant_name(prefix: tuple[str, ...]) -> str:
    return "/".join(prefix) if prefix else "flat"


def _split_temporal_prefix(
    raw_prefix: tuple[str, ...],
    frame_prefixes: set[tuple[str, ...]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not raw_prefix:
        return (), ()
    for frame_prefix in sorted(frame_prefixes, key=len, reverse=True):
        if _starts_with(raw_prefix, frame_prefix):
            return frame_prefix, tuple(raw_prefix[len(frame_prefix):])
    if () in frame_prefixes:
        return (), raw_prefix
    return raw_prefix[:-1], raw_prefix[-1:]


def _split_multiview_prefix(
    raw_prefix: tuple[str, ...],
    *,
    frame_prefixes: set[tuple[str, ...]],
    temporal_keys: set[tuple[tuple[str, ...], tuple[str, ...]]],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    matching_frame_prefixes = [
        frame_prefix for frame_prefix in frame_prefixes if _starts_with(raw_prefix, frame_prefix)
    ]
    for frame_prefix in sorted(matching_frame_prefixes, key=len, reverse=True):
        remainder = tuple(raw_prefix[len(frame_prefix):])
        matching_temporal_prefixes = [
            temporal_prefix
            for candidate_frame_prefix, temporal_prefix in temporal_keys
            if candidate_frame_prefix == frame_prefix and _starts_with(remainder, temporal_prefix)
        ]
        if matching_temporal_prefixes:
            temporal_prefix = max(matching_temporal_prefixes, key=len)
            return frame_prefix, temporal_prefix, tuple(remainder[len(temporal_prefix):])
    frame_prefix, temporal_prefix = _split_temporal_prefix(raw_prefix, frame_prefixes)
    return frame_prefix, temporal_prefix, ()


def _starts_with(value: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    return len(prefix) <= len(value) and value[: len(prefix)] == prefix


def _index_frame_predictions_from_paths(paths: Iterable[Path], variant_root: Path) -> dict[Any, Any]:
    predictions: dict[Any, Any] = {}
    for result_path in sorted(paths):
        prediction = _load_frame_prediction(result_path)
        patient_video_pairs = _candidate_patient_video_pairs(result_path, variant_root, prediction.view_id)
        for patient_id, video_id in patient_video_pairs:
            image_keys = _frame_prediction_image_keys(prediction, patient_id=patient_id, video_id=video_id)
            for image_key in image_keys:
                if image_key:
                    predictions.setdefault((patient_id, video_id, image_key), prediction)
    return predictions


def _index_temporal_predictions_from_paths(
    paths: Iterable[Path],
    temporal_results_root: Path,
) -> dict[tuple[str, str], tuple[bool | None, float | None]]:
    predictions: dict[tuple[str, str], tuple[bool | None, float | None]] = {}
    for result_path in sorted(paths):
        patient_video_pair = _cadica_patient_video_from_parts(_relative_parts(result_path, temporal_results_root))
        if patient_video_pair is None:
            continue
        patient_id, video_id = patient_video_pair
        predictions.setdefault((patient_id.lower(), video_id.lower()), _extract_temporal_prediction(result_path))
    return predictions


def _extract_temporal_prediction(result_path: Path) -> tuple[bool | None, float | None]:
    payload = _read_json_object(result_path)
    final_lesion = payload.get("final_lesion")
    if not isinstance(final_lesion, dict):
        return False, 0.0
    degrees = final_lesion.get("degrees") if isinstance(final_lesion.get("degrees"), dict) else {}
    score = _coerce_float(degrees.get("median")) or _coerce_float(degrees.get("max")) or 1.0
    return True, score


def _index_multiview_predictions_from_paths(
    paths: Iterable[Path],
    multiview_results_root: Path,
    *,
    multiview_min_score: float,
) -> dict[str, MultiViewPrediction]:
    predictions: dict[str, MultiViewPrediction] = {}
    for result_path in sorted(paths):
        patient_id = _cadica_patient_id_from_multiview_path(result_path, multiview_results_root)
        if patient_id is None:
            continue
        payload = _read_json_object(result_path)
        predictions.setdefault(
            patient_id,
            _extract_multiview_prediction(result_path, payload, multiview_min_score=multiview_min_score),
        )
    return predictions


def _cadica_patient_id_from_multiview_path(result_path: Path, root: Path) -> str | None:
    for part in reversed(_relative_parts(result_path, root)[:-1]):
        if CADICA_PATIENT_ID_RE.fullmatch(part):
            return part
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _experiment_identity_columns(experiment: _ExperimentVariant) -> dict[str, Any]:
    row = {
        "frame_variant": experiment.frame_variant,
        "temporal_variant": experiment.temporal_variant,
        "multiview_variant": experiment.multiview_variant,
    }
    for column in FRAME_PARAMETER_COLUMNS:
        row[column] = experiment.frame_parameters.get(column)
    for column in TEMPORAL_PARAMETER_COLUMNS:
        row[column] = experiment.temporal_parameters.get(column)
    return row


def parse_float_list(value: str) -> list[float]:
    values: list[float] = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        try:
            parsed = float(item)
        except ValueError as exc:
            raise ValueError(f"Invalid float value: {item!r}.") from exc
        if parsed < 0.0:
            raise ValueError(f"Threshold values must be >= 0, got {parsed}.")
        values.append(parsed)
    if not values:
        raise ValueError("At least one numeric value is required.")
    return values


def parse_video_prediction_sources(value: str) -> list[str]:
    sources = [item.strip() for item in value.split(",") if item.strip()]
    if not sources:
        raise ValueError("At least one video prediction source is required.")
    invalid_sources = sorted(set(sources) - VIDEO_PREDICTION_SOURCES)
    if invalid_sources:
        raise ValueError(
            "video_prediction_sources must contain only "
            f"{sorted(VIDEO_PREDICTION_SOURCES)}, got {invalid_sources}."
        )
    return sources


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a supervised CADICA threshold sweep against existing prediction JSONs.",
    )
    parser.add_argument("--manifest", required=True, help="Path to CADICA prepared manifest.csv or manifest.jsonl.")
    parser.add_argument("--frame-results-root", required=True, help="Root containing frame-level stenosis JSON outputs.")
    parser.add_argument("--output-root", required=True, help="Directory where CADICA sweep outputs will be written.")
    parser.add_argument(
        "--temporal-results-root",
        help="Optional root containing view_temporal_fusion.json outputs grouped by patient/video.",
    )
    parser.add_argument(
        "--multiview-results-root",
        help="Optional root containing case_multiview_fusion.json outputs grouped by patient/case.",
    )
    parser.add_argument(
        "--frame-min-degrees",
        help="Comma-separated frame degree thresholds. Defaults to 0.0 through 1.0 in 0.05 steps.",
    )
    parser.add_argument(
        "--box-margins-px",
        help="Comma-separated CADICA box margins in pixels. Defaults to 0,5,10.",
    )
    parser.add_argument(
        "--video-prediction-sources",
        help="Comma-separated sources: frame_any, temporal_final. Defaults depend on --temporal-results-root.",
    )
    parser.add_argument(
        "--multiview-min-scores",
        "--sweep-multiview-min-scores",
        dest="multiview_min_scores",
        help="Comma-separated multi-view score thresholds. Defaults to 0.0 through 1.0 in 0.05 steps when multi-view results are supplied.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes used to evaluate sweep combinations. Defaults to 1.",
    )
    parser.add_argument(
        "--evaluate-matched-frames-only",
        action="store_true",
        help=(
            "Evaluate only manifest frames that have a matching frame prediction JSON. "
            "By default, unmatched manifest frames are preserved as no-prediction frames."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        outputs = run_cadica_threshold_sweep(
            manifest=args.manifest,
            frame_results_root=args.frame_results_root,
            output_root=args.output_root,
            temporal_results_root=args.temporal_results_root,
            multiview_results_root=args.multiview_results_root,
            frame_min_degrees=None if args.frame_min_degrees is None else parse_float_list(args.frame_min_degrees),
            box_margins_px=None if args.box_margins_px is None else parse_float_list(args.box_margins_px),
            video_prediction_sources=(
                None
                if args.video_prediction_sources is None
                else parse_video_prediction_sources(args.video_prediction_sources)
            ),
            multiview_min_scores=(
                None if args.multiview_min_scores is None else parse_float_list(args.multiview_min_scores)
            ),
            workers=args.workers,
            evaluate_matched_frames_only=args.evaluate_matched_frames_only,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"CADICA threshold sweep failed: {exc}", file=sys.stderr)
        return 2

    print(f"CADICA threshold sweep CSV: {outputs['cadica_threshold_sweep_csv']}")
    print(f"CADICA threshold sweep summary: {outputs['cadica_threshold_sweep_summary_json']}")
    print(f"CADICA benchmark diagnostics: {outputs['cadica_benchmark_diagnostics_json']}")
    return 0


def _resolve_float_values(values: list[float] | None, *, default: list[float], name: str) -> list[float]:
    resolved = list(default if values is None else values)
    if not resolved:
        raise ValueError(f"{name} must contain at least one value.")
    negative_values = [value for value in resolved if value < 0.0]
    if negative_values:
        raise ValueError(f"{name} values must be >= 0, got {negative_values}.")
    return resolved


def _resolve_workers(workers: int) -> int:
    if workers < 1:
        raise ValueError("workers must be >= 1.")
    return workers


def _resolve_video_prediction_sources(
    sources: list[str] | None,
    temporal_results_root: Path | None,
) -> list[str]:
    resolved = list(sources) if sources is not None else _default_video_prediction_sources(temporal_results_root)
    if not resolved:
        raise ValueError("video_prediction_sources must contain at least one value.")
    invalid_sources = sorted(set(resolved) - VIDEO_PREDICTION_SOURCES)
    if invalid_sources:
        raise ValueError(
            "video_prediction_sources must contain only "
            f"{sorted(VIDEO_PREDICTION_SOURCES)}, got {invalid_sources}."
        )
    if temporal_results_root is None and "temporal_final" in resolved:
        raise ValueError("video_prediction_source temporal_final requires temporal_results_root.")
    return resolved


def _default_video_prediction_sources(temporal_results_root: Path | None) -> list[str]:
    if temporal_results_root is None:
        return ["frame_any"]
    return ["frame_any", "temporal_final"]


def _resolve_multiview_min_scores(
    values: list[float] | None,
    multiview_results_root: Path | None,
) -> list[float | None]:
    if multiview_results_root is None:
        if values is not None:
            raise ValueError("multiview_min_scores requires multiview_results_root.")
        return [None]
    return _resolve_float_values(
        values,
        default=DEFAULT_MULTIVIEW_MIN_SCORES,
        name="multiview_min_scores",
    )


def _first_multiview_row_score(multiview_min_scores: list[float | None]) -> float:
    for score in multiview_min_scores:
        if score is not None:
            return score
    return 0.0


def _matched_manifest_frames(manifest_frames: list[Any], frame_predictions: dict[Any, Any]) -> list[Any]:
    return [
        manifest_frame
        for manifest_frame in manifest_frames
        if _lookup_frame_prediction(manifest_frame, frame_predictions) is not None
    ]


def _build_cadica_sweep_diagnostics(
    *,
    manifest_frames: list[Any],
    frame_variants: list[_FrameVariant],
    temporal_variants: list[_TemporalVariant],
    multiview_variants: list[_MultiviewVariant],
    experiments: list[_ExperimentVariant],
    skipped_combinations: list[dict[str, Any]],
    frame_results_root: Path,
    temporal_results_root: Path | None,
    multiview_results_root: Path | None,
    frame_min_degrees: list[float],
    evaluate_matched_frames_only: bool,
) -> dict[str, Any]:
    frame_paths = sorted(path for path in frame_results_root.rglob(f"*{FRAME_RESULT_SUFFIX}") if path.is_file())
    temporal_paths = (
        []
        if temporal_results_root is None
        else sorted(path for path in temporal_results_root.rglob(TEMPORAL_RESULT_FILENAME) if path.is_file())
    )
    multiview_paths = (
        []
        if multiview_results_root is None
        else sorted(path for path in multiview_results_root.rglob(MULTIVIEW_RESULT_FILENAME) if path.is_file())
    )
    all_frame_predictions: dict[Any, Any] = {}
    for frame_variant in frame_variants:
        for key, prediction in frame_variant.predictions.items():
            all_frame_predictions.setdefault((*frame_variant.prefix, *key), prediction)
    frame_sources = _unique_frame_prediction_sources(all_frame_predictions)
    frame_json_summary = _summarize_frame_jsons(frame_paths, frame_results_root)
    manifest_by_variant = _manifest_matching_by_variant(
        manifest_frames=manifest_frames,
        frame_variants=frame_variants,
        experiments=experiments,
        frame_min_degrees=frame_min_degrees,
        evaluate_matched_frames_only=evaluate_matched_frames_only,
    )
    first_manifest_summary = next(iter(manifest_by_variant.values()), None) or {
        "evaluate_matched_frames_only": evaluate_matched_frames_only,
        "manifest_frame_count": len(manifest_frames),
        "matched_manifest_frames": 0,
        "evaluated_frame_count": 0,
    }
    temporal_summary = _summarize_temporal_results_by_variant(
        temporal_variants=temporal_variants,
        manifest_frames=manifest_frames,
        frame_variants=frame_variants,
        frame_min_degrees=frame_min_degrees,
    )
    warnings = _variant_diagnostics_warnings(
        frame_variants=frame_variants,
        temporal_variants=temporal_variants,
        multiview_variants=multiview_variants,
        experiments=experiments,
        skipped_combinations=skipped_combinations,
    )
    return {
        "variant_counts": {
            "frame_variants_found": len(frame_variants),
            "temporal_variants_found": len(temporal_variants),
            "multiview_variants_found": len(multiview_variants),
            "valid_experiment_combinations": len(experiments),
            "valid_full_experiment_combinations": sum(1 for experiment in experiments if experiment.multiview_result_paths),
        },
        "variants": {
            "frame_variants": [variant.name for variant in frame_variants],
            "temporal_variants": sorted({variant.name for variant in temporal_variants}),
            "multiview_variants": sorted({variant.name for variant in multiview_variants}),
        },
        "skipped_missing_combinations": skipped_combinations,
        "warnings": warnings,
        "frame_results": {
            **frame_json_summary,
            "variant_count": len(frame_variants),
            "indexed_prediction_key_count": sum(len(variant.predictions) for variant in frame_variants),
            "unique_indexed_prediction_file_count": len(frame_sources),
            "created_key_examples": _frame_key_examples(
                frame_variants[0].predictions if frame_variants else {},
                frame_variants[0].root if frame_variants else frame_results_root,
            ),
        },
        "manifest_matching": {
            **first_manifest_summary,
            "per_frame_variant": manifest_by_variant,
        },
        "temporal_results": temporal_summary,
        "multiview_results": {
            "json_file_count": len(multiview_paths),
            "variant_count": len(multiview_variants),
            "per_variant": {
                f"{_variant_name(variant.frame_prefix)}::{_variant_name(variant.temporal_prefix)}::{variant.name}": {
                    "frame_variant": _variant_name(variant.frame_prefix),
                    "temporal_variant": _variant_name(variant.temporal_prefix),
                    "multiview_variant": variant.name,
                    "json_file_count": len(variant.result_paths),
                }
                for variant in multiview_variants
            },
        },
        "staleness_and_root_hints": _summarize_staleness_and_roots(
            frame_paths=frame_paths,
            temporal_paths=temporal_paths,
            frame_results_root=frame_results_root,
            temporal_results_root=temporal_results_root,
        ),
    }


def _summarize_frame_jsons(frame_paths: list[Path], frame_results_root: Path) -> dict[str, Any]:
    nonempty_raw = 0
    parsed_point_files = 0
    examples: list[dict[str, Any]] = []
    for result_path in frame_paths:
        payload = _read_json_object(result_path)
        raw_points = payload.get("stenosis_points")
        raw_count = len(raw_points) if isinstance(raw_points, list) else 0
        parsed_count = sum(1 for item in raw_points if isinstance(item, dict)) if isinstance(raw_points, list) else 0
        if raw_count > 0:
            nonempty_raw += 1
        if parsed_count > 0:
            parsed_point_files += 1
        if len(examples) < 10:
            frame_payload = payload.get("frame") if isinstance(payload.get("frame"), dict) else {}
            examples.append(
                {
                    "relative_path": _display_relative_path(result_path, frame_results_root),
                    "image_name": frame_payload.get("image_name"),
                    "view_id": frame_payload.get("view_id"),
                    "stenosis_point_count": raw_count,
                    "max_degree": _max_raw_degree(raw_points),
                }
            )
    return {
        "json_file_count": len(frame_paths),
        "jsons_with_nonempty_stenosis_points": nonempty_raw,
        "jsons_with_dict_stenosis_points": parsed_point_files,
        "frame_json_examples": examples,
    }


def _summarize_manifest_matches(
    *,
    manifest_frames: list[Any],
    evaluation_manifest_frames: list[Any],
    frame_predictions: dict[Any, Any],
    frame_min_degrees: list[float],
    evaluate_matched_frames_only: bool,
) -> dict[str, Any]:
    matched_count = 0
    matched_with_points = 0
    threshold_counts = {_threshold_key(value): 0 for value in frame_min_degrees}
    unmatched_examples: list[dict[str, Any]] = []

    for manifest_frame in manifest_frames:
        prediction = _lookup_frame_prediction(manifest_frame, frame_predictions)
        if prediction is None:
            if len(unmatched_examples) < 10:
                unmatched_examples.append(
                    {
                        "patient_id": manifest_frame.patient_id,
                        "video_id": manifest_frame.video_id,
                        "frame_id": manifest_frame.frame_id,
                        "prepared_image_name": manifest_frame.prepared_image_name,
                        "prepared_image_stem": manifest_frame.prepared_image_stem,
                    }
                )
            continue

        matched_count += 1
        if prediction.points:
            matched_with_points += 1
        for threshold in frame_min_degrees:
            if _threshold_points(prediction.points, threshold):
                threshold_counts[_threshold_key(threshold)] += 1

    return {
        "evaluate_matched_frames_only": evaluate_matched_frames_only,
        "manifest_frame_count": len(manifest_frames),
        "positive_manifest_frame_count": sum(1 for frame in manifest_frames if frame.frame_label == "positive"),
        "matched_manifest_frames": matched_count,
        "unmatched_manifest_frames": len(manifest_frames) - matched_count,
        "skipped_unmatched_manifest_frames": len(manifest_frames) - len(evaluation_manifest_frames),
        "evaluated_frame_count": len(evaluation_manifest_frames),
        "positive_evaluated_frame_count": sum(
            1 for frame in evaluation_manifest_frames if frame.frame_label == "positive"
        ),
        "negative_evaluated_frame_count": sum(
            1 for frame in evaluation_manifest_frames if frame.frame_label == "negative"
        ),
        "matched_manifest_frames_with_stenosis_points": matched_with_points,
        "matched_manifest_frames_with_thresholded_points_by_frame_min_degree": threshold_counts,
        "unmatched_manifest_frame_examples": unmatched_examples,
    }


def _manifest_matching_by_variant(
    *,
    manifest_frames: list[Any],
    frame_variants: list[_FrameVariant],
    experiments: list[_ExperimentVariant],
    frame_min_degrees: list[float],
    evaluate_matched_frames_only: bool,
) -> dict[str, dict[str, Any]]:
    experiment_by_frame_prefix: dict[tuple[str, ...], _ExperimentVariant] = {}
    for experiment in experiments:
        experiment_by_frame_prefix.setdefault(experiment.frame_prefix, experiment)

    summaries: dict[str, dict[str, Any]] = {}
    for frame_variant in frame_variants:
        experiment = experiment_by_frame_prefix.get(frame_variant.prefix)
        evaluation_manifest_frames = (
            experiment.evaluation_manifest_frames
            if experiment is not None
            else _matched_manifest_frames(manifest_frames, frame_variant.predictions)
            if evaluate_matched_frames_only
            else manifest_frames
        )
        summaries[frame_variant.name] = _summarize_manifest_matches(
            manifest_frames=manifest_frames,
            evaluation_manifest_frames=evaluation_manifest_frames,
            frame_predictions=frame_variant.predictions,
            frame_min_degrees=frame_min_degrees,
            evaluate_matched_frames_only=evaluate_matched_frames_only,
        )
    return summaries


def _summarize_temporal_results_by_variant(
    *,
    temporal_variants: list[_TemporalVariant],
    manifest_frames: list[Any],
    frame_variants: list[_FrameVariant],
    frame_min_degrees: list[float],
) -> dict[str, Any]:
    if not temporal_variants:
        return {
            "json_file_count": 0,
            "indexed_patient_video_count": 0,
            "temporal_final_positive_count": 0,
            "note": "No temporal_results_root was supplied or no temporal result JSONs were found.",
            "per_variant": {},
        }

    frame_by_prefix = {variant.prefix: variant for variant in frame_variants}
    per_variant: dict[str, Any] = {}
    aggregate_positive_thresholds = {_threshold_key(value): 0 for value in frame_min_degrees}
    aggregate = {
        "json_file_count": 0,
        "indexed_patient_video_count": 0,
        "temporal_final_positive_count": 0,
        "positive_temporal_with_any_matched_frame_stenosis_points": 0,
    }
    examples: list[dict[str, Any]] = []
    unindexed_examples: list[str] = []

    for temporal_variant in temporal_variants:
        frame_variant = frame_by_prefix.get(temporal_variant.frame_prefix)
        summary = _summarize_temporal_results(
            temporal_paths=list(temporal_variant.result_paths),
            temporal_results_root=temporal_variant.root,
            manifest_frames=manifest_frames,
            frame_predictions={} if frame_variant is None else frame_variant.predictions,
            frame_min_degrees=frame_min_degrees,
        )
        key = f"{_variant_name(temporal_variant.frame_prefix)}::{temporal_variant.name}"
        per_variant[key] = {
            "frame_variant": _variant_name(temporal_variant.frame_prefix),
            "temporal_variant": temporal_variant.name,
            **summary,
        }
        for aggregate_key in aggregate:
            aggregate[aggregate_key] += int(summary.get(aggregate_key) or 0)
        threshold_counts = summary.get("positive_temporal_with_thresholded_frame_by_frame_min_degree")
        if isinstance(threshold_counts, dict):
            for threshold_key, count in threshold_counts.items():
                aggregate_positive_thresholds[threshold_key] = aggregate_positive_thresholds.get(threshold_key, 0) + int(count)
        unindexed_examples.extend(str(item) for item in summary.get("unindexed_temporal_json_examples", [])[:10])
        examples.extend(
            item
            for item in summary.get("positive_temporal_without_underlying_positive_frame_examples", [])
            if isinstance(item, dict)
        )

    return {
        **aggregate,
        "unindexed_temporal_json_examples": unindexed_examples[:10],
        "positive_temporal_with_thresholded_frame_by_frame_min_degree": aggregate_positive_thresholds,
        "positive_temporal_without_underlying_positive_frame_examples": examples[:10],
        "variant_count": len(temporal_variants),
        "per_variant": per_variant,
    }


def _variant_diagnostics_warnings(
    *,
    frame_variants: list[_FrameVariant],
    temporal_variants: list[_TemporalVariant],
    multiview_variants: list[_MultiviewVariant],
    experiments: list[_ExperimentVariant],
    skipped_combinations: list[dict[str, Any]],
) -> dict[str, Any]:
    warning_items: list[str] = []
    if len(frame_variants) > 1 or len(temporal_variants) > 1 or len(multiview_variants) > 1:
        warning_items.append("Variant-aware evaluation is active; prediction dictionaries are scoped per experiment.")
    if not experiments:
        warning_items.append("No valid experiment combinations were found.")
    for skipped in skipped_combinations:
        reason = str(skipped.get("reason") or "")
        if reason == "temporal_result_has_no_corresponding_frame_result":
            warning_items.append(
                "Temporal result has no corresponding frame result: "
                f"{skipped.get('frame_prefix')} / {skipped.get('temporal_variant')}"
            )
        elif reason == "multiview_result_has_no_corresponding_temporal_result":
            warning_items.append(
                "Multiview result has no corresponding temporal result: "
                f"{skipped.get('frame_variant')} / {skipped.get('temporal_variant')}"
            )
    return {
        "variants_collapsed": False,
        "variant_collapse_warning": "",
        "items": warning_items,
    }


def _summarize_temporal_results(
    *,
    temporal_paths: list[Path],
    temporal_results_root: Path | None,
    manifest_frames: list[Any],
    frame_predictions: dict[Any, Any],
    frame_min_degrees: list[float],
) -> dict[str, Any]:
    if temporal_results_root is None:
        return {
            "json_file_count": 0,
            "indexed_patient_video_count": 0,
            "temporal_final_positive_count": 0,
            "note": "No temporal_results_root was supplied.",
        }

    manifest_by_video: dict[tuple[str, str], list[Any]] = {}
    for manifest_frame in manifest_frames:
        key = (manifest_frame.patient_id.lower(), manifest_frame.video_id.lower())
        manifest_by_video.setdefault(key, []).append(manifest_frame)

    indexed_pairs: set[tuple[str, str]] = set()
    unindexed_examples: list[str] = []
    positive_count = 0
    positives_with_raw_frame_points = 0
    positives_with_thresholded_frames = {_threshold_key(value): 0 for value in frame_min_degrees}
    positive_examples_without_min_threshold_frame: list[dict[str, Any]] = []
    min_threshold = min(frame_min_degrees) if frame_min_degrees else 0.0

    for temporal_path in temporal_paths:
        relative_parts = _relative_parts(temporal_path, temporal_results_root)
        patient_video_pair = _cadica_patient_video_from_parts(relative_parts)
        if patient_video_pair is None:
            if len(unindexed_examples) < 10:
                unindexed_examples.append(_display_relative_path(temporal_path, temporal_results_root))
            continue

        patient_id, video_id = patient_video_pair
        indexed_pairs.add((patient_id.lower(), video_id.lower()))
        payload = _read_json_object(temporal_path)
        final_lesion = payload.get("final_lesion")
        if not isinstance(final_lesion, dict):
            continue

        positive_count += 1
        manifest_video_frames = manifest_by_video.get((patient_id.lower(), video_id.lower()), [])
        has_raw_frame_points = any(
            (prediction := _lookup_frame_prediction(frame, frame_predictions)) is not None and bool(prediction.points)
            for frame in manifest_video_frames
        )
        if has_raw_frame_points:
            positives_with_raw_frame_points += 1

        has_min_threshold_frame = False
        for threshold in frame_min_degrees:
            has_thresholded_frame = any(
                (prediction := _lookup_frame_prediction(frame, frame_predictions)) is not None
                and bool(_threshold_points(prediction.points, threshold))
                for frame in manifest_video_frames
            )
            if has_thresholded_frame:
                positives_with_thresholded_frames[_threshold_key(threshold)] += 1
            if threshold == min_threshold:
                has_min_threshold_frame = has_thresholded_frame

        if not has_min_threshold_frame and len(positive_examples_without_min_threshold_frame) < 10:
            positive_examples_without_min_threshold_frame.append(
                {
                    "patient_id": patient_id,
                    "video_id": video_id,
                    "relative_path": _display_relative_path(temporal_path, temporal_results_root),
                    "matched_manifest_frames_for_video": len(manifest_video_frames),
                    "has_raw_frame_points": has_raw_frame_points,
                    "frame_min_degree": min_threshold,
                }
            )

    return {
        "json_file_count": len(temporal_paths),
        "indexed_patient_video_count": len(indexed_pairs),
        "unindexed_temporal_json_examples": unindexed_examples,
        "temporal_final_positive_count": positive_count,
        "positive_temporal_with_any_matched_frame_stenosis_points": positives_with_raw_frame_points,
        "positive_temporal_with_thresholded_frame_by_frame_min_degree": positives_with_thresholded_frames,
        "positive_temporal_without_underlying_positive_frame_examples": positive_examples_without_min_threshold_frame,
    }


def _summarize_staleness_and_roots(
    *,
    frame_paths: list[Path],
    temporal_paths: list[Path],
    frame_results_root: Path,
    temporal_results_root: Path | None,
) -> dict[str, Any]:
    frame_latest = _latest_mtime(frame_paths)
    temporal_latest = _latest_mtime(temporal_paths)
    temporal_oldest = _oldest_mtime(temporal_paths)
    temporal_older_than_latest_frame = (
        0 if frame_latest is None else sum(1 for path in temporal_paths if path.stat().st_mtime < frame_latest)
    )
    frame_prefixes = _frame_variant_prefixes(frame_paths, frame_results_root)
    temporal_frame_prefixes = (
        set() if temporal_results_root is None else _temporal_frame_variant_prefixes(temporal_paths, temporal_results_root)
    )
    missing_frame_prefixes = sorted(prefix for prefix in temporal_frame_prefixes if prefix not in frame_prefixes)
    return {
        "newest_frame_result_mtime": _format_mtime(frame_latest),
        "oldest_temporal_result_mtime": _format_mtime(temporal_oldest),
        "newest_temporal_result_mtime": _format_mtime(temporal_latest),
        "newest_temporal_older_than_newest_frame": (
            None if frame_latest is None or temporal_latest is None else temporal_latest < frame_latest
        ),
        "temporal_files_older_than_newest_frame_count": temporal_older_than_latest_frame,
        "frame_results_root_parent": str(frame_results_root.resolve().parent),
        "temporal_results_root_parent": None if temporal_results_root is None else str(temporal_results_root.resolve().parent),
        "roots_share_parent": (
            None
            if temporal_results_root is None
            else frame_results_root.resolve().parent == temporal_results_root.resolve().parent
        ),
        "frame_variant_prefix_examples": [_prefix_text(prefix) for prefix in sorted(frame_prefixes)[:10]],
        "temporal_frame_variant_prefix_examples": [
            _prefix_text(prefix) for prefix in sorted(temporal_frame_prefixes)[:10]
        ],
        "temporal_frame_prefixes_without_frame_results": [
            _prefix_text(prefix) for prefix in missing_frame_prefixes[:10]
        ],
    }


def _unique_frame_prediction_sources(frame_predictions: dict[Any, Any]) -> dict[Path, Any]:
    sources: dict[Path, Any] = {}
    for prediction in frame_predictions.values():
        sources.setdefault(prediction.source_path, prediction)
    return sources


def _frame_key_examples(frame_predictions: dict[Any, Any], frame_results_root: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for (patient_id, video_id, image_key), prediction in sorted(frame_predictions.items(), key=lambda item: item[0]):
        if len(examples) >= 10:
            break
        examples.append(
            {
                "patient_id": patient_id,
                "video_id": video_id,
                "image_key": image_key,
                "relative_path": _display_relative_path(prediction.source_path, frame_results_root),
                "parsed_stenosis_point_count": len(prediction.points),
            }
        )
    return examples


def _max_raw_degree(raw_points: object) -> float | None:
    if not isinstance(raw_points, list):
        return None
    degrees = []
    for raw_point in raw_points:
        if not isinstance(raw_point, dict):
            continue
        try:
            degrees.append(float(raw_point.get("degree")))
        except (TypeError, ValueError):
            continue
    return max(degrees) if degrees else None


def _result_mtimes(paths: list[Path]) -> list[float]:
    return [path.stat().st_mtime for path in paths]


def _latest_mtime(paths: list[Path]) -> float | None:
    mtimes = _result_mtimes(paths)
    return max(mtimes) if mtimes else None


def _oldest_mtime(paths: list[Path]) -> float | None:
    mtimes = _result_mtimes(paths)
    return min(mtimes) if mtimes else None


def _format_mtime(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def _relative_parts(path: Path, root: Path) -> tuple[str, ...]:
    try:
        return path.resolve().relative_to(root.resolve()).parts
    except ValueError:
        return path.parts


def _display_relative_path(path: Path, root: Path) -> str:
    try:
        relative_path = path.resolve().relative_to(root.resolve())
    except ValueError:
        relative_path = path
    return relative_path.as_posix()


def _cadica_patient_video_index(parts: tuple[str, ...]) -> int | None:
    for index, part in enumerate(parts[:-1]):
        if CADICA_PATIENT_ID_RE.fullmatch(part) and CADICA_VIDEO_ID_RE.fullmatch(parts[index + 1]):
            return index
    return None


def _frame_variant_prefixes(paths: list[Path], root: Path) -> set[tuple[str, ...]]:
    prefixes: set[tuple[str, ...]] = set()
    for path in paths:
        parts = _relative_parts(path, root)
        patient_index = _cadica_patient_video_index(parts)
        if patient_index is not None:
            prefixes.add(parts[:patient_index])
    return prefixes


def _temporal_frame_variant_prefixes(paths: list[Path], root: Path) -> set[tuple[str, ...]]:
    prefixes: set[tuple[str, ...]] = set()
    for path in paths:
        parts = _relative_parts(path, root)
        patient_index = _cadica_patient_video_index(parts)
        if patient_index is None:
            continue
        temporal_prefix = parts[:patient_index]
        prefixes.add(temporal_prefix[:-1] if temporal_prefix else ())
    return prefixes


def _prefix_text(prefix: tuple[str, ...]) -> str:
    return "/".join(prefix) if prefix else "(flat)"


def _threshold_key(value: float) -> str:
    return f"{value:g}"


def _write_multiview_review_rows(
    *,
    multiview_patient_rows_path: Path,
    multiview_side_rows_path: Path,
    manifest_frames: list[Any],
    experiments: list[_ExperimentVariant],
    multiview_min_score: float,
) -> dict[str, Path]:
    multiview_experiments = [experiment for experiment in experiments if experiment.multiview_result_paths]
    if not multiview_experiments:
        return {}

    patient_rows: list[dict[str, Any]] = []
    side_rows: list[dict[str, Any]] = []
    for experiment in multiview_experiments:
        predictions = _index_multiview_predictions_from_paths(
            experiment.multiview_result_paths,
            experiment.multiview_root or multiview_patient_rows_path.parent,
            multiview_min_score=multiview_min_score,
        )
        experiment_patient_rows, experiment_side_rows, _score_rule = _build_multiview_rows(
            manifest_frames,
            multiview_results_root=experiment.multiview_root,
            multiview_min_score=multiview_min_score,
            multiview_predictions=predictions,
        )
        identity = _experiment_identity_columns(experiment)
        patient_rows.extend({**identity, **row} for row in experiment_patient_rows)
        side_rows.extend({**identity, **row} for row in experiment_side_rows)

    patient_fields = [*IDENTITY_COLUMNS, *MULTIVIEW_PATIENT_ROW_FIELDS]
    side_fields = [*IDENTITY_COLUMNS, *MULTIVIEW_SIDE_ROW_FIELDS]
    _write_csv(multiview_patient_rows_path, patient_rows, fieldnames=patient_fields)
    outputs = {"cadica_multiview_patient_rows_csv": multiview_patient_rows_path}
    if side_rows:
        _write_csv(multiview_side_rows_path, side_rows, fieldnames=side_fields)
        outputs["cadica_multiview_side_rows_csv"] = multiview_side_rows_path
    return outputs


def _sweep_row_from_summary(summary: dict[str, Any], *, multiview_min_score: float | None) -> dict[str, Any]:
    frame_metrics = summary["frame_binary_metrics"]
    localization_metrics = summary["frame_localization_metrics"]
    box_metrics = summary["box_detection_metrics"]
    video_metrics = summary["video_binary_metrics"]
    patient_metrics = summary["patient_binary_metrics"]
    multiview_patient_metrics = summary.get("multiview_patient_binary_metrics") or {}
    multiview_side_metrics = summary.get("multiview_side_binary_metrics") or {}
    config = summary["config"]
    return {
        "frame_min_degree": config["frame_min_degree"],
        "box_margin_px": config["box_margin_px"],
        "video_prediction_source": config["video_prediction_source"],
        "multiview_min_score": multiview_min_score,
        **_binary_metric_columns(frame_metrics, prefix="frame", include_fnr=True, include_predicted_positive=True),
        **localization_metrics,
        **box_metrics,
        **_binary_metric_columns(video_metrics, prefix="video", include_fnr=True, include_predicted_positive=False),
        **_binary_metric_columns(patient_metrics, prefix="patient", include_fnr=False, include_predicted_positive=False),
        **_binary_metric_columns(
            multiview_patient_metrics,
            prefix="multiview_patient",
            include_fnr=True,
            include_predicted_positive=False,
        ),
        **_binary_metric_columns(
            multiview_side_metrics,
            prefix="multiview_side",
            include_fnr=False,
            include_predicted_positive=False,
            include_rates=False,
            include_accuracy=False,
        ),
    }


def _build_long_metrics_rows(sweep_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _key, grouped_rows in _group_sweep_rows(sweep_rows).items():
        source_rows = {str(row.get("video_prediction_source") or ""): row for row in grouped_rows}
        preferred_row = _preferred_source_row(grouped_rows)
        if preferred_row is None:
            continue

        rows.append(_long_row_from_sweep_row(preferred_row, stage="frame", metric_prefix="frame"))
        if "frame_any" in source_rows:
            rows.append(
                _long_row_from_sweep_row(
                    source_rows["frame_any"],
                    stage="temporal_frame_any",
                    metric_prefix="video",
                    video_prediction_source="frame_any",
                )
            )
        if "temporal_final" in source_rows:
            rows.append(
                _long_row_from_sweep_row(
                    source_rows["temporal_final"],
                    stage="temporal_final",
                    metric_prefix="video",
                    video_prediction_source="temporal_final",
                )
            )
        rows.append(_long_row_from_sweep_row(preferred_row, stage="patient", metric_prefix="patient"))
        if _metric_value(preferred_row, "multiview_patient_total_evaluated") is not None:
            rows.append(_long_row_from_sweep_row(preferred_row, stage="multiview_patient", metric_prefix="multiview_patient"))
        if _metric_value(preferred_row, "multiview_side_total_evaluated") is not None:
            rows.append(_long_row_from_sweep_row(preferred_row, stage="multiview_side", metric_prefix="multiview_side"))
    return rows


def _long_row_from_sweep_row(
    row: dict[str, Any],
    *,
    stage: str,
    metric_prefix: str,
    video_prediction_source: str | None = None,
) -> dict[str, Any]:
    true_positive = _metric_value(row, f"{metric_prefix}_TP")
    false_positive = _metric_value(row, f"{metric_prefix}_FP")
    true_negative = _metric_value(row, f"{metric_prefix}_TN")
    false_negative = _metric_value(row, f"{metric_prefix}_FN")
    long_row = {
        "stage": stage,
        **{column: row.get(column) for column in IDENTITY_COLUMNS},
        "frame_min_degree": row.get("frame_min_degree"),
        "box_margin_px": row.get("box_margin_px"),
        "video_prediction_source": video_prediction_source or row.get("video_prediction_source"),
        "multiview_min_score": row.get("multiview_min_score"),
        "precision": _metric_value(row, f"{metric_prefix}_precision"),
        "recall": _metric_value(row, f"{metric_prefix}_recall"),
        "specificity": _metric_value(row, f"{metric_prefix}_specificity"),
        "f1": _metric_value(row, f"{metric_prefix}_F1"),
        "balanced_accuracy": _metric_value(row, f"{metric_prefix}_balanced_accuracy"),
        "TP": _count_or_none(true_positive),
        "FP": _count_or_none(false_positive),
        "TN": _count_or_none(true_negative),
        "FN": _count_or_none(false_negative),
        "total_evaluated": _count_or_none(_metric_value(row, f"{metric_prefix}_total_evaluated")),
        "positive_count": _count_or_none(_sum_optional(true_positive, false_negative)),
        "negative_count": _count_or_none(_sum_optional(true_negative, false_positive)),
    }
    return long_row


def _build_wide_metrics_rows(long_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for long_row in long_rows:
        key = _wide_group_key(long_row)
        wide_row = rows_by_key.setdefault(
            key,
            {
                **{column: long_row.get(column) for column in IDENTITY_COLUMNS},
                "frame_min_degree": long_row.get("frame_min_degree"),
                "box_margin_px": long_row.get("box_margin_px"),
                "video_prediction_source": long_row.get("video_prediction_source"),
                "multiview_min_score": long_row.get("multiview_min_score"),
            },
        )
        stage = str(long_row.get("stage") or "")
        if stage == "temporal_final":
            wide_row["video_prediction_source"] = "temporal_final"
        for metric_name in LONG_METRIC_COLUMNS:
            wide_row[f"{stage}_{metric_name}"] = long_row.get(metric_name)
    return list(rows_by_key.values())


def _best_experiments_by_stage(long_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked_rows: list[dict[str, Any]] = []
    for stage in [stage for stage, _prefix in STAGE_PREFIXES]:
        candidates = [row for row in long_rows if row.get("stage") == stage and _numeric(row.get("f1")) is not None]
        for rank, row in enumerate(sorted(candidates, key=_long_ranking_key), start=1):
            ranked_rows.append({"rank": rank, **row})
    return ranked_rows


def _best_overall_experiments(wide_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(wide_rows, key=_overall_ranking_key)
    rows: list[dict[str, Any]] = []
    for rank, row in enumerate(ranked, start=1):
        metric_stage = "multiview_side" if _numeric(row.get("multiview_side_f1")) is not None else "multiview_patient"
        rows.append({"rank": rank, "overall_rank_metric_stage": metric_stage, **row})
    return rows


def _parameter_effect_rows(
    long_rows: list[dict[str, Any]],
    *,
    group_columns: list[str],
    stage: str,
) -> list[dict[str, Any]]:
    rows = [row for row in long_rows if row.get("stage") == stage]
    if not rows and stage == "temporal_final":
        rows = [row for row in long_rows if row.get("stage") == "temporal_frame_any"]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(column) for column in group_columns), []).append(row)

    effect_rows: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: tuple("" if value is None else str(value) for value in item[0])):
        best_f1 = min(group_rows, key=_long_ranking_key)
        best_balanced = min(group_rows, key=lambda row: _long_ranking_key_for_metric(row, "balanced_accuracy"))
        effect_row: dict[str, Any] = {
            column: key[index] for index, column in enumerate(group_columns)
        }
        effect_row.update(
            {
                "stage": stage,
                "count_experiments": len(group_rows),
                "max_f1": _max_metric(group_rows, "f1"),
                "max_balanced_accuracy": _max_metric(group_rows, "balanced_accuracy"),
                "best_config": _config_label(best_f1),
                "best_balanced_accuracy_config": _config_label(best_balanced),
            }
        )
        for metric in ("precision", "recall", "specificity", "f1", "balanced_accuracy", "TP", "FP", "TN", "FN"):
            effect_row[f"mean_{metric}"] = _mean_metric(group_rows, metric)
        effect_rows.append(effect_row)
    return effect_rows


def _temporal_parameter_effect_rows(long_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_columns in (["min_supporting_frames"], ["min_persistence_ratio"], TEMPORAL_PARAMETER_COLUMNS):
        group_rows = _parameter_effect_rows(long_rows, group_columns=group_columns, stage="temporal_final")
        group_by = ",".join(group_columns)
        for row in group_rows:
            for column in TEMPORAL_PARAMETER_COLUMNS:
                row.setdefault(column, None)
            row["group_by"] = group_by
        rows.extend(group_rows)
    return rows


def _frame_parameter_effect_rows(wide_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in wide_rows:
        grouped.setdefault(tuple(row.get(column) for column in FRAME_PARAMETER_COLUMNS), []).append(row)

    effect_rows: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items(), key=lambda item: tuple("" if value is None else str(value) for value in item[0])):
        best_frame = min(rows, key=lambda row: _wide_stage_ranking_key(row, "frame"))
        best_downstream = min(rows, key=lambda row: _wide_stage_ranking_key(row, "multiview_side"))
        effect_row: dict[str, Any] = {column: key[index] for index, column in enumerate(FRAME_PARAMETER_COLUMNS)}
        effect_row.update(
            {
                "count_experiments": len(rows),
                "mean_frame_precision": _mean_wide_metric(rows, "frame_precision"),
                "mean_frame_recall": _mean_wide_metric(rows, "frame_recall"),
                "mean_frame_specificity": _mean_wide_metric(rows, "frame_specificity"),
                "mean_frame_f1": _mean_wide_metric(rows, "frame_f1"),
                "mean_frame_balanced_accuracy": _mean_wide_metric(rows, "frame_balanced_accuracy"),
                "max_frame_f1": _max_wide_metric(rows, "frame_f1"),
                "max_frame_balanced_accuracy": _max_wide_metric(rows, "frame_balanced_accuracy"),
                "mean_multiview_side_f1": _mean_wide_metric(rows, "multiview_side_f1"),
                "max_multiview_side_f1": _max_wide_metric(rows, "multiview_side_f1"),
                "best_frame_config": _config_label(best_frame),
                "best_downstream_config": _config_label(best_downstream),
            }
        )
        effect_rows.append(effect_row)
    return effect_rows


def _stage_progression_rows(wide_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for wide_row in wide_rows:
        temporal_prefix = "temporal_final" if _numeric(wide_row.get("temporal_final_f1")) is not None else "temporal_frame_any"
        progression = {
            **{column: wide_row.get(column) for column in IDENTITY_COLUMNS},
            "frame_min_degree": wide_row.get("frame_min_degree"),
            "box_margin_px": wide_row.get("box_margin_px"),
            "video_prediction_source": "temporal_final" if temporal_prefix == "temporal_final" else "frame_any",
            "multiview_min_score": wide_row.get("multiview_min_score"),
            "frame_f1": wide_row.get("frame_f1"),
            "temporal_f1": wide_row.get(f"{temporal_prefix}_f1"),
            "multiview_side_f1": wide_row.get("multiview_side_f1"),
        }
        for metric in ("precision", "recall", "specificity", "balanced_accuracy"):
            frame_value = _numeric(wide_row.get(f"frame_{metric}"))
            temporal_value = _numeric(wide_row.get(f"{temporal_prefix}_{metric}"))
            multiview_value = _numeric(wide_row.get(f"multiview_side_{metric}"))
            progression[f"frame_{metric}"] = frame_value
            progression[f"temporal_{metric}"] = temporal_value
            progression[f"multiview_side_{metric}"] = multiview_value
            progression[f"delta_frame_to_temporal_{metric}"] = _delta(frame_value, temporal_value)
            progression[f"delta_temporal_to_multiview_{metric}"] = _delta(temporal_value, multiview_value)
            progression[f"delta_frame_to_multiview_{metric}"] = _delta(frame_value, multiview_value)
        progression["delta_frame_to_temporal_f1"] = _delta(_numeric(progression["frame_f1"]), _numeric(progression["temporal_f1"]))
        progression["delta_temporal_to_multiview_f1"] = _delta(_numeric(progression["temporal_f1"]), _numeric(progression["multiview_side_f1"]))
        progression["delta_frame_to_multiview_f1"] = _delta(_numeric(progression["frame_f1"]), _numeric(progression["multiview_side_f1"]))
        rows.append(progression)
    return rows


def _group_sweep_rows(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_sweep_group_key(row), []).append(row)
    return grouped


def _sweep_group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(column) for column in [*IDENTITY_COLUMNS, "frame_min_degree", "box_margin_px", "multiview_min_score"])


def _wide_group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(column) for column in [*IDENTITY_COLUMNS, "frame_min_degree", "box_margin_px", "multiview_min_score"])


def _preferred_source_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    for row in rows:
        if row.get("video_prediction_source") == "temporal_final":
            return row
    return rows[0]


def _long_ranking_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, int, int]:
    return _long_ranking_key_for_metric(row, "f1")


def _long_ranking_key_for_metric(row: dict[str, Any], metric: str) -> tuple[float, float, float, float, float, int, int]:
    primary = _numeric(row.get(metric))
    return (
        -_rank_number(primary),
        -_rank_number(_numeric(row.get("balanced_accuracy"))),
        -_rank_number(_numeric(row.get("recall"))),
        -_rank_number(_numeric(row.get("precision"))),
        -_rank_number(_numeric(row.get("specificity"))),
        _rank_count(row.get("FP")),
        _rank_count(row.get("FN")),
    )


def _overall_ranking_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, int, int]:
    stage = "multiview_side" if _numeric(row.get("multiview_side_f1")) is not None else "multiview_patient"
    return _wide_stage_ranking_key(row, stage)


def _wide_stage_ranking_key(row: dict[str, Any], stage: str) -> tuple[float, float, float, float, float, int, int]:
    return (
        -_rank_number(_numeric(row.get(f"{stage}_f1"))),
        -_rank_number(_numeric(row.get(f"{stage}_balanced_accuracy"))),
        -_rank_number(_numeric(row.get(f"{stage}_recall"))),
        -_rank_number(_numeric(row.get(f"{stage}_precision"))),
        -_rank_number(_numeric(row.get(f"{stage}_specificity"))),
        _rank_count(row.get(f"{stage}_FP")),
        _rank_count(row.get(f"{stage}_FN")),
    )


def _config_label(row: dict[str, Any]) -> str:
    parts = [
        f"frame_variant={row.get('frame_variant')}",
        f"temporal_variant={row.get('temporal_variant')}",
        f"frame_min_degree={row.get('frame_min_degree')}",
        f"box_margin_px={row.get('box_margin_px')}",
    ]
    if row.get("multiview_min_score") not in (None, ""):
        parts.append(f"multiview_min_score={row.get('multiview_min_score')}")
    return "; ".join(parts)


def _mean_metric(rows: list[dict[str, Any]], metric: str) -> float | None:
    return _mean([_numeric(row.get(metric)) for row in rows])


def _max_metric(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = [_numeric(row.get(metric)) for row in rows]
    valid = [value for value in values if value is not None]
    return max(valid) if valid else None


def _mean_wide_metric(rows: list[dict[str, Any]], metric: str) -> float | None:
    return _mean([_numeric(row.get(metric)) for row in rows])


def _max_wide_metric(rows: list[dict[str, Any]], metric: str) -> float | None:
    valid = [value for value in (_numeric(row.get(metric)) for row in rows) if value is not None]
    return max(valid) if valid else None


def _mean(values: Iterable[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return None if not valid else sum(valid) / len(valid)


def _numeric(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rank_number(value: float | None) -> float:
    return float("-inf") if value is None else value


def _rank_count(value: Any) -> int:
    number = _numeric(value)
    return 10**12 if number is None else int(number)


def _count_or_none(value: float | None) -> int | None:
    return None if value is None else int(value)


def _sum_optional(first: float | None, second: float | None) -> float | None:
    if first is None or second is None:
        return None
    return first + second


def _delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return after - before


def _wide_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fields = [*IDENTITY_COLUMNS, "frame_min_degree", "box_margin_px", "video_prediction_source", "multiview_min_score"]
    for stage, _metric_prefix in STAGE_PREFIXES:
        for metric in LONG_METRIC_COLUMNS:
            fields.append(f"{stage}_{metric}")
    return _fields_with_extras(fields, rows)


def _best_by_stage_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    return _fields_with_extras(["rank", *LONG_CSV_FIELDS], rows)


def _best_overall_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    return _fields_with_extras(["rank", "overall_rank_metric_stage", *_wide_fieldnames(rows)], rows)


def _effect_fieldnames(rows: list[dict[str, Any]], group_columns: list[str]) -> list[str]:
    fields = [
        "group_by",
        *group_columns,
        "stage",
        "count_experiments",
        "mean_precision",
        "mean_recall",
        "mean_specificity",
        "mean_f1",
        "mean_balanced_accuracy",
        "max_f1",
        "max_balanced_accuracy",
        "mean_TP",
        "mean_FP",
        "mean_TN",
        "mean_FN",
        "best_config",
        "best_balanced_accuracy_config",
    ]
    return _fields_with_extras(fields, rows)


def _frame_effect_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fields = [
        *FRAME_PARAMETER_COLUMNS,
        "count_experiments",
        "mean_frame_precision",
        "mean_frame_recall",
        "mean_frame_specificity",
        "mean_frame_f1",
        "mean_frame_balanced_accuracy",
        "max_frame_f1",
        "max_frame_balanced_accuracy",
        "mean_multiview_side_f1",
        "max_multiview_side_f1",
        "best_frame_config",
        "best_downstream_config",
    ]
    return _fields_with_extras(fields, rows)


def _stage_progression_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fields = [
        *IDENTITY_COLUMNS,
        "frame_min_degree",
        "box_margin_px",
        "video_prediction_source",
        "multiview_min_score",
        "frame_f1",
        "temporal_f1",
        "multiview_side_f1",
    ]
    for metric in ("precision", "recall", "specificity", "balanced_accuracy"):
        fields.extend(
            [
                f"frame_{metric}",
                f"temporal_{metric}",
                f"multiview_side_{metric}",
                f"delta_frame_to_temporal_{metric}",
                f"delta_temporal_to_multiview_{metric}",
                f"delta_frame_to_multiview_{metric}",
            ]
        )
    fields.extend(["delta_frame_to_temporal_f1", "delta_temporal_to_multiview_f1", "delta_frame_to_multiview_f1"])
    return _fields_with_extras(fields, rows)


def _fields_with_extras(fields: list[str], rows: list[dict[str, Any]]) -> list[str]:
    resolved = list(dict.fromkeys(fields))
    for row in rows:
        for field in row:
            if field not in resolved:
                resolved.append(field)
    return resolved


def _binary_metric_columns(
    metrics: dict[str, Any],
    *,
    prefix: str,
    include_fnr: bool,
    include_predicted_positive: bool,
    include_rates: bool = True,
    include_accuracy: bool = True,
) -> dict[str, Any]:
    if not metrics:
        row = {
            f"{prefix}_total_evaluated": None,
            f"{prefix}_TP": None,
            f"{prefix}_FP": None,
            f"{prefix}_TN": None,
            f"{prefix}_FN": None,
            f"{prefix}_precision": None,
            f"{prefix}_recall": None,
            f"{prefix}_specificity": None,
            f"{prefix}_F1": None,
            f"{prefix}_balanced_accuracy": None,
        }
        if include_accuracy:
            row[f"{prefix}_accuracy"] = None
        if include_rates:
            row[f"{prefix}_false_positive_rate"] = None
        if include_fnr:
            row[f"{prefix}_false_negative_rate"] = None
        if include_predicted_positive:
            row[f"{prefix}_predicted_positive_count"] = None
        return row

    row = {
        f"{prefix}_total_evaluated": metrics["total_evaluated"],
        f"{prefix}_TP": metrics["TP"],
        f"{prefix}_FP": metrics["FP"],
        f"{prefix}_TN": metrics["TN"],
        f"{prefix}_FN": metrics["FN"],
        f"{prefix}_precision": metrics["precision"],
        f"{prefix}_recall": metrics["recall"],
        f"{prefix}_specificity": metrics["specificity"],
        f"{prefix}_F1": metrics["f1"],
        f"{prefix}_balanced_accuracy": metrics["balanced_accuracy"],
    }
    if include_accuracy:
        row[f"{prefix}_accuracy"] = metrics["accuracy"]
    if include_rates:
        row[f"{prefix}_false_positive_rate"] = metrics["false_positive_rate"]
    if include_fnr:
        row[f"{prefix}_false_negative_rate"] = metrics["false_negative_rate"]
    if include_predicted_positive:
        row[f"{prefix}_predicted_positive_count"] = metrics["predicted_positive"]
    return row


def _build_sweep_summary(
    *,
    rows: list[dict[str, Any]],
    manifest: Path,
    frame_results_root: Path,
    output_root: Path,
    temporal_results_root: Path | None,
    multiview_results_root: Path | None,
    frame_min_degrees: list[float],
    box_margins_px: list[float],
    video_prediction_sources: list[str],
    multiview_min_scores: list[float | None],
    experiments: list[_ExperimentVariant],
    multiview_row_paths: dict[str, Path],
    workers: int,
    evaluate_matched_frames_only: bool,
) -> dict[str, Any]:
    best_frame_f1 = _best_metric(rows, "frame_F1")
    best_video_f1 = _best_metric(rows, "video_F1")
    best_patient_f1 = _best_metric(rows, "patient_F1")
    best_multiview_patient_f1 = _best_metric(rows, "multiview_patient_F1")
    return {
        "note": CADICA_SUPERVISED_NOTE,
        "config": {
            "manifest": str(manifest),
            "frame_results_root": str(frame_results_root),
            "output_root": str(output_root),
            "temporal_results_root": None if temporal_results_root is None else str(temporal_results_root),
            "multiview_results_root": None if multiview_results_root is None else str(multiview_results_root),
            "multiview_evaluated": multiview_results_root is not None,
            "frame_min_degrees": frame_min_degrees,
            "box_margins_px": box_margins_px,
            "video_prediction_sources": video_prediction_sources,
            "multiview_min_scores": multiview_min_scores,
            "frame_variants": sorted({experiment.frame_variant for experiment in experiments}),
            "temporal_variants": sorted({experiment.temporal_variant for experiment in experiments}),
            "multiview_variants": sorted({experiment.multiview_variant for experiment in experiments if experiment.multiview_variant}),
            "workers": workers,
            "evaluate_matched_frames_only": evaluate_matched_frames_only,
            "multiview_prediction_rule": (
                "predicted positive when lesion is present and score >= threshold; "
                "if a result lacks an explicit lesion-present field, score >= threshold is used"
            ),
            "multiview_row_paths": {key: str(path) for key, path in sorted(multiview_row_paths.items())},
        },
        "evaluated_sweep_combinations": len(rows),
        "valid_experiment_combinations": len(experiments),
        "valid_full_experiment_combinations": sum(1 for experiment in experiments if experiment.multiview_result_paths),
        "multiview_evaluated": multiview_results_root is not None,
        "best_operating_points": {
            "best_frame_F1": best_frame_f1,
            "best_frame_balanced_accuracy": _best_metric(rows, "frame_balanced_accuracy"),
            "best_frame_recall_with_specificity_at_least_0_80": _best_filtered_max(
                rows,
                filter_metric="frame_specificity",
                minimum=0.80,
                optimize_metric="frame_recall",
            ),
            "lowest_frame_false_positive_rate_with_recall_at_least_0_70": _best_low_false_positive_rate(rows),
            "best_box_recall_with_frame_specificity_at_least_0_80": _best_filtered_max(
                rows,
                filter_metric="frame_specificity",
                minimum=0.80,
                optimize_metric="box_recall",
                tie_metric="frame_precision",
            ),
            "best_localization_recall_with_frame_specificity_at_least_0_80": _best_filtered_max(
                rows,
                filter_metric="frame_specificity",
                minimum=0.80,
                optimize_metric="localization_recall_on_positive_frames",
                tie_metric="frame_precision",
            ),
            "best_video_F1": best_video_f1,
            "best_patient_F1": best_patient_f1,
            "best_multiview_patient_F1": best_multiview_patient_f1,
            "best_multiview_patient_balanced_accuracy": _best_metric(rows, "multiview_patient_balanced_accuracy"),
            "best_multiview_side_F1": _best_metric(rows, "multiview_side_F1"),
        },
        "stage_best_operating_point_comparison": {
            "frame": best_frame_f1,
            "temporal_video": best_video_f1,
            "temporal_patient": best_patient_f1,
            "multiview_patient": best_multiview_patient_f1,
        },
    }


def _best_metric(rows: Iterable[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if _metric_value(row, metric) is not None]
    if not candidates:
        return None
    return dict(
        max(
            candidates,
            key=lambda row: (
                _metric_value(row, metric) or float("-inf"),
                -float(row["frame_min_degree"]),
            ),
        )
    )


def _best_filtered_max(
    rows: Iterable[dict[str, Any]],
    *,
    filter_metric: str,
    minimum: float,
    optimize_metric: str,
    tie_metric: str | None = None,
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if (_metric_value(row, filter_metric) is not None)
        and (_metric_value(row, filter_metric) or 0.0) >= minimum
        and _metric_value(row, optimize_metric) is not None
    ]
    if not candidates:
        return None

    def sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
        tie_value = 0.0 if tie_metric is None else (_metric_value(row, tie_metric) or float("-inf"))
        return (
            _metric_value(row, optimize_metric) or float("-inf"),
            tie_value,
            -float(row["frame_min_degree"]),
        )

    return dict(max(candidates, key=sort_key))


def _best_low_false_positive_rate(rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if (_metric_value(row, "frame_recall") is not None)
        and (_metric_value(row, "frame_recall") or 0.0) >= 0.70
        and _metric_value(row, "frame_false_positive_rate") is not None
    ]
    if not candidates:
        return None
    return dict(
        min(
            candidates,
            key=lambda row: (
                _metric_value(row, "frame_false_positive_rate") or float("inf"),
                -(_metric_value(row, "frame_recall") or float("-inf")),
                float(row["frame_min_degree"]),
            ),
        )
    )


def _metric_value(row: dict[str, Any], metric: str) -> float | None:
    value = row.get(metric)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field_name: _csv_value(row.get(field_name)) for field_name in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

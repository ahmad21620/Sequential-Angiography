from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
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
    TEMPORAL_RESULT_FILENAME,
    VIDEO_PREDICTION_SOURCES,
    _build_patient_rows,
    _build_summary,
    _build_video_rows,
    _build_multiview_rows,
    _cadica_patient_video_from_parts,
    _csv_value,
    _evaluate_frames,
    _index_frame_predictions,
    _lookup_frame_prediction,
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
    frame_min_degree: float
    box_margin_px: float
    video_prediction_source: str
    multiview_min_score: float | None


_WORKER_CONTEXT: dict[str, Any] = {}


SWEEP_CSV_FIELDS = [
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
    frame_predictions = _index_frame_predictions(resolved_frame_root)
    evaluation_manifest_frames = (
        _matched_manifest_frames(manifest_frames, frame_predictions)
        if evaluate_matched_frames_only
        else manifest_frames
    )
    diagnostics = _build_cadica_sweep_diagnostics(
        manifest_frames=manifest_frames,
        evaluation_manifest_frames=evaluation_manifest_frames,
        frame_predictions=frame_predictions,
        frame_results_root=resolved_frame_root,
        temporal_results_root=resolved_temporal_root,
        frame_min_degrees=resolved_frame_min_degrees,
        evaluate_matched_frames_only=evaluate_matched_frames_only,
    )

    jobs = _build_sweep_jobs(
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
        manifest_frames=evaluation_manifest_frames,
        frame_predictions=frame_predictions,
        workers=resolved_workers,
    )

    csv_path = resolved_output_root / "cadica_threshold_sweep.csv"
    summary_path = resolved_output_root / "cadica_threshold_sweep_summary.json"
    diagnostics_path = resolved_output_root / "cadica_benchmark_diagnostics.json"
    multiview_patient_rows_path = resolved_output_root / "cadica_multiview_patient_rows.csv"
    multiview_side_rows_path = resolved_output_root / "cadica_multiview_side_rows.csv"
    _write_csv(csv_path, sweep_rows, fieldnames=SWEEP_CSV_FIELDS)
    _write_json(diagnostics_path, diagnostics)
    multiview_row_paths = _write_multiview_review_rows(
        multiview_patient_rows_path=multiview_patient_rows_path,
        multiview_side_rows_path=multiview_side_rows_path,
        manifest_frames=evaluation_manifest_frames,
        multiview_results_root=resolved_multiview_root,
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
            multiview_row_paths=multiview_row_paths,
            workers=resolved_workers,
            evaluate_matched_frames_only=evaluate_matched_frames_only,
        ),
    )
    outputs = {
        "cadica_threshold_sweep_csv": csv_path,
        "cadica_threshold_sweep_summary_json": summary_path,
        "cadica_benchmark_diagnostics_json": diagnostics_path,
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
    manifest_frames: list[Any],
    frame_predictions: dict[Any, Any],
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
                        manifest_frames=manifest_frames,
                        frame_predictions=frame_predictions,
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
            manifest_frames,
            frame_predictions,
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
    manifest_frames: list[Any],
    frame_predictions: dict[Any, Any],
) -> None:
    _WORKER_CONTEXT.clear()
    _WORKER_CONTEXT.update(
        {
            "manifest": manifest,
            "frame_results_root": frame_results_root,
            "output_root": output_root,
            "temporal_results_root": temporal_results_root,
            "multiview_results_root": multiview_results_root,
            "manifest_frames": manifest_frames,
            "frame_predictions": frame_predictions,
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
        manifest_frames=_WORKER_CONTEXT["manifest_frames"],
        frame_predictions=_WORKER_CONTEXT["frame_predictions"],
    )


def _evaluate_sweep_job(
    job: _SweepEvaluationJob,
    *,
    manifest: Path,
    frame_results_root: Path,
    output_root: Path,
    temporal_results_root: Path | None,
    multiview_results_root: Path | None,
    manifest_frames: list[Any],
    frame_predictions: dict[Any, Any],
) -> dict[str, Any]:
    frame_rows, box_rows, _unmatched_point_rows = _evaluate_frames(
        manifest_frames,
        frame_predictions,
        frame_min_degree=job.frame_min_degree,
        box_margin_px=job.box_margin_px,
    )
    video_rows = _build_video_rows(
        frame_rows,
        temporal_results_root=temporal_results_root,
        video_prediction_source=job.video_prediction_source,
    )
    patient_rows = _build_patient_rows(video_rows)
    multiview_patient_rows, multiview_side_rows, multiview_score_rule = _build_multiview_rows(
        manifest_frames,
        multiview_results_root=multiview_results_root,
        multiview_min_score=0.0 if job.multiview_min_score is None else job.multiview_min_score,
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
        temporal_results_root=temporal_results_root,
        multiview_results_root=multiview_results_root,
        frame_min_degree=job.frame_min_degree,
        box_margin_px=job.box_margin_px,
        video_prediction_source=job.video_prediction_source,
        multiview_min_score=0.0 if job.multiview_min_score is None else job.multiview_min_score,
        multiview_score_rule=multiview_score_rule,
        write_review_images=False,
        max_review_images=0,
        review_image_root=None,
    )
    return _sweep_row_from_summary(summary, multiview_min_score=job.multiview_min_score)


def _build_sweep_jobs(
    *,
    frame_min_degrees: list[float],
    box_margins_px: list[float],
    video_prediction_sources: list[str],
    multiview_min_scores: list[float | None],
) -> list[_SweepEvaluationJob]:
    jobs: list[_SweepEvaluationJob] = []
    combinations = itertools.product(
        frame_min_degrees,
        box_margins_px,
        video_prediction_sources,
        multiview_min_scores,
    )
    for frame_min_degree, box_margin_px, video_prediction_source, multiview_min_score in combinations:
        jobs.append(
            _SweepEvaluationJob(
                frame_min_degree=frame_min_degree,
                box_margin_px=box_margin_px,
                video_prediction_source=video_prediction_source,
                multiview_min_score=multiview_min_score,
            )
        )
    return jobs


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
    evaluation_manifest_frames: list[Any],
    frame_predictions: dict[Any, Any],
    frame_results_root: Path,
    temporal_results_root: Path | None,
    frame_min_degrees: list[float],
    evaluate_matched_frames_only: bool,
) -> dict[str, Any]:
    frame_paths = sorted(path for path in frame_results_root.rglob(f"*{FRAME_RESULT_SUFFIX}") if path.is_file())
    temporal_paths = (
        []
        if temporal_results_root is None
        else sorted(path for path in temporal_results_root.rglob(TEMPORAL_RESULT_FILENAME) if path.is_file())
    )
    frame_sources = _unique_frame_prediction_sources(frame_predictions)
    frame_json_summary = _summarize_frame_jsons(frame_paths, frame_results_root)
    manifest_summary = _summarize_manifest_matches(
        manifest_frames=manifest_frames,
        evaluation_manifest_frames=evaluation_manifest_frames,
        frame_predictions=frame_predictions,
        frame_min_degrees=frame_min_degrees,
        evaluate_matched_frames_only=evaluate_matched_frames_only,
    )
    temporal_summary = _summarize_temporal_results(
        temporal_paths=temporal_paths,
        temporal_results_root=temporal_results_root,
        manifest_frames=manifest_frames,
        frame_predictions=frame_predictions,
        frame_min_degrees=frame_min_degrees,
    )
    return {
        "frame_results": {
            **frame_json_summary,
            "indexed_prediction_key_count": len(frame_predictions),
            "unique_indexed_prediction_file_count": len(frame_sources),
            "created_key_examples": _frame_key_examples(frame_predictions, frame_results_root),
        },
        "manifest_matching": manifest_summary,
        "temporal_results": temporal_summary,
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
    multiview_results_root: Path | None,
    multiview_min_score: float,
) -> dict[str, Path]:
    if multiview_results_root is None:
        return {}
    patient_rows, side_rows, _score_rule = _build_multiview_rows(
        manifest_frames,
        multiview_results_root=multiview_results_root,
        multiview_min_score=multiview_min_score,
    )
    _write_csv(multiview_patient_rows_path, patient_rows, fieldnames=MULTIVIEW_PATIENT_ROW_FIELDS)
    outputs = {"cadica_multiview_patient_rows_csv": multiview_patient_rows_path}
    if side_rows:
        _write_csv(multiview_side_rows_path, side_rows, fieldnames=MULTIVIEW_SIDE_ROW_FIELDS)
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
            "workers": workers,
            "evaluate_matched_frames_only": evaluate_matched_frames_only,
            "multiview_prediction_rule": (
                "predicted positive when lesion is present and score >= threshold; "
                "if a result lacks an explicit lesion-present field, score >= threshold is used"
            ),
            "multiview_row_paths": {key: str(path) for key, path in sorted(multiview_row_paths.items())},
        },
        "evaluated_sweep_combinations": len(rows),
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

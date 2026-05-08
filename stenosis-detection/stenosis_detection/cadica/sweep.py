from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import sys
from typing import Any, Iterable

from .benchmark import (
    CADICA_SUPERVISED_NOTE,
    VIDEO_PREDICTION_SOURCES,
    _build_patient_rows,
    _build_summary,
    _build_video_rows,
    _csv_value,
    _evaluate_frames,
    _index_frame_predictions,
    _validate_inputs,
    load_cadica_manifest,
)


DEFAULT_FRAME_MIN_DEGREES = [round(index * 0.05, 2) for index in range(21)]
DEFAULT_BOX_MARGINS_PX = [0.0, 5.0, 10.0]

SWEEP_CSV_FIELDS = [
    "frame_min_degree",
    "box_margin_px",
    "video_prediction_source",
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
]


def run_cadica_threshold_sweep(
    *,
    manifest: str | Path,
    frame_results_root: str | Path,
    output_root: str | Path,
    temporal_results_root: str | Path | None = None,
    frame_min_degrees: list[float] | None = None,
    box_margins_px: list[float] | None = None,
    video_prediction_sources: list[str] | None = None,
) -> dict[str, Path]:
    resolved_manifest = Path(manifest)
    resolved_frame_root = Path(frame_results_root)
    resolved_output_root = Path(output_root)
    resolved_temporal_root = None if temporal_results_root is None else Path(temporal_results_root)
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

    _validate_inputs(resolved_manifest, resolved_frame_root, resolved_temporal_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    manifest_frames = load_cadica_manifest(resolved_manifest)
    frame_predictions = _index_frame_predictions(resolved_frame_root)

    sweep_rows: list[dict[str, Any]] = []
    for frame_min_degree, box_margin_px in itertools.product(resolved_frame_min_degrees, resolved_box_margins):
        frame_rows, box_rows, _unmatched_point_rows = _evaluate_frames(
            manifest_frames,
            frame_predictions,
            frame_min_degree=frame_min_degree,
            box_margin_px=box_margin_px,
        )
        for video_prediction_source in resolved_sources:
            video_rows = _build_video_rows(
                frame_rows,
                temporal_results_root=resolved_temporal_root,
                video_prediction_source=video_prediction_source,
            )
            patient_rows = _build_patient_rows(video_rows)
            summary = _build_summary(
                frame_rows,
                box_rows,
                video_rows,
                patient_rows,
                manifest=resolved_manifest,
                frame_results_root=resolved_frame_root,
                output_root=resolved_output_root,
                temporal_results_root=resolved_temporal_root,
                frame_min_degree=frame_min_degree,
                box_margin_px=box_margin_px,
                video_prediction_source=video_prediction_source,
                write_review_images=False,
                max_review_images=0,
                review_image_root=None,
            )
            sweep_rows.append(_sweep_row_from_summary(summary))

    csv_path = resolved_output_root / "cadica_threshold_sweep.csv"
    summary_path = resolved_output_root / "cadica_threshold_sweep_summary.json"
    _write_csv(csv_path, sweep_rows, fieldnames=SWEEP_CSV_FIELDS)
    _write_json(
        summary_path,
        _build_sweep_summary(
            rows=sweep_rows,
            manifest=resolved_manifest,
            frame_results_root=resolved_frame_root,
            output_root=resolved_output_root,
            temporal_results_root=resolved_temporal_root,
            frame_min_degrees=resolved_frame_min_degrees,
            box_margins_px=resolved_box_margins,
            video_prediction_sources=resolved_sources,
        ),
    )
    return {
        "cadica_threshold_sweep_csv": csv_path,
        "cadica_threshold_sweep_summary_json": summary_path,
    }


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
            frame_min_degrees=None if args.frame_min_degrees is None else parse_float_list(args.frame_min_degrees),
            box_margins_px=None if args.box_margins_px is None else parse_float_list(args.box_margins_px),
            video_prediction_sources=(
                None
                if args.video_prediction_sources is None
                else parse_video_prediction_sources(args.video_prediction_sources)
            ),
        )
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"CADICA threshold sweep failed: {exc}", file=sys.stderr)
        return 2

    print(f"CADICA threshold sweep CSV: {outputs['cadica_threshold_sweep_csv']}")
    print(f"CADICA threshold sweep summary: {outputs['cadica_threshold_sweep_summary_json']}")
    return 0


def _resolve_float_values(values: list[float] | None, *, default: list[float], name: str) -> list[float]:
    resolved = list(default if values is None else values)
    if not resolved:
        raise ValueError(f"{name} must contain at least one value.")
    negative_values = [value for value in resolved if value < 0.0]
    if negative_values:
        raise ValueError(f"{name} values must be >= 0, got {negative_values}.")
    return resolved


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


def _sweep_row_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    frame_metrics = summary["frame_binary_metrics"]
    localization_metrics = summary["frame_localization_metrics"]
    box_metrics = summary["box_detection_metrics"]
    video_metrics = summary["video_binary_metrics"]
    patient_metrics = summary["patient_binary_metrics"]
    config = summary["config"]
    return {
        "frame_min_degree": config["frame_min_degree"],
        "box_margin_px": config["box_margin_px"],
        "video_prediction_source": config["video_prediction_source"],
        **_binary_metric_columns(frame_metrics, prefix="frame", include_fnr=True, include_predicted_positive=True),
        **localization_metrics,
        **box_metrics,
        **_binary_metric_columns(video_metrics, prefix="video", include_fnr=True, include_predicted_positive=False),
        **_binary_metric_columns(patient_metrics, prefix="patient", include_fnr=False, include_predicted_positive=False),
    }


def _binary_metric_columns(
    metrics: dict[str, Any],
    *,
    prefix: str,
    include_fnr: bool,
    include_predicted_positive: bool,
) -> dict[str, Any]:
    row = {
        f"{prefix}_total_evaluated": metrics["total_evaluated"],
        f"{prefix}_TP": metrics["TP"],
        f"{prefix}_FP": metrics["FP"],
        f"{prefix}_TN": metrics["TN"],
        f"{prefix}_FN": metrics["FN"],
        f"{prefix}_accuracy": metrics["accuracy"],
        f"{prefix}_precision": metrics["precision"],
        f"{prefix}_recall": metrics["recall"],
        f"{prefix}_specificity": metrics["specificity"],
        f"{prefix}_F1": metrics["f1"],
        f"{prefix}_false_positive_rate": metrics["false_positive_rate"],
        f"{prefix}_balanced_accuracy": metrics["balanced_accuracy"],
    }
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
    frame_min_degrees: list[float],
    box_margins_px: list[float],
    video_prediction_sources: list[str],
) -> dict[str, Any]:
    return {
        "note": CADICA_SUPERVISED_NOTE,
        "config": {
            "manifest": str(manifest),
            "frame_results_root": str(frame_results_root),
            "output_root": str(output_root),
            "temporal_results_root": None if temporal_results_root is None else str(temporal_results_root),
            "frame_min_degrees": frame_min_degrees,
            "box_margins_px": box_margins_px,
            "video_prediction_sources": video_prediction_sources,
        },
        "evaluated_sweep_combinations": len(rows),
        "best_operating_points": {
            "best_frame_F1": _best_metric(rows, "frame_F1"),
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
            "best_video_F1": _best_metric(rows, "video_F1"),
            "best_patient_F1": _best_metric(rows, "patient_F1"),
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

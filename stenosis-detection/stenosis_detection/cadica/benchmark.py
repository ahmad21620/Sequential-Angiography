from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping


FRAME_RESULT_SUFFIX = "_stenosis_results.json"
TEMPORAL_RESULT_FILENAME = "view_temporal_fusion.json"
MULTIVIEW_RESULT_FILENAME = "case_multiview_fusion.json"
VIDEO_PREDICTION_SOURCES = {"frame_any", "temporal_final"}
MULTIVIEW_SIDES = ("left", "right")
PROJECTION_SIDE_BY_GROUP = {
    "LCA": "left",
    "LCA2": "left",
    "RCA": "right",
}
CADICA_PATIENT_ID_RE = re.compile(r"^p\d+$", re.IGNORECASE)
CADICA_VIDEO_ID_RE = re.compile(r"^v\d+$", re.IGNORECASE)
REVIEW_CATEGORY_DIRS = {
    "false_positive": "false_positive",
    "false_negative": "false_negative",
    "true_positive_localized": "true_positive_localized",
    "true_positive_nonlocalized": "true_positive_nonlocalized",
}
CADICA_SUPERVISED_NOTE = (
    "This is a supervised CADICA frame/video benchmark using CADICA bounding-box annotations. "
    "It is separate from EHR weak-label agreement metrics."
)


@dataclass(frozen=True, slots=True)
class CadicaBenchmarkOutputs:
    frame_rows_csv: Path
    frame_rows_jsonl: Path
    box_rows_csv: Path
    video_rows_csv: Path
    patient_rows_csv: Path
    summary_json: Path
    false_positive_frames_csv: Path
    false_negative_frames_csv: Path
    missed_gt_boxes_csv: Path
    unmatched_predicted_points_csv: Path
    multiview_patient_rows_csv: Path | None = None
    multiview_side_rows_csv: Path | None = None
    review_image_root: Path | None = None

    def to_dict(self) -> dict[str, str]:
        payload = {
            "frame_rows_csv": str(self.frame_rows_csv),
            "frame_rows_jsonl": str(self.frame_rows_jsonl),
            "box_rows_csv": str(self.box_rows_csv),
            "video_rows_csv": str(self.video_rows_csv),
            "patient_rows_csv": str(self.patient_rows_csv),
            "summary_json": str(self.summary_json),
            "false_positive_frames_csv": str(self.false_positive_frames_csv),
            "false_negative_frames_csv": str(self.false_negative_frames_csv),
            "missed_gt_boxes_csv": str(self.missed_gt_boxes_csv),
            "unmatched_predicted_points_csv": str(self.unmatched_predicted_points_csv),
        }
        if self.multiview_patient_rows_csv is not None:
            payload["multiview_patient_rows_csv"] = str(self.multiview_patient_rows_csv)
        if self.multiview_side_rows_csv is not None:
            payload["multiview_side_rows_csv"] = str(self.multiview_side_rows_csv)
        if self.review_image_root is not None:
            payload["review_image_root"] = str(self.review_image_root)
        return payload


@dataclass(frozen=True, slots=True)
class CadicaBenchmarkResult:
    frame_rows: list[dict[str, Any]]
    box_rows: list[dict[str, Any]]
    video_rows: list[dict[str, Any]]
    patient_rows: list[dict[str, Any]]
    multiview_patient_rows: list[dict[str, Any]]
    multiview_side_rows: list[dict[str, Any]]
    summary: dict[str, Any]
    outputs: CadicaBenchmarkOutputs


@dataclass(frozen=True, slots=True)
class ManifestFrame:
    patient_id: str
    video_id: str
    frame_id: int
    prepared_image_name: str
    prepared_image_stem: str
    original_image_name: str
    original_image_path: Path | None
    prepared_image_path: Path | None
    frame_label: str
    video_label: str
    coronary_side: str | None
    projection_group: str | None
    boxes: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class PredictedPoint:
    x: float
    y: float
    degree: float
    severity: str | None


@dataclass(frozen=True, slots=True)
class FramePrediction:
    source_path: Path
    image_name: str
    image_stem: str
    view_id: str
    points: list[PredictedPoint]


@dataclass(frozen=True, slots=True)
class MultiViewPrediction:
    source_path: Path
    predicted_positive: bool
    score: float | None
    lesion_present: bool | None
    score_rule: str
    side_predictions: dict[str, "MultiViewSidePrediction"]
    skip_reason: str = ""


@dataclass(frozen=True, slots=True)
class MultiViewSidePrediction:
    predicted_positive: bool
    score: float | None
    lesion_present: bool | None
    score_rule: str
    skip_reason: str = ""


def run_cadica_benchmark(
    *,
    manifest: str | Path,
    frame_results_root: str | Path,
    output_root: str | Path,
    temporal_results_root: str | Path | None = None,
    multiview_results_root: str | Path | None = None,
    frame_min_degree: float = 0.0,
    box_margin_px: float = 5.0,
    video_prediction_source: str = "frame_any",
    multiview_min_score: float = 0.0,
    write_review_images: bool = False,
    max_review_images: int = 100,
    review_image_root: str | Path | None = None,
) -> CadicaBenchmarkResult:
    if frame_min_degree < 0.0:
        raise ValueError("frame_min_degree must be >= 0.0.")
    if box_margin_px < 0.0:
        raise ValueError("box_margin_px must be >= 0.")
    if video_prediction_source not in VIDEO_PREDICTION_SOURCES:
        raise ValueError(
            f"video_prediction_source must be one of {sorted(VIDEO_PREDICTION_SOURCES)}, "
            f"got {video_prediction_source!r}."
        )
    if video_prediction_source == "temporal_final" and temporal_results_root is None:
        raise ValueError("--video-prediction-source temporal_final requires --temporal-results-root.")
    if multiview_min_score < 0.0:
        raise ValueError("multiview_min_score must be >= 0.0.")
    if max_review_images < 0:
        raise ValueError("max_review_images must be >= 0.")

    manifest_path = Path(manifest)
    resolved_frame_results_root = Path(frame_results_root)
    resolved_output_root = Path(output_root)
    resolved_temporal_root = None if temporal_results_root is None else Path(temporal_results_root)
    resolved_multiview_root = None if multiview_results_root is None else Path(multiview_results_root)
    resolved_review_image_root = (
        None
        if not write_review_images
        else Path(review_image_root) if review_image_root is not None else resolved_output_root / "review_images"
    )
    _validate_inputs(manifest_path, resolved_frame_results_root, resolved_temporal_root, resolved_multiview_root)

    manifest_frames = load_cadica_manifest(manifest_path)
    frame_predictions = _index_frame_predictions(resolved_frame_results_root)
    frame_rows, box_rows, unmatched_point_rows = _evaluate_frames(
        manifest_frames,
        frame_predictions,
        frame_min_degree=frame_min_degree,
        box_margin_px=box_margin_px,
    )
    video_rows = _build_video_rows(
        frame_rows,
        temporal_results_root=resolved_temporal_root,
        video_prediction_source=video_prediction_source,
    )
    patient_rows = _build_patient_rows(video_rows)
    multiview_patient_rows, multiview_side_rows, multiview_score_rule = _build_multiview_rows(
        manifest_frames,
        multiview_results_root=resolved_multiview_root,
        multiview_min_score=multiview_min_score,
    )
    summary = _build_summary(
        frame_rows,
        box_rows,
        video_rows,
        patient_rows,
        multiview_patient_rows,
        multiview_side_rows,
        manifest=manifest_path,
        frame_results_root=resolved_frame_results_root,
        output_root=resolved_output_root,
        temporal_results_root=resolved_temporal_root,
        multiview_results_root=resolved_multiview_root,
        frame_min_degree=frame_min_degree,
        box_margin_px=box_margin_px,
        video_prediction_source=video_prediction_source,
        multiview_min_score=multiview_min_score,
        multiview_score_rule=multiview_score_rule,
        write_review_images=write_review_images,
        max_review_images=max_review_images,
        review_image_root=resolved_review_image_root,
    )
    outputs = save_cadica_benchmark_outputs(
        output_root=resolved_output_root,
        frame_rows=frame_rows,
        box_rows=box_rows,
        video_rows=video_rows,
        patient_rows=patient_rows,
        multiview_patient_rows=multiview_patient_rows,
        multiview_side_rows=multiview_side_rows,
        summary=summary,
        unmatched_point_rows=unmatched_point_rows,
        review_image_root=resolved_review_image_root,
    )
    if write_review_images and resolved_review_image_root is not None:
        write_cadica_review_images(
            manifest_frames=manifest_frames,
            frame_predictions=frame_predictions,
            frame_rows=frame_rows,
            review_image_root=resolved_review_image_root,
            frame_min_degree=frame_min_degree,
            max_review_images=max_review_images,
        )
    return CadicaBenchmarkResult(
        frame_rows=frame_rows,
        box_rows=box_rows,
        video_rows=video_rows,
        patient_rows=patient_rows,
        multiview_patient_rows=multiview_patient_rows,
        multiview_side_rows=multiview_side_rows,
        summary=summary,
        outputs=outputs,
    )


def load_cadica_manifest(path: str | Path) -> list[ManifestFrame]:
    manifest_path = Path(path)
    if manifest_path.suffix.lower() == ".jsonl":
        raw_rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    else:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            raw_rows = list(csv.DictReader(handle))

    return [_load_manifest_frame(row, row_number=index + 1, manifest_path=manifest_path) for index, row in enumerate(raw_rows)]


def save_cadica_benchmark_outputs(
    *,
    output_root: str | Path,
    frame_rows: list[dict[str, Any]],
    box_rows: list[dict[str, Any]],
    video_rows: list[dict[str, Any]],
    patient_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    unmatched_point_rows: list[dict[str, Any]],
    multiview_patient_rows: list[dict[str, Any]] | None = None,
    multiview_side_rows: list[dict[str, Any]] | None = None,
    review_image_root: Path | None = None,
) -> CadicaBenchmarkOutputs:
    resolved_output_root = Path(output_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    outputs = CadicaBenchmarkOutputs(
        frame_rows_csv=resolved_output_root / "cadica_frame_rows.csv",
        frame_rows_jsonl=resolved_output_root / "cadica_frame_rows.jsonl",
        box_rows_csv=resolved_output_root / "cadica_box_rows.csv",
        video_rows_csv=resolved_output_root / "cadica_video_rows.csv",
        patient_rows_csv=resolved_output_root / "cadica_patient_rows.csv",
        summary_json=resolved_output_root / "cadica_summary.json",
        false_positive_frames_csv=resolved_output_root / "false_positive_frames.csv",
        false_negative_frames_csv=resolved_output_root / "false_negative_frames.csv",
        missed_gt_boxes_csv=resolved_output_root / "missed_gt_boxes.csv",
        unmatched_predicted_points_csv=resolved_output_root / "unmatched_predicted_points.csv",
        multiview_patient_rows_csv=(
            resolved_output_root / "cadica_multiview_patient_rows.csv"
            if multiview_patient_rows is not None
            else None
        ),
        multiview_side_rows_csv=(
            resolved_output_root / "cadica_multiview_side_rows.csv"
            if multiview_side_rows
            else None
        ),
        review_image_root=review_image_root,
    )

    _write_csv(outputs.frame_rows_csv, frame_rows, fieldnames=FRAME_ROW_FIELDS)
    _write_jsonl(outputs.frame_rows_jsonl, frame_rows)
    _write_csv(outputs.box_rows_csv, box_rows, fieldnames=BOX_ROW_FIELDS)
    _write_csv(outputs.video_rows_csv, video_rows, fieldnames=VIDEO_ROW_FIELDS)
    _write_csv(outputs.patient_rows_csv, patient_rows, fieldnames=PATIENT_ROW_FIELDS)
    if outputs.multiview_patient_rows_csv is not None and multiview_patient_rows is not None:
        _write_csv(
            outputs.multiview_patient_rows_csv,
            multiview_patient_rows,
            fieldnames=MULTIVIEW_PATIENT_ROW_FIELDS,
        )
    if outputs.multiview_side_rows_csv is not None and multiview_side_rows is not None:
        _write_csv(
            outputs.multiview_side_rows_csv,
            multiview_side_rows,
            fieldnames=MULTIVIEW_SIDE_ROW_FIELDS,
        )
    _write_json(outputs.summary_json, summary)
    _write_csv(
        outputs.false_positive_frames_csv,
        [row for row in frame_rows if row["outcome"] == "FP"],
        fieldnames=FRAME_ROW_FIELDS,
    )
    _write_csv(
        outputs.false_negative_frames_csv,
        [row for row in frame_rows if row["outcome"] == "FN"],
        fieldnames=FRAME_ROW_FIELDS,
    )
    _write_csv(outputs.missed_gt_boxes_csv, [row for row in box_rows if not row["matched"]], fieldnames=BOX_ROW_FIELDS)
    _write_csv(outputs.unmatched_predicted_points_csv, unmatched_point_rows, fieldnames=UNMATCHED_POINT_ROW_FIELDS)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a supervised CADICA benchmark against existing stenosis pipeline outputs.",
    )
    parser.add_argument("--manifest", required=True, help="Path to CADICA prepared manifest.csv or manifest.jsonl.")
    parser.add_argument("--frame-results-root", required=True, help="Root containing frame-level stenosis JSON outputs.")
    parser.add_argument("--output-root", required=True, help="Directory where CADICA benchmark outputs will be written.")
    parser.add_argument(
        "--temporal-results-root",
        help="Optional root containing view_temporal_fusion.json outputs grouped by patient/video.",
    )
    parser.add_argument(
        "--multiview-results-root",
        help="Optional root containing case_multiview_fusion.json outputs grouped by patient/case.",
    )
    parser.add_argument(
        "--frame-min-degree",
        type=float,
        default=0.0,
        help="Minimum stenosis point degree required to count a frame prediction as positive.",
    )
    parser.add_argument(
        "--box-margin-px",
        type=float,
        default=5.0,
        help="Pixel margin used to expand CADICA boxes for point-in-box localization.",
    )
    parser.add_argument(
        "--video-prediction-source",
        choices=sorted(VIDEO_PREDICTION_SOURCES),
        default="frame_any",
        help="Use any positive frame or temporal final lesion presence for video-level predictions.",
    )
    parser.add_argument(
        "--multiview-min-score",
        type=float,
        default=0.0,
        help="Minimum multi-view score required to count a case-level prediction as positive.",
    )
    parser.add_argument(
        "--write-review-images",
        action="store_true",
        help="Write simple visual review images for selected CADICA benchmark outcomes.",
    )
    parser.add_argument(
        "--max-review-images",
        type=int,
        default=100,
        help="Maximum total number of review images to write when --write-review-images is enabled.",
    )
    parser.add_argument(
        "--review-image-root",
        help="Optional review image output root. Defaults to <output-root>/review_images.",
    )
    parser.add_argument(
        "--write-threshold-sweep",
        action="store_true",
        help="Also write a supervised CADICA threshold sweep CSV/JSON under --output-root.",
    )
    parser.add_argument(
        "--sweep-frame-min-degrees",
        help="Comma-separated frame degree thresholds for --write-threshold-sweep.",
    )
    parser.add_argument(
        "--sweep-box-margins-px",
        help="Comma-separated CADICA box margins in pixels for --write-threshold-sweep.",
    )
    parser.add_argument(
        "--sweep-video-prediction-sources",
        help="Comma-separated sources for --write-threshold-sweep: frame_any, temporal_final.",
    )
    parser.add_argument(
        "--sweep-multiview-min-scores",
        help="Comma-separated multi-view score thresholds for --write-threshold-sweep.",
    )
    parser.add_argument(
        "--sweep-workers",
        type=int,
        default=1,
        help="Number of worker processes used when --write-threshold-sweep is enabled.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    sweep_outputs: dict[str, Path] | None = None
    try:
        result = run_cadica_benchmark(
            manifest=args.manifest,
            frame_results_root=args.frame_results_root,
            output_root=args.output_root,
            temporal_results_root=args.temporal_results_root,
            multiview_results_root=args.multiview_results_root,
            frame_min_degree=args.frame_min_degree,
            box_margin_px=args.box_margin_px,
            video_prediction_source=args.video_prediction_source,
            multiview_min_score=args.multiview_min_score,
            write_review_images=args.write_review_images,
            max_review_images=args.max_review_images,
            review_image_root=args.review_image_root,
        )
        if args.write_threshold_sweep:
            from .sweep import parse_float_list, parse_video_prediction_sources, run_cadica_threshold_sweep

            sweep_outputs = run_cadica_threshold_sweep(
                manifest=args.manifest,
                frame_results_root=args.frame_results_root,
                output_root=args.output_root,
                temporal_results_root=args.temporal_results_root,
                multiview_results_root=args.multiview_results_root,
                frame_min_degrees=(
                    None
                    if args.sweep_frame_min_degrees is None
                    else parse_float_list(args.sweep_frame_min_degrees)
                ),
                box_margins_px=(
                    None if args.sweep_box_margins_px is None else parse_float_list(args.sweep_box_margins_px)
                ),
                video_prediction_sources=(
                    None
                    if args.sweep_video_prediction_sources is None
                    else parse_video_prediction_sources(args.sweep_video_prediction_sources)
                ),
                multiview_min_scores=(
                    None
                    if args.sweep_multiview_min_scores is None
                    else parse_float_list(args.sweep_multiview_min_scores)
                ),
                workers=args.sweep_workers,
            )
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"CADICA benchmark failed: {exc}", file=sys.stderr)
        return 2

    frame_metrics = result.summary["frame_binary_metrics"]
    box_metrics = result.summary["box_detection_metrics"]
    print(f"CADICA frame rows: {result.outputs.frame_rows_csv}")
    print(f"Evaluated frames: {frame_metrics['total_evaluated']}")
    print(f"Frame F1: {_format_optional_float(frame_metrics['f1'])}")
    print(f"Box recall: {_format_optional_float(box_metrics['box_recall'])}")
    if result.summary["config"]["multiview_evaluated"]:
        multiview_metrics = result.summary["multiview_patient_binary_metrics"] or {}
        print(f"Multi-view patient F1: {_format_optional_float(multiview_metrics.get('f1'))}")
    print(f"Summary: {result.outputs.summary_json}")
    if sweep_outputs is not None:
        print(f"Threshold sweep CSV: {sweep_outputs['cadica_threshold_sweep_csv']}")
        print(f"Threshold sweep summary: {sweep_outputs['cadica_threshold_sweep_summary_json']}")
    if result.outputs.review_image_root is not None:
        print(f"Review images: {result.outputs.review_image_root}")
    return 0


def _evaluate_frames(
    manifest_frames: list[ManifestFrame],
    frame_predictions: dict[tuple[str, str, str], FramePrediction],
    *,
    frame_min_degree: float,
    box_margin_px: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    frame_rows: list[dict[str, Any]] = []
    box_rows: list[dict[str, Any]] = []
    unmatched_point_rows: list[dict[str, Any]] = []

    for manifest_frame in manifest_frames:
        prediction = _lookup_frame_prediction(manifest_frame, frame_predictions)
        predicted_points = [] if prediction is None else _threshold_points(prediction.points, frame_min_degree)
        score = 0.0 if prediction is None else max((point.degree for point in prediction.points), default=0.0)
        predicted_positive = bool(predicted_points)
        match_result = _match_points_to_boxes(predicted_points, manifest_frame.boxes, margin=box_margin_px)
        label_target = _frame_label_target(manifest_frame.frame_label)
        outcome = "" if label_target is None else _binary_outcome(label_target=label_target, predicted_positive=predicted_positive)
        skip_reason = "unknown_frame_label" if label_target is None else ""

        frame_row = {
            "patient_id": manifest_frame.patient_id,
            "video_id": manifest_frame.video_id,
            "frame_id": manifest_frame.frame_id,
            "prepared_image_name": manifest_frame.prepared_image_name,
            "original_image_name": manifest_frame.original_image_name,
            "frame_label": manifest_frame.frame_label,
            "video_label": manifest_frame.video_label,
            "predicted_positive": predicted_positive,
            "score": score,
            "predicted_point_count": len(predicted_points),
            "gt_box_count": len(manifest_frame.boxes),
            "localized_positive": bool(match_result["localized_point_indices"]),
            "matched_gt_box_count": len(match_result["matched_box_indices"]),
            "unmatched_gt_box_count": len(manifest_frame.boxes) - len(match_result["matched_box_indices"]),
            "unmatched_predicted_point_count": len(predicted_points) - len(match_result["localized_point_indices"]),
            "outcome": outcome,
            "skip_reason": skip_reason,
        }
        frame_rows.append(frame_row)

        for box_index, box in enumerate(manifest_frame.boxes):
            matched_point = match_result["box_matches"].get(box_index)
            box_rows.append(
                {
                    "patient_id": manifest_frame.patient_id,
                    "video_id": manifest_frame.video_id,
                    "frame_id": manifest_frame.frame_id,
                    "box_index": box_index,
                    "x": _optional_float(box.get("x")),
                    "y": _optional_float(box.get("y")),
                    "w": _optional_float(box.get("w")),
                    "h": _optional_float(box.get("h")),
                    "category": box.get("category"),
                    "matched": matched_point is not None,
                    "matched_point_x": None if matched_point is None else matched_point.x,
                    "matched_point_y": None if matched_point is None else matched_point.y,
                    "matched_point_degree": None if matched_point is None else matched_point.degree,
                    "source_path": box.get("source_path"),
                }
            )

        for point_index, point in enumerate(predicted_points):
            if point_index in match_result["localized_point_indices"]:
                continue
            unmatched_point_rows.append(
                {
                    "patient_id": manifest_frame.patient_id,
                    "video_id": manifest_frame.video_id,
                    "frame_id": manifest_frame.frame_id,
                    "prepared_image_name": manifest_frame.prepared_image_name,
                    "frame_label": manifest_frame.frame_label,
                    "video_label": manifest_frame.video_label,
                    "point_index": point_index,
                    "x": point.x,
                    "y": point.y,
                    "degree": point.degree,
                    "severity": point.severity,
                }
            )

    return frame_rows, box_rows, unmatched_point_rows


def _build_video_rows(
    frame_rows: list[dict[str, Any]],
    *,
    temporal_results_root: Path | None,
    video_prediction_source: str,
    temporal_predictions: Mapping[tuple[str, str], tuple[bool | None, float | None]] | None = None,
) -> list[dict[str, Any]]:
    rows_by_video: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for frame_row in frame_rows:
        rows_by_video[(str(frame_row["patient_id"]), str(frame_row["video_id"]))].append(frame_row)

    video_rows: list[dict[str, Any]] = []
    for (patient_id, video_id), rows in sorted(rows_by_video.items()):
        video_label = _resolve_video_label(rows)
        label_target = _video_label_target(video_label)
        frame_predicted_positive = any(bool(row["predicted_positive"]) for row in rows)
        frame_score = max((_optional_float(row.get("score")) or 0.0 for row in rows), default=0.0)
        if temporal_predictions is None:
            temporal_present, temporal_score = _load_temporal_prediction(temporal_results_root, patient_id, video_id)
        else:
            temporal_present, temporal_score = temporal_predictions.get((patient_id.lower(), video_id.lower()), (False, 0.0))
        if video_prediction_source == "temporal_final":
            predicted_positive = bool(temporal_present)
            score = temporal_score
        else:
            predicted_positive = frame_predicted_positive
            score = frame_score
        outcome = "" if label_target is None else _binary_outcome(label_target=label_target, predicted_positive=predicted_positive)
        video_rows.append(
            {
                "patient_id": patient_id,
                "video_id": video_id,
                "video_label": video_label,
                "predicted_positive": predicted_positive,
                "score": score,
                "positive_frame_count": sum(1 for row in rows if row["frame_label"] == "positive"),
                "predicted_positive_frame_count": sum(1 for row in rows if bool(row["predicted_positive"])),
                "temporal_final_lesion_present": (
                    temporal_present if temporal_results_root is not None or temporal_predictions is not None else None
                ),
                "outcome": outcome,
            }
        )
    return video_rows


def _build_patient_rows(video_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for video_row in video_rows:
        rows_by_patient[str(video_row["patient_id"])].append(video_row)

    patient_rows: list[dict[str, Any]] = []
    for patient_id, rows in sorted(rows_by_patient.items()):
        label_positive = any(row["video_label"] == "lesion" for row in rows)
        predicted_positive = any(bool(row["predicted_positive"]) for row in rows)
        patient_rows.append(
            {
                "patient_id": patient_id,
                "label_positive": label_positive,
                "predicted_positive": predicted_positive,
                "lesion_video_count": sum(1 for row in rows if row["video_label"] == "lesion"),
                "predicted_positive_video_count": sum(1 for row in rows if bool(row["predicted_positive"])),
                "outcome": _binary_outcome(label_target=label_positive, predicted_positive=predicted_positive),
            }
        )
    return patient_rows


def _build_multiview_rows(
    manifest_frames: list[ManifestFrame],
    *,
    multiview_results_root: Path | None,
    multiview_min_score: float,
    multiview_predictions: Mapping[str, MultiViewPrediction] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    if multiview_results_root is None and multiview_predictions is None:
        return [], [], None

    predictions = (
        dict(multiview_predictions)
        if multiview_predictions is not None
        else _index_multiview_predictions(multiview_results_root, multiview_min_score=multiview_min_score)
    )
    patient_label_rows = _multiview_patient_label_rows(manifest_frames)
    patient_rows = [
        _multiview_patient_row(label_row, predictions.get(label_row["patient_id"]))
        for label_row in patient_label_rows
    ]
    side_label_rows, side_skip_reason = _multiview_side_label_rows(manifest_frames)
    side_rows = [
        _multiview_side_row(label_row, predictions.get(label_row["patient_id"]))
        for label_row in side_label_rows
    ]
    if not side_rows and side_skip_reason:
        for row in patient_rows:
            row["side_skip_reason"] = side_skip_reason
    score_rule = _summarize_multiview_score_rule(predictions.values())
    return patient_rows, side_rows, score_rule


def _multiview_patient_label_rows(manifest_frames: list[ManifestFrame]) -> list[dict[str, Any]]:
    rows_by_patient: dict[str, list[ManifestFrame]] = defaultdict(list)
    for manifest_frame in manifest_frames:
        rows_by_patient[manifest_frame.patient_id].append(manifest_frame)

    label_rows: list[dict[str, Any]] = []
    for patient_id, rows in sorted(rows_by_patient.items()):
        label_positive = _manifest_rows_label_target(rows)
        lesion_video_ids = {row.video_id for row in rows if row.video_label == "lesion"}
        label_rows.append(
            {
                "patient_id": patient_id,
                "label_positive": label_positive,
                "lesion_video_count": len(lesion_video_ids),
                "positive_frame_count": sum(1 for row in rows if row.frame_label == "positive"),
            }
        )
    return label_rows


def _multiview_side_label_rows(manifest_frames: list[ManifestFrame]) -> tuple[list[dict[str, Any]], str | None]:
    if not any(_manifest_frame_side(row) is not None for row in manifest_frames):
        return [], "missing_side_metadata"

    rows_by_patient_side: dict[tuple[str, str], list[ManifestFrame]] = defaultdict(list)
    for manifest_frame in manifest_frames:
        side = _manifest_frame_side(manifest_frame)
        if side is None:
            continue
        rows_by_patient_side[(manifest_frame.patient_id, side)].append(manifest_frame)

    label_rows: list[dict[str, Any]] = []
    for (patient_id, coronary_side), rows in sorted(rows_by_patient_side.items()):
        label_positive = _manifest_rows_label_target(rows)
        lesion_video_ids = {row.video_id for row in rows if row.video_label == "lesion"}
        label_rows.append(
            {
                "patient_id": patient_id,
                "coronary_side": coronary_side,
                "label_positive": label_positive,
                "lesion_video_count": len(lesion_video_ids),
                "positive_frame_count": sum(1 for row in rows if row.frame_label == "positive"),
            }
        )
    return label_rows, None


def _manifest_rows_label_target(rows: list[ManifestFrame]) -> bool | None:
    if any(row.video_label == "lesion" or row.frame_label == "positive" for row in rows):
        return True
    known_negative_rows = [
        row
        for row in rows
        if row.video_label == "nonlesion" or row.frame_label == "negative"
    ]
    if known_negative_rows and len(known_negative_rows) == len(rows):
        return False
    return None


def _manifest_frame_side(manifest_frame: ManifestFrame) -> str | None:
    side = (manifest_frame.coronary_side or "").strip().lower()
    if side in MULTIVIEW_SIDES:
        return side
    group = (manifest_frame.projection_group or "").strip().upper()
    return PROJECTION_SIDE_BY_GROUP.get(group)


def _multiview_patient_row(
    label_row: dict[str, Any],
    prediction: MultiViewPrediction | None,
) -> dict[str, Any]:
    label_target = label_row["label_positive"]
    predicted_positive = False if prediction is None else prediction.predicted_positive
    skip_reason = _multiview_skip_reason(label_target, prediction)
    return {
        "patient_id": label_row["patient_id"],
        "label_positive": label_target,
        "predicted_positive": predicted_positive,
        "score": None if prediction is None else prediction.score,
        "outcome": "" if skip_reason else _binary_outcome(label_target=label_target, predicted_positive=predicted_positive),
        "lesion_video_count": label_row["lesion_video_count"],
        "positive_frame_count": label_row["positive_frame_count"],
        "multiview_json_path": "" if prediction is None else str(prediction.source_path),
        "skip_reason": skip_reason,
    }


def _multiview_side_row(
    label_row: dict[str, Any],
    prediction: MultiViewPrediction | None,
) -> dict[str, Any]:
    side = str(label_row["coronary_side"])
    side_prediction = None if prediction is None else prediction.side_predictions.get(side)
    label_target = label_row["label_positive"]
    predicted_positive = False if side_prediction is None else side_prediction.predicted_positive
    skip_reason = _multiview_skip_reason(label_target, prediction)
    if not skip_reason and side_prediction is None:
        skip_reason = "missing_side_prediction"
    return {
        "patient_id": label_row["patient_id"],
        "coronary_side": side,
        "label_positive": label_target,
        "predicted_positive": predicted_positive,
        "score": None if side_prediction is None else side_prediction.score,
        "outcome": "" if skip_reason else _binary_outcome(label_target=label_target, predicted_positive=predicted_positive),
        "lesion_video_count": label_row["lesion_video_count"],
        "positive_frame_count": label_row["positive_frame_count"],
        "multiview_json_path": "" if prediction is None else str(prediction.source_path),
        "skip_reason": skip_reason,
    }


def _multiview_skip_reason(label_target: bool | None, prediction: MultiViewPrediction | None) -> str:
    if label_target is None:
        return "unknown_label"
    if prediction is None:
        return "missing_multiview_result"
    return prediction.skip_reason


def _index_multiview_predictions(
    multiview_results_root: Path,
    *,
    multiview_min_score: float,
) -> dict[str, MultiViewPrediction]:
    predictions: dict[str, MultiViewPrediction] = {}
    for result_path in sorted(multiview_results_root.rglob(MULTIVIEW_RESULT_FILENAME)):
        if not result_path.is_file():
            continue
        patient_id = _multiview_patient_id_from_path(result_path, multiview_results_root)
        payload = _read_json_object(result_path)
        predictions.setdefault(
            patient_id,
            _extract_multiview_prediction(result_path, payload, multiview_min_score=multiview_min_score),
        )
    return predictions


def _multiview_patient_id_from_path(result_path: Path, multiview_results_root: Path) -> str:
    try:
        relative_path = result_path.resolve().relative_to(multiview_results_root.resolve())
    except ValueError:
        relative_path = result_path
    patient_id = _cadica_patient_id_from_parts(relative_path.parts)
    if patient_id is not None:
        return patient_id
    if len(relative_path.parts) >= 2:
        return relative_path.parts[0]
    return result_path.parent.name


def _extract_multiview_prediction(
    result_path: Path,
    payload: dict[str, Any],
    *,
    multiview_min_score: float,
) -> MultiViewPrediction:
    if payload.get("split_by_coronary_side") is True and isinstance(payload.get("side_results"), dict):
        side_predictions: dict[str, MultiViewSidePrediction] = {}
        for side in MULTIVIEW_SIDES:
            side_payload = payload["side_results"].get(side)
            if isinstance(side_payload, dict):
                side_predictions[side] = _extract_multiview_side_prediction(
                    side_payload,
                    multiview_min_score=multiview_min_score,
                )
        best_side_prediction = _select_best_multiview_side(side_predictions)
        predicted_positive = any(side_prediction.predicted_positive for side_prediction in side_predictions.values())
        return MultiViewPrediction(
            source_path=result_path,
            predicted_positive=predicted_positive,
            score=None if best_side_prediction is None else best_side_prediction.score,
            lesion_present=None if best_side_prediction is None else best_side_prediction.lesion_present,
            score_rule="split_side_any_positive",
            side_predictions=side_predictions,
            skip_reason="" if side_predictions else "missing_side_results",
        )

    side_predictions = _side_predictions_from_overall_payload(payload, multiview_min_score=multiview_min_score)
    single_prediction = _extract_multiview_side_prediction(payload, multiview_min_score=multiview_min_score)
    return MultiViewPrediction(
        source_path=result_path,
        predicted_positive=single_prediction.predicted_positive,
        score=single_prediction.score,
        lesion_present=single_prediction.lesion_present,
        score_rule=single_prediction.score_rule,
        side_predictions=side_predictions,
        skip_reason=single_prediction.skip_reason,
    )


def _extract_multiview_side_prediction(
    payload: dict[str, Any],
    *,
    multiview_min_score: float,
) -> MultiViewSidePrediction:
    lesion_present = _extract_lesion_present(payload)
    score = _extract_multiview_score(payload)
    score_for_threshold = 0.0 if score is None else score
    if lesion_present is None:
        predicted_positive = score_for_threshold >= multiview_min_score
        score_rule = "score_threshold_without_explicit_lesion_present"
    else:
        predicted_positive = lesion_present and score_for_threshold >= multiview_min_score
        score_rule = "explicit_lesion_present_and_score_threshold"
    return MultiViewSidePrediction(
        predicted_positive=predicted_positive,
        score=score,
        lesion_present=lesion_present,
        score_rule=score_rule,
    )


def _extract_lesion_present(payload: dict[str, Any]) -> bool | None:
    for key in ("lesion_present", "final_lesion_present", "case_lesion_present"):
        value = payload.get(key)
        if isinstance(value, bool):
            return value
    overall_result = payload.get("overall_result")
    if isinstance(overall_result, dict):
        nested = _extract_lesion_present(overall_result)
        if nested is not None:
            return nested
    if "final_case_lesion" in payload:
        return isinstance(payload.get("final_case_lesion"), dict)
    if "final_lesion" in payload:
        return isinstance(payload.get("final_lesion"), dict)
    return None


def _extract_multiview_score(payload: dict[str, Any]) -> float | None:
    overall_result = payload.get("overall_result")
    if isinstance(overall_result, dict):
        nested_score = _extract_multiview_score(overall_result)
        if nested_score is not None:
            return nested_score

    final_case_lesion = payload.get("final_case_lesion") if isinstance(payload.get("final_case_lesion"), dict) else {}
    final_lesion = payload.get("final_lesion") if isinstance(payload.get("final_lesion"), dict) else {}
    confidence = payload.get("confidence") if isinstance(payload.get("confidence"), dict) else {}
    lesion_confidence = (
        final_case_lesion.get("confidence")
        if isinstance(final_case_lesion.get("confidence"), dict)
        else final_lesion.get("confidence") if isinstance(final_lesion.get("confidence"), dict) else {}
    )
    degrees = (
        final_case_lesion.get("degrees")
        if isinstance(final_case_lesion.get("degrees"), dict)
        else final_lesion.get("degrees") if isinstance(final_lesion.get("degrees"), dict) else {}
    )
    for value in (
        confidence.get("score"),
        lesion_confidence.get("score"),
        payload.get("confidence_score"),
        payload.get("score"),
        final_case_lesion.get("total_score"),
        final_lesion.get("total_score"),
        degrees.get("median"),
        degrees.get("max"),
        payload.get("degree"),
    ):
        score = _optional_float(value)
        if score is not None:
            return score
    return 1.0 if _extract_lesion_present(payload) is True else 0.0


def _side_predictions_from_overall_payload(
    payload: dict[str, Any],
    *,
    multiview_min_score: float,
) -> dict[str, MultiViewSidePrediction]:
    side_predictions: dict[str, MultiViewSidePrediction] = {}
    overall_result = payload.get("overall_result")
    if isinstance(overall_result, dict):
        side_predictions.update(
            _side_predictions_from_overall_payload(overall_result, multiview_min_score=multiview_min_score)
        )
    for key in ("side_predictions", "side_results"):
        raw_side_payloads = payload.get(key)
        if not isinstance(raw_side_payloads, dict):
            continue
        for side in MULTIVIEW_SIDES:
            side_payload = raw_side_payloads.get(side)
            if isinstance(side_payload, dict):
                side_predictions[side] = _extract_multiview_side_prediction(
                    side_payload,
                    multiview_min_score=multiview_min_score,
                )
    return side_predictions


def _select_best_multiview_side(
    side_predictions: dict[str, MultiViewSidePrediction],
) -> MultiViewSidePrediction | None:
    if not side_predictions:
        return None
    return max(
        side_predictions.values(),
        key=lambda side_prediction: (
            side_prediction.predicted_positive,
            -1.0 if side_prediction.score is None else side_prediction.score,
        ),
    )


def _summarize_multiview_score_rule(predictions: Iterable[MultiViewPrediction]) -> str | None:
    rules = sorted({prediction.score_rule for prediction in predictions if prediction.score_rule})
    if not rules:
        return None
    return ",".join(rules)


def _multiview_side_summary_skip_reason(
    multiview_patient_rows: list[dict[str, Any]],
    multiview_side_rows: list[dict[str, Any]],
    *,
    multiview_results_root: Path | None,
) -> str | None:
    if multiview_results_root is None or multiview_side_rows:
        return None
    for row in multiview_patient_rows:
        reason = row.get("side_skip_reason")
        if reason:
            return str(reason)
    return "missing_side_metadata"


def _build_summary(
    frame_rows: list[dict[str, Any]],
    box_rows: list[dict[str, Any]],
    video_rows: list[dict[str, Any]],
    patient_rows: list[dict[str, Any]],
    multiview_patient_rows: list[dict[str, Any]],
    multiview_side_rows: list[dict[str, Any]],
    *,
    manifest: Path,
    frame_results_root: Path,
    output_root: Path,
    temporal_results_root: Path | None,
    multiview_results_root: Path | None,
    frame_min_degree: float,
    box_margin_px: float,
    video_prediction_source: str,
    multiview_min_score: float,
    multiview_score_rule: str | None,
    write_review_images: bool,
    max_review_images: int,
    review_image_root: Path | None,
) -> dict[str, Any]:
    evaluated_positive_frames = [row for row in frame_rows if row["frame_label"] == "positive"]
    localized_positive_frames = [row for row in evaluated_positive_frames if row["localized_positive"]]
    matched_gt_boxes = sum(1 for row in box_rows if bool(row["matched"]))
    total_gt_boxes = len(box_rows)
    return {
        "note": CADICA_SUPERVISED_NOTE,
        "config": {
            "manifest": str(manifest),
            "frame_results_root": str(frame_results_root),
            "output_root": str(output_root),
            "temporal_results_root": None if temporal_results_root is None else str(temporal_results_root),
            "multiview_results_root": None if multiview_results_root is None else str(multiview_results_root),
            "frame_min_degree": float(frame_min_degree),
            "box_margin_px": float(box_margin_px),
            "video_prediction_source": video_prediction_source,
            "multiview_min_score": float(multiview_min_score),
            "multiview_evaluated": multiview_results_root is not None,
            "multiview_score_rule": multiview_score_rule,
            "write_review_images": write_review_images,
            "max_review_images": int(max_review_images),
            "review_image_root": None if review_image_root is None else str(review_image_root),
        },
        "frame_binary_metrics": _binary_metrics_from_rows(frame_rows),
        "frame_localization_metrics": {
            "positive_evaluated_frames": len(evaluated_positive_frames),
            "positive_frames_with_localized_prediction": len(localized_positive_frames),
            "localization_recall_on_positive_frames": _safe_divide(
                len(localized_positive_frames),
                len(evaluated_positive_frames),
            ),
            "localized_predicted_positive_frames": sum(1 for row in frame_rows if row["localized_positive"]),
            "unmatched_predicted_points": sum(int(row["unmatched_predicted_point_count"]) for row in frame_rows),
        },
        "box_detection_metrics": {
            "total_gt_boxes": total_gt_boxes,
            "matched_gt_boxes": matched_gt_boxes,
            "missed_gt_boxes": total_gt_boxes - matched_gt_boxes,
            "box_recall": _safe_divide(matched_gt_boxes, total_gt_boxes),
        },
        "video_binary_metrics": _binary_metrics_from_rows(video_rows),
        "patient_binary_metrics": _binary_metrics_from_rows(patient_rows),
        "multiview_patient_binary_metrics": (
            _binary_metrics_from_rows(multiview_patient_rows) if multiview_results_root is not None else None
        ),
        "multiview_side_binary_metrics": (
            _binary_metrics_from_rows(multiview_side_rows) if multiview_side_rows else None
        ),
        "multiview_evaluated": multiview_results_root is not None,
        "multiview_side_skip_reason": _multiview_side_summary_skip_reason(
            multiview_patient_rows,
            multiview_side_rows,
            multiview_results_root=multiview_results_root,
        ),
        "counts": {
            "frame_label_counts": dict(sorted(Counter(str(row["frame_label"]) for row in frame_rows).items())),
            "video_label_counts": dict(sorted(Counter(str(row["video_label"]) for row in video_rows).items())),
            "frame_skip_reason_counts": dict(
                sorted(Counter(str(row["skip_reason"] or "none") for row in frame_rows).items())
            ),
            "multiview_patient_skip_reason_counts": dict(
                sorted(Counter(str(row.get("skip_reason") or "none") for row in multiview_patient_rows).items())
            ),
            "multiview_side_skip_reason_counts": dict(
                sorted(Counter(str(row.get("skip_reason") or "none") for row in multiview_side_rows).items())
            ),
        },
    }


def _binary_metrics_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated_rows = [row for row in rows if row.get("outcome")]
    true_positive = sum(1 for row in evaluated_rows if row["outcome"] == "TP")
    false_positive = sum(1 for row in evaluated_rows if row["outcome"] == "FP")
    true_negative = sum(1 for row in evaluated_rows if row["outcome"] == "TN")
    false_negative = sum(1 for row in evaluated_rows if row["outcome"] == "FN")
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    specificity = _safe_divide(true_negative, true_negative + false_positive)
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    balanced_accuracy = None if recall is None or specificity is None else (recall + specificity) / 2.0
    return {
        "total_rows": len(rows),
        "total_evaluated": len(evaluated_rows),
        "skipped": len(rows) - len(evaluated_rows),
        "positive_labels": true_positive + false_negative,
        "negative_labels": true_negative + false_positive,
        "predicted_positive": true_positive + false_positive,
        "predicted_negative": true_negative + false_negative,
        "TP": true_positive,
        "FP": false_positive,
        "TN": true_negative,
        "FN": false_negative,
        "accuracy": _safe_divide(true_positive + true_negative, len(evaluated_rows)),
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "false_positive_rate": _safe_divide(false_positive, false_positive + true_negative),
        "false_negative_rate": _safe_divide(false_negative, false_negative + true_positive),
        "balanced_accuracy": balanced_accuracy,
    }


def _index_frame_predictions(frame_results_root: Path) -> dict[tuple[str, str, str], FramePrediction]:
    predictions: dict[tuple[str, str, str], FramePrediction] = {}
    for result_path in sorted(frame_results_root.rglob(f"*{FRAME_RESULT_SUFFIX}")):
        if not result_path.is_file():
            continue
        prediction = _load_frame_prediction(result_path)
        patient_video_pairs = _candidate_patient_video_pairs(result_path, frame_results_root, prediction.view_id)
        for patient_id, video_id in patient_video_pairs:
            image_keys = _frame_prediction_image_keys(prediction, patient_id=patient_id, video_id=video_id)
            for image_key in image_keys:
                if image_key:
                    predictions.setdefault((patient_id, video_id, image_key), prediction)
    if not predictions:
        raise FileNotFoundError(f"No frame-level '*{FRAME_RESULT_SUFFIX}' files were found under: {frame_results_root}")
    return predictions


def _frame_prediction_image_keys(
    prediction: FramePrediction,
    *,
    patient_id: str,
    video_id: str,
) -> set[str]:
    image_stem = prediction.image_stem or Path(prediction.image_name).stem
    image_keys = {prediction.image_name, image_stem, Path(prediction.image_name).stem}
    image_keys.update(_cadica_slice_alias_keys(image_stem, prediction.image_name, patient_id=patient_id, video_id=video_id))
    return image_keys


def _cadica_slice_alias_keys(
    image_stem: str,
    image_name: str,
    *,
    patient_id: str,
    video_id: str,
) -> set[str]:
    pattern = re.compile(
        rf"^{re.escape(patient_id)}_{re.escape(video_id)}_(\d+)$",
        re.IGNORECASE,
    )
    match = pattern.fullmatch(image_stem)
    if match is None:
        return set()

    slice_stem = f"slice_{match.group(1)}"
    aliases = {slice_stem, f"{slice_stem}.png"}
    suffix = Path(image_name).suffix
    if suffix:
        aliases.add(f"{slice_stem}{suffix}")
    return aliases


def write_cadica_review_images(
    *,
    manifest_frames: list[ManifestFrame],
    frame_predictions: dict[tuple[str, str, str], FramePrediction],
    frame_rows: list[dict[str, Any]],
    review_image_root: str | Path,
    frame_min_degree: float,
    max_review_images: int,
) -> dict[str, int]:
    resolved_review_root = Path(review_image_root)
    for directory_name in REVIEW_CATEGORY_DIRS.values():
        (resolved_review_root / directory_name).mkdir(parents=True, exist_ok=True)

    written_counts = {category: 0 for category in REVIEW_CATEGORY_DIRS}
    if max_review_images <= 0:
        return written_counts

    manifest_by_key = {_manifest_key(manifest_frame): manifest_frame for manifest_frame in manifest_frames}
    written_total = 0
    for frame_row in frame_rows:
        category = _review_category(frame_row)
        if category is None:
            continue
        if written_total >= max_review_images:
            break

        manifest_frame = manifest_by_key.get(_frame_row_key(frame_row))
        if manifest_frame is None:
            continue
        prediction = _lookup_frame_prediction(manifest_frame, frame_predictions)
        image_path = _review_source_image_path(manifest_frame)
        if image_path is None:
            continue

        output_path = resolved_review_root / REVIEW_CATEGORY_DIRS[category] / _review_image_name(manifest_frame)
        predicted_points = [] if prediction is None else _threshold_points(prediction.points, frame_min_degree)
        if _write_review_image(
            image_path=image_path,
            output_path=output_path,
            manifest_frame=manifest_frame,
            predicted_points=predicted_points,
            frame_row=frame_row,
        ):
            written_counts[category] += 1
            written_total += 1

    return written_counts


def _write_review_image(
    *,
    image_path: Path,
    output_path: Path,
    manifest_frame: ManifestFrame,
    predicted_points: list[PredictedPoint],
    frame_row: dict[str, Any],
) -> bool:
    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return False

    canvas = image.copy()
    for box in manifest_frame.boxes:
        x = _optional_float(box.get("x"))
        y = _optional_float(box.get("y"))
        w = _optional_float(box.get("w"))
        h = _optional_float(box.get("h"))
        if x is None or y is None or w is None or h is None:
            continue
        top_left = (max(0, int(round(x))), max(0, int(round(y))))
        bottom_right = (max(0, int(round(x + w))), max(0, int(round(y + h))))
        cv2.rectangle(canvas, top_left, bottom_right, (70, 220, 70), thickness=2, lineType=cv2.LINE_AA)

    for point in predicted_points:
        center = (int(round(point.x)), int(round(point.y)))
        cv2.circle(canvas, center, radius=5, color=(255, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, center, radius=8, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    text = (
        f"{manifest_frame.patient_id}/{manifest_frame.video_id} "
        f"{manifest_frame.prepared_image_name} | label={manifest_frame.frame_label} "
        f"score={float(frame_row['score']):.3f} outcome={frame_row['outcome']}"
    )
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (min(canvas.shape[1] - 1, 760), 30), (10, 14, 18), thickness=-1)
    cv2.addWeighted(overlay, 0.62, canvas, 0.38, 0.0, dst=canvas)
    cv2.putText(canvas, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 248, 252), 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), canvas))


def _review_category(frame_row: dict[str, Any]) -> str | None:
    outcome = frame_row.get("outcome")
    if outcome == "FP":
        return "false_positive"
    if outcome == "FN":
        return "false_negative"
    if outcome == "TP" and bool(frame_row.get("localized_positive")):
        return "true_positive_localized"
    if outcome == "TP":
        return "true_positive_nonlocalized"
    return None


def _review_source_image_path(manifest_frame: ManifestFrame) -> Path | None:
    for candidate in (manifest_frame.original_image_path, manifest_frame.prepared_image_path):
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def _review_image_name(manifest_frame: ManifestFrame) -> str:
    patient_id = _safe_filename_part(manifest_frame.patient_id)
    video_id = _safe_filename_part(manifest_frame.video_id)
    stem = _safe_filename_part(manifest_frame.prepared_image_stem)
    return f"{patient_id}_{video_id}_{stem}.png"


def _safe_filename_part(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value)
    return cleaned or "unknown"


def _manifest_key(manifest_frame: ManifestFrame) -> tuple[str, str, int]:
    return manifest_frame.patient_id, manifest_frame.video_id, manifest_frame.frame_id


def _frame_row_key(frame_row: dict[str, Any]) -> tuple[str, str, int]:
    return str(frame_row["patient_id"]), str(frame_row["video_id"]), int(frame_row["frame_id"])


def _load_frame_prediction(result_path: Path) -> FramePrediction:
    payload = _read_json_object(result_path)
    frame_payload = payload.get("frame") if isinstance(payload.get("frame"), dict) else {}
    image_name = str(frame_payload.get("image_name") or result_path.name.removesuffix(FRAME_RESULT_SUFFIX) + ".png")
    image_stem = str(frame_payload.get("image_stem") or Path(image_name).stem)
    view_id = str(frame_payload.get("view_id") or "")
    points = [
        point
        for raw_point in payload.get("stenosis_points", [])
        if isinstance(raw_point, dict)
        for point in [_load_predicted_point(raw_point)]
        if point is not None
    ]
    return FramePrediction(
        source_path=result_path,
        image_name=image_name,
        image_stem=image_stem,
        view_id=view_id,
        points=points,
    )


def _candidate_patient_video_pairs(result_path: Path, frame_results_root: Path, view_id: str) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    try:
        relative_path = result_path.resolve().relative_to(frame_results_root.resolve())
    except ValueError:
        relative_path = result_path

    parts = relative_path.parts
    patient_video_pair = _cadica_patient_video_from_parts(parts)
    if patient_video_pair is not None:
        pairs.add(patient_video_pair)
    if len(parts) >= 3:
        pairs.add((parts[0], parts[-2]))

    clean_view_parts = [part for part in view_id.replace("\\", "/").split("/") if part]
    if len(clean_view_parts) >= 2:
        pairs.add((clean_view_parts[0], clean_view_parts[-1]))
    elif len(clean_view_parts) == 1 and len(parts) >= 2:
        pairs.add((parts[0], clean_view_parts[0]))
    return pairs


def _load_predicted_point(payload: dict[str, Any]) -> PredictedPoint | None:
    x = _optional_float(payload.get("x"))
    y = _optional_float(payload.get("y"))
    degree = _optional_float(payload.get("degree"))
    if x is None or y is None or degree is None:
        return None
    severity = payload.get("severity")
    return PredictedPoint(
        x=x,
        y=y,
        degree=degree,
        severity=severity.strip() if isinstance(severity, str) and severity.strip() else None,
    )


def _threshold_points(points: list[PredictedPoint], frame_min_degree: float) -> list[PredictedPoint]:
    return [point for point in points if point.degree >= frame_min_degree]


def _lookup_frame_prediction(
    manifest_frame: ManifestFrame,
    frame_predictions: dict[tuple[str, str, str], FramePrediction],
) -> FramePrediction | None:
    for image_key in (manifest_frame.prepared_image_name, manifest_frame.prepared_image_stem):
        prediction = frame_predictions.get((manifest_frame.patient_id, manifest_frame.video_id, image_key))
        if prediction is not None:
            return prediction
    return None


def _match_points_to_boxes(
    points: list[PredictedPoint],
    boxes: list[dict[str, Any]],
    *,
    margin: float,
) -> dict[str, Any]:
    localized_point_indices: set[int] = set()
    matched_box_indices: set[int] = set()
    box_matches: dict[int, PredictedPoint] = {}

    for point_index, point in enumerate(points):
        containing_box_indices = [
            box_index
            for box_index, box in enumerate(boxes)
            if _point_in_box(point, box, margin=margin)
        ]
        if not containing_box_indices:
            continue
        localized_point_indices.add(point_index)
        for box_index in containing_box_indices:
            matched_box_indices.add(box_index)
            box_matches.setdefault(box_index, point)

    return {
        "localized_point_indices": localized_point_indices,
        "matched_box_indices": matched_box_indices,
        "box_matches": box_matches,
    }


def _point_in_box(point: PredictedPoint, box: dict[str, Any], *, margin: float) -> bool:
    x = _optional_float(box.get("x"))
    y = _optional_float(box.get("y"))
    w = _optional_float(box.get("w"))
    h = _optional_float(box.get("h"))
    if x is None or y is None or w is None or h is None:
        return False
    return x - margin <= point.x <= x + w + margin and y - margin <= point.y <= y + h + margin


def _load_manifest_frame(row: dict[str, Any], *, row_number: int, manifest_path: Path) -> ManifestFrame:
    try:
        frame_id = int(row["frame_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{manifest_path}:{row_number}: invalid or missing frame_id.") from exc

    prepared_image_name = str(row.get("prepared_image_name") or f"slice_{frame_id:05d}.png")
    original_image_path = _manifest_path(row.get("original_image_path"), base_dir=manifest_path.parent)
    prepared_image_path = _manifest_path(row.get("prepared_relative_path"), base_dir=manifest_path.parent)
    gt_boxes = _parse_json_field(row.get("gt_boxes_json"), default=[])
    if not isinstance(gt_boxes, list):
        raise ValueError(f"{manifest_path}:{row_number}: gt_boxes_json must decode to a list.")
    return ManifestFrame(
        patient_id=str(row.get("patient_id") or ""),
        video_id=str(row.get("video_id") or ""),
        frame_id=frame_id,
        prepared_image_name=prepared_image_name,
        prepared_image_stem=str(row.get("prepared_image_stem") or Path(prepared_image_name).stem),
        original_image_name=str(row.get("original_image_name") or ""),
        original_image_path=original_image_path,
        prepared_image_path=prepared_image_path,
        frame_label=str(row.get("frame_label") or "unknown"),
        video_label=str(row.get("video_label") or "unknown"),
        coronary_side=_optional_text(row.get("coronary_side")),
        projection_group=_optional_text(row.get("projection_group")),
        boxes=[box for box in gt_boxes if isinstance(box, dict)],
    )


def _manifest_path(value: object, *, base_dir: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value.strip())
    if path.is_absolute():
        return path
    return base_dir / path


def _parse_json_field(value: object, *, default: Any) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, str):
            return json.loads(parsed)
        return parsed
    return value


def _load_temporal_prediction(
    temporal_results_root: Path | None,
    patient_id: str,
    video_id: str,
) -> tuple[bool | None, float | None]:
    if temporal_results_root is None:
        return None, None
    temporal_path = _temporal_result_path(temporal_results_root, patient_id, video_id)
    if not temporal_path.is_file():
        return False, 0.0
    payload = _read_json_object(temporal_path)
    final_lesion = payload.get("final_lesion")
    if not isinstance(final_lesion, dict):
        return False, 0.0
    degrees = final_lesion.get("degrees") if isinstance(final_lesion.get("degrees"), dict) else {}
    score = _optional_float(degrees.get("median")) or _optional_float(degrees.get("max")) or 1.0
    return True, score


def _temporal_result_path(temporal_results_root: Path, patient_id: str, video_id: str) -> Path:
    flat_path = temporal_results_root / patient_id / video_id / TEMPORAL_RESULT_FILENAME
    if flat_path.is_file():
        return flat_path
    nested_path = _temporal_prediction_index(str(temporal_results_root.resolve())).get(
        (patient_id.lower(), video_id.lower())
    )
    return flat_path if nested_path is None else nested_path


@lru_cache(maxsize=16)
def _temporal_prediction_index(temporal_results_root: str) -> dict[tuple[str, str], Path]:
    root = Path(temporal_results_root)
    index: dict[tuple[str, str], Path] = {}
    for result_path in sorted(root.rglob(TEMPORAL_RESULT_FILENAME)):
        if not result_path.is_file():
            continue
        try:
            relative_parts = result_path.relative_to(root).parts
        except ValueError:
            relative_parts = result_path.parts
        patient_video_pair = _cadica_patient_video_from_parts(relative_parts)
        if patient_video_pair is None:
            continue
        patient_id, video_id = patient_video_pair
        index.setdefault((patient_id.lower(), video_id.lower()), result_path)
    return index


def _resolve_video_label(rows: list[dict[str, Any]]) -> str:
    counts = Counter(str(row["video_label"]) for row in rows)
    if counts.get("lesion", 0) > 0:
        return "lesion"
    if counts.get("nonlesion", 0) > 0:
        return "nonlesion"
    return counts.most_common(1)[0][0] if counts else "unknown"


def _frame_label_target(frame_label: str) -> bool | None:
    if frame_label == "positive":
        return True
    if frame_label == "negative":
        return False
    return None


def _cadica_patient_video_from_parts(parts: tuple[str, ...]) -> tuple[str, str] | None:
    for index, part in enumerate(parts[:-1]):
        if CADICA_PATIENT_ID_RE.fullmatch(part) and CADICA_VIDEO_ID_RE.fullmatch(parts[index + 1]):
            return part, parts[index + 1]
    return None


def _cadica_patient_id_from_parts(parts: tuple[str, ...]) -> str | None:
    for part in reversed(parts[:-1]):
        if CADICA_PATIENT_ID_RE.fullmatch(part):
            return part
    return None


def _video_label_target(video_label: str) -> bool | None:
    if video_label == "lesion":
        return True
    if video_label == "nonlesion":
        return False
    return None


def _binary_outcome(*, label_target: bool, predicted_positive: bool) -> str:
    if label_target and predicted_positive:
        return "TP"
    if not label_target and predicted_positive:
        return "FP"
    if not label_target and not predicted_positive:
        return "TN"
    return "FN"


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected top-level JSON object.")
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field_name: _csv_value(row.get(field_name)) for field_name in fieldnames})


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or stripped.lower() in {"none", "unknown", "nan"}:
        return None
    return stripped


def _safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _format_optional_float(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _validate_inputs(
    manifest: Path,
    frame_results_root: Path,
    temporal_results_root: Path | None,
    multiview_results_root: Path | None = None,
) -> None:
    if not manifest.exists():
        raise FileNotFoundError(f"CADICA manifest does not exist: {manifest}")
    if not manifest.is_file():
        raise ValueError(f"CADICA manifest path is not a file: {manifest}")
    if not frame_results_root.exists():
        raise FileNotFoundError(f"Frame results root does not exist: {frame_results_root}")
    if not frame_results_root.is_dir():
        raise NotADirectoryError(f"Frame results root is not a directory: {frame_results_root}")
    if temporal_results_root is not None and not temporal_results_root.exists():
        raise FileNotFoundError(f"Temporal results root does not exist: {temporal_results_root}")
    if temporal_results_root is not None and not temporal_results_root.is_dir():
        raise NotADirectoryError(f"Temporal results root is not a directory: {temporal_results_root}")
    if multiview_results_root is not None and not multiview_results_root.exists():
        raise FileNotFoundError(f"Multi-view results root does not exist: {multiview_results_root}")
    if multiview_results_root is not None and not multiview_results_root.is_dir():
        raise NotADirectoryError(f"Multi-view results root is not a directory: {multiview_results_root}")


FRAME_ROW_FIELDS = [
    "patient_id",
    "video_id",
    "frame_id",
    "prepared_image_name",
    "original_image_name",
    "frame_label",
    "video_label",
    "predicted_positive",
    "score",
    "predicted_point_count",
    "gt_box_count",
    "localized_positive",
    "matched_gt_box_count",
    "unmatched_gt_box_count",
    "unmatched_predicted_point_count",
    "outcome",
    "skip_reason",
]

BOX_ROW_FIELDS = [
    "patient_id",
    "video_id",
    "frame_id",
    "box_index",
    "x",
    "y",
    "w",
    "h",
    "category",
    "matched",
    "matched_point_x",
    "matched_point_y",
    "matched_point_degree",
    "source_path",
]

VIDEO_ROW_FIELDS = [
    "patient_id",
    "video_id",
    "video_label",
    "predicted_positive",
    "score",
    "positive_frame_count",
    "predicted_positive_frame_count",
    "temporal_final_lesion_present",
    "outcome",
]

PATIENT_ROW_FIELDS = [
    "patient_id",
    "label_positive",
    "predicted_positive",
    "lesion_video_count",
    "predicted_positive_video_count",
    "outcome",
]

MULTIVIEW_PATIENT_ROW_FIELDS = [
    "patient_id",
    "label_positive",
    "predicted_positive",
    "score",
    "outcome",
    "lesion_video_count",
    "positive_frame_count",
    "multiview_json_path",
    "skip_reason",
]

MULTIVIEW_SIDE_ROW_FIELDS = [
    "patient_id",
    "coronary_side",
    "label_positive",
    "predicted_positive",
    "score",
    "outcome",
    "lesion_video_count",
    "positive_frame_count",
    "multiview_json_path",
    "skip_reason",
]

UNMATCHED_POINT_ROW_FIELDS = [
    "patient_id",
    "video_id",
    "frame_id",
    "prepared_image_name",
    "frame_label",
    "video_label",
    "point_index",
    "x",
    "y",
    "degree",
    "severity",
]


if __name__ == "__main__":
    raise SystemExit(main())

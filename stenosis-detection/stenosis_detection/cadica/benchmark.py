from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any


FRAME_RESULT_SUFFIX = "_stenosis_results.json"
TEMPORAL_RESULT_FILENAME = "view_temporal_fusion.json"
VIDEO_PREDICTION_SOURCES = {"frame_any", "temporal_final"}
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
        if self.review_image_root is not None:
            payload["review_image_root"] = str(self.review_image_root)
        return payload


@dataclass(frozen=True, slots=True)
class CadicaBenchmarkResult:
    frame_rows: list[dict[str, Any]]
    box_rows: list[dict[str, Any]]
    video_rows: list[dict[str, Any]]
    patient_rows: list[dict[str, Any]]
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


def run_cadica_benchmark(
    *,
    manifest: str | Path,
    frame_results_root: str | Path,
    output_root: str | Path,
    temporal_results_root: str | Path | None = None,
    frame_min_degree: float = 0.0,
    box_margin_px: float = 5.0,
    video_prediction_source: str = "frame_any",
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
    if max_review_images < 0:
        raise ValueError("max_review_images must be >= 0.")

    manifest_path = Path(manifest)
    resolved_frame_results_root = Path(frame_results_root)
    resolved_output_root = Path(output_root)
    resolved_temporal_root = None if temporal_results_root is None else Path(temporal_results_root)
    resolved_review_image_root = (
        None
        if not write_review_images
        else Path(review_image_root) if review_image_root is not None else resolved_output_root / "review_images"
    )
    _validate_inputs(manifest_path, resolved_frame_results_root, resolved_temporal_root)

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
    summary = _build_summary(
        frame_rows,
        box_rows,
        video_rows,
        patient_rows,
        manifest=manifest_path,
        frame_results_root=resolved_frame_results_root,
        output_root=resolved_output_root,
        temporal_results_root=resolved_temporal_root,
        frame_min_degree=frame_min_degree,
        box_margin_px=box_margin_px,
        video_prediction_source=video_prediction_source,
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
        review_image_root=review_image_root,
    )

    _write_csv(outputs.frame_rows_csv, frame_rows, fieldnames=FRAME_ROW_FIELDS)
    _write_jsonl(outputs.frame_rows_jsonl, frame_rows)
    _write_csv(outputs.box_rows_csv, box_rows, fieldnames=BOX_ROW_FIELDS)
    _write_csv(outputs.video_rows_csv, video_rows, fieldnames=VIDEO_ROW_FIELDS)
    _write_csv(outputs.patient_rows_csv, patient_rows, fieldnames=PATIENT_ROW_FIELDS)
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = run_cadica_benchmark(
            manifest=args.manifest,
            frame_results_root=args.frame_results_root,
            output_root=args.output_root,
            temporal_results_root=args.temporal_results_root,
            frame_min_degree=args.frame_min_degree,
            box_margin_px=args.box_margin_px,
            video_prediction_source=args.video_prediction_source,
            write_review_images=args.write_review_images,
            max_review_images=args.max_review_images,
            review_image_root=args.review_image_root,
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
    print(f"Summary: {result.outputs.summary_json}")
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
        temporal_present, temporal_score = _load_temporal_prediction(temporal_results_root, patient_id, video_id)
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
                "temporal_final_lesion_present": temporal_present if temporal_results_root is not None else None,
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


def _build_summary(
    frame_rows: list[dict[str, Any]],
    box_rows: list[dict[str, Any]],
    video_rows: list[dict[str, Any]],
    patient_rows: list[dict[str, Any]],
    *,
    manifest: Path,
    frame_results_root: Path,
    output_root: Path,
    temporal_results_root: Path | None,
    frame_min_degree: float,
    box_margin_px: float,
    video_prediction_source: str,
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
            "frame_min_degree": float(frame_min_degree),
            "box_margin_px": float(box_margin_px),
            "video_prediction_source": video_prediction_source,
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
        "counts": {
            "frame_label_counts": dict(sorted(Counter(str(row["frame_label"]) for row in frame_rows).items())),
            "video_label_counts": dict(sorted(Counter(str(row["video_label"]) for row in video_rows).items())),
            "frame_skip_reason_counts": dict(
                sorted(Counter(str(row["skip_reason"] or "none") for row in frame_rows).items())
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
        image_keys = {prediction.image_name, prediction.image_stem, Path(prediction.image_name).stem}
        for patient_id, video_id in patient_video_pairs:
            for image_key in image_keys:
                if image_key:
                    predictions.setdefault((patient_id, video_id, image_key), prediction)
    if not predictions:
        raise FileNotFoundError(f"No frame-level '*{FRAME_RESULT_SUFFIX}' files were found under: {frame_results_root}")
    return predictions


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
    temporal_path = temporal_results_root / patient_id / video_id / TEMPORAL_RESULT_FILENAME
    if not temporal_path.is_file():
        return False, 0.0
    payload = _read_json_object(temporal_path)
    final_lesion = payload.get("final_lesion")
    if not isinstance(final_lesion, dict):
        return False, 0.0
    degrees = final_lesion.get("degrees") if isinstance(final_lesion.get("degrees"), dict) else {}
    score = _optional_float(degrees.get("median")) or _optional_float(degrees.get("max")) or 1.0
    return True, score


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


def _safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _format_optional_float(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _validate_inputs(manifest: Path, frame_results_root: Path, temporal_results_root: Path | None) -> None:
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

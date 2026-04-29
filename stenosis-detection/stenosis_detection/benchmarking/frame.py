from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .io import BenchmarkIOError
from .labels import load_weak_labels_jsonl
from .metrics import summarize_rows
from .models import BenchmarkRow, BinaryMetricSummary, WeakLabel
from .severity import severity_row_fields, summarize_severity_agreement
from .sweep import write_threshold_sweep


FRAME_RESULT_SUFFIX = "_stenosis_results.json"
SEVERITY_RANK = {
    "none": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3,
    "occlusion": 4,
}


@dataclass(frozen=True, slots=True)
class FrameBenchmarkOutputs:
    frame_rows_csv: Path
    frame_rows_jsonl: Path
    frame_summary_json: Path
    frame_summary_csv: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "frame_rows_csv": str(self.frame_rows_csv),
            "frame_rows_jsonl": str(self.frame_rows_jsonl),
            "frame_summary_json": str(self.frame_summary_json),
            "frame_summary_csv": str(self.frame_summary_csv),
        }


@dataclass(frozen=True, slots=True)
class FrameBenchmarkResult:
    frame_rows: list[dict[str, Any]]
    frame_summary: BinaryMetricSummary
    output_paths: FrameBenchmarkOutputs
    threshold_sweep_paths: dict[str, Path]


def run_frame_level_benchmark(
    *,
    results_root: str | Path,
    weak_labels_path: str | Path,
    output_root: str | Path,
    frame_min_degree: float = 0.0,
    include_unclear_labels: bool = False,
    write_threshold_sweep_report: bool = False,
) -> FrameBenchmarkResult:
    """Benchmark existing frame-level JSON outputs against case-level weak labels."""
    resolved_results_root = Path(results_root)
    resolved_output_root = Path(output_root)
    _validate_frame_benchmark_inputs(
        resolved_results_root,
        frame_min_degree=frame_min_degree,
    )

    weak_labels = load_weak_labels_jsonl(weak_labels_path)
    frame_paths = discover_frame_result_paths(resolved_results_root)
    frame_rows = [
        _build_frame_row(
            frame_path,
            results_root=resolved_results_root,
            weak_labels=weak_labels,
            frame_min_degree=frame_min_degree,
            include_unclear_labels=include_unclear_labels,
        )
        for frame_path in frame_paths
    ]
    frame_summary = _summarize_dict_rows(
        frame_rows,
        prediction_column="predicted_positive",
        outcome_column="weak_outcome",
        skip_column="skip_reason",
        labels=weak_labels,
    )

    output_paths = save_frame_benchmark_outputs(
        output_root=resolved_output_root,
        frame_rows=frame_rows,
        frame_summary=frame_summary,
        frame_min_degree=frame_min_degree,
        include_unclear_labels=include_unclear_labels,
    )
    threshold_sweep_paths = (
        write_threshold_sweep(
            rows=frame_rows,
            output_root=resolved_output_root,
            level="frame",
            score_columns={"frame_score": "score"},
        )
        if write_threshold_sweep_report
        else {}
    )

    return FrameBenchmarkResult(
        frame_rows=frame_rows,
        frame_summary=frame_summary,
        output_paths=output_paths,
        threshold_sweep_paths=threshold_sweep_paths,
    )


def discover_frame_result_paths(results_root: str | Path) -> list[Path]:
    resolved_results_root = Path(results_root)
    if not resolved_results_root.exists():
        raise FileNotFoundError(f"Frame results root does not exist: {resolved_results_root}")
    if not resolved_results_root.is_dir():
        raise NotADirectoryError(f"Frame results root is not a directory: {resolved_results_root}")

    frame_paths = sorted(
        path
        for path in resolved_results_root.rglob(f"*{FRAME_RESULT_SUFFIX}")
        if path.is_file()
    )
    if not frame_paths:
        raise FileNotFoundError(f"No frame-level '*{FRAME_RESULT_SUFFIX}' files were found under: {resolved_results_root}")
    return frame_paths


def save_frame_benchmark_outputs(
    *,
    output_root: str | Path,
    frame_rows: list[dict[str, Any]],
    frame_summary: BinaryMetricSummary,
    frame_min_degree: float,
    include_unclear_labels: bool,
) -> FrameBenchmarkOutputs:
    resolved_output_root = Path(output_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    frame_rows_csv = resolved_output_root / "frame_rows.csv"
    frame_rows_jsonl = resolved_output_root / "frame_rows.jsonl"
    frame_summary_json = resolved_output_root / "frame_summary.json"
    frame_summary_csv = resolved_output_root / "frame_summary.csv"

    _write_csv(frame_rows_csv, frame_rows, fieldnames=FRAME_ROW_FIELDS)
    _write_jsonl(frame_rows_jsonl, frame_rows)
    _write_json(
        frame_summary_json,
        {
            "level": "frame",
            "frame_min_degree": float(frame_min_degree),
            "include_unclear_labels": include_unclear_labels,
            "summary": frame_summary.to_dict(),
            "severity_agreement": summarize_severity_agreement(frame_rows),
        },
    )
    _write_csv(
        frame_summary_csv,
        [
            {
                "level": "frame",
                "frame_min_degree": float(frame_min_degree),
                "include_unclear_labels": include_unclear_labels,
                **frame_summary.to_dict(),
            }
        ],
    )

    return FrameBenchmarkOutputs(
        frame_rows_csv=frame_rows_csv,
        frame_rows_jsonl=frame_rows_jsonl,
        frame_summary_json=frame_summary_json,
        frame_summary_csv=frame_summary_csv,
    )


def _build_frame_row(
    frame_path: Path,
    *,
    results_root: Path,
    weak_labels: dict[str, WeakLabel],
    frame_min_degree: float,
    include_unclear_labels: bool,
) -> dict[str, Any]:
    relative_path = _relative_frame_path(frame_path, results_root)
    case_id, sequence_id = _derive_case_and_sequence_id(relative_path)
    payload = _read_json_object(frame_path)
    stenosis_points = payload.get("stenosis_points")
    if not isinstance(stenosis_points, list):
        stenosis_points = []

    degrees = [_coerce_float(point.get("degree")) for point in stenosis_points if isinstance(point, dict)]
    degrees = [degree for degree in degrees if degree is not None]
    max_degree = max(degrees, default=0.0)
    predicted_positive = len(stenosis_points) > 0 and max_degree >= frame_min_degree
    label = weak_labels.get(case_id)
    weak_outcome, skip_reason, label_target = _weak_outcome(
        predicted_positive,
        label,
        include_unclear_labels=include_unclear_labels,
    )
    frame_payload = payload.get("frame") if isinstance(payload.get("frame"), dict) else {}
    predicted_severity = _worst_severity(stenosis_points)

    return {
        "case_id": case_id,
        "sequence_id": sequence_id,
        "relative_path": relative_path.as_posix(),
        "source_path": str(frame_path),
        "frame_id": _frame_id(payload, frame_path),
        "frame_index": frame_payload.get("frame_index"),
        "image_name": frame_payload.get("image_name"),
        "stenosis_point_count": len(stenosis_points),
        "predicted_positive": predicted_positive,
        "score": max_degree,
        "max_stenosis_degree": max_degree,
        "predicted_severity": predicted_severity,
        "frame_min_degree": float(frame_min_degree),
        **_label_fields(label, label_target=label_target),
        **severity_row_fields(
            label_severity=None if label is None else label.severity,
            pipeline_severity=predicted_severity,
            label_target=label_target,
            predicted_positive=predicted_positive,
        ),
        "weak_outcome": weak_outcome,
        "skip_reason": skip_reason,
    }


def _summarize_dict_rows(
    rows: Iterable[dict[str, Any]],
    *,
    prediction_column: str,
    outcome_column: str,
    skip_column: str,
    labels: dict[str, WeakLabel],
) -> BinaryMetricSummary:
    row_list = list(rows)
    metric_rows = [
        BenchmarkRow(
            case_id=str(row["case_id"]),
            predicted_positive=bool(row[prediction_column]),
            label_target=row.get("label_target"),
            label_status=str(row.get("label_status") or "missing"),
            outcome=row.get(outcome_column),
            skip_reason=row.get(skip_column),
            source_path=row.get("source_path"),
            prediction_level=str(row.get("prediction_level") or ""),
            prediction_score=row.get("score"),
            prediction_confidence=None,
            label_stenosis_exists=row.get("label_stenosis_exists"),
            label_severity=row.get("label_severity"),
            label_confidence=row.get("label_confidence"),
            label_excel_row=row.get("label_excel_row"),
            label_source_path=row.get("label_source_path"),
        )
        for row in row_list
    ]
    predicted_case_ids = {str(row["case_id"]) for row in row_list}
    return summarize_rows(
        metric_rows,
        total_predictions=len(row_list),
        labels_without_prediction=len(set(labels) - predicted_case_ids),
    )


def _weak_outcome(
    predicted_positive: bool,
    label: WeakLabel | None,
    *,
    include_unclear_labels: bool,
) -> tuple[str | None, str | None, bool | None]:
    _ = include_unclear_labels
    if label is None:
        return None, "missing_label", None
    if label.target is None:
        return None, "unclear_label", None
    if label.target and predicted_positive:
        return "TP", None, True
    if not label.target and predicted_positive:
        return "FP", None, False
    if not label.target and not predicted_positive:
        return "TN", None, False
    return "FN", None, True


def _label_fields(label: WeakLabel | None, *, label_target: bool | None) -> dict[str, Any]:
    if label is None:
        return {
            "label_target": None,
            "label_status": "missing",
            "label_stenosis_exists": None,
            "label_severity": None,
            "label_confidence": None,
            "label_excel_row": None,
            "label_column_U": None,
            "label_max_percent": None,
            "label_source_path": None,
        }

    return {
        "label_target": label_target,
        "label_status": label.status,
        "label_stenosis_exists": label.stenosis_exists,
        "label_severity": label.severity,
        "label_confidence": label.confidence,
        "label_excel_row": label.excel_row,
        "label_column_U": label.column_u,
        "label_max_percent": label.max_percent,
        "label_source_path": label.source_path,
    }


def _relative_frame_path(frame_path: Path, results_root: Path) -> Path:
    try:
        return frame_path.resolve().relative_to(results_root.resolve())
    except ValueError as exc:
        raise BenchmarkIOError(f"Frame path is not under results root: {frame_path}") from exc


def _derive_case_and_sequence_id(relative_path: Path) -> tuple[str, str]:
    parts = relative_path.parts
    if len(parts) < 2:
        raise BenchmarkIOError(
            f"Frame result must live under a case directory relative to --results-root: {relative_path}"
        )
    case_id = parts[0]
    sequence_id = Path(*parts[1:-1]).as_posix() if len(parts) > 2 else ""
    return case_id, sequence_id


def _read_json_object(json_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise BenchmarkIOError(f"{json_path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise BenchmarkIOError(f"{json_path}: expected top-level JSON object.")
    return payload


def _frame_id(payload: dict[str, Any], frame_path: Path) -> str:
    frame_payload = payload.get("frame")
    if isinstance(frame_payload, dict):
        for key in ("image_stem", "image_name"):
            value = frame_payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return frame_path.name.removesuffix(FRAME_RESULT_SUFFIX)


def _worst_severity(stenosis_points: list[Any]) -> str | None:
    severities = [
        point.get("severity")
        for point in stenosis_points
        if isinstance(point, dict) and isinstance(point.get("severity"), str) and point.get("severity").strip()
    ]
    if not severities:
        return None
    return max(severities, key=lambda severity: SEVERITY_RANK.get(severity.strip().lower(), -1)).strip().lower()


def _coerce_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str] | None = None) -> None:
    resolved_fieldnames = fieldnames or _union_fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=resolved_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field_name: _csv_value(row.get(field_name)) for field_name in resolved_fieldnames})


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _union_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for field_name in row:
            if field_name not in fieldnames:
                fieldnames.append(field_name)
    return fieldnames


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    return value


def _validate_frame_benchmark_inputs(
    results_root: Path,
    *,
    frame_min_degree: float,
) -> None:
    if frame_min_degree < 0.0:
        raise ValueError("frame_min_degree must be >= 0.0.")
    if not results_root.exists():
        raise FileNotFoundError(f"Frame results root does not exist: {results_root}")
    if not results_root.is_dir():
        raise NotADirectoryError(f"Frame results root is not a directory: {results_root}")


FRAME_ROW_FIELDS = [
    "case_id",
    "sequence_id",
    "relative_path",
    "source_path",
    "frame_id",
    "frame_index",
    "image_name",
    "stenosis_point_count",
    "predicted_positive",
    "score",
    "max_stenosis_degree",
    "predicted_severity",
    "pipeline_severity",
    "weak_label_severity_normalized",
    "severity_match",
    "frame_min_degree",
    "label_target",
    "label_status",
    "label_stenosis_exists",
    "label_severity",
    "label_confidence",
    "label_excel_row",
    "label_column_U",
    "label_max_percent",
    "label_source_path",
    "weak_outcome",
    "skip_reason",
]

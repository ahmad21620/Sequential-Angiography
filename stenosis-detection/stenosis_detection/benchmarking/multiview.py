from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .frame import (
    _coerce_float,
    _label_fields,
    _read_json_object,
    _summarize_dict_rows,
    _weak_outcome,
    _write_csv,
    _write_json,
    _write_jsonl,
)
from .io import BenchmarkIOError
from .labels import load_weak_labels_jsonl
from .models import BinaryMetricSummary, WeakLabel
from .sweep import write_threshold_sweep
from .temporal import _coerce_int


MULTIVIEW_RESULT_FILENAME = "case_multiview_fusion.json"


@dataclass(frozen=True, slots=True)
class MultiViewBenchmarkOutputs:
    case_rows_csv: Path
    case_rows_jsonl: Path
    case_summary_json: Path
    case_summary_csv: Path
    false_positive_cases_csv: Path
    false_negative_cases_csv: Path
    true_positive_cases_csv: Path
    true_negative_cases_csv: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "multiview_case_rows_csv": str(self.case_rows_csv),
            "multiview_case_rows_jsonl": str(self.case_rows_jsonl),
            "multiview_case_summary_json": str(self.case_summary_json),
            "multiview_case_summary_csv": str(self.case_summary_csv),
            "false_positive_cases_csv": str(self.false_positive_cases_csv),
            "false_negative_cases_csv": str(self.false_negative_cases_csv),
            "true_positive_cases_csv": str(self.true_positive_cases_csv),
            "true_negative_cases_csv": str(self.true_negative_cases_csv),
        }


@dataclass(frozen=True, slots=True)
class MultiViewBenchmarkResult:
    case_rows: list[dict[str, Any]]
    case_summary: BinaryMetricSummary
    breakdowns: dict[str, dict[str, dict[str, Any]]]
    output_paths: MultiViewBenchmarkOutputs
    threshold_sweep_paths: dict[str, Path]


def run_multiview_level_benchmark(
    *,
    results_root: str | Path,
    weak_labels_path: str | Path,
    output_root: str | Path,
    multiview_min_confidence_score: float = 0.0,
    multiview_min_degree: float = 0.0,
    require_distinct_supporting_view: bool = False,
    include_unclear_labels: bool = False,
    write_threshold_sweep_report: bool = False,
) -> MultiViewBenchmarkResult:
    resolved_results_root = Path(results_root)
    _validate_multiview_inputs(
        resolved_results_root,
        multiview_min_confidence_score=multiview_min_confidence_score,
        multiview_min_degree=multiview_min_degree,
    )

    weak_labels = load_weak_labels_jsonl(weak_labels_path)
    case_rows = [
        _build_case_row(
            result_path,
            results_root=resolved_results_root,
            weak_labels=weak_labels,
            multiview_min_confidence_score=multiview_min_confidence_score,
            multiview_min_degree=multiview_min_degree,
            require_distinct_supporting_view=require_distinct_supporting_view,
            include_unclear_labels=include_unclear_labels,
        )
        for result_path in discover_multiview_result_paths(resolved_results_root)
    ]
    case_summary = _summarize_dict_rows(
        case_rows,
        prediction_column="predicted_positive",
        outcome_column="weak_outcome",
        skip_column="skip_reason",
        labels=weak_labels,
    )
    breakdowns = {
        "by_weak_label_severity": _build_breakdown(case_rows, "label_severity", labels=weak_labels),
        "by_weak_label_confidence": _build_breakdown(case_rows, "label_confidence", labels=weak_labels),
        "by_pipeline_confidence_label": _build_breakdown(case_rows, "confidence_label", labels=weak_labels),
    }

    output_paths = save_multiview_benchmark_outputs(
        output_root=output_root,
        case_rows=case_rows,
        case_summary=case_summary,
        breakdowns=breakdowns,
        multiview_min_confidence_score=multiview_min_confidence_score,
        multiview_min_degree=multiview_min_degree,
        require_distinct_supporting_view=require_distinct_supporting_view,
        include_unclear_labels=include_unclear_labels,
    )
    threshold_sweep_paths = (
        write_threshold_sweep(
            rows=case_rows,
            output_root=output_root,
            level="multiview",
            score_columns={
                "confidence_score": "confidence_score",
                "total_score": "total_score",
            },
        )
        if write_threshold_sweep_report
        else {}
    )
    return MultiViewBenchmarkResult(
        case_rows=case_rows,
        case_summary=case_summary,
        breakdowns=breakdowns,
        output_paths=output_paths,
        threshold_sweep_paths=threshold_sweep_paths,
    )


def discover_multiview_result_paths(results_root: str | Path) -> list[Path]:
    resolved_results_root = Path(results_root)
    if not resolved_results_root.exists():
        raise FileNotFoundError(f"Multi-view results root does not exist: {resolved_results_root}")
    if not resolved_results_root.is_dir():
        raise NotADirectoryError(f"Multi-view results root is not a directory: {resolved_results_root}")

    paths = sorted(path for path in resolved_results_root.rglob(MULTIVIEW_RESULT_FILENAME) if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No '{MULTIVIEW_RESULT_FILENAME}' files were found under: {resolved_results_root}")
    return paths


def save_multiview_benchmark_outputs(
    *,
    output_root: str | Path,
    case_rows: list[dict[str, Any]],
    case_summary: BinaryMetricSummary,
    breakdowns: dict[str, dict[str, dict[str, Any]]],
    multiview_min_confidence_score: float,
    multiview_min_degree: float,
    require_distinct_supporting_view: bool,
    include_unclear_labels: bool,
) -> MultiViewBenchmarkOutputs:
    resolved_output_root = Path(output_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    case_rows_csv = resolved_output_root / "multiview_case_rows.csv"
    case_rows_jsonl = resolved_output_root / "multiview_case_rows.jsonl"
    case_summary_json = resolved_output_root / "multiview_case_summary.json"
    case_summary_csv = resolved_output_root / "multiview_case_summary.csv"
    false_positive_cases_csv = resolved_output_root / "false_positive_cases.csv"
    false_negative_cases_csv = resolved_output_root / "false_negative_cases.csv"
    true_positive_cases_csv = resolved_output_root / "true_positive_cases.csv"
    true_negative_cases_csv = resolved_output_root / "true_negative_cases.csv"

    _write_csv(case_rows_csv, case_rows, fieldnames=MULTIVIEW_CASE_ROW_FIELDS)
    _write_jsonl(case_rows_jsonl, case_rows)
    _write_json(
        case_summary_json,
        {
            "level": "multiview_case",
            "multiview_min_confidence_score": float(multiview_min_confidence_score),
            "multiview_min_degree": float(multiview_min_degree),
            "require_distinct_supporting_view": require_distinct_supporting_view,
            "include_unclear_labels": include_unclear_labels,
            "summary": case_summary.to_dict(),
            "breakdowns": breakdowns,
        },
    )
    _write_csv(
        case_summary_csv,
        [
            {
                "level": "multiview_case",
                "multiview_min_confidence_score": float(multiview_min_confidence_score),
                "multiview_min_degree": float(multiview_min_degree),
                "require_distinct_supporting_view": require_distinct_supporting_view,
                "include_unclear_labels": include_unclear_labels,
                **case_summary.to_dict(),
            }
        ],
    )
    _write_csv(false_positive_cases_csv, _filter_rows(case_rows, "FP"), fieldnames=MULTIVIEW_CASE_ROW_FIELDS)
    _write_csv(false_negative_cases_csv, _filter_rows(case_rows, "FN"), fieldnames=MULTIVIEW_CASE_ROW_FIELDS)
    _write_csv(true_positive_cases_csv, _filter_rows(case_rows, "TP"), fieldnames=MULTIVIEW_CASE_ROW_FIELDS)
    _write_csv(true_negative_cases_csv, _filter_rows(case_rows, "TN"), fieldnames=MULTIVIEW_CASE_ROW_FIELDS)

    return MultiViewBenchmarkOutputs(
        case_rows_csv=case_rows_csv,
        case_rows_jsonl=case_rows_jsonl,
        case_summary_json=case_summary_json,
        case_summary_csv=case_summary_csv,
        false_positive_cases_csv=false_positive_cases_csv,
        false_negative_cases_csv=false_negative_cases_csv,
        true_positive_cases_csv=true_positive_cases_csv,
        true_negative_cases_csv=true_negative_cases_csv,
    )


def _build_case_row(
    result_path: Path,
    *,
    results_root: Path,
    weak_labels: dict[str, WeakLabel],
    multiview_min_confidence_score: float,
    multiview_min_degree: float,
    require_distinct_supporting_view: bool,
    include_unclear_labels: bool,
) -> dict[str, Any]:
    relative_path = _relative_path(result_path, results_root)
    path_case_id = _path_case_id(relative_path)
    payload = _read_json_object(result_path)
    json_case_id = payload.get("case_id")
    if isinstance(json_case_id, str) and json_case_id.strip():
        case_id = json_case_id.strip()
        if case_id != path_case_id:
            raise BenchmarkIOError(f"{result_path}: JSON case_id '{case_id}' does not match path case_id '{path_case_id}'.")
    else:
        case_id = path_case_id

    final_case_lesion = payload.get("final_case_lesion") if isinstance(payload.get("final_case_lesion"), dict) else None
    degrees = final_case_lesion.get("degrees") if isinstance(final_case_lesion, dict) and isinstance(final_case_lesion.get("degrees"), dict) else {}
    confidence = payload.get("confidence") if isinstance(payload.get("confidence"), dict) else {}
    fusion_metadata = payload.get("fusion_metadata") if isinstance(payload.get("fusion_metadata"), dict) else {}
    supporting_views = payload.get("supporting_views") if isinstance(payload.get("supporting_views"), list) else []
    distinct_supporting_view_ids = (
        final_case_lesion.get("distinct_supporting_view_ids")
        if isinstance(final_case_lesion, dict) and isinstance(final_case_lesion.get("distinct_supporting_view_ids"), list)
        else []
    )

    final_degree = _coerce_float(degrees.get("median")) if final_case_lesion is not None else 0.0
    final_max_degree = _coerce_float(degrees.get("max")) if final_case_lesion is not None else 0.0
    total_score = _coerce_float(final_case_lesion.get("total_score")) if final_case_lesion is not None else 0.0
    confidence_score = _coerce_float(confidence.get("score"))
    final_degree = 0.0 if final_degree is None else final_degree
    final_max_degree = 0.0 if final_max_degree is None else final_max_degree
    total_score = 0.0 if total_score is None else total_score
    confidence_score = 0.0 if confidence_score is None else confidence_score
    distinct_supporting_view_count = len(distinct_supporting_view_ids)
    predicted_positive = (
        final_case_lesion is not None
        and final_degree >= multiview_min_degree
        and confidence_score >= multiview_min_confidence_score
        and (not require_distinct_supporting_view or distinct_supporting_view_count > 0)
    )

    label = weak_labels.get(case_id)
    weak_outcome, skip_reason, label_target = _weak_outcome(
        predicted_positive,
        label,
        include_unclear_labels=include_unclear_labels,
    )

    return {
        "case_id": case_id,
        "path_case_id": path_case_id,
        "relative_path": relative_path.as_posix(),
        "source_path": str(result_path),
        "predicted_positive": predicted_positive,
        "final_severity": final_case_lesion.get("severity") if final_case_lesion is not None else None,
        "final_degree": final_degree,
        "final_max_degree": final_max_degree,
        "total_score": total_score,
        "confidence_score": confidence_score,
        "confidence_label": confidence.get("label"),
        "supporting_view_count": len(supporting_views),
        "distinct_supporting_view_count": distinct_supporting_view_count,
        "total_candidate_count": _coerce_int(fusion_metadata.get("total_candidate_count")),
        "multiview_min_confidence_score": float(multiview_min_confidence_score),
        "multiview_min_degree": float(multiview_min_degree),
        "require_distinct_supporting_view": require_distinct_supporting_view,
        **_label_fields(label, label_target=label_target),
        "weak_outcome": weak_outcome,
        "skip_reason": skip_reason,
    }


def _build_breakdown(
    rows: list[dict[str, Any]],
    group_column: str,
    *,
    labels: dict[str, WeakLabel],
) -> dict[str, dict[str, Any]]:
    breakdown: dict[str, dict[str, Any]] = {}
    group_values = sorted({str(row[group_column]) for row in rows if row.get(group_column) not in (None, "")})
    for group_value in group_values:
        group_rows = [row for row in rows if str(row.get(group_column)) == group_value]
        group_case_ids = {str(row["case_id"]) for row in group_rows}
        group_labels = {case_id: labels[case_id] for case_id in group_case_ids if case_id in labels}
        breakdown[group_value] = _summarize_dict_rows(
            group_rows,
            prediction_column="predicted_positive",
            outcome_column="weak_outcome",
            skip_column="skip_reason",
            labels=group_labels,
        ).to_dict()
    return breakdown


def _filter_rows(rows: list[dict[str, Any]], outcome: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("weak_outcome") == outcome]


def _relative_path(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise BenchmarkIOError(f"Multi-view result path is not under results root: {path}") from exc


def _path_case_id(relative_path: Path) -> str:
    parts = relative_path.parts
    if len(parts) < 2:
        raise BenchmarkIOError(
            f"Multi-view result must live under a case directory relative to --results-root: {relative_path}"
        )
    return parts[0]


def _validate_multiview_inputs(
    results_root: Path,
    *,
    multiview_min_confidence_score: float,
    multiview_min_degree: float,
) -> None:
    if not 0.0 <= multiview_min_confidence_score <= 1.0:
        raise ValueError("multiview_min_confidence_score must be in the range [0.0, 1.0].")
    if multiview_min_degree < 0.0:
        raise ValueError("multiview_min_degree must be >= 0.0.")
    if not results_root.exists():
        raise FileNotFoundError(f"Multi-view results root does not exist: {results_root}")
    if not results_root.is_dir():
        raise NotADirectoryError(f"Multi-view results root is not a directory: {results_root}")


MULTIVIEW_CASE_ROW_FIELDS = [
    "case_id",
    "path_case_id",
    "relative_path",
    "source_path",
    "predicted_positive",
    "final_severity",
    "final_degree",
    "final_max_degree",
    "total_score",
    "confidence_score",
    "confidence_label",
    "supporting_view_count",
    "distinct_supporting_view_count",
    "total_candidate_count",
    "multiview_min_confidence_score",
    "multiview_min_degree",
    "require_distinct_supporting_view",
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

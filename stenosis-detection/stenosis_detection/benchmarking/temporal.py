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
from .severity import severity_row_fields, summarize_severity_agreement
from .sweep import write_threshold_sweep


TEMPORAL_RESULT_FILENAME = "view_temporal_fusion.json"


@dataclass(frozen=True, slots=True)
class TemporalBenchmarkOutputs:
    sequence_rows_csv: Path
    sequence_rows_jsonl: Path
    sequence_summary_json: Path
    sequence_summary_csv: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "temporal_sequence_rows_csv": str(self.sequence_rows_csv),
            "temporal_sequence_rows_jsonl": str(self.sequence_rows_jsonl),
            "temporal_sequence_summary_json": str(self.sequence_summary_json),
            "temporal_sequence_summary_csv": str(self.sequence_summary_csv),
        }


@dataclass(frozen=True, slots=True)
class TemporalBenchmarkResult:
    sequence_rows: list[dict[str, Any]]
    sequence_summary: BinaryMetricSummary
    output_paths: TemporalBenchmarkOutputs
    threshold_sweep_paths: dict[str, Path]


def run_temporal_level_benchmark(
    *,
    results_root: str | Path,
    weak_labels_path: str | Path,
    output_root: str | Path,
    temporal_min_degree: float = 0.0,
    temporal_min_persistence_ratio: float = 0.0,
    include_unclear_labels: bool = False,
    write_threshold_sweep_report: bool = False,
) -> TemporalBenchmarkResult:
    resolved_results_root = Path(results_root)
    _validate_temporal_inputs(
        resolved_results_root,
        temporal_min_degree=temporal_min_degree,
        temporal_min_persistence_ratio=temporal_min_persistence_ratio,
    )

    weak_labels = load_weak_labels_jsonl(weak_labels_path)
    sequence_rows = [
        _build_sequence_row(
            temporal_path,
            results_root=resolved_results_root,
            weak_labels=weak_labels,
            temporal_min_degree=temporal_min_degree,
            temporal_min_persistence_ratio=temporal_min_persistence_ratio,
            include_unclear_labels=include_unclear_labels,
        )
        for temporal_path in discover_temporal_result_paths(resolved_results_root)
    ]
    sequence_summary = _summarize_dict_rows(
        sequence_rows,
        prediction_column="predicted_positive",
        outcome_column="weak_outcome",
        skip_column="skip_reason",
        labels=weak_labels,
    )

    output_paths = save_temporal_benchmark_outputs(
        output_root=output_root,
        sequence_rows=sequence_rows,
        sequence_summary=sequence_summary,
        temporal_min_degree=temporal_min_degree,
        temporal_min_persistence_ratio=temporal_min_persistence_ratio,
        include_unclear_labels=include_unclear_labels,
    )
    threshold_sweep_paths = (
        write_threshold_sweep(
            rows=sequence_rows,
            output_root=output_root,
            level="temporal",
            score_columns={
                "temporal_score": "score",
                "persistence_ratio": "persistence_ratio",
            },
        )
        if write_threshold_sweep_report
        else {}
    )

    return TemporalBenchmarkResult(
        sequence_rows=sequence_rows,
        sequence_summary=sequence_summary,
        output_paths=output_paths,
        threshold_sweep_paths=threshold_sweep_paths,
    )


def discover_temporal_result_paths(results_root: str | Path) -> list[Path]:
    resolved_results_root = Path(results_root)
    if not resolved_results_root.exists():
        raise FileNotFoundError(f"Temporal results root does not exist: {resolved_results_root}")
    if not resolved_results_root.is_dir():
        raise NotADirectoryError(f"Temporal results root is not a directory: {resolved_results_root}")

    paths = sorted(path for path in resolved_results_root.rglob(TEMPORAL_RESULT_FILENAME) if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No '{TEMPORAL_RESULT_FILENAME}' files were found under: {resolved_results_root}")
    return paths


def save_temporal_benchmark_outputs(
    *,
    output_root: str | Path,
    sequence_rows: list[dict[str, Any]],
    sequence_summary: BinaryMetricSummary,
    temporal_min_degree: float,
    temporal_min_persistence_ratio: float,
    include_unclear_labels: bool,
) -> TemporalBenchmarkOutputs:
    resolved_output_root = Path(output_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    sequence_rows_csv = resolved_output_root / "temporal_sequence_rows.csv"
    sequence_rows_jsonl = resolved_output_root / "temporal_sequence_rows.jsonl"
    sequence_summary_json = resolved_output_root / "temporal_sequence_summary.json"
    sequence_summary_csv = resolved_output_root / "temporal_sequence_summary.csv"

    _write_csv(sequence_rows_csv, sequence_rows, fieldnames=TEMPORAL_SEQUENCE_ROW_FIELDS)
    _write_jsonl(sequence_rows_jsonl, sequence_rows)
    _write_json(
        sequence_summary_json,
        {
            "level": "temporal_sequence",
            "temporal_min_degree": float(temporal_min_degree),
            "temporal_min_persistence_ratio": float(temporal_min_persistence_ratio),
            "include_unclear_labels": include_unclear_labels,
            "summary": sequence_summary.to_dict(),
            "severity_agreement": summarize_severity_agreement(sequence_rows),
        },
    )
    _write_csv(
        sequence_summary_csv,
        [
            {
                "level": "temporal_sequence",
                "temporal_min_degree": float(temporal_min_degree),
                "temporal_min_persistence_ratio": float(temporal_min_persistence_ratio),
                "include_unclear_labels": include_unclear_labels,
                **sequence_summary.to_dict(),
            }
        ],
    )

    return TemporalBenchmarkOutputs(
        sequence_rows_csv=sequence_rows_csv,
        sequence_rows_jsonl=sequence_rows_jsonl,
        sequence_summary_json=sequence_summary_json,
        sequence_summary_csv=sequence_summary_csv,
    )


def _build_sequence_row(
    temporal_path: Path,
    *,
    results_root: Path,
    weak_labels: dict[str, WeakLabel],
    temporal_min_degree: float,
    temporal_min_persistence_ratio: float,
    include_unclear_labels: bool,
) -> dict[str, Any]:
    relative_path = _relative_path(temporal_path, results_root)
    case_id, sequence_id = _derive_case_and_sequence_id(relative_path)
    payload = _read_json_object(temporal_path)
    final_lesion = payload.get("final_lesion") if isinstance(payload.get("final_lesion"), dict) else None
    fusion = payload.get("fusion") if isinstance(payload.get("fusion"), dict) else {}
    degrees = final_lesion.get("degrees") if isinstance(final_lesion, dict) and isinstance(final_lesion.get("degrees"), dict) else {}

    score = _coerce_float(degrees.get("median")) if final_lesion is not None else 0.0
    max_degree = _coerce_float(degrees.get("max")) if final_lesion is not None else 0.0
    persistence_ratio = _coerce_float(final_lesion.get("persistence_ratio")) if final_lesion is not None else 0.0
    score = 0.0 if score is None else score
    max_degree = 0.0 if max_degree is None else max_degree
    persistence_ratio = 0.0 if persistence_ratio is None else persistence_ratio
    predicted_positive = (
        final_lesion is not None
        and score >= temporal_min_degree
        and persistence_ratio >= temporal_min_persistence_ratio
    )
    predicted_severity = final_lesion.get("severity") if final_lesion is not None else None

    label = weak_labels.get(case_id)
    weak_outcome, skip_reason, label_target = _weak_outcome(
        predicted_positive,
        label,
        include_unclear_labels=include_unclear_labels,
    )

    return {
        "case_id": case_id,
        "sequence_id": sequence_id,
        "relative_path": relative_path.as_posix(),
        "source_path": str(temporal_path),
        "json_view_id": payload.get("view_id"),
        "predicted_positive": predicted_positive,
        "persistent_lesion_count": _coerce_int(fusion.get("persistent_lesion_count")),
        "score": score,
        "max_degree": max_degree,
        "predicted_severity": predicted_severity,
        "persistence_ratio": persistence_ratio,
        "supporting_frame_count": _coerce_int(final_lesion.get("supporting_frame_count")) if final_lesion is not None else 0,
        "total_frame_count": _coerce_int(final_lesion.get("total_frame_count")) if final_lesion is not None else 0,
        "temporal_min_degree": float(temporal_min_degree),
        "temporal_min_persistence_ratio": float(temporal_min_persistence_ratio),
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


def _relative_path(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise BenchmarkIOError(f"Temporal result path is not under results root: {path}") from exc


def _derive_case_and_sequence_id(relative_path: Path) -> tuple[str, str]:
    parts = relative_path.parts
    if len(parts) < 2:
        raise BenchmarkIOError(
            f"Temporal result must live under a case directory relative to --results-root: {relative_path}"
        )
    case_id = parts[0]
    sequence_id = Path(*parts[1:-1]).as_posix() if len(parts) > 2 else ""
    return case_id, sequence_id


def _coerce_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _validate_temporal_inputs(
    results_root: Path,
    *,
    temporal_min_degree: float,
    temporal_min_persistence_ratio: float,
) -> None:
    if temporal_min_degree < 0.0:
        raise ValueError("temporal_min_degree must be >= 0.0.")
    if not 0.0 <= temporal_min_persistence_ratio <= 1.0:
        raise ValueError("temporal_min_persistence_ratio must be in the range [0.0, 1.0].")
    if not results_root.exists():
        raise FileNotFoundError(f"Temporal results root does not exist: {results_root}")
    if not results_root.is_dir():
        raise NotADirectoryError(f"Temporal results root is not a directory: {results_root}")


TEMPORAL_SEQUENCE_ROW_FIELDS = [
    "case_id",
    "sequence_id",
    "relative_path",
    "source_path",
    "json_view_id",
    "predicted_positive",
    "persistent_lesion_count",
    "score",
    "max_degree",
    "predicted_severity",
    "pipeline_severity",
    "weak_label_severity_normalized",
    "severity_match",
    "persistence_ratio",
    "supporting_frame_count",
    "total_frame_count",
    "temporal_min_degree",
    "temporal_min_persistence_ratio",
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

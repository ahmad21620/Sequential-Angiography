from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .models import BenchmarkResult, PipelinePrediction


class BenchmarkIOError(ValueError):
    pass


CaseIdResolver = Callable[[dict[str, Any], Path], str | None]


def load_pipeline_predictions(
    paths: str | Path | Sequence[str | Path],
    *,
    case_id_resolver: CaseIdResolver | None = None,
) -> list[PipelinePrediction]:
    """Load existing pipeline JSON outputs and convert them to binary predictions."""
    predictions: list[PipelinePrediction] = []
    for json_path in _iter_json_paths(paths):
        payload = _read_json_object(json_path)
        predictions.append(
            extract_pipeline_prediction(
                payload,
                source_path=json_path,
                case_id_resolver=case_id_resolver,
            )
        )
    return predictions


def extract_pipeline_prediction(
    payload: dict[str, Any],
    *,
    source_path: str | Path | None = None,
    case_id: str | None = None,
    case_id_resolver: CaseIdResolver | None = None,
) -> PipelinePrediction:
    """Extract a case-level binary prediction from a known pipeline JSON payload."""
    resolved_path = None if source_path is None else Path(source_path)
    resolved_case_id = case_id or _extract_case_id(payload, resolved_path, case_id_resolver=case_id_resolver)
    if resolved_case_id is None:
        context = "pipeline payload" if resolved_path is None else str(resolved_path)
        raise BenchmarkIOError(f"{context}: could not determine case_id for benchmarking.")

    prediction_level, predicted_positive, score, confidence = _extract_prediction_fields(payload)
    return PipelinePrediction(
        case_id=resolved_case_id,
        predicted_positive=predicted_positive,
        source_path=None if resolved_path is None else str(resolved_path),
        prediction_level=prediction_level,
        score=score,
        confidence=confidence,
    )


def save_benchmark_result(
    result: BenchmarkResult,
    output_dir: str | Path,
    *,
    rows_filename: str = "benchmark_rows.jsonl",
    summary_filename: str = "benchmark_summary.json",
) -> dict[str, Path]:
    """Save detailed benchmark rows as JSONL and aggregate metrics as JSON."""
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    rows_path = resolved_output_dir / rows_filename
    summary_path = resolved_output_dir / summary_filename

    rows_path.write_text(
        "".join(json.dumps(row.to_dict(), sort_keys=True) + "\n" for row in result.rows),
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(result.summary.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return {"rows_jsonl": rows_path, "summary_json": summary_path}


def _iter_json_paths(paths: str | Path | Sequence[str | Path]) -> Iterable[Path]:
    if isinstance(paths, (str, Path)):
        path = Path(paths)
        if path.is_dir():
            yield from sorted(candidate for candidate in path.rglob("*.json") if candidate.is_file())
            return
        yield path
        return

    for raw_path in paths:
        yield from _iter_json_paths(raw_path)


def _read_json_object(json_path: Path) -> dict[str, Any]:
    if not json_path.exists():
        raise FileNotFoundError(f"Pipeline JSON file does not exist: {json_path}")
    if not json_path.is_file():
        raise BenchmarkIOError(f"Pipeline JSON path is not a file: {json_path}")

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise BenchmarkIOError(f"{json_path}: invalid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise BenchmarkIOError(f"{json_path}: expected top-level JSON object.")
    return payload


def _extract_case_id(
    payload: dict[str, Any],
    source_path: Path | None,
    *,
    case_id_resolver: CaseIdResolver | None,
) -> str | None:
    if case_id_resolver is not None and source_path is not None:
        resolved_case_id = case_id_resolver(payload, source_path)
        if resolved_case_id:
            return str(resolved_case_id)

    for candidate in (
        payload.get("case_id"),
        _nested_get(payload, "frame", "case_id"),
        _nested_get(payload, "metadata", "case_id"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    return None


def _extract_prediction_fields(payload: dict[str, Any]) -> tuple[str, bool, float | None, str | None]:
    if "final_case_lesion" in payload:
        confidence_payload = payload.get("confidence") if isinstance(payload.get("confidence"), dict) else {}
        return (
            "case",
            payload.get("final_case_lesion") is not None,
            _optional_float(confidence_payload.get("score")),
            _optional_string(confidence_payload.get("label")),
        )

    if "final_lesion" in payload:
        final_lesion = payload.get("final_lesion")
        return (
            "sequence",
            final_lesion is not None,
            _lesion_score(final_lesion),
            _lesion_severity(final_lesion),
        )

    if "stenosis_points" in payload or "counts" in payload:
        stenosis_points = payload.get("stenosis_points")
        counts = payload.get("counts") if isinstance(payload.get("counts"), dict) else {}
        count_value = counts.get("stenosis_points")
        if isinstance(stenosis_points, list):
            predicted_positive = len(stenosis_points) > 0
        elif count_value is not None:
            predicted_positive = _optional_float(count_value) not in (None, 0.0)
        else:
            predicted_positive = False
        return ("frame", predicted_positive, None, None)

    raise BenchmarkIOError("Unsupported pipeline JSON schema for benchmarking.")


def _nested_get(payload: dict[str, Any], parent: str, child: str) -> object:
    parent_payload = payload.get(parent)
    if not isinstance(parent_payload, dict):
        return None
    return parent_payload.get(child)


def _lesion_score(payload: object) -> float | None:
    if not isinstance(payload, dict):
        return None
    degrees_payload = payload.get("degrees")
    if isinstance(degrees_payload, dict):
        return _optional_float(degrees_payload.get("median"))
    return _optional_float(payload.get("median_degree"))


def _lesion_severity(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    return _optional_string(payload.get("severity"))


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

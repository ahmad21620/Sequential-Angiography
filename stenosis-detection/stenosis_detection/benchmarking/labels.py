from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .models import WeakLabel


POSITIVE_SEVERITIES = {"mild", "moderate", "severe", "occlusion"}
NEGATIVE_SEVERITIES = {"none"}


class WeakLabelLoadError(ValueError):
    pass


def normalize_weak_label(stenosis_exists: object, severity: object) -> tuple[bool | None, str]:
    """Normalize one EHR weak label into a binary target or unclear."""
    normalized_exists = _normalize_token(stenosis_exists)
    normalized_severity = _normalize_token(severity)

    positive_evidence = normalized_exists == "yes" or normalized_severity in POSITIVE_SEVERITIES
    negative_evidence = normalized_exists == "no" or normalized_severity in NEGATIVE_SEVERITIES

    if positive_evidence and negative_evidence:
        return None, "conflicting_positive_and_negative_evidence"
    if positive_evidence:
        return True, "positive_stenosis_exists_or_positive_severity"
    if negative_evidence:
        return False, "negative_stenosis_exists_or_none_severity"
    return None, "unclear_label"


def load_weak_labels_jsonl(path: str | Path) -> dict[str, WeakLabel]:
    """Load weak EHR labels from a JSONL file and index them by case_id."""
    resolved_path = Path(path)
    labels: dict[str, WeakLabel] = {}

    if not resolved_path.exists():
        raise FileNotFoundError(f"Weak-label JSONL file does not exist: {resolved_path}")
    if not resolved_path.is_file():
        raise WeakLabelLoadError(f"Weak-label path is not a file: {resolved_path}")

    for line_number, raw_line in enumerate(resolved_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise WeakLabelLoadError(f"{resolved_path}:{line_number}: invalid JSONL row: {exc}") from exc

        label = weak_label_from_json(payload, source_path=str(resolved_path), line_number=line_number)
        if label.case_id in labels:
            raise WeakLabelLoadError(f"{resolved_path}:{line_number}: duplicate case_id '{label.case_id}'.")
        labels[label.case_id] = label

    return labels


def weak_label_from_json(
    payload: object,
    *,
    source_path: str | None = None,
    line_number: int | None = None,
) -> WeakLabel:
    if not isinstance(payload, dict):
        raise WeakLabelLoadError(f"{_label_context(source_path, line_number)}: expected a JSON object.")

    case_id = _require_non_empty_string(payload, "case_id", context=_label_context(source_path, line_number))
    stenosis_exists = _optional_string(payload.get("stenosis_exists"))
    severity = _optional_string(payload.get("severity"))
    target, reason = normalize_weak_label(stenosis_exists, severity)

    return WeakLabel(
        case_id=case_id,
        column_u=_optional_string(payload.get("column_U")),
        excel_row=_optional_int(payload.get("excel_row")),
        stenosis_exists=stenosis_exists,
        severity=severity,
        max_percent=_optional_float(payload.get("max_percent")),
        evidence=_optional_string(payload.get("evidence")),
        confidence=_optional_string(payload.get("confidence")),
        raw_model_output=payload.get("raw_model_output"),
        target=target,
        normalization_reason=reason,
        source_path=source_path,
        line_number=line_number,
    )


def index_weak_labels(labels: Iterable[WeakLabel]) -> dict[str, WeakLabel]:
    indexed: dict[str, WeakLabel] = {}
    for label in labels:
        if label.case_id in indexed:
            raise WeakLabelLoadError(f"Duplicate case_id '{label.case_id}'.")
        indexed[label.case_id] = label
    return indexed


def _normalize_token(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value)


def _require_non_empty_string(payload: dict[str, Any], field_name: str, *, context: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise WeakLabelLoadError(f"{context}: missing required string field '{field_name}'.")
    return value.strip()


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise WeakLabelLoadError(f"Expected integer-compatible value, got {value!r}.") from exc


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise WeakLabelLoadError(f"Expected float-compatible value, got {value!r}.") from exc


def _label_context(source_path: str | None, line_number: int | None) -> str:
    if source_path is None:
        return "weak label"
    if line_number is None:
        return source_path
    return f"{source_path}:{line_number}"

from __future__ import annotations

from typing import Any

from .labels import POSITIVE_SEVERITIES


def severity_row_fields(
    *,
    label_severity: str | None,
    pipeline_severity: str | None,
    label_target: bool | None,
    predicted_positive: bool,
) -> dict[str, Any]:
    normalized_label_severity = _normalize_severity(label_severity)
    normalized_pipeline_severity = _normalize_severity(pipeline_severity)
    comparable = (
        label_target is True
        and predicted_positive
        and normalized_label_severity in POSITIVE_SEVERITIES
        and normalized_pipeline_severity in POSITIVE_SEVERITIES
    )
    return {
        "weak_label_severity_normalized": normalized_label_severity,
        "pipeline_severity": normalized_pipeline_severity,
        "severity_match": (normalized_label_severity == normalized_pipeline_severity) if comparable else None,
    }


def summarize_severity_agreement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive_label_rows = [
        row
        for row in rows
        if row.get("label_target") is True
        and _normalize_severity(row.get("weak_label_severity_normalized") or row.get("label_severity")) in POSITIVE_SEVERITIES
    ]
    comparable_rows = [row for row in rows if row.get("severity_match") is not None]
    exact_match_count = sum(1 for row in comparable_rows if row.get("severity_match") is True)

    return {
        "evaluated_positive_label_count": len(positive_label_rows),
        "comparable_positive_prediction_count": len(comparable_rows),
        "exact_match_count": exact_match_count,
        "exact_match_rate": _safe_ratio(exact_match_count, len(comparable_rows)),
        "exact_match_percent": _percent(exact_match_count, len(comparable_rows)),
        "weak_label_severity_distribution": _distribution(
            _normalize_severity(row.get("weak_label_severity_normalized") or row.get("label_severity"))
            for row in positive_label_rows
        ),
        "pipeline_severity_distribution": _distribution(row.get("pipeline_severity") for row in comparable_rows),
        "by_weak_label_severity": _by_weak_label_severity(comparable_rows),
        "confusion_counts": _confusion_counts(comparable_rows),
    }


def _by_weak_label_severity(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    severities = sorted({_normalize_severity(row.get("weak_label_severity_normalized") or row.get("label_severity")) for row in rows})
    for severity in severities:
        if severity is None:
            continue
        severity_rows = [
            row
            for row in rows
            if _normalize_severity(row.get("weak_label_severity_normalized") or row.get("label_severity")) == severity
        ]
        exact_match_count = sum(1 for row in severity_rows if row.get("severity_match") is True)
        summary[severity] = {
            "count": len(severity_rows),
            "percent_of_comparable": _percent(len(severity_rows), len(rows)),
            "exact_match_count": exact_match_count,
            "exact_match_percent": _percent(exact_match_count, len(severity_rows)),
            "pipeline_severity_distribution": _distribution(row.get("pipeline_severity") for row in severity_rows),
        }
    return summary


def _confusion_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        weak_severity = _normalize_severity(row.get("weak_label_severity_normalized") or row.get("label_severity"))
        pipeline_severity = _normalize_severity(row.get("pipeline_severity"))
        if weak_severity is None or pipeline_severity is None:
            continue
        counts.setdefault(weak_severity, {})
        counts[weak_severity][pipeline_severity] = counts[weak_severity].get(pipeline_severity, 0) + 1
    return counts


def _distribution(values) -> dict[str, dict[str, float | int]]:
    normalized_values = [value for value in (_normalize_severity(value) for value in values) if value is not None]
    total = len(normalized_values)
    distribution: dict[str, dict[str, float | int]] = {}
    for value in sorted(set(normalized_values)):
        count = normalized_values.count(value)
        distribution[value] = {
            "count": count,
            "percent": _percent(count, total),
        }
    return distribution


def _normalize_severity(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized or None


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _percent(numerator: int, denominator: int) -> float | None:
    ratio = _safe_ratio(numerator, denominator)
    if ratio is None:
        return None
    return float(ratio * 100.0)

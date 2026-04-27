from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .metrics import summarize_rows
from .models import BenchmarkRow


SWEEP_THRESHOLDS = [round(index * 0.05, 2) for index in range(21)]
SWEEP_FIELDS = [
    "score_name",
    "threshold",
    "TP",
    "FP",
    "TN",
    "FN",
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "F1",
    "false_positive_rate",
    "false_negative_rate",
    "balanced_accuracy",
    "predicted_positive_count",
]


def write_threshold_sweep(
    *,
    rows: list[dict[str, Any]],
    output_root: str | Path,
    level: str,
    score_columns: dict[str, str],
) -> dict[str, Path]:
    sweep_rows: list[dict[str, Any]] = []
    for score_name, score_column in score_columns.items():
        sweep_rows.extend(_build_sweep_rows(rows, score_name=score_name, score_column=score_column))

    resolved_output_root = Path(output_root)
    csv_path = resolved_output_root / f"threshold_sweep_{level}.csv"
    summary_path = resolved_output_root / "threshold_sweep_summary.json"
    _write_csv(csv_path, sweep_rows, fieldnames=SWEEP_FIELDS)
    _write_json(summary_path, {"level": level, "operating_points": _suggest_operating_points(sweep_rows)})
    return {"threshold_sweep_csv": csv_path, "threshold_sweep_summary_json": summary_path}


def _build_sweep_rows(
    rows: list[dict[str, Any]],
    *,
    score_name: str,
    score_column: str,
) -> list[dict[str, Any]]:
    sweep_rows: list[dict[str, Any]] = []
    for threshold in SWEEP_THRESHOLDS:
        metric_rows = [
            BenchmarkRow(
                case_id=str(row["case_id"]),
                predicted_positive=_score_is_positive(row.get(score_column), threshold),
                label_target=row.get("label_target"),
                label_status=str(row.get("label_status") or "missing"),
                outcome=_outcome(_score_is_positive(row.get(score_column), threshold), row.get("label_target")),
                skip_reason=row.get("skip_reason") if row.get("label_target") is None else None,
            )
            for row in rows
        ]
        summary = summarize_rows(metric_rows)
        sweep_rows.append(
            {
                "score_name": score_name,
                "threshold": threshold,
                "TP": summary.true_positive,
                "FP": summary.false_positive,
                "TN": summary.true_negative,
                "FN": summary.false_negative,
                "accuracy": summary.accuracy,
                "precision": summary.precision,
                "recall": summary.recall,
                "specificity": summary.specificity,
                "F1": summary.f1,
                "false_positive_rate": summary.false_positive_rate,
                "false_negative_rate": summary.false_negative_rate,
                "balanced_accuracy": summary.balanced_accuracy,
                "predicted_positive_count": summary.predicted_positive,
            }
        )
    return sweep_rows


def _score_is_positive(score: object, threshold: float) -> bool:
    try:
        value = float(score)
    except (TypeError, ValueError):
        value = 0.0
    if threshold == 0.0:
        return value > 0.0
    return value >= threshold


def _outcome(predicted_positive: bool, label_target: object) -> str | None:
    if label_target is True and predicted_positive:
        return "TP"
    if label_target is False and predicted_positive:
        return "FP"
    if label_target is False and not predicted_positive:
        return "TN"
    if label_target is True and not predicted_positive:
        return "FN"
    return None


def _suggest_operating_points(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    suggestions: dict[str, dict[str, Any]] = {}
    score_names = sorted({str(row["score_name"]) for row in rows})
    for score_name in score_names:
        score_rows = [row for row in rows if row["score_name"] == score_name]
        suggestions[score_name] = {
            "best_f1": _best_by(score_rows, "F1"),
            "best_balanced_accuracy": _best_by(score_rows, "balanced_accuracy"),
            "highest_recall_with_specificity_at_least_0_80": _best_filtered(
                score_rows,
                filter_metric="specificity",
                minimum=0.80,
                optimize_metric="recall",
                prefer_low_threshold=True,
            ),
            "lowest_false_positive_rate_with_recall_at_least_0_70": _best_filtered(
                score_rows,
                filter_metric="recall",
                minimum=0.70,
                optimize_metric="false_positive_rate",
                prefer_low_value=True,
                prefer_low_threshold=False,
            ),
        }
    return suggestions


def _best_by(rows: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get(metric) is not None]
    if not candidates:
        return None
    return dict(max(candidates, key=lambda row: (float(row[metric]), -float(row["threshold"]))))


def _best_filtered(
    rows: list[dict[str, Any]],
    *,
    filter_metric: str,
    minimum: float,
    optimize_metric: str,
    prefer_low_value: bool = False,
    prefer_low_threshold: bool = True,
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row.get(filter_metric) is not None
        and float(row[filter_metric]) >= minimum
        and row.get(optimize_metric) is not None
    ]
    if not candidates:
        return None

    if prefer_low_value:
        threshold_key = float if prefer_low_threshold else lambda value: -float(value)
        return dict(min(candidates, key=lambda row: (float(row[optimize_metric]), threshold_key(row["threshold"]))))

    threshold_key = (lambda value: -float(value)) if prefer_low_threshold else float
    return dict(max(candidates, key=lambda row: (float(row[optimize_metric]), threshold_key(row["threshold"]))))


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field_name: "" if row.get(field_name) is None else row.get(field_name) for field_name in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

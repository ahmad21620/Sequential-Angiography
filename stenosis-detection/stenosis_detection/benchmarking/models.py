from __future__ import annotations

from dataclasses import dataclass
from typing import Any


WEAK_LABEL_SUMMARY_NOTE = (
    "These metrics compare pipeline predictions against case-level weak EHR labels. "
    "Frame-level and sequence-level metrics should be interpreted as weak agreement metrics, "
    "not true localization accuracy."
)


@dataclass(frozen=True, slots=True)
class WeakLabel:
    case_id: str
    stenosis_exists: str | None
    severity: str | None
    target: bool | None
    normalization_reason: str
    source_path: str | None = None
    line_number: int | None = None
    column_u: str | None = None
    excel_row: int | None = None
    max_percent: float | None = None
    evidence: str | None = None
    confidence: str | None = None
    raw_model_output: Any = None

    @property
    def status(self) -> str:
        if self.target is True:
            return "positive"
        if self.target is False:
            return "negative"
        return "unclear"

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "stenosis_exists": self.stenosis_exists,
            "severity": self.severity,
            "target": self.target,
            "status": self.status,
            "normalization_reason": self.normalization_reason,
            "source_path": self.source_path,
            "line_number": self.line_number,
            "column_U": self.column_u,
            "excel_row": self.excel_row,
            "max_percent": self.max_percent,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "raw_model_output": self.raw_model_output,
        }


@dataclass(frozen=True, slots=True)
class PipelinePrediction:
    case_id: str
    predicted_positive: bool
    source_path: str | None = None
    prediction_level: str | None = None
    score: float | None = None
    confidence: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "predicted_positive": self.predicted_positive,
            "source_path": self.source_path,
            "prediction_level": self.prediction_level,
            "score": self.score,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkRow:
    case_id: str
    predicted_positive: bool
    label_target: bool | None
    label_status: str
    outcome: str | None
    skip_reason: str | None
    source_path: str | None = None
    prediction_level: str | None = None
    prediction_score: float | None = None
    prediction_confidence: str | None = None
    label_stenosis_exists: str | None = None
    label_severity: str | None = None
    label_confidence: str | None = None
    label_excel_row: int | None = None
    label_source_path: str | None = None
    used_unclear_label: bool = False

    @property
    def evaluated(self) -> bool:
        return self.outcome is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "source_path": self.source_path,
            "prediction_level": self.prediction_level,
            "predicted_positive": self.predicted_positive,
            "prediction_score": self.prediction_score,
            "prediction_confidence": self.prediction_confidence,
            "label_target": self.label_target,
            "label_status": self.label_status,
            "label_stenosis_exists": self.label_stenosis_exists,
            "label_severity": self.label_severity,
            "label_confidence": self.label_confidence,
            "label_excel_row": self.label_excel_row,
            "label_source_path": self.label_source_path,
            "used_unclear_label": self.used_unclear_label,
            "outcome": self.outcome,
            "skip_reason": self.skip_reason,
            "evaluated": self.evaluated,
        }


@dataclass(frozen=True, slots=True)
class BinaryMetricSummary:
    total_predictions: int
    total_evaluated: int
    skipped_missing_label: int
    skipped_unclear_label: int
    positive_labels: int
    negative_labels: int
    predicted_positive: int
    predicted_negative: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    accuracy: float | None
    precision: float | None
    recall: float | None
    specificity: float | None
    f1: float | None
    false_positive_rate: float | None
    false_negative_rate: float | None
    negative_predictive_value: float | None
    balanced_accuracy: float | None
    labels_without_prediction: int = 0
    unclear_labels_included: int = 0
    note: str = WEAK_LABEL_SUMMARY_NOTE

    @property
    def ppv(self) -> float | None:
        return self.precision

    @property
    def sensitivity(self) -> float | None:
        return self.recall

    def to_dict(self) -> dict[str, Any]:
        return {
            "note": self.note,
            "total_predictions": self.total_predictions,
            "total_evaluated": self.total_evaluated,
            "skipped_missing_label": self.skipped_missing_label,
            "skipped_unclear_label": self.skipped_unclear_label,
            "positive_labels": self.positive_labels,
            "negative_labels": self.negative_labels,
            "predicted_positive": self.predicted_positive,
            "predicted_negative": self.predicted_negative,
            "TP": self.true_positive,
            "FP": self.false_positive,
            "TN": self.true_negative,
            "FN": self.false_negative,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "ppv": self.ppv,
            "recall": self.recall,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "f1": self.f1,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "negative_predictive_value": self.negative_predictive_value,
            "balanced_accuracy": self.balanced_accuracy,
            "labels_without_prediction": self.labels_without_prediction,
            "unclear_labels_included": self.unclear_labels_included,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    rows: list[BenchmarkRow]
    summary: BinaryMetricSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary.to_dict(),
            "rows": [row.to_dict() for row in self.rows],
        }

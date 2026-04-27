from __future__ import annotations

from typing import Iterable, Mapping

from .labels import index_weak_labels
from .models import BenchmarkResult, BenchmarkRow, BinaryMetricSummary, PipelinePrediction, WeakLabel


def evaluate_binary_predictions(
    predictions: Iterable[PipelinePrediction],
    labels: Mapping[str, WeakLabel] | Iterable[WeakLabel],
    *,
    include_unclear: bool = False,
    unclear_target: bool | None = None,
) -> BenchmarkResult:
    """Compare pipeline binary predictions with normalized weak EHR labels."""
    if include_unclear and unclear_target is None:
        raise ValueError("unclear_target must be provided when include_unclear=True.")

    indexed_labels = dict(labels) if isinstance(labels, Mapping) else index_weak_labels(labels)
    prediction_list = list(predictions)
    predicted_case_ids: set[str] = set()
    rows: list[BenchmarkRow] = []
    unclear_labels_included = 0

    for prediction in prediction_list:
        predicted_case_ids.add(prediction.case_id)
        label = indexed_labels.get(prediction.case_id)
        if label is None:
            rows.append(_skipped_row(prediction, label_status="missing", skip_reason="missing_label"))
            continue

        label_target = label.target
        used_unclear_label = False
        if label_target is None:
            if not include_unclear:
                rows.append(_skipped_row(prediction, label=label, label_status="unclear", skip_reason="unclear_label"))
                continue
            label_target = unclear_target
            used_unclear_label = True
            unclear_labels_included += 1

        outcome = _binary_outcome(label_target=label_target, predicted_positive=prediction.predicted_positive)
        rows.append(
            BenchmarkRow(
                case_id=prediction.case_id,
                predicted_positive=prediction.predicted_positive,
                label_target=label_target,
                label_status="unclear" if used_unclear_label else label.status,
                outcome=outcome,
                skip_reason=None,
                source_path=prediction.source_path,
                prediction_level=prediction.prediction_level,
                prediction_score=prediction.score,
                prediction_confidence=prediction.confidence,
                label_stenosis_exists=label.stenosis_exists,
                label_severity=label.severity,
                label_confidence=label.confidence,
                label_excel_row=label.excel_row,
                label_source_path=label.source_path,
                used_unclear_label=used_unclear_label,
            )
        )

    summary = summarize_rows(
        rows,
        total_predictions=len(prediction_list),
        labels_without_prediction=len(set(indexed_labels) - predicted_case_ids),
        unclear_labels_included=unclear_labels_included,
    )
    return BenchmarkResult(rows=rows, summary=summary)


def summarize_rows(
    rows: Iterable[BenchmarkRow],
    *,
    total_predictions: int | None = None,
    labels_without_prediction: int = 0,
    unclear_labels_included: int = 0,
) -> BinaryMetricSummary:
    row_list = list(rows)
    evaluated_rows = [row for row in row_list if row.evaluated]
    true_positive = sum(1 for row in evaluated_rows if row.outcome == "TP")
    false_positive = sum(1 for row in evaluated_rows if row.outcome == "FP")
    true_negative = sum(1 for row in evaluated_rows if row.outcome == "TN")
    false_negative = sum(1 for row in evaluated_rows if row.outcome == "FN")

    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    specificity = _safe_divide(true_negative, true_negative + false_positive)

    if precision is None or recall is None or (precision + recall) == 0.0:
        f1 = None
    else:
        f1 = float(2.0 * precision * recall / (precision + recall))

    if recall is None or specificity is None:
        balanced_accuracy = None
    else:
        balanced_accuracy = float((recall + specificity) / 2.0)

    return BinaryMetricSummary(
        total_predictions=len(row_list) if total_predictions is None else total_predictions,
        total_evaluated=len(evaluated_rows),
        skipped_missing_label=sum(1 for row in row_list if row.skip_reason == "missing_label"),
        skipped_unclear_label=sum(1 for row in row_list if row.skip_reason == "unclear_label"),
        positive_labels=sum(1 for row in evaluated_rows if row.label_target is True),
        negative_labels=sum(1 for row in evaluated_rows if row.label_target is False),
        predicted_positive=sum(1 for row in evaluated_rows if row.predicted_positive),
        predicted_negative=sum(1 for row in evaluated_rows if not row.predicted_positive),
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
        accuracy=_safe_divide(true_positive + true_negative, len(evaluated_rows)),
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1=f1,
        false_positive_rate=_safe_divide(false_positive, false_positive + true_negative),
        false_negative_rate=_safe_divide(false_negative, false_negative + true_positive),
        negative_predictive_value=_safe_divide(true_negative, true_negative + false_negative),
        balanced_accuracy=balanced_accuracy,
        labels_without_prediction=labels_without_prediction,
        unclear_labels_included=unclear_labels_included,
    )


def _binary_outcome(*, label_target: bool, predicted_positive: bool) -> str:
    if label_target and predicted_positive:
        return "TP"
    if not label_target and predicted_positive:
        return "FP"
    if not label_target and not predicted_positive:
        return "TN"
    return "FN"


def _skipped_row(
    prediction: PipelinePrediction,
    *,
    label_status: str | None = None,
    skip_reason: str,
    label: WeakLabel | None = None,
) -> BenchmarkRow:
    return BenchmarkRow(
        case_id=prediction.case_id,
        predicted_positive=prediction.predicted_positive,
        label_target=None,
        label_status=label.status if label is not None else (label_status or "missing"),
        outcome=None,
        skip_reason=skip_reason,
        source_path=prediction.source_path,
        prediction_level=prediction.prediction_level,
        prediction_score=prediction.score,
        prediction_confidence=prediction.confidence,
        label_stenosis_exists=None if label is None else label.stenosis_exists,
        label_severity=None if label is None else label.severity,
        label_confidence=None if label is None else label.confidence,
        label_excel_row=None if label is None else label.excel_row,
        label_source_path=None if label is None else label.source_path,
    )


def _safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)

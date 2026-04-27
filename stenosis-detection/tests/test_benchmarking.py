from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stenosis_detection.benchmarking import (
    WEAK_LABEL_SUMMARY_NOTE,
    PipelinePrediction,
    evaluate_binary_predictions,
    load_weak_labels_jsonl,
    normalize_weak_label,
)


class BenchmarkingTests(unittest.TestCase):
    def test_load_weak_labels_jsonl_normalizes_binary_targets(self) -> None:
        rows = [
            {
                "case_id": "case_positive_exists",
                "column_U": "U text",
                "excel_row": 91749,
                "stenosis_exists": "yes",
                "severity": "unclear",
                "max_percent": None,
                "evidence": "positive mention",
                "confidence": "high",
                "raw_model_output": {"raw": True},
            },
            {
                "case_id": "case_positive_severity",
                "stenosis_exists": "unclear",
                "severity": "moderate",
                "max_percent": 62,
                "confidence": "medium",
                "raw_model_output": {},
            },
            {
                "case_id": "case_negative",
                "stenosis_exists": "no",
                "severity": "none",
                "max_percent": 0,
                "confidence": "high",
                "raw_model_output": {},
            },
            {
                "case_id": "case_unclear",
                "stenosis_exists": "unclear",
                "severity": "unclear",
                "max_percent": None,
                "confidence": "low",
                "raw_model_output": {},
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = Path(temp_dir) / "labels.jsonl"
            jsonl_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

            labels = load_weak_labels_jsonl(jsonl_path)

        self.assertTrue(labels["case_positive_exists"].target)
        self.assertTrue(labels["case_positive_severity"].target)
        self.assertFalse(labels["case_negative"].target)
        self.assertIsNone(labels["case_unclear"].target)
        self.assertEqual(labels["case_positive_exists"].column_u, "U text")
        self.assertEqual(labels["case_positive_exists"].excel_row, 91749)
        self.assertEqual(labels["case_positive_exists"].status, "positive")

    def test_normalize_weak_label_marks_conflicts_unclear(self) -> None:
        target, reason = normalize_weak_label("yes", "none")

        self.assertIsNone(target)
        self.assertEqual(reason, "conflicting_positive_and_negative_evidence")

    def test_evaluate_binary_predictions_counts_confusion_and_skips(self) -> None:
        labels = {
            "tp": self._label("tp", True),
            "fp": self._label("fp", False),
            "tn": self._label("tn", False),
            "fn": self._label("fn", True),
            "unclear": self._label("unclear", None),
            "without_prediction": self._label("without_prediction", True),
        }
        predictions = [
            PipelinePrediction(case_id="tp", predicted_positive=True, prediction_level="case"),
            PipelinePrediction(case_id="fp", predicted_positive=True, prediction_level="case"),
            PipelinePrediction(case_id="tn", predicted_positive=False, prediction_level="case"),
            PipelinePrediction(case_id="fn", predicted_positive=False, prediction_level="case"),
            PipelinePrediction(case_id="unclear", predicted_positive=True, prediction_level="case"),
            PipelinePrediction(case_id="missing", predicted_positive=True, prediction_level="case"),
        ]

        result = evaluate_binary_predictions(predictions, labels)
        summary = result.summary

        self.assertEqual(summary.total_predictions, 6)
        self.assertEqual(summary.total_evaluated, 4)
        self.assertEqual(summary.skipped_missing_label, 1)
        self.assertEqual(summary.skipped_unclear_label, 1)
        self.assertEqual(summary.positive_labels, 2)
        self.assertEqual(summary.negative_labels, 2)
        self.assertEqual(summary.predicted_positive, 2)
        self.assertEqual(summary.predicted_negative, 2)
        self.assertEqual(summary.true_positive, 1)
        self.assertEqual(summary.false_positive, 1)
        self.assertEqual(summary.true_negative, 1)
        self.assertEqual(summary.false_negative, 1)
        self.assertAlmostEqual(summary.accuracy or 0.0, 0.5)
        self.assertAlmostEqual(summary.precision or 0.0, 0.5)
        self.assertAlmostEqual(summary.recall or 0.0, 0.5)
        self.assertAlmostEqual(summary.specificity or 0.0, 0.5)
        self.assertAlmostEqual(summary.f1 or 0.0, 0.5)
        self.assertAlmostEqual(summary.false_positive_rate or 0.0, 0.5)
        self.assertAlmostEqual(summary.false_negative_rate or 0.0, 0.5)
        self.assertAlmostEqual(summary.negative_predictive_value or 0.0, 0.5)
        self.assertAlmostEqual(summary.balanced_accuracy or 0.0, 0.5)
        self.assertEqual(summary.labels_without_prediction, 1)
        self.assertEqual(summary.to_dict()["note"], WEAK_LABEL_SUMMARY_NOTE)

    def test_evaluate_binary_predictions_can_include_unclear_with_explicit_target(self) -> None:
        labels = {"unclear": self._label("unclear", None)}
        predictions = [PipelinePrediction(case_id="unclear", predicted_positive=True)]

        result = evaluate_binary_predictions(predictions, labels, include_unclear=True, unclear_target=True)

        self.assertEqual(result.summary.total_evaluated, 1)
        self.assertEqual(result.summary.unclear_labels_included, 1)
        self.assertEqual(result.summary.true_positive, 1)
        self.assertTrue(result.rows[0].used_unclear_label)

    def _label(self, case_id: str, target: bool | None):
        from stenosis_detection.benchmarking.models import WeakLabel

        if target is True:
            stenosis_exists = "yes"
            severity = "moderate"
            reason = "test_positive"
        elif target is False:
            stenosis_exists = "no"
            severity = "none"
            reason = "test_negative"
        else:
            stenosis_exists = "unclear"
            severity = "unclear"
            reason = "test_unclear"

        return WeakLabel(
            case_id=case_id,
            stenosis_exists=stenosis_exists,
            severity=severity,
            target=target,
            normalization_reason=reason,
        )


if __name__ == "__main__":
    unittest.main()

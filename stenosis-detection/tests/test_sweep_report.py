from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stenosis_detection.benchmarking.sweep_report import (
    VariantMetric,
    pareto_front,
    parse_frame_variant,
    parse_temporal_variant,
    rank_variants,
    run_sweep_report,
)


class SweepReportTests(unittest.TestCase):
    def test_parse_frame_variant_name(self) -> None:
        parsed = parse_frame_variant(
            "radius_outside_fraction_threshold_0p1"
            "__radius_min_outside_samples_2"
            "__stenosis_threshold_0p25"
            "__average_radius_threshold_4"
        )

        self.assertEqual(parsed["radius_outside_fraction_threshold"], 0.1)
        self.assertEqual(parsed["radius_min_outside_samples"], 2)
        self.assertEqual(parsed["stenosis_threshold"], 0.25)
        self.assertEqual(parsed["average_radius_threshold"], 4.0)

    def test_parse_temporal_variant_name(self) -> None:
        parsed = parse_temporal_variant("min_supporting_frames_2__min_persistence_ratio_0p2")

        self.assertEqual(parsed["min_supporting_frames"], 2)
        self.assertEqual(parsed["min_persistence_ratio"], 0.2)

    def test_rank_variants_uses_binary_final_metric_order(self) -> None:
        lower_recall = self._variant("a", f1=0.8, recall=0.6, precision=0.95, balanced_accuracy=0.8, fp=0, fn=4)
        higher_recall = self._variant("b", f1=0.8, recall=0.7, precision=0.80, balanced_accuracy=0.7, fp=2, fn=3)
        lower_f1 = self._variant("c", f1=0.7, recall=1.0, precision=1.0, balanced_accuracy=1.0, fp=0, fn=0)

        ranked = rank_variants([lower_f1, lower_recall, higher_recall])

        self.assertEqual([variant.frame_variant for variant in ranked], ["b", "a", "c"])

    def test_pareto_front_filters_precision_recall_dominated_variants(self) -> None:
        dominated = self._variant("a", precision=0.80, recall=0.50, f1=0.62)
        dominates = self._variant("b", precision=0.80, recall=0.60, f1=0.69)
        tradeoff = self._variant("c", precision=0.70, recall=0.75, f1=0.72)

        front = pareto_front([dominated, dominates, tradeoff])

        self.assertNotIn(dominated, front)
        self.assertIn(dominates, front)
        self.assertIn(tradeoff, front)

    def test_report_generation_on_tiny_benchmark_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            sweep_root = temp_root / "sweep"
            benchmark_root = sweep_root / "benchmark_results"
            output_root = temp_root / "final_report"
            frame_variant = (
                "radius_outside_fraction_threshold_0p1"
                "__radius_min_outside_samples_2"
                "__stenosis_threshold_0p25"
                "__average_radius_threshold_4"
            )
            temporal_variant = "min_supporting_frames_2__min_persistence_ratio_0p2"
            sweep_root.mkdir()

            self._write_summary(
                benchmark_root / "frame" / frame_variant / "frame_summary.json",
                tp=1,
                fp=1,
                tn=1,
                fn=0,
            )
            self._write_summary(
                benchmark_root / "temporal" / frame_variant / temporal_variant / "temporal_sequence_summary.json",
                tp=1,
                fp=0,
                tn=1,
                fn=1,
            )
            multiview_root = benchmark_root / "multiview" / frame_variant / temporal_variant
            self._write_summary(
                multiview_root / "multiview_case_summary.json",
                tp=1,
                fp=0,
                tn=1,
                fn=0,
            )
            self._write_case_rows(multiview_root / "multiview_case_rows.csv")

            artifacts = run_sweep_report(
                sweep_root=sweep_root,
                output_root=output_root,
                top_k=1,
            )

            self.assertTrue(artifacts.reports["markdown"].is_file())
            self.assertTrue(artifacts.reports["html"].is_file())
            self.assertTrue(artifacts.reports["summary_json"].is_file())
            self.assertTrue((output_root / "plots" / "02_multiview_precision_recall_scatter.png").is_file())
            all_rows = self._read_csv(output_root / "tables" / "all_variant_metrics.csv")
            case_rows = self._read_csv(output_root / "tables" / "final_selected_variant_cases.csv")
            self.assertEqual(len(all_rows), 3)
            self.assertEqual(len(case_rows), 2)
            self.assertIn("selected_side", case_rows[0])
            self.assertNotIn("final_severity", case_rows[0])
            self.assertNotIn("severity agreement", artifacts.reports["markdown"].read_text(encoding="utf-8").lower())

    def _variant(
        self,
        name: str,
        *,
        f1: float = 0.0,
        recall: float = 0.0,
        precision: float = 0.0,
        balanced_accuracy: float = 0.0,
        fp: int = 0,
        fn: int = 0,
    ) -> VariantMetric:
        metrics = {
            "f1": f1,
            "recall": recall,
            "precision": precision,
            "balanced_accuracy": balanced_accuracy,
            "FP": fp,
            "FN": fn,
            "TP": 0,
            "TN": 0,
        }
        return VariantMetric(
            level="multiview",
            frame_variant=name,
            temporal_variant="min_supporting_frames_2__min_persistence_ratio_0p2",
            result_summary_path=Path(f"{name}.json"),
            row_count=1,
            frame_params=parse_frame_variant(name),
            temporal_params=parse_temporal_variant("min_supporting_frames_2__min_persistence_ratio_0p2"),
            metrics=metrics,
        )

    def _write_summary(self, path: Path, *, tp: int, fp: int, tn: int, fn: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        precision = None if tp + fp == 0 else tp / (tp + fp)
        recall = None if tp + fn == 0 else tp / (tp + fn)
        specificity = None if tn + fp == 0 else tn / (tn + fp)
        f1 = None if not precision or not recall else 2 * precision * recall / (precision + recall)
        summary = {
            "total_predictions": tp + fp + tn + fn,
            "total_evaluated": tp + fp + tn + fn,
            "skipped_missing_label": 0,
            "skipped_unclear_label": 0,
            "positive_labels": tp + fn,
            "negative_labels": tn + fp,
            "predicted_positive": tp + fp,
            "predicted_negative": tn + fn,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "accuracy": (tp + tn) / (tp + fp + tn + fn),
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "balanced_accuracy": None if recall is None or specificity is None else (recall + specificity) / 2,
            "false_positive_rate": None if specificity is None else 1 - specificity,
            "false_negative_rate": None if recall is None else 1 - recall,
            "negative_predictive_value": None if tn + fn == 0 else tn / (tn + fn),
            "labels_without_prediction": 0,
        }
        path.write_text(json.dumps({"summary": summary}, indent=2), encoding="utf-8")

    def _write_case_rows(self, path: Path) -> None:
        fieldnames = [
            "case_id",
            "predicted_positive",
            "label_target",
            "weak_outcome",
            "confidence_score",
            "final_degree",
            "final_max_degree",
            "supporting_view_count",
            "distinct_supporting_view_count",
            "selected_side",
            "source_path",
        ]
        rows = [
            {
                "case_id": "case_1",
                "predicted_positive": "True",
                "label_target": "True",
                "weak_outcome": "TP",
                "confidence_score": "0.9",
                "final_degree": "0.7",
                "final_max_degree": "0.8",
                "supporting_view_count": "1",
                "distinct_supporting_view_count": "1",
                "selected_side": "left",
                "source_path": "case_1/case_multiview_fusion.json",
            },
            {
                "case_id": "case_2",
                "predicted_positive": "False",
                "label_target": "False",
                "weak_outcome": "TN",
                "confidence_score": "0.0",
                "final_degree": "0.0",
                "final_max_degree": "0.0",
                "supporting_view_count": "0",
                "distinct_supporting_view_count": "0",
                "selected_side": "right",
                "source_path": "case_2/case_multiview_fusion.json",
            },
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _read_csv(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()

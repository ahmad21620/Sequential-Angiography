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

from stenosis_detection.benchmarking import BenchmarkIOError, run_multiview_level_benchmark


class MultiViewBenchmarkingTests(unittest.TestCase):
    def test_multiview_benchmark_writes_case_outputs_and_breakdowns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            results_root = temp_root / "case_results"
            output_root = temp_root / "benchmarks"
            weak_labels_path = temp_root / "weak_labels.jsonl"

            self._write_multiview_result(
                results_root / "case_001" / "case_multiview_fusion.json",
                case_id="case_001",
                final_case_lesion={
                    "severity": "moderate",
                    "degrees": {"median": 0.65, "max": 0.82},
                    "distinct_supporting_view_ids": ["view_02"],
                    "total_score": 0.93,
                },
                confidence={"score": 0.91, "label": "high"},
                supporting_views=["view_02"],
                total_candidate_count=2,
            )
            self._write_multiview_result(
                results_root / "case_002" / "case_multiview_fusion.json",
                case_id="case_002",
                final_case_lesion={
                    "severity": "mild",
                    "degrees": {"median": 0.40, "max": 0.55},
                    "distinct_supporting_view_ids": [],
                    "total_score": 0.50,
                },
                confidence={"score": 0.50, "label": "medium"},
                supporting_views=[],
                total_candidate_count=1,
            )
            self._write_multiview_result(
                results_root / "case_003" / "case_multiview_fusion.json",
                case_id="case_003",
                final_case_lesion=None,
                confidence={"score": 0.0, "label": "low"},
                supporting_views=[],
                total_candidate_count=0,
            )
            self._write_multiview_result(
                results_root / "case_004" / "case_multiview_fusion.json",
                case_id="case_004",
                final_case_lesion=None,
                confidence={"score": 0.0, "label": "low"},
                supporting_views=[],
                total_candidate_count=0,
            )
            self._write_multiview_result(
                results_root / "case_005" / "case_multiview_fusion.json",
                case_id="case_005",
                final_case_lesion=None,
                confidence={"score": 0.0, "label": "low"},
                supporting_views=[],
                total_candidate_count=0,
            )
            self._write_weak_labels(
                weak_labels_path,
                [
                    {"case_id": "case_001", "stenosis_exists": "yes", "severity": "moderate", "confidence": "high"},
                    {"case_id": "case_002", "stenosis_exists": "no", "severity": "none", "confidence": "high"},
                    {"case_id": "case_003", "stenosis_exists": "yes", "severity": "mild", "confidence": "medium"},
                    {"case_id": "case_004", "stenosis_exists": "no", "severity": "none", "confidence": "low"},
                    {"case_id": "case_006", "stenosis_exists": "yes", "severity": "severe", "confidence": "low"},
                ],
            )

            result = run_multiview_level_benchmark(
                results_root=results_root,
                weak_labels_path=weak_labels_path,
                output_root=output_root,
                write_threshold_sweep_report=True,
            )

            self.assertTrue((output_root / "multiview_case_rows.csv").is_file())
            self.assertTrue((output_root / "multiview_case_rows.jsonl").is_file())
            self.assertTrue((output_root / "multiview_case_summary.json").is_file())
            self.assertTrue((output_root / "multiview_case_summary.csv").is_file())
            self.assertTrue((output_root / "false_positive_cases.csv").is_file())
            self.assertTrue((output_root / "false_negative_cases.csv").is_file())
            self.assertTrue((output_root / "true_positive_cases.csv").is_file())
            self.assertTrue((output_root / "true_negative_cases.csv").is_file())
            self.assertTrue((output_root / "threshold_sweep_multiview.csv").is_file())
            self.assertTrue((output_root / "threshold_sweep_summary.json").is_file())

            self.assertEqual(len(result.case_rows), 5)
            first_row = result.case_rows[0]
            self.assertEqual(first_row["case_id"], "case_001")
            self.assertTrue(first_row["predicted_positive"])
            self.assertEqual(first_row["final_severity"], "moderate")
            self.assertAlmostEqual(first_row["final_degree"], 0.65)
            self.assertAlmostEqual(first_row["final_max_degree"], 0.82)
            self.assertAlmostEqual(first_row["total_score"], 0.93)
            self.assertAlmostEqual(first_row["confidence_score"], 0.91)
            self.assertEqual(first_row["confidence_label"], "high")
            self.assertEqual(first_row["supporting_view_count"], 1)
            self.assertEqual(first_row["distinct_supporting_view_count"], 1)
            self.assertEqual(first_row["total_candidate_count"], 2)
            self.assertEqual(first_row["weak_outcome"], "TP")

            self.assertEqual(result.case_summary.total_predictions, 5)
            self.assertEqual(result.case_summary.total_evaluated, 4)
            self.assertEqual(result.case_summary.true_positive, 1)
            self.assertEqual(result.case_summary.false_positive, 1)
            self.assertEqual(result.case_summary.false_negative, 1)
            self.assertEqual(result.case_summary.true_negative, 1)
            self.assertEqual(result.case_summary.skipped_missing_label, 1)
            self.assertEqual(result.case_summary.labels_without_prediction, 1)

            self.assertIn("moderate", result.breakdowns["by_weak_label_severity"])
            self.assertIn("high", result.breakdowns["by_weak_label_confidence"])
            self.assertIn("medium", result.breakdowns["by_pipeline_confidence_label"])

            summary_payload = json.loads((output_root / "multiview_case_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary_payload["summary"]["TP"], 1)
            self.assertIn("by_pipeline_confidence_label", summary_payload["breakdowns"])

            self.assertEqual(len(self._read_csv(output_root / "true_positive_cases.csv")), 1)
            self.assertEqual(len(self._read_csv(output_root / "false_positive_cases.csv")), 1)
            self.assertEqual(len(self._read_csv(output_root / "false_negative_cases.csv")), 1)
            self.assertEqual(len(self._read_csv(output_root / "true_negative_cases.csv")), 1)
            sweep_rows = self._read_csv(output_root / "threshold_sweep_multiview.csv")
            self.assertEqual(len(sweep_rows), 42)
            sweep_summary = json.loads((output_root / "threshold_sweep_summary.json").read_text(encoding="utf-8"))
            self.assertIn("confidence_score", sweep_summary["operating_points"])
            self.assertIn("total_score", sweep_summary["operating_points"])

    def test_multiview_threshold_options_filter_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            results_root = temp_root / "case_results"
            weak_labels_path = temp_root / "weak_labels.jsonl"
            self._write_multiview_result(
                results_root / "case_001" / "case_multiview_fusion.json",
                case_id="case_001",
                final_case_lesion={
                    "severity": "moderate",
                    "degrees": {"median": 0.60, "max": 0.70},
                    "distinct_supporting_view_ids": [],
                },
                confidence={"score": 0.70, "label": "medium"},
                supporting_views=[],
                total_candidate_count=1,
            )
            self._write_weak_labels(
                weak_labels_path,
                [{"case_id": "case_001", "stenosis_exists": "yes", "severity": "moderate", "confidence": "high"}],
            )

            result = run_multiview_level_benchmark(
                results_root=results_root,
                weak_labels_path=weak_labels_path,
                output_root=temp_root / "benchmarks",
                multiview_min_confidence_score=0.80,
                multiview_min_degree=0.50,
                require_distinct_supporting_view=True,
            )

            self.assertFalse(result.case_rows[0]["predicted_positive"])
            self.assertEqual(result.case_rows[0]["weak_outcome"], "FN")

    def test_multiview_benchmark_rejects_mismatched_json_case_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            results_root = temp_root / "case_results"
            weak_labels_path = temp_root / "weak_labels.jsonl"
            self._write_multiview_result(
                results_root / "case_001" / "case_multiview_fusion.json",
                case_id="different_case",
                final_case_lesion=None,
                confidence={"score": 0.0, "label": "low"},
                supporting_views=[],
                total_candidate_count=0,
            )
            self._write_weak_labels(
                weak_labels_path,
                [{"case_id": "case_001", "stenosis_exists": "no", "severity": "none", "confidence": "high"}],
            )

            with self.assertRaises(BenchmarkIOError):
                run_multiview_level_benchmark(
                    results_root=results_root,
                    weak_labels_path=weak_labels_path,
                    output_root=temp_root / "benchmarks",
                )

    def _write_multiview_result(
        self,
        path: Path,
        *,
        case_id: str,
        final_case_lesion: dict[str, object] | None,
        confidence: dict[str, object],
        supporting_views: list[str],
        total_candidate_count: int,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "case_id": case_id,
            "final_case_lesion": final_case_lesion,
            "confidence": confidence,
            "supporting_views": supporting_views,
            "fusion_metadata": {"total_candidate_count": total_candidate_count},
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _write_weak_labels(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "column_U": "",
                        "excel_row": 1,
                        "max_percent": None,
                        "evidence": "",
                        "raw_model_output": {},
                        **row,
                    }
                )
                for row in rows
            ),
            encoding="utf-8",
        )

    def _read_csv(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()

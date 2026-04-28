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

from stenosis_detection.benchmarking import run_temporal_level_benchmark


class TemporalBenchmarkingTests(unittest.TestCase):
    def test_temporal_benchmark_writes_sequence_and_case_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            results_root = temp_root / "temporal_results"
            output_root = temp_root / "benchmarks"
            weak_labels_path = temp_root / "weak_labels.jsonl"

            self._write_temporal_result(
                results_root / "case_001" / "view_01" / "view_temporal_fusion.json",
                view_id="view_01",
                final_lesion={
                    "severity": "moderate",
                    "persistence_ratio": 0.75,
                    "supporting_frame_count": 9,
                    "total_frame_count": 12,
                    "degrees": {"median": 0.65, "max": 0.82},
                },
                persistent_lesion_count=2,
            )
            self._write_temporal_result(
                results_root / "case_001" / "view_02" / "view_temporal_fusion.json",
                view_id="view_02",
                final_lesion=None,
                persistent_lesion_count=0,
            )
            self._write_temporal_result(
                results_root / "case_002" / "view_a" / "view_temporal_fusion.json",
                view_id="view_a",
                final_lesion={
                    "severity": "mild",
                    "persistence_ratio": 0.50,
                    "supporting_frame_count": 6,
                    "total_frame_count": 12,
                    "degrees": {"median": 0.55, "max": 0.70},
                },
                persistent_lesion_count=1,
            )
            self._write_temporal_result(
                results_root / "case_003" / "view_z" / "view_temporal_fusion.json",
                view_id="view_z",
                final_lesion=None,
                persistent_lesion_count=0,
            )
            self._write_weak_labels(
                weak_labels_path,
                [
                    {"case_id": "case_001", "stenosis_exists": "yes", "severity": "moderate", "confidence": "high"},
                    {"case_id": "case_002", "stenosis_exists": "no", "severity": "none", "confidence": "high"},
                    {"case_id": "case_004", "stenosis_exists": "yes", "severity": "mild", "confidence": "low"},
                ],
            )

            result = run_temporal_level_benchmark(
                results_root=results_root,
                weak_labels_path=weak_labels_path,
                output_root=output_root,
                write_threshold_sweep_report=True,
            )

            self.assertTrue((output_root / "temporal_sequence_rows.csv").is_file())
            self.assertTrue((output_root / "temporal_sequence_rows.jsonl").is_file())
            self.assertTrue((output_root / "temporal_sequence_summary.json").is_file())
            self.assertTrue((output_root / "temporal_sequence_summary.csv").is_file())
            self.assertTrue((output_root / "case_from_temporal_rows.csv").is_file())
            self.assertTrue((output_root / "case_from_temporal_summary.json").is_file())
            self.assertTrue((output_root / "threshold_sweep_temporal.csv").is_file())
            self.assertTrue((output_root / "threshold_sweep_summary.json").is_file())

            self.assertEqual(len(result.sequence_rows), 4)
            first_row = result.sequence_rows[0]
            self.assertEqual(first_row["case_id"], "case_001")
            self.assertEqual(first_row["sequence_id"], "view_01")
            self.assertTrue(first_row["predicted_positive"])
            self.assertEqual(first_row["persistent_lesion_count"], 2)
            self.assertAlmostEqual(first_row["score"], 0.65)
            self.assertAlmostEqual(first_row["max_degree"], 0.82)
            self.assertAlmostEqual(first_row["persistence_ratio"], 0.75)
            self.assertEqual(first_row["supporting_frame_count"], 9)
            self.assertEqual(first_row["total_frame_count"], 12)
            self.assertEqual(first_row["weak_outcome"], "TP")

            self.assertEqual(result.sequence_summary.total_predictions, 4)
            self.assertEqual(result.sequence_summary.total_evaluated, 3)
            self.assertEqual(result.sequence_summary.true_positive, 1)
            self.assertEqual(result.sequence_summary.false_positive, 1)
            self.assertEqual(result.sequence_summary.false_negative, 1)
            self.assertEqual(result.sequence_summary.skipped_missing_label, 1)
            self.assertEqual(result.sequence_summary.labels_without_prediction, 1)

            case_row = self._row_by_key(result.case_rows, "case_id", "case_001")
            self.assertEqual(case_row["total_sequences"], 2)
            self.assertEqual(case_row["positive_sequences"], 1)
            self.assertAlmostEqual(case_row["positive_sequence_ratio"], 0.5)
            self.assertAlmostEqual(case_row["max_temporal_score"], 0.65)
            self.assertAlmostEqual(case_row["max_temporal_degree"], 0.82)
            self.assertEqual(case_row["best_sequence_id"], "view_01")
            self.assertTrue(case_row["predicted_positive_any_sequence"])
            self.assertTrue(case_row["predicted_positive_ratio_threshold"])
            self.assertEqual(case_row["weak_outcome_any_sequence"], "TP")
            self.assertEqual(case_row["weak_outcome_ratio_threshold"], "TP")

            saved_sequence_rows = self._read_csv(output_root / "temporal_sequence_rows.csv")
            self.assertEqual(len(saved_sequence_rows), 4)
            self.assertEqual(saved_sequence_rows[0]["case_id"], "case_001")
            self.assertEqual(saved_sequence_rows[0]["sequence_id"], "view_01")

            summary_payload = json.loads((output_root / "temporal_sequence_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary_payload["summary"]["TP"], 1)
            self.assertIn("weak EHR labels", summary_payload["summary"]["note"])
            self.assertEqual(summary_payload["severity_agreement"]["comparable_positive_prediction_count"], 1)
            self.assertEqual(summary_payload["severity_agreement"]["exact_match_count"], 1)
            self.assertEqual(summary_payload["severity_agreement"]["pipeline_severity_distribution"]["moderate"]["percent"], 100.0)
            sweep_rows = self._read_csv(output_root / "threshold_sweep_temporal.csv")
            self.assertEqual(len(sweep_rows), 42)
            sweep_summary = json.loads((output_root / "threshold_sweep_summary.json").read_text(encoding="utf-8"))
            self.assertIn("temporal_score", sweep_summary["operating_points"])
            self.assertIn("persistence_ratio", sweep_summary["operating_points"])

    def test_temporal_thresholds_filter_final_lesion_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            results_root = temp_root / "temporal_results"
            weak_labels_path = temp_root / "weak_labels.jsonl"
            self._write_temporal_result(
                results_root / "case_001" / "view_01" / "view_temporal_fusion.json",
                view_id="view_01",
                final_lesion={
                    "severity": "mild",
                    "persistence_ratio": 0.20,
                    "supporting_frame_count": 2,
                    "total_frame_count": 12,
                    "degrees": {"median": 0.30, "max": 0.45},
                },
                persistent_lesion_count=1,
            )
            self._write_weak_labels(
                weak_labels_path,
                [{"case_id": "case_001", "stenosis_exists": "yes", "severity": "mild", "confidence": "medium"}],
            )

            result = run_temporal_level_benchmark(
                results_root=results_root,
                weak_labels_path=weak_labels_path,
                output_root=temp_root / "benchmarks",
                temporal_min_degree=0.50,
                temporal_min_persistence_ratio=0.25,
            )

            self.assertFalse(result.sequence_rows[0]["predicted_positive"])
            self.assertEqual(result.sequence_rows[0]["weak_outcome"], "FN")
            self.assertEqual(result.case_rows[0]["positive_sequences"], 0)

    def _write_temporal_result(
        self,
        path: Path,
        *,
        view_id: str,
        final_lesion: dict[str, object] | None,
        persistent_lesion_count: int,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "view_id": view_id,
            "fusion": {"persistent_lesion_count": persistent_lesion_count},
            "final_lesion": final_lesion,
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

    def _row_by_key(self, rows: list[dict[str, object]], key: str, value: object) -> dict[str, object]:
        for row in rows:
            if row[key] == value:
                return row
        self.fail(f"Missing row with {key}={value!r}")

    def _read_csv(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()

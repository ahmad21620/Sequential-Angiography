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

from stenosis_detection.benchmarking import run_frame_level_benchmark


class FrameBenchmarkingTests(unittest.TestCase):
    def test_frame_benchmark_writes_frame_sequence_and_case_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            results_root = temp_root / "frame_results"
            output_root = temp_root / "benchmarks"
            weak_labels_path = temp_root / "weak_labels.jsonl"

            self._write_frame_result(
                results_root / "case_001" / "view_01" / "slice_0001_stenosis_results.json",
                frame_index=1,
                stenosis_points=[
                    {"x": 10, "y": 20, "degree": 0.40, "severity": "mild"},
                    {"x": 11, "y": 21, "degree": 0.70, "severity": "moderate"},
                ],
            )
            self._write_frame_result(
                results_root / "case_001" / "view_01" / "slice_0002_stenosis_results.json",
                frame_index=2,
                stenosis_points=[],
            )
            self._write_frame_result(
                results_root / "case_002" / "view_a" / "slice_0001_stenosis_results.json",
                frame_index=1,
                stenosis_points=[{"x": 1, "y": 2, "degree": 0.90, "severity": "severe"}],
            )
            self._write_frame_result(
                results_root / "case_003" / "view_z" / "slice_0001_stenosis_results.json",
                frame_index=1,
                stenosis_points=[],
            )
            self._write_weak_labels(
                weak_labels_path,
                [
                    {"case_id": "case_001", "stenosis_exists": "yes", "severity": "moderate", "confidence": "high"},
                    {"case_id": "case_002", "stenosis_exists": "no", "severity": "none", "confidence": "high"},
                    {"case_id": "case_004", "stenosis_exists": "yes", "severity": "mild", "confidence": "low"},
                ],
            )

            result = run_frame_level_benchmark(
                results_root=results_root,
                weak_labels_path=weak_labels_path,
                output_root=output_root,
                write_threshold_sweep_report=True,
            )

            self.assertTrue((output_root / "frame_rows.csv").is_file())
            self.assertTrue((output_root / "frame_rows.jsonl").is_file())
            self.assertTrue((output_root / "frame_summary.json").is_file())
            self.assertTrue((output_root / "frame_summary.csv").is_file())
            self.assertTrue((output_root / "sequence_from_frame_rows.csv").is_file())
            self.assertTrue((output_root / "sequence_from_frame_summary.json").is_file())
            self.assertTrue((output_root / "case_from_frame_rows.csv").is_file())
            self.assertTrue((output_root / "case_from_frame_summary.json").is_file())
            self.assertTrue((output_root / "threshold_sweep_frame.csv").is_file())
            self.assertTrue((output_root / "threshold_sweep_summary.json").is_file())

            self.assertEqual(len(result.frame_rows), 4)
            first_row = result.frame_rows[0]
            self.assertEqual(first_row["case_id"], "case_001")
            self.assertEqual(first_row["sequence_id"], "view_01")
            self.assertTrue(first_row["predicted_positive"])
            self.assertAlmostEqual(first_row["score"], 0.70)
            self.assertEqual(first_row["predicted_severity"], "moderate")
            self.assertEqual(first_row["weak_outcome"], "TP")

            self.assertEqual(result.frame_summary.total_predictions, 4)
            self.assertEqual(result.frame_summary.total_evaluated, 3)
            self.assertEqual(result.frame_summary.true_positive, 1)
            self.assertEqual(result.frame_summary.false_positive, 1)
            self.assertEqual(result.frame_summary.false_negative, 1)
            self.assertEqual(result.frame_summary.skipped_missing_label, 1)
            self.assertEqual(result.frame_summary.labels_without_prediction, 1)

            sequence_row = self._row_by_key(result.sequence_rows, "case_id", "case_001")
            self.assertEqual(sequence_row["sequence_id"], "view_01")
            self.assertEqual(sequence_row["total_frames"], 2)
            self.assertEqual(sequence_row["positive_frames"], 1)
            self.assertEqual(sequence_row["negative_frames"], 1)
            self.assertAlmostEqual(sequence_row["positive_frame_ratio"], 0.5)
            self.assertAlmostEqual(sequence_row["max_frame_degree"], 0.70)
            self.assertAlmostEqual(sequence_row["mean_positive_frame_degree"], 0.70)
            self.assertTrue(sequence_row["predicted_positive_any_frame"])
            self.assertTrue(sequence_row["predicted_positive_ratio_threshold"])
            self.assertEqual(sequence_row["weak_outcome_any_frame"], "TP")
            self.assertEqual(sequence_row["weak_outcome_ratio_threshold"], "TP")

            case_row = self._row_by_key(result.case_rows, "case_id", "case_001")
            self.assertEqual(case_row["total_frames"], 2)
            self.assertEqual(case_row["positive_frames"], 1)
            self.assertEqual(case_row["positive_sequences"], 1)
            self.assertEqual(case_row["total_sequences"], 1)
            self.assertTrue(case_row["predicted_positive_any_frame"])
            self.assertTrue(case_row["predicted_positive_ratio_threshold"])
            self.assertEqual(case_row["weak_outcome_any_frame"], "TP")

            saved_frame_rows = self._read_csv(output_root / "frame_rows.csv")
            self.assertEqual(len(saved_frame_rows), 4)
            self.assertEqual(saved_frame_rows[0]["case_id"], "case_001")
            self.assertEqual(saved_frame_rows[0]["sequence_id"], "view_01")

            frame_summary_payload = json.loads((output_root / "frame_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(frame_summary_payload["summary"]["TP"], 1)
            self.assertIn("weak EHR labels", frame_summary_payload["summary"]["note"])
            self.assertEqual(frame_summary_payload["severity_agreement"]["comparable_positive_prediction_count"], 1)
            self.assertEqual(frame_summary_payload["severity_agreement"]["exact_match_count"], 1)
            self.assertEqual(frame_summary_payload["severity_agreement"]["weak_label_severity_distribution"]["moderate"]["percent"], 100.0)
            sweep_rows = self._read_csv(output_root / "threshold_sweep_frame.csv")
            self.assertEqual(len(sweep_rows), 21)
            sweep_summary = json.loads((output_root / "threshold_sweep_summary.json").read_text(encoding="utf-8"))
            self.assertIn("frame_score", sweep_summary["operating_points"])

    def test_frame_min_degree_filters_low_degree_positive_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            results_root = temp_root / "frame_results"
            weak_labels_path = temp_root / "weak_labels.jsonl"
            self._write_frame_result(
                results_root / "case_001" / "view_01" / "slice_0001_stenosis_results.json",
                frame_index=1,
                stenosis_points=[{"x": 10, "y": 20, "degree": 0.20, "severity": "mild"}],
            )
            self._write_weak_labels(
                weak_labels_path,
                [{"case_id": "case_001", "stenosis_exists": "yes", "severity": "mild", "confidence": "medium"}],
            )

            result = run_frame_level_benchmark(
                results_root=results_root,
                weak_labels_path=weak_labels_path,
                output_root=temp_root / "benchmarks",
                frame_min_degree=0.50,
            )

            self.assertFalse(result.frame_rows[0]["predicted_positive"])
            self.assertEqual(result.frame_rows[0]["weak_outcome"], "FN")
            self.assertEqual(result.case_rows[0]["positive_frames"], 0)

    def _write_frame_result(self, path: Path, *, frame_index: int, stenosis_points: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        image_stem = path.name.removesuffix("_stenosis_results.json")
        payload = {
            "frame": {
                "image_name": f"{image_stem}.png",
                "image_stem": image_stem,
                "frame_index": frame_index,
                "view_id": "unused_for_benchmark_path_derivation",
            },
            "counts": {
                "stenosis_points": len(stenosis_points),
            },
            "stenosis_points": stenosis_points,
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

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

from stenosis_detection.cadica.report import CadicaReportError, run_cadica_report


class CadicaReportTests(unittest.TestCase):
    def test_report_generation_from_multiview_sweep_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            benchmark_root = root / "cadica_benchmark"
            output_root = root / "final_report"
            self._write_sweep_outputs(benchmark_root, include_multiview=True)

            artifacts = run_cadica_report(
                benchmark_root=benchmark_root,
                output_root=output_root,
                top_k=1,
            )

            self.assertTrue(artifacts.reports["markdown"].is_file())
            self.assertTrue(artifacts.reports["html"].is_file())
            self.assertTrue(artifacts.reports["summary_json"].is_file())
            self.assertTrue((output_root / "tables" / "cadica_threshold_sweep_metrics.csv").is_file())
            self.assertTrue((output_root / "tables" / "best_operating_points.csv").is_file())
            self.assertTrue((output_root / "tables" / "top_primary_operating_points.csv").is_file())
            self.assertTrue((output_root / "tables" / "pareto_primary_operating_points.csv").is_file())
            self.assertTrue((output_root / "tables" / "selected_multiview_patient_rows.csv").is_file())
            self.assertTrue((output_root / "plots" / "01_precision_recall_scatter.png").is_file())
            self.assertTrue((output_root / "plots" / "05_multiview_threshold_curve.png").is_file())

            report_text = artifacts.reports["markdown"].read_text(encoding="utf-8")
            summary = json.loads(artifacts.reports["summary_json"].read_text(encoding="utf-8"))
            top_rows = self._read_csv(output_root / "tables" / "top_primary_operating_points.csv")
            selected_rows = self._read_csv(output_root / "tables" / "selected_multiview_patient_rows.csv")

            self.assertIn("Final CADICA Benchmark Report", report_text)
            self.assertIn("does not use weak labels", report_text)
            self.assertIn("does not read `views.json`", report_text)
            self.assertEqual(summary["primary_metric_level"], "multiview_patient")
            self.assertEqual(top_rows[0]["multiview_min_score"], "0.5")
            self.assertEqual(selected_rows[0]["selected_multiview_min_score"], "0.5")
            self.assertEqual(selected_rows[0]["selected_outcome"], "FN")

    def test_report_falls_back_to_patient_metrics_without_multiview(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            benchmark_root = root / "cadica_benchmark"
            output_root = root / "final_report"
            self._write_sweep_outputs(benchmark_root, include_multiview=False)

            artifacts = run_cadica_report(
                benchmark_root=benchmark_root,
                output_root=output_root,
                top_k=2,
            )

            summary = json.loads(artifacts.reports["summary_json"].read_text(encoding="utf-8"))
            self.assertEqual(summary["primary_metric_level"], "patient")
            self.assertFalse(summary["overview"]["multiview_evaluated"])
            self.assertIn("Multi-view metrics were not found", artifacts.warnings[0])
            self.assertFalse((output_root / "tables" / "selected_multiview_patient_rows.csv").exists())

    def test_report_requires_threshold_sweep_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark_root = Path(temp_dir) / "cadica_benchmark"
            benchmark_root.mkdir()

            with self.assertRaises(FileNotFoundError):
                run_cadica_report(benchmark_root=benchmark_root)

    def test_report_rejects_empty_threshold_sweep_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark_root = Path(temp_dir) / "cadica_benchmark"
            benchmark_root.mkdir()
            (benchmark_root / "cadica_threshold_sweep.csv").write_text("frame_min_degree\n", encoding="utf-8")

            with self.assertRaises(CadicaReportError):
                run_cadica_report(benchmark_root=benchmark_root)

    def _write_sweep_outputs(self, benchmark_root: Path, *, include_multiview: bool) -> None:
        benchmark_root.mkdir(parents=True, exist_ok=True)
        rows = [
            self._sweep_row(
                frame_min_degree="0.0",
                box_margin_px="0",
                video_prediction_source="frame_any",
                multiview_min_score="0.0" if include_multiview else "",
                frame=(1, 1, 1, 0),
                video=(1, 1, 1, 0),
                patient=(1, 1, 1, 0),
                multiview_patient=(1, 1, 1, 0) if include_multiview else None,
                multiview_side=(1, 0, 1, 1) if include_multiview else None,
            ),
            self._sweep_row(
                frame_min_degree="0.5",
                box_margin_px="5",
                video_prediction_source="frame_any",
                multiview_min_score="0.5" if include_multiview else "",
                frame=(1, 0, 2, 0),
                video=(1, 0, 2, 0),
                patient=(1, 0, 2, 0),
                multiview_patient=(1, 0, 2, 0) if include_multiview else None,
                multiview_side=(1, 0, 2, 0) if include_multiview else None,
            ),
        ]
        self._write_csv(benchmark_root / "cadica_threshold_sweep.csv", rows)
        summary = {
            "config": {
                "multiview_evaluated": include_multiview,
                "note": "Supervised CADICA threshold sweep.",
            },
            "evaluated_sweep_combinations": len(rows),
        }
        (benchmark_root / "cadica_threshold_sweep_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (benchmark_root / "cadica_summary.json").write_text(
            json.dumps({"config": {"multiview_evaluated": include_multiview}}),
            encoding="utf-8",
        )
        if include_multiview:
            self._write_csv(
                benchmark_root / "cadica_multiview_patient_rows.csv",
                [
                    {
                        "patient_id": "p1",
                        "label_positive": "true",
                        "predicted_positive": "true",
                        "score": "0.4",
                        "outcome": "TP",
                        "lesion_video_count": "1",
                        "positive_frame_count": "1",
                        "multiview_json_path": "multiview/p1/case_multiview_fusion.json",
                        "skip_reason": "",
                    },
                    {
                        "patient_id": "p2",
                        "label_positive": "false",
                        "predicted_positive": "false",
                        "score": "0.0",
                        "outcome": "TN",
                        "lesion_video_count": "0",
                        "positive_frame_count": "0",
                        "multiview_json_path": "multiview/p2/case_multiview_fusion.json",
                        "skip_reason": "",
                    },
                ],
            )

    def _sweep_row(
        self,
        *,
        frame_min_degree: str,
        box_margin_px: str,
        video_prediction_source: str,
        multiview_min_score: str,
        frame: tuple[int, int, int, int],
        video: tuple[int, int, int, int],
        patient: tuple[int, int, int, int],
        multiview_patient: tuple[int, int, int, int] | None,
        multiview_side: tuple[int, int, int, int] | None,
    ) -> dict[str, object]:
        row: dict[str, object] = {
            "frame_min_degree": frame_min_degree,
            "box_margin_px": box_margin_px,
            "video_prediction_source": video_prediction_source,
            "multiview_min_score": multiview_min_score,
            "positive_evaluated_frames": 1,
            "positive_frames_with_localized_prediction": 1,
            "localization_recall_on_positive_frames": 1.0,
            "localized_predicted_positive_frames": 1,
            "unmatched_predicted_points": 0,
            "total_gt_boxes": 1,
            "matched_gt_boxes": 1,
            "missed_gt_boxes": 0,
            "box_recall": 1.0,
        }
        row.update(self._metric_columns("frame", frame, extra={"predicted_positive_count": frame[0] + frame[1]}))
        row.update(self._metric_columns("video", video))
        row.update(self._metric_columns("patient", patient))
        row.update(self._metric_columns("multiview_patient", multiview_patient))
        row.update(self._metric_columns("multiview_side", multiview_side, include_accuracy=False, include_rates=False))
        return row

    def _metric_columns(
        self,
        prefix: str,
        counts: tuple[int, int, int, int] | None,
        *,
        extra: dict[str, object] | None = None,
        include_accuracy: bool = True,
        include_rates: bool = True,
    ) -> dict[str, object]:
        tp, fp, tn, fn = counts if counts is not None else (None, None, None, None)
        metrics = self._metrics(tp, fp, tn, fn) if counts is not None else {}
        columns: dict[str, object] = {
            f"{prefix}_total_evaluated": "" if counts is None else sum(counts),
            f"{prefix}_TP": "" if tp is None else tp,
            f"{prefix}_FP": "" if fp is None else fp,
            f"{prefix}_TN": "" if tn is None else tn,
            f"{prefix}_FN": "" if fn is None else fn,
            f"{prefix}_precision": metrics.get("precision", ""),
            f"{prefix}_recall": metrics.get("recall", ""),
            f"{prefix}_specificity": metrics.get("specificity", ""),
            f"{prefix}_F1": metrics.get("F1", ""),
            f"{prefix}_balanced_accuracy": metrics.get("balanced_accuracy", ""),
        }
        if include_accuracy:
            columns[f"{prefix}_accuracy"] = metrics.get("accuracy", "")
        if include_rates:
            columns[f"{prefix}_false_positive_rate"] = metrics.get("false_positive_rate", "")
            columns[f"{prefix}_false_negative_rate"] = metrics.get("false_negative_rate", "")
        if extra:
            for key, value in extra.items():
                columns[f"{prefix}_{key}"] = value
        return columns

    def _metrics(self, tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "F1": f1,
            "balanced_accuracy": (recall + specificity) / 2,
            "false_positive_rate": 1 - specificity,
            "false_negative_rate": 1 - recall,
        }

    def _write_csv(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames: list[str] = []
        for row in rows:
            for field in row:
                if field not in fieldnames:
                    fieldnames.append(field)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _read_csv(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()

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

from stenosis_detection.cadica.benchmark import run_cadica_benchmark
from stenosis_detection.cadica.sweep import parse_float_list, run_cadica_threshold_sweep


class CadicaThresholdSweepTests(unittest.TestCase):
    def test_sweep_writes_csv_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            manifest_path, frame_results_root = self._write_synthetic_dataset(temp_root)
            output_root = temp_root / "sweep"

            outputs = run_cadica_threshold_sweep(
                manifest=manifest_path,
                frame_results_root=frame_results_root,
                output_root=output_root,
                frame_min_degrees=[0.0, 0.5],
                box_margins_px=[0.0, 5.0],
            )

            self.assertTrue(outputs["cadica_threshold_sweep_csv"].is_file())
            self.assertTrue(outputs["cadica_threshold_sweep_summary_json"].is_file())
            rows = self._read_csv(output_root / "cadica_threshold_sweep.csv")
            summary = json.loads((output_root / "cadica_threshold_sweep_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 4)
            self.assertEqual(summary["evaluated_sweep_combinations"], 4)

    def test_thresholds_and_box_margins_change_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            manifest_path, frame_results_root = self._write_synthetic_dataset(temp_root)
            output_root = temp_root / "sweep"

            run_cadica_threshold_sweep(
                manifest=manifest_path,
                frame_results_root=frame_results_root,
                output_root=output_root,
                frame_min_degrees=[0.0, 0.5],
                box_margins_px=[0.0, 5.0],
            )

            rows = {
                (float(row["frame_min_degree"]), float(row["box_margin_px"])): row
                for row in self._read_csv(output_root / "cadica_threshold_sweep.csv")
            }
            low_threshold_no_margin = rows[(0.0, 0.0)]
            low_threshold_margin = rows[(0.0, 5.0)]
            high_threshold_margin = rows[(0.5, 5.0)]

            self.assertEqual(int(low_threshold_no_margin["frame_FP"]), 1)
            self.assertEqual(int(low_threshold_no_margin["frame_FN"]), 0)
            self.assertEqual(int(high_threshold_margin["frame_FP"]), 0)
            self.assertEqual(int(high_threshold_margin["frame_FN"]), 1)
            self.assertEqual(float(low_threshold_no_margin["box_recall"]), 0.0)
            self.assertEqual(float(low_threshold_margin["box_recall"]), 1.0)

    def test_unknown_frame_labels_are_excluded_from_frame_binary_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            manifest_path, frame_results_root = self._write_synthetic_dataset(temp_root)
            output_root = temp_root / "sweep"

            run_cadica_threshold_sweep(
                manifest=manifest_path,
                frame_results_root=frame_results_root,
                output_root=output_root,
                frame_min_degrees=[0.0],
                box_margins_px=[0.0],
            )

            row = self._read_csv(output_root / "cadica_threshold_sweep.csv")[0]
            self.assertEqual(int(row["frame_total_evaluated"]), 2)
            self.assertEqual(int(row["frame_predicted_positive_count"]), 2)

    def test_temporal_final_source_requires_temporal_results_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            manifest_path, frame_results_root = self._write_synthetic_dataset(temp_root)

            with self.assertRaisesRegex(ValueError, "temporal_final requires temporal_results_root"):
                run_cadica_threshold_sweep(
                    manifest=manifest_path,
                    frame_results_root=frame_results_root,
                    output_root=temp_root / "sweep",
                    video_prediction_sources=["temporal_final"],
                )

    def test_summary_selects_best_frame_f1(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            manifest_path, frame_results_root = self._write_synthetic_dataset(temp_root)
            output_root = temp_root / "sweep"

            run_cadica_threshold_sweep(
                manifest=manifest_path,
                frame_results_root=frame_results_root,
                output_root=output_root,
                frame_min_degrees=[0.0, 0.5],
                box_margins_px=[5.0],
            )

            summary = json.loads((output_root / "cadica_threshold_sweep_summary.json").read_text(encoding="utf-8"))
            best_frame_f1 = summary["best_operating_points"]["best_frame_F1"]
            self.assertEqual(float(best_frame_f1["frame_min_degree"]), 0.0)
            self.assertAlmostEqual(float(best_frame_f1["frame_F1"]), 2 / 3)

    def test_normal_cadica_benchmark_still_works(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            manifest_path, frame_results_root = self._write_synthetic_dataset(temp_root)
            output_root = temp_root / "benchmark"

            result = run_cadica_benchmark(
                manifest=manifest_path,
                frame_results_root=frame_results_root,
                output_root=output_root,
                frame_min_degree=0.0,
                box_margin_px=5.0,
            )

            self.assertTrue((output_root / "cadica_summary.json").is_file())
            self.assertEqual(result.summary["frame_binary_metrics"]["total_evaluated"], 2)
            self.assertEqual(result.summary["frame_binary_metrics"]["TP"], 1)
            self.assertEqual(result.summary["frame_binary_metrics"]["FP"], 1)

    def test_parse_float_list(self) -> None:
        self.assertEqual(parse_float_list("0,0.05,0.1"), [0.0, 0.05, 0.1])
        with self.assertRaisesRegex(ValueError, "must be >= 0"):
            parse_float_list("0,-0.1")

    def _write_synthetic_dataset(self, root: Path) -> tuple[Path, Path]:
        manifest_path = root / "manifest.csv"
        frame_results_root = root / "frame_results"
        rows = [
            self._manifest_row(
                "p1",
                "v1",
                1,
                "lesion",
                "positive",
                [{"x": 10, "y": 10, "w": 10, "h": 10, "category": "stenosis", "source_path": "gt/1.txt"}],
            ),
            self._manifest_row("p1", "v2", 1, "nonlesion", "negative", []),
            self._manifest_row("p1", "v1", 2, "lesion", "unknown", []),
        ]
        self._write_manifest_rows(manifest_path, rows)
        self._write_frame_result(
            frame_results_root / "p1" / "v1" / "slice_00001_stenosis_results.json",
            patient_id="p1",
            video_id="v1",
            frame_id=1,
            points=[{"x": 22, "y": 15, "degree": 0.30, "severity": "moderate"}],
        )
        self._write_frame_result(
            frame_results_root / "p1" / "v2" / "slice_00001_stenosis_results.json",
            patient_id="p1",
            video_id="v2",
            frame_id=1,
            points=[{"x": 100, "y": 100, "degree": 0.10, "severity": "mild"}],
        )
        self._write_frame_result(
            frame_results_root / "p1" / "v1" / "slice_00002_stenosis_results.json",
            patient_id="p1",
            video_id="v1",
            frame_id=2,
            points=[{"x": 50, "y": 50, "degree": 0.90, "severity": "severe"}],
        )
        return manifest_path, frame_results_root

    def _write_manifest_rows(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _manifest_row(
        self,
        patient_id: str,
        video_id: str,
        frame_id: int,
        video_label: str,
        frame_label: str,
        boxes: list[dict[str, object]],
    ) -> dict[str, object]:
        return {
            "patient_id": patient_id,
            "video_id": video_id,
            "frame_id": frame_id,
            "original_image_path": "",
            "original_image_name": f"{patient_id}_{video_id}_{frame_id:05d}.png",
            "prepared_relative_path": f"keyframes/{patient_id}/{video_id}/slice_{frame_id:05d}.png",
            "prepared_image_name": f"slice_{frame_id:05d}.png",
            "is_selected_frame": "true",
            "video_label": video_label,
            "frame_label": frame_label,
            "box_count": len(boxes),
            "gt_box_source_paths": json.dumps([box["source_path"] for box in boxes]),
            "gt_boxes_json": json.dumps(boxes),
        }

    def _write_frame_result(
        self,
        path: Path,
        *,
        patient_id: str,
        video_id: str,
        frame_id: int,
        points: list[dict[str, object]],
    ) -> None:
        image_name = f"slice_{frame_id:05d}.png"
        payload = {
            "frame": {
                "image_name": image_name,
                "image_stem": image_name.removesuffix(".png"),
                "frame_index": frame_id,
                "view_id": f"{patient_id}/{video_id}",
                "width": 512,
                "height": 512,
            },
            "stenosis_points": points,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _read_csv(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()

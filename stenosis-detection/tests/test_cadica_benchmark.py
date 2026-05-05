from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stenosis_detection.cadica.benchmark import run_cadica_benchmark


class CadicaBenchmarkTests(unittest.TestCase):
    def test_supervised_benchmark_outputs_frame_localization_and_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            manifest_path = temp_root / "manifest.csv"
            frame_results_root = temp_root / "frame_results"
            output_root = temp_root / "benchmark"
            self._write_manifest(manifest_path)
            self._write_frame_result(
                frame_results_root / "p1" / "v1" / "slice_00001_stenosis_results.json",
                patient_id="p1",
                video_id="v1",
                frame_id=1,
                points=[{"x": 15, "y": 15, "degree": 0.70, "severity": "moderate"}],
            )
            self._write_frame_result(
                frame_results_root / "p1" / "v1" / "slice_00002_stenosis_results.json",
                patient_id="p1",
                video_id="v1",
                frame_id=2,
                points=[{"x": 200, "y": 200, "degree": 0.80, "severity": "severe"}],
            )
            self._write_frame_result(
                frame_results_root / "p1" / "v1" / "slice_00003_stenosis_results.json",
                patient_id="p1",
                video_id="v1",
                frame_id=3,
                points=[],
            )
            self._write_frame_result(
                frame_results_root / "p1" / "v2" / "slice_00001_stenosis_results.json",
                patient_id="p1",
                video_id="v2",
                frame_id=1,
                points=[{"x": 50, "y": 50, "degree": 0.60, "severity": "moderate"}],
            )
            self._write_frame_result(
                frame_results_root / "p1" / "v2" / "slice_00002_stenosis_results.json",
                patient_id="p1",
                video_id="v2",
                frame_id=2,
                points=[],
            )
            self._write_frame_result(
                frame_results_root / "p2" / "v1" / "slice_00001_stenosis_results.json",
                patient_id="p2",
                video_id="v1",
                frame_id=1,
                points=[],
            )

            result = run_cadica_benchmark(
                manifest=manifest_path,
                frame_results_root=frame_results_root,
                output_root=output_root,
                frame_min_degree=0.5,
                box_margin_px=5,
            )

            self.assertTrue((output_root / "cadica_frame_rows.csv").is_file())
            self.assertTrue((output_root / "cadica_frame_rows.jsonl").is_file())
            self.assertTrue((output_root / "cadica_box_rows.csv").is_file())
            self.assertTrue((output_root / "cadica_video_rows.csv").is_file())
            self.assertTrue((output_root / "cadica_patient_rows.csv").is_file())
            self.assertTrue((output_root / "cadica_summary.json").is_file())

            frame_by_key = {
                (row["patient_id"], row["video_id"], int(row["frame_id"])): row
                for row in result.frame_rows
            }
            self.assertEqual(frame_by_key[("p1", "v1", 1)]["outcome"], "TP")
            self.assertTrue(frame_by_key[("p1", "v1", 1)]["localized_positive"])
            self.assertEqual(frame_by_key[("p1", "v1", 1)]["matched_gt_box_count"], 1)
            self.assertEqual(frame_by_key[("p1", "v1", 2)]["outcome"], "TP")
            self.assertFalse(frame_by_key[("p1", "v1", 2)]["localized_positive"])
            self.assertEqual(frame_by_key[("p1", "v1", 2)]["unmatched_gt_box_count"], 1)
            self.assertEqual(frame_by_key[("p1", "v2", 1)]["outcome"], "FP")
            self.assertEqual(frame_by_key[("p1", "v2", 2)]["outcome"], "TN")
            self.assertEqual(frame_by_key[("p1", "v1", 3)]["outcome"], "")
            self.assertEqual(frame_by_key[("p1", "v1", 3)]["skip_reason"], "unknown_frame_label")

            summary = result.summary
            self.assertIn("supervised CADICA frame/video benchmark", summary["note"])
            self.assertEqual(summary["frame_binary_metrics"]["total_evaluated"], 5)
            self.assertEqual(summary["frame_binary_metrics"]["TP"], 2)
            self.assertEqual(summary["frame_binary_metrics"]["FP"], 1)
            self.assertEqual(summary["frame_binary_metrics"]["TN"], 2)
            self.assertEqual(summary["frame_binary_metrics"]["FN"], 0)
            self.assertAlmostEqual(summary["frame_binary_metrics"]["precision"] or 0.0, 2 / 3)
            self.assertAlmostEqual(summary["frame_localization_metrics"]["localization_recall_on_positive_frames"] or 0.0, 0.5)
            self.assertEqual(summary["box_detection_metrics"]["total_gt_boxes"], 2)
            self.assertEqual(summary["box_detection_metrics"]["matched_gt_boxes"], 1)
            self.assertAlmostEqual(summary["box_detection_metrics"]["box_recall"] or 0.0, 0.5)
            self.assertEqual(summary["counts"]["frame_skip_reason_counts"], {"none": 5, "unknown_frame_label": 1})

            box_by_frame = {(row["patient_id"], row["video_id"], int(row["frame_id"])): row for row in result.box_rows}
            self.assertTrue(box_by_frame[("p1", "v1", 1)]["matched"])
            self.assertEqual(box_by_frame[("p1", "v1", 1)]["matched_point_x"], 15.0)
            self.assertFalse(box_by_frame[("p1", "v1", 2)]["matched"])

            video_by_key = {(row["patient_id"], row["video_id"]): row for row in result.video_rows}
            self.assertEqual(video_by_key[("p1", "v1")]["outcome"], "TP")
            self.assertEqual(video_by_key[("p1", "v2")]["outcome"], "FP")
            self.assertEqual(video_by_key[("p2", "v1")]["outcome"], "TN")
            self.assertEqual(video_by_key[("p1", "v1")]["positive_frame_count"], 2)
            self.assertEqual(video_by_key[("p1", "v1")]["predicted_positive_frame_count"], 2)

            patient_by_id = {row["patient_id"]: row for row in result.patient_rows}
            self.assertEqual(patient_by_id["p1"]["outcome"], "TP")
            self.assertEqual(patient_by_id["p2"]["outcome"], "TN")

            self.assertEqual(len(self._read_csv(output_root / "false_positive_frames.csv")), 1)
            self.assertEqual(len(self._read_csv(output_root / "false_negative_frames.csv")), 0)
            self.assertEqual(len(self._read_csv(output_root / "missed_gt_boxes.csv")), 1)
            self.assertEqual(len(self._read_csv(output_root / "unmatched_predicted_points.csv")), 2)

    def test_review_images_are_written_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            manifest_path = temp_root / "manifest.csv"
            frame_results_root = temp_root / "frame_results"
            output_root = temp_root / "benchmark"
            image_root = temp_root / "images"
            rows = [
                self._manifest_row(
                    "p1",
                    "v1",
                    1,
                    "lesion",
                    "positive",
                    [{"x": 10, "y": 10, "w": 20, "h": 20, "category": "stenosis", "source_path": "gt/1.txt"}],
                    image_root=image_root,
                ),
                self._manifest_row(
                    "p1",
                    "v1",
                    2,
                    "lesion",
                    "positive",
                    [{"x": 10, "y": 10, "w": 20, "h": 20, "category": "stenosis", "source_path": "gt/2.txt"}],
                    image_root=image_root,
                ),
                self._manifest_row("p1", "v2", 1, "nonlesion", "negative", [], image_root=image_root),
                self._manifest_row(
                    "p1",
                    "v3",
                    1,
                    "lesion",
                    "positive",
                    [{"x": 10, "y": 10, "w": 20, "h": 20, "category": "stenosis", "source_path": "gt/3.txt"}],
                    image_root=image_root,
                ),
            ]
            self._write_manifest_rows(manifest_path, rows)
            for row in rows:
                self._write_test_image(Path(str(row["original_image_path"])))

            self._write_frame_result(
                frame_results_root / "p1" / "v1" / "slice_00001_stenosis_results.json",
                patient_id="p1",
                video_id="v1",
                frame_id=1,
                points=[{"x": 15, "y": 15, "degree": 0.70, "severity": "moderate"}],
            )
            self._write_frame_result(
                frame_results_root / "p1" / "v1" / "slice_00002_stenosis_results.json",
                patient_id="p1",
                video_id="v1",
                frame_id=2,
                points=[{"x": 50, "y": 50, "degree": 0.80, "severity": "severe"}],
            )
            self._write_frame_result(
                frame_results_root / "p1" / "v2" / "slice_00001_stenosis_results.json",
                patient_id="p1",
                video_id="v2",
                frame_id=1,
                points=[{"x": 30, "y": 30, "degree": 0.60, "severity": "moderate"}],
            )
            self._write_frame_result(
                frame_results_root / "p1" / "v3" / "slice_00001_stenosis_results.json",
                patient_id="p1",
                video_id="v3",
                frame_id=1,
                points=[],
            )

            result = run_cadica_benchmark(
                manifest=manifest_path,
                frame_results_root=frame_results_root,
                output_root=output_root,
                write_review_images=True,
                max_review_images=10,
            )

            review_root = output_root / "review_images"
            self.assertEqual(result.outputs.review_image_root, review_root)
            self.assertEqual(len(list((review_root / "false_positive").glob("*.png"))), 1)
            self.assertEqual(len(list((review_root / "false_negative").glob("*.png"))), 1)
            self.assertEqual(len(list((review_root / "true_positive_localized").glob("*.png"))), 1)
            self.assertEqual(len(list((review_root / "true_positive_nonlocalized").glob("*.png"))), 1)

    def _write_manifest(self, path: Path) -> None:
        rows = [
            self._manifest_row("p1", "v1", 1, "lesion", "positive", [{"x": 10, "y": 10, "w": 20, "h": 20, "category": "stenosis", "source_path": "gt/p1_v1_00001.txt"}]),
            self._manifest_row("p1", "v1", 2, "lesion", "positive", [{"x": 100, "y": 100, "w": 10, "h": 10, "category": "stenosis", "source_path": "gt/p1_v1_00002.txt"}]),
            self._manifest_row("p1", "v1", 3, "lesion", "unknown", []),
            self._manifest_row("p1", "v2", 1, "nonlesion", "negative", []),
            self._manifest_row("p1", "v2", 2, "nonlesion", "negative", []),
            self._manifest_row("p2", "v1", 1, "nonlesion", "negative", []),
        ]
        self._write_manifest_rows(path, rows)

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
        image_root: Path | None = None,
    ) -> dict[str, object]:
        original_image_path = (
            image_root / patient_id / video_id / f"{patient_id}_{video_id}_{frame_id:05d}.png"
            if image_root is not None
            else Path(f"/CADICA/selectedVideos/{patient_id}/{video_id}/input/{patient_id}_{video_id}_{frame_id:05d}.png")
        )
        return {
            "patient_id": patient_id,
            "video_id": video_id,
            "frame_id": frame_id,
            "original_image_path": str(original_image_path),
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
        image_stem = image_name.removesuffix(".png")
        payload = {
            "frame": {
                "image_name": image_name,
                "image_stem": image_stem,
                "frame_index": frame_id,
                "view_id": f"{patient_id}/{video_id}",
                "width": 512,
                "height": 512,
            },
            "stenosis_points": points,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _write_test_image(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.full((80, 80, 3), 36, dtype=np.uint8)
        cv2.line(image, (8, 40), (72, 40), color=(90, 90, 90), thickness=3)
        self.assertTrue(cv2.imwrite(str(path), image))

    def _read_csv(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()

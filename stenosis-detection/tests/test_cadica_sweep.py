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
from stenosis_detection.yolo.schema import YoloDetection, YoloDetectorMetadata, build_yolo_frame_payload


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
                workers=2,
            )

            self.assertTrue(outputs["cadica_threshold_sweep_csv"].is_file())
            self.assertTrue(outputs["cadica_threshold_sweep_summary_json"].is_file())
            rows = self._read_csv(output_root / "cadica_threshold_sweep.csv")
            summary = json.loads((output_root / "cadica_threshold_sweep_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 4)
            self.assertEqual(summary["evaluated_sweep_combinations"], 4)
            self.assertEqual(summary["config"]["workers"], 2)

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

    def test_threshold_sweep_accepts_yolo_frame_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            manifest_path = temp_root / "manifest.csv"
            frame_results_root = temp_root / "yolo_frame_results"
            output_root = temp_root / "sweep"
            self._write_manifest_rows(
                manifest_path,
                [
                    self._manifest_row(
                        "p1",
                        "v1",
                        1,
                        "lesion",
                        "positive",
                        [{"x": 10, "y": 10, "w": 20, "h": 20, "category": "stenosis", "source_path": "gt/1.txt"}],
                    )
                ],
            )
            self._write_yolo_frame_result(
                frame_results_root / "p1" / "v1" / "slice_00001_stenosis_results.json",
                patient_id="p1",
                video_id="v1",
                frame_id=1,
                confidence=0.83,
            )

            run_cadica_threshold_sweep(
                manifest=manifest_path,
                frame_results_root=frame_results_root,
                output_root=output_root,
                frame_min_degrees=[0.5],
                box_margins_px=[0.0],
            )

            row = self._read_csv(output_root / "cadica_threshold_sweep.csv")[0]
            self.assertEqual(int(row["frame_TP"]), 1)
            self.assertEqual(int(row["matched_gt_boxes"]), 1)
            self.assertAlmostEqual(float(row["box_recall"]), 1.0)

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

    def test_threshold_sweep_writes_frame_temporal_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            manifest_path = temp_root / "manifest.csv"
            frame_results_root = temp_root / "frame_results"
            temporal_results_root = temp_root / "temporal_results"
            output_root = temp_root / "sweep"
            frame_variant = (
                "radius_outside_fraction_threshold_0p1"
                "__radius_min_outside_samples_2"
                "__stenosis_threshold_0p35"
                "__average_radius_threshold_3"
            )
            temporal_variant = "min_supporting_frames_2__min_persistence_ratio_0p2"

            self._write_manifest_rows(
                manifest_path,
                [
                    self._manifest_row(
                        "p1",
                        "v1",
                        1,
                        "lesion",
                        "positive",
                        [{"x": 10, "y": 10, "w": 10, "h": 10, "category": "stenosis", "source_path": "gt/1.txt"}],
                    )
                ],
            )
            self._write_frame_result(
                frame_results_root / frame_variant / "p1" / "v1" / "slice_00001_stenosis_results.json",
                patient_id="p1",
                video_id="v1",
                frame_id=1,
                points=[],
            )
            self._write_temporal_result(
                temporal_results_root
                / frame_variant
                / temporal_variant
                / "p1"
                / "v1"
                / "view_temporal_fusion.json"
            )

            outputs = run_cadica_threshold_sweep(
                manifest=manifest_path,
                frame_results_root=frame_results_root,
                temporal_results_root=temporal_results_root,
                output_root=output_root,
                frame_min_degrees=[0.0, 0.5],
                box_margins_px=[0.0],
                video_prediction_sources=["frame_any", "temporal_final"],
            )

            diagnostics = json.loads(outputs["cadica_benchmark_diagnostics_json"].read_text(encoding="utf-8"))
            self.assertEqual(diagnostics["frame_results"]["json_file_count"], 1)
            self.assertEqual(diagnostics["frame_results"]["jsons_with_nonempty_stenosis_points"], 0)
            self.assertEqual(diagnostics["manifest_matching"]["matched_manifest_frames"], 1)
            self.assertEqual(
                diagnostics["manifest_matching"]["matched_manifest_frames_with_thresholded_points_by_frame_min_degree"],
                {"0": 0, "0.5": 0},
            )
            self.assertEqual(diagnostics["temporal_results"]["json_file_count"], 1)
            self.assertEqual(diagnostics["temporal_results"]["temporal_final_positive_count"], 1)
            self.assertEqual(
                diagnostics["temporal_results"]["positive_temporal_with_any_matched_frame_stenosis_points"],
                0,
            )
            self.assertEqual(
                len(diagnostics["temporal_results"]["positive_temporal_without_underlying_positive_frame_examples"]),
                1,
            )

    def test_evaluate_matched_frames_only_skips_unprocessed_manifest_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            manifest_path = temp_root / "manifest.csv"
            frame_results_root = temp_root / "frame_results"
            self._write_manifest_rows(
                manifest_path,
                [
                    self._manifest_row(
                        "p1",
                        "v1",
                        1,
                        "lesion",
                        "positive",
                        [{"x": 10, "y": 10, "w": 10, "h": 10, "category": "stenosis", "source_path": "gt/1.txt"}],
                    ),
                    self._manifest_row("p1", "v1", 2, "lesion", "negative", []),
                    self._manifest_row(
                        "p1",
                        "v1",
                        3,
                        "lesion",
                        "positive",
                        [{"x": 30, "y": 30, "w": 10, "h": 10, "category": "stenosis", "source_path": "gt/3.txt"}],
                    ),
                ],
            )
            self._write_frame_result(
                frame_results_root / "p1" / "v1" / "slice_00001_stenosis_results.json",
                patient_id="p1",
                video_id="v1",
                frame_id=1,
                points=[{"x": 15, "y": 15, "degree": 0.7, "severity": "moderate"}],
            )
            self._write_frame_result(
                frame_results_root / "p1" / "v1" / "slice_00002_stenosis_results.json",
                patient_id="p1",
                video_id="v1",
                frame_id=2,
                points=[],
            )

            default_output_root = temp_root / "sweep_default"
            run_cadica_threshold_sweep(
                manifest=manifest_path,
                frame_results_root=frame_results_root,
                output_root=default_output_root,
                frame_min_degrees=[0.5],
                box_margins_px=[0.0],
            )
            default_row = self._read_csv(default_output_root / "cadica_threshold_sweep.csv")[0]
            self.assertEqual(int(default_row["frame_total_evaluated"]), 3)
            self.assertEqual(int(default_row["frame_TP"]), 1)
            self.assertEqual(int(default_row["frame_TN"]), 1)
            self.assertEqual(int(default_row["frame_FN"]), 1)

            matched_only_output_root = temp_root / "sweep_matched_only"
            outputs = run_cadica_threshold_sweep(
                manifest=manifest_path,
                frame_results_root=frame_results_root,
                output_root=matched_only_output_root,
                frame_min_degrees=[0.5],
                box_margins_px=[0.0],
                evaluate_matched_frames_only=True,
            )
            matched_only_row = self._read_csv(matched_only_output_root / "cadica_threshold_sweep.csv")[0]
            self.assertEqual(int(matched_only_row["frame_total_evaluated"]), 2)
            self.assertEqual(int(matched_only_row["frame_TP"]), 1)
            self.assertEqual(int(matched_only_row["frame_TN"]), 1)
            self.assertEqual(int(matched_only_row["frame_FN"]), 0)

            diagnostics = json.loads(outputs["cadica_benchmark_diagnostics_json"].read_text(encoding="utf-8"))
            self.assertTrue(diagnostics["manifest_matching"]["evaluate_matched_frames_only"])
            self.assertEqual(diagnostics["manifest_matching"]["manifest_frame_count"], 3)
            self.assertEqual(diagnostics["manifest_matching"]["matched_manifest_frames"], 2)
            self.assertEqual(diagnostics["manifest_matching"]["skipped_unmatched_manifest_frames"], 1)
            self.assertEqual(diagnostics["manifest_matching"]["evaluated_frame_count"], 2)
            self.assertEqual(diagnostics["manifest_matching"]["positive_evaluated_frame_count"], 1)
            self.assertEqual(diagnostics["manifest_matching"]["negative_evaluated_frame_count"], 1)

    def test_workers_must_be_positive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            manifest_path, frame_results_root = self._write_synthetic_dataset(temp_root)

            with self.assertRaisesRegex(ValueError, "workers must be >= 1"):
                run_cadica_threshold_sweep(
                    manifest=manifest_path,
                    frame_results_root=frame_results_root,
                    output_root=temp_root / "sweep",
                    workers=0,
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

    def test_benchmark_includes_multiview_patient_and_side_metrics_without_views_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            manifest_path, frame_results_root, multiview_results_root = self._write_multiview_dataset(
                temp_root,
                include_side_metadata=True,
            )
            output_root = temp_root / "benchmark"

            result = run_cadica_benchmark(
                manifest=manifest_path,
                frame_results_root=frame_results_root,
                multiview_results_root=multiview_results_root,
                output_root=output_root,
                multiview_min_score=0.5,
            )

            self.assertFalse((multiview_results_root / "p1" / "views.json").exists())
            self.assertTrue((output_root / "cadica_multiview_patient_rows.csv").is_file())
            self.assertTrue((output_root / "cadica_multiview_side_rows.csv").is_file())
            self.assertTrue(result.summary["config"]["multiview_evaluated"])
            patient_metrics = result.summary["multiview_patient_binary_metrics"]
            self.assertEqual(patient_metrics["TP"], 1)
            self.assertEqual(patient_metrics["FP"], 1)
            self.assertEqual(patient_metrics["TN"], 1)
            self.assertEqual(patient_metrics["FN"], 1)
            side_metrics = result.summary["multiview_side_binary_metrics"]
            self.assertEqual(side_metrics["TP"], 1)
            self.assertEqual(side_metrics["FP"], 1)
            self.assertEqual(side_metrics["TN"], 1)
            self.assertEqual(side_metrics["FN"], 1)

    def test_multiview_side_metrics_are_skipped_without_side_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            manifest_path, frame_results_root, multiview_results_root = self._write_multiview_dataset(
                temp_root,
                include_side_metadata=False,
            )
            output_root = temp_root / "benchmark"

            result = run_cadica_benchmark(
                manifest=manifest_path,
                frame_results_root=frame_results_root,
                multiview_results_root=multiview_results_root,
                output_root=output_root,
            )

            self.assertTrue((output_root / "cadica_multiview_patient_rows.csv").is_file())
            self.assertFalse((output_root / "cadica_multiview_side_rows.csv").exists())
            self.assertIsNone(result.summary["multiview_side_binary_metrics"])
            self.assertEqual(
                result.summary["counts"]["multiview_patient_skip_reason_counts"].get("none"),
                4,
            )

    def test_sweep_contains_multiview_columns_and_threshold_changes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            manifest_path, frame_results_root, multiview_results_root = self._write_multiview_dataset(
                temp_root,
                include_side_metadata=True,
            )
            output_root = temp_root / "sweep"

            run_cadica_threshold_sweep(
                manifest=manifest_path,
                frame_results_root=frame_results_root,
                multiview_results_root=multiview_results_root,
                output_root=output_root,
                frame_min_degrees=[0.0],
                box_margins_px=[0.0],
                multiview_min_scores=[0.0, 0.5],
            )

            rows = {
                float(row["multiview_min_score"]): row
                for row in self._read_csv(output_root / "cadica_threshold_sweep.csv")
            }
            self.assertTrue((output_root / "cadica_multiview_patient_rows.csv").is_file())
            self.assertTrue((output_root / "cadica_multiview_side_rows.csv").is_file())
            self.assertIn("multiview_patient_F1", rows[0.0])
            self.assertEqual(int(rows[0.0]["multiview_patient_TP"]), 2)
            self.assertEqual(int(rows[0.0]["multiview_patient_FN"]), 0)
            self.assertEqual(int(rows[0.5]["multiview_patient_TP"]), 1)
            self.assertEqual(int(rows[0.5]["multiview_patient_FN"]), 1)

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

    def _write_multiview_dataset(self, root: Path, *, include_side_metadata: bool) -> tuple[Path, Path, Path]:
        manifest_path = root / "manifest.csv"
        frame_results_root = root / "frame_results"
        multiview_results_root = root / "multiview_results"
        rows = [
            self._manifest_row(
                "p1",
                "v1",
                1,
                "lesion",
                "positive",
                [{"x": 10, "y": 10, "w": 10, "h": 10, "category": "stenosis", "source_path": "gt/p1.txt"}],
                coronary_side="left" if include_side_metadata else None,
                projection_group="LCA" if include_side_metadata else None,
            ),
            self._manifest_row(
                "p2",
                "v1",
                1,
                "nonlesion",
                "negative",
                [],
                coronary_side="right" if include_side_metadata else None,
                projection_group="RCA" if include_side_metadata else None,
            ),
            self._manifest_row(
                "p3",
                "v1",
                1,
                "lesion",
                "positive",
                [{"x": 20, "y": 20, "w": 10, "h": 10, "category": "stenosis", "source_path": "gt/p3.txt"}],
                coronary_side="right" if include_side_metadata else None,
                projection_group="RCA" if include_side_metadata else None,
            ),
            self._manifest_row(
                "p4",
                "v1",
                1,
                "nonlesion",
                "negative",
                [],
                coronary_side="left" if include_side_metadata else None,
                projection_group="LCA" if include_side_metadata else None,
            ),
        ]
        self._write_manifest_rows(manifest_path, rows)
        for row in rows:
            self._write_frame_result(
                frame_results_root
                / str(row["patient_id"])
                / str(row["video_id"])
                / "slice_00001_stenosis_results.json",
                patient_id=str(row["patient_id"]),
                video_id=str(row["video_id"]),
                frame_id=1,
                points=[],
            )

        self._write_multiview_result(
            multiview_results_root / "p1" / "case_multiview_fusion.json",
            side_scores={"left": 0.9},
        )
        self._write_multiview_result(
            multiview_results_root / "p2" / "case_multiview_fusion.json",
            side_scores={"right": 0.8},
        )
        self._write_multiview_result(
            multiview_results_root / "p3" / "case_multiview_fusion.json",
            side_scores={"right": 0.4},
        )
        self._write_multiview_result(
            multiview_results_root / "p4" / "case_multiview_fusion.json",
            side_scores={"left": None},
        )
        return manifest_path, frame_results_root, multiview_results_root

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
        coronary_side: str | None = None,
        projection_group: str | None = None,
    ) -> dict[str, object]:
        row: dict[str, object] = {
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
        if coronary_side is not None:
            row["coronary_side"] = coronary_side
        if projection_group is not None:
            row["projection_group"] = projection_group
        return row

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

    def _write_yolo_frame_result(
        self,
        path: Path,
        *,
        patient_id: str,
        video_id: str,
        frame_id: int,
        confidence: float,
    ) -> None:
        image_name = f"slice_{frame_id:05d}.png"
        detection = YoloDetection(
            bbox_xyxy_zero_based=(9.0, 9.0, 18.0, 18.0),
            confidence=confidence,
            class_id=0,
            class_name="Stenosis",
            image_width=512,
            image_height=512,
        )
        payload = build_yolo_frame_payload(
            image_path=Path("work/cadica_prepared/keyframes") / patient_id / video_id / image_name,
            mask_path=None,
            detector=YoloDetectorMetadata(
                name="yolov8",
                weights="runs/stenosis/best.pt",
                imgsz=1024,
                conf=0.25,
                iou=0.7,
                device="cpu",
            ),
            view_id=f"{patient_id}/{video_id}",
            image_width=512,
            image_height=512,
            detections=[detection],
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _write_multiview_result(self, path: Path, *, side_scores: dict[str, float | None]) -> None:
        side_results: dict[str, object] = {}
        for side, score in side_scores.items():
            side_results[side] = {
                "case_id": f"{path.parent.name}:{side}",
                "final_case_lesion": None if score is None else self._final_case_lesion(score),
                "confidence": {"score": 0.0 if score is None else score, "label": "medium"},
                "supporting_views": [] if score is None else [f"{side}_view"],
                "fusion_metadata": {"total_candidate_count": 0 if score is None else 1},
            }
        payload = {
            "case_id": path.parent.name,
            "split_by_coronary_side": True,
            "side_results": side_results,
            "skipped_sides": [side for side in ("left", "right") if side not in side_results],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _write_temporal_result(self, path: Path) -> None:
        payload = {
            "view_id": "p1/v1",
            "final_lesion": {
                "severity": "moderate",
                "degrees": {"median": 0.7, "max": 0.8},
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _final_case_lesion(self, score: float) -> dict[str, object]:
        return {
            "severity": "moderate",
            "degrees": {"median": score, "max": score},
            "total_score": score,
            "confidence": {"score": score, "label": "medium"},
            "distinct_supporting_view_ids": ["view_a"],
        }

    def _read_csv(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stenosis_detection.multiview import MultiViewFusionConfig
from stenosis_detection.parameter_sweep import (
    FrameSweepVariant,
    TemporalSweepVariant,
    _is_temporal_variant_complete,
    build_frame_sweep_variants,
    build_temporal_sweep_variants,
    build_yolo_frame_sweep_variants,
    run_multiview_sweep,
    run_parameter_sweep,
)
from stenosis_detection.pipeline import PipelineConfig
from stenosis_detection.temporal import TemporalFusionConfig
from stenosis_detection.yolo.inference import YoloBatchSummary
from stenosis_detection.yolo.schema import YoloDetection, build_yolo_frame_payload
import stenosis_detection.parameter_sweep as parameter_sweep_module


class ParameterSweepTests(unittest.TestCase):
    def test_build_frame_sweep_variants_crosses_all_frame_parameters(self) -> None:
        variants = build_frame_sweep_variants(
            PipelineConfig(),
            stenosis_thresholds=[0.25, 0.35],
            average_radius_thresholds=[4.0],
            radius_outside_fraction_thresholds=[0.05, 0.10],
            radius_min_outside_samples_values=[2, 3],
        )

        self.assertEqual(len(variants), 8)
        self.assertIn(
            "radius_outside_fraction_threshold_0p05"
            "__radius_min_outside_samples_2"
            "__stenosis_threshold_0p25"
            "__average_radius_threshold_4",
            [variant.name for variant in variants],
        )

    def test_build_temporal_sweep_variants_crosses_temporal_parameters(self) -> None:
        variants = build_temporal_sweep_variants(
            min_supporting_frames_values=[2, 3],
            min_persistence_ratios=[0.25, 0.50],
        )

        self.assertEqual(len(variants), 4)
        self.assertEqual(
            variants[-1].name,
            "min_supporting_frames_3__min_persistence_ratio_0p5",
        )

    def test_build_yolo_frame_sweep_variants_crosses_yolo_parameters(self) -> None:
        variants = build_yolo_frame_sweep_variants(
            yolo_weights="best.pt",
            yolo_conf_thresholds=[0.15, 0.25],
            yolo_iou_thresholds=[0.50, 0.70],
            yolo_imgsz_values=[1024],
            device="0",
            batch_size=8,
        )

        self.assertEqual(len(variants), 4)
        self.assertIn(
            "detector_yolo__yolo_conf_0p25__yolo_iou_0p70__yolo_imgsz_1024",
            [variant.name for variant in variants],
        )
        self.assertTrue(all(variant.detector == "yolo" for variant in variants))
        self.assertTrue(all(variant.config.batch_size == 8 for variant in variants))

    def test_vessel_sweep_requires_masks_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            images_root = temp_root / "images"
            images_root.mkdir()

            with self.assertRaisesRegex(ValueError, "masks_root is required"):
                run_parameter_sweep(
                    frame_detector="vessel",
                    images_root=images_root,
                    output_root=temp_root / "sweep",
                )

    def test_yolo_sweep_without_masks_creates_temporal_and_multiview_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            images_root = self._write_multiview_case_tree(temp_root / "cases", view_ids=["view_01"])
            output_root = temp_root / "yolo_sweep"
            frame_variant = "detector_yolo__yolo_conf_0p25__yolo_iou_0p70__yolo_imgsz_1024"
            temporal_variant = "min_supporting_frames_1__min_persistence_ratio_0p25"

            original_process_yolo_tree = parameter_sweep_module.process_yolo_tree
            calls = []
            parameter_sweep_module.process_yolo_tree = self._fake_yolo_process_tree(calls)
            try:
                result = run_parameter_sweep(
                    frame_detector="yolo",
                    images_root=images_root,
                    masks_root=None,
                    output_root=output_root,
                    yolo_weights="fake.pt",
                    yolo_conf_thresholds=[0.25],
                    yolo_iou_thresholds=[0.70],
                    yolo_imgsz_values=[1024],
                    min_supporting_frames_values=[1],
                    min_persistence_ratios=[0.25],
                    allow_variable_frame_count=True,
                    run_multiview=True,
                    multiview_case_root_tree=images_root,
                    write_debug_images=False,
                    write_temporal_images=False,
                    workers=1,
                    temporal_workers=1,
                    yolo_batch_size=4,
                )
            finally:
                parameter_sweep_module.process_yolo_tree = original_process_yolo_tree

            frame_json = (
                output_root
                / "frame_results"
                / frame_variant
                / "case_a"
                / "view_01"
                / "slice_0001_stenosis_results.json"
            )
            temporal_json = (
                output_root
                / "temporal_results"
                / frame_variant
                / temporal_variant
                / "case_a"
                / "view_01"
                / "view_temporal_fusion.json"
            )
            multiview_json = (
                output_root
                / "multiview_results"
                / frame_variant
                / temporal_variant
                / "case_a"
                / "case_multiview_fusion.json"
            )

            self.assertTrue(frame_json.is_file())
            self.assertTrue(temporal_json.is_file())
            self.assertTrue(multiview_json.is_file())
            self.assertEqual(result.frame_summary.failed, 0)
            self.assertEqual(result.temporal_summary.failed_jobs, 0)
            self.assertIsNotNone(result.multiview_summary)
            self.assertEqual(result.multiview_summary.failed_cases, 0)
            self.assertEqual(calls[0]["masks_root"], None)
            self.assertEqual(calls[0]["output_root"], output_root / "frame_results" / frame_variant)
            self.assertEqual(calls[0]["batch_size"], 4)

            temporal_payload = json.loads(temporal_json.read_text(encoding="utf-8"))
            self.assertEqual(temporal_payload["view_id"], "case_a/view_01")
            multiview_payload = json.loads(multiview_json.read_text(encoding="utf-8"))
            self.assertEqual(multiview_payload["case_id"], "case_a")

    def test_temporal_variant_complete_can_skip_summary_png_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "view_temporal_fusion.json"
            output_path.write_text("{}", encoding="utf-8")

            self.assertFalse(
                _is_temporal_variant_complete(
                    output_path,
                    write_temporal_images=True,
                    write_video=False,
                    video_format="mp4",
                )
            )
            self.assertTrue(
                _is_temporal_variant_complete(
                    output_path,
                    write_temporal_images=False,
                    write_video=False,
                    video_format="mp4",
                )
            )

    def test_multiview_sweep_runs_once_per_temporal_variant_and_skips_existing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            case_root_tree = self._write_multiview_case_tree(temp_root / "cases", view_ids=["view_01"])
            temporal_results_root = temp_root / "temporal_results"
            output_root = temp_root / "multiview_results"
            frame_variants = [FrameSweepVariant("frame_a", PipelineConfig())]
            temporal_variants = [
                TemporalSweepVariant("temporal_a", TemporalFusionConfig()),
                TemporalSweepVariant("temporal_b", TemporalFusionConfig(min_supporting_frames=3)),
            ]
            for temporal_variant in temporal_variants:
                self._write_temporal_result(
                    temporal_results_root / "frame_a" / temporal_variant.name / "case_a" / "view_01" / "view_temporal_fusion.json",
                    view_id="view_01",
                )

            first_summary = run_multiview_sweep(
                case_root_tree=case_root_tree,
                temporal_results_root=temporal_results_root,
                output_root=output_root,
                frame_variants=frame_variants,
                temporal_variants=temporal_variants,
                config=MultiViewFusionConfig(),
                split_by_coronary_side=False,
                skip_existing=True,
            )
            second_summary = run_multiview_sweep(
                case_root_tree=case_root_tree,
                temporal_results_root=temporal_results_root,
                output_root=output_root,
                frame_variants=frame_variants,
                temporal_variants=temporal_variants,
                config=MultiViewFusionConfig(),
                split_by_coronary_side=False,
                skip_existing=True,
            )

            self.assertEqual(first_summary.total_jobs, 2)
            self.assertEqual(first_summary.processed_cases, 2)
            self.assertEqual(first_summary.failed_cases, 0)
            self.assertEqual(second_summary.processed_cases, 0)
            self.assertEqual(second_summary.skipped_cases, 2)
            for temporal_variant in temporal_variants:
                result_path = output_root / "frame_a" / temporal_variant.name / "case_a" / "case_multiview_fusion.json"
                self.assertTrue(result_path.is_file())
                self.assertTrue(result_path.with_name("case_multiview_fusion_summary.png").is_file())
                self.assertTrue(result_path.with_name("case_multiview_fusion_support_matrix.png").is_file())

    def test_multiview_sweep_can_split_cadica_projection_groups_by_coronary_side(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            case_root_tree = self._write_multiview_case_tree(
                temp_root / "cases",
                view_ids=["left_01", "right_01"],
                projection_metadata={
                    "left_01": {"projection_group": "LCA", "projection_groups": ["LCA"], "coronary_side": "left"},
                    "right_01": {"projection_group": "RCA", "projection_groups": ["RCA"], "coronary_side": "right"},
                },
            )
            temporal_results_root = temp_root / "temporal_results"
            output_root = temp_root / "multiview_results"
            frame_variants = [FrameSweepVariant("frame_a", PipelineConfig())]
            temporal_variants = [TemporalSweepVariant("temporal_a", TemporalFusionConfig())]
            self._write_temporal_result(
                temporal_results_root / "frame_a" / "temporal_a" / "case_a" / "left_01" / "view_temporal_fusion.json",
                view_id="left_01",
            )
            self._write_temporal_result(
                temporal_results_root / "frame_a" / "temporal_a" / "case_a" / "right_01" / "view_temporal_fusion.json",
                view_id="right_01",
            )

            summary = run_multiview_sweep(
                case_root_tree=case_root_tree,
                temporal_results_root=temporal_results_root,
                output_root=output_root,
                frame_variants=frame_variants,
                temporal_variants=temporal_variants,
                config=MultiViewFusionConfig(view_diversity_mode="projection_group"),
                split_by_coronary_side=True,
                skip_existing=True,
            )

            result_path = output_root / "frame_a" / "temporal_a" / "case_a" / "case_multiview_fusion.json"
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(summary.processed_cases, 1)
            self.assertEqual(summary.failed_cases, 0)
            self.assertTrue(payload["split_by_coronary_side"])
            self.assertEqual(sorted(payload["side_results"]), ["left", "right"])
            self.assertEqual(payload["unknown_views"], [])

    def _write_multiview_case_tree(
        self,
        case_root_tree: Path,
        *,
        view_ids: list[str],
        projection_metadata: dict[str, dict[str, object]] | None = None,
    ) -> Path:
        case_root = case_root_tree / "case_a"
        case_root.mkdir(parents=True)
        views = []
        for index, view_id in enumerate(view_ids):
            view_dir = case_root / view_id
            view_dir.mkdir()
            (view_dir / "slice_0001.png").write_bytes(b"placeholder")
            view = {
                "view_id": view_id,
                "sequence_id": view_id,
                "rao_lao": float(index * 45),
                "cra_cau": 0.0,
                "temporal_fusion_json": "stale/path.json",
            }
            if projection_metadata is not None and view_id in projection_metadata:
                view.update(
                    {
                        "angle_status": "missing",
                        "projection_status": "known",
                        **projection_metadata[view_id],
                    }
                )
            views.append(view)

        (case_root / "views.json").write_text(
            json.dumps({"case_id": "case_a", "views": views}, indent=2),
            encoding="utf-8",
        )
        return case_root_tree

    def _fake_yolo_process_tree(self, calls: list[dict[str, object]]):
        def fake_process_yolo_tree(
            jobs,
            output_root,
            *,
            images_root,
            masks_root=None,
            config,
            skip_existing=True,
            workers=1,
            write_review_images=False,
            model=None,
        ):
            output_root = Path(output_root)
            calls.append(
                {
                    "output_root": output_root,
                    "masks_root": masks_root,
                    "conf": config.conf,
                    "iou": config.iou,
                    "imgsz": config.imgsz,
                    "batch_size": config.batch_size,
                }
            )
            for job in jobs:
                detection = YoloDetection(
                    bbox_xyxy_zero_based=(9.0, 9.0, 11.0, 11.0),
                    confidence=float(config.conf),
                    class_id=0,
                    class_name="Stenosis",
                    image_width=32,
                    image_height=32,
                )
                payload = build_yolo_frame_payload(
                    image_path=job.image_path,
                    mask_path=job.mask_path,
                    detector=config.detector_metadata(),
                    image_width=32,
                    image_height=32,
                    view_id=job.relative_dir.as_posix(),
                    skeleton_points_xy=[],
                    detections=[detection],
                )
                output_path = output_root / job.relative_dir / f"{job.image_stem}_stenosis_results.json"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            return YoloBatchSummary(
                images_root=Path(images_root),
                masks_root=None if masks_root is None else Path(masks_root),
                output_root=output_root,
                total_jobs=len(jobs),
                workers=1,
                processed=len(jobs),
                skipped_existing=0,
                failed=0,
                failures=[],
                review_images_saved=write_review_images,
            )

        return fake_process_yolo_tree

    def _write_temporal_result(self, output_path: Path, *, view_id: str) -> None:
        output_path.parent.mkdir(parents=True)
        lesion = {
            "lesion_id": 1,
            "track_id": 10,
            "severity": "moderate",
            "supporting_frame_count": 2,
            "total_frame_count": 3,
            "persistence_ratio": 0.67,
            "frame_indices": [1, 2],
            "degrees": {"median": 0.70, "max": 0.80},
            "stability": {"degree_std": 0.05, "max_frame_gap": 1},
            "positions": {},
        }
        output_path.write_text(
            json.dumps(
                {
                    "view_id": view_id,
                    "final_lesion": lesion,
                    "persistent_lesions": [lesion],
                }
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()

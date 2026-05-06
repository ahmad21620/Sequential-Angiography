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
    run_multiview_sweep,
)
from stenosis_detection.pipeline import PipelineConfig
from stenosis_detection.temporal import TemporalFusionConfig


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

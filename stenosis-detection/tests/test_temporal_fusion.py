from __future__ import annotations

import io
from contextlib import redirect_stdout
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_temporal_fusion as temporal_cli

from stenosis_detection.temporal.fusion import build_persistent_lesions
from stenosis_detection.temporal.loader import load_view_sequence
from stenosis_detection.temporal.mapping import map_observations_to_reference_centerline
from stenosis_detection.temporal.models import (
    FrameLevelResult,
    FrameRegistration,
    LesionObservation,
    LesionTrack,
    PersistentLesion,
    ReferenceFrameSelection,
    ViewLevelResult,
)
from stenosis_detection.temporal.reference import select_reference_frame
from stenosis_detection.temporal.video import create_view_demo_frames
from stenosis_detection.temporal.visualization import INFO_PANEL_WIDTH, create_view_summary_visualization


class TemporalFusionTests(unittest.TestCase):
    def test_loader_parses_and_sorts_view_sequence(self) -> None:
        fixture_dir = Path(__file__).resolve().parent / "data" / "view_sequence"
        view_sequence = load_view_sequence(fixture_dir, expected_frame_count=2)

        self.assertEqual(view_sequence.view_id, "study/series_a")
        self.assertEqual([frame.frame_index for frame in view_sequence.frames], [1, 2])
        self.assertEqual(view_sequence.frames[0].image_name, "slice_0001.png")
        self.assertEqual(view_sequence.frames[0].width, 600)
        self.assertEqual(view_sequence.frames[0].height, 800)
        self.assertEqual(view_sequence.frames[0].observation_count, 1)
        self.assertAlmostEqual(view_sequence.frames[1].observations[0].degree, 0.6)

    def test_reference_frame_selection_prefers_largest_support_then_middle_frame(self) -> None:
        frame_results = [
            self._make_frame_result(image_name="slice_0001.png", frame_index=1, skeleton_points=[[1, 1], [2, 1], [3, 1]]),
            self._make_frame_result(image_name="slice_0002.png", frame_index=2, skeleton_points=[[1, 1], [2, 1], [3, 1], [4, 1], [5, 1]]),
            self._make_frame_result(image_name="slice_0003.png", frame_index=3, skeleton_points=[[1, 1], [2, 1], [3, 1], [4, 1], [5, 1]]),
        ]

        selection = select_reference_frame(frame_results)

        self.assertEqual(selection.frame.image_name, "slice_0002.png")
        self.assertEqual(selection.method, "largest_centerline_support")
        self.assertEqual(selection.score_name, "skeleton_point_count")
        self.assertEqual(selection.support_point_count, 5)
        self.assertEqual(selection.middle_frame_distance, 0)

    def test_mapping_projects_observation_onto_reference_centerline(self) -> None:
        reference_frame = self._make_frame_result(
            image_name="slice_0002.png",
            frame_index=2,
            skeleton_points=[[1, 1], [2, 1], [3, 1], [4, 1], [5, 1]],
        )
        frame_result = self._make_frame_result(
            image_name="slice_0001.png",
            frame_index=1,
            skeleton_points=[[1, 1], [2, 1], [3, 1], [4, 1], [5, 1]],
            observations=[
                self._make_observation(
                    frame_index=1,
                    image_name="slice_0001.png",
                    point_xy=[3, 2],
                    degree=0.55,
                )
            ],
        )
        registration = FrameRegistration(
            image_name="slice_0001.png",
            frame_index=1,
            reference_image_name="slice_0002.png",
            reference_frame_index=2,
            transform_matrix=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
            method="reference_identity",
            status="success",
            score=1.0,
        )

        mapped_observations = map_observations_to_reference_centerline(frame_result, reference_frame, registration)

        self.assertEqual(len(mapped_observations), 1)
        mapped_observation = mapped_observations[0]
        self.assertEqual(mapped_observation.centerline_position_status, "ok")
        self.assertEqual(mapped_observation.centerline_component_id, 1)
        self.assertEqual(mapped_observation.reference_centerline_point_xy.tolist(), [3, 1])
        self.assertAlmostEqual(mapped_observation.centerline_distance or 0.0, 1.0)
        self.assertAlmostEqual(mapped_observation.centerline_position or 0.0, 0.5)

    def test_persistent_lesions_filter_unstable_tracks_and_keep_robust_summary(self) -> None:
        persistent_track = LesionTrack(
            track_id=1,
            observations=[
                self._make_observation(
                    frame_index=1,
                    image_name="slice_0001.png",
                    point_xy=[10, 10],
                    degree=0.4,
                    registered_point_xy=[10.0, 10.0],
                    reference_centerline_point_xy=[10, 10],
                    centerline_component_id=1,
                    centerline_position=0.20,
                ),
                self._make_observation(
                    frame_index=2,
                    image_name="slice_0002.png",
                    point_xy=[11, 10],
                    degree=0.6,
                    registered_point_xy=[10.5, 10.0],
                    reference_centerline_point_xy=[10, 10],
                    centerline_component_id=1,
                    centerline_position=0.22,
                ),
                self._make_observation(
                    frame_index=4,
                    image_name="slice_0004.png",
                    point_xy=[10, 11],
                    degree=0.8,
                    registered_point_xy=[10.0, 10.5],
                    reference_centerline_point_xy=[10, 11],
                    centerline_component_id=1,
                    centerline_position=0.21,
                ),
            ],
        )
        unstable_track = LesionTrack(
            track_id=2,
            observations=[
                self._make_observation(
                    frame_index=3,
                    image_name="slice_0003.png",
                    point_xy=[30, 30],
                    degree=0.9,
                    registered_point_xy=[30.0, 30.0],
                    reference_centerline_point_xy=[30, 30],
                    centerline_component_id=2,
                    centerline_position=0.75,
                )
            ],
        )

        persistent_lesions = build_persistent_lesions(
            [persistent_track, unstable_track],
            total_frame_count=4,
            min_supporting_frames=2,
            min_persistence_ratio=0.5,
        )

        self.assertEqual(len(persistent_lesions), 1)
        persistent_lesion = persistent_lesions[0]
        self.assertEqual(persistent_lesion.track_id, 1)
        self.assertEqual(persistent_lesion.supporting_frame_count, 3)
        self.assertAlmostEqual(persistent_lesion.persistence_ratio, 0.75)
        self.assertAlmostEqual(persistent_lesion.median_degree, 0.6)
        self.assertAlmostEqual(persistent_lesion.max_degree, 0.8)
        self.assertAlmostEqual(persistent_lesion.median_centerline_position or 0.0, 0.21)
        self.assertEqual(persistent_lesion.severity, "moderate")

    def test_summary_visualization_builds_canvas_for_view_result(self) -> None:
        view_result = self._make_view_result_for_rendering()

        summary_image = create_view_summary_visualization(view_result)

        self.assertEqual(summary_image.ndim, 3)
        self.assertEqual(summary_image.shape[2], 3)
        self.assertEqual(summary_image.shape[0], view_result.reference_frame.height)
        self.assertEqual(summary_image.shape[1], view_result.reference_frame.width + INFO_PANEL_WIDTH)

    def test_summary_visualization_grows_to_fit_info_panel(self) -> None:
        view_result = self._make_view_result_for_rendering()
        view_result.reference_frame.width = 220
        view_result.reference_frame.height = 220
        view_result.persistent_lesions = view_result.persistent_lesions * 8

        summary_image = create_view_summary_visualization(view_result)

        self.assertGreater(summary_image.shape[0], view_result.reference_frame.height)
        self.assertEqual(summary_image.shape[1], view_result.reference_frame.width + INFO_PANEL_WIDTH)

    def test_video_renderer_builds_minimal_demo_frames(self) -> None:
        view_result = self._make_view_result_for_rendering()

        demo_frames = create_view_demo_frames(view_result)

        self.assertEqual(len(demo_frames), len(view_result.frames))
        self.assertEqual(demo_frames[0].ndim, 3)
        self.assertEqual(demo_frames[0].shape[2], 3)
        self.assertEqual(demo_frames[0].shape[0], view_result.reference_frame.height)
        self.assertEqual(demo_frames[0].shape[1], view_result.reference_frame.width)

    def test_cli_build_view_output_path_mirrors_view_id_under_output_root(self) -> None:
        output_path = temporal_cli._build_view_output_path(
            "OutputTemporal",
            self._make_view_sequence(),
            result_source=["Output/study/series_a/slice_0001_stenosis_results.json"],
        )

        self.assertEqual(output_path, Path("OutputTemporal") / "study" / "series_a" / "view_temporal_fusion.json")

    def test_cli_build_view_output_path_mirrors_results_root_layout_when_available(self) -> None:
        output_path = temporal_cli._build_view_output_path(
            "OutputTemporal",
            self._make_view_sequence(),
            result_source="Output",
        )

        self.assertEqual(output_path, Path("OutputTemporal") / "study" / "series_a" / "view_temporal_fusion.json")

    def test_cli_build_view_output_path_falls_back_for_empty_view_id(self) -> None:
        output_path = temporal_cli._build_view_output_path(
            "OutputTemporal",
            self._make_view_sequence(view_id=""),
            result_source=["Output/slice_0001_stenosis_results.json"],
        )

        self.assertEqual(output_path, Path("OutputTemporal") / "view" / "view_temporal_fusion.json")

    def test_cli_is_view_fully_processed_requires_complete_non_video_output_set(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "study" / "series_a" / "view_temporal_fusion.json"
            summary_path = output_path.with_name("view_temporal_fusion_summary.png")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            output_path.write_text("{}", encoding="utf-8")
            self.assertFalse(
                temporal_cli._is_view_fully_processed(
                    output_path,
                    write_video=False,
                    video_format="mp4",
                )
            )

            summary_path.write_bytes(b"png")
            self.assertTrue(
                temporal_cli._is_view_fully_processed(
                    output_path,
                    write_video=False,
                    video_format="mp4",
                )
            )

    def test_cli_is_view_fully_processed_requires_video_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "study" / "series_a" / "view_temporal_fusion.json"
            summary_path = output_path.with_name("view_temporal_fusion_summary.png")
            video_path = output_path.with_name("view_temporal_fusion_demo.mp4")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            output_path.write_text("{}", encoding="utf-8")
            summary_path.write_bytes(b"png")
            self.assertFalse(
                temporal_cli._is_view_fully_processed(
                    output_path,
                    write_video=True,
                    video_format="mp4",
                )
            )

            video_path.write_bytes(b"mp4")
            self.assertTrue(
                temporal_cli._is_view_fully_processed(
                    output_path,
                    write_video=True,
                    video_format="mp4",
                )
            )

    def test_cli_filters_and_tallies_frame_count_mismatches(self) -> None:
        valid_view = self._make_view_sequence(view_id="study/series_a")
        short_view = self._make_view_sequence(view_id="study/series_b")
        short_view.frames = short_view.frames[:1]

        accepted, skipped = temporal_cli._filter_view_sequences_by_expected_frame_count(
            [valid_view, short_view],
            expected_frame_count=2,
        )

        self.assertEqual([view.view_id for view in accepted], ["study/series_a"])
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0].view_id, "study/series_b")
        self.assertEqual(skipped[0].frame_count, 1)
        self.assertEqual(skipped[0].expected_frame_count, 2)

        messages: list[str] = []
        temporal_cli._print_frame_count_skip_summary(skipped, log=messages.append)

        self.assertEqual(
            messages,
            [
                "Skipped 1 view with frame-count mismatches (expected 2 frame results).",
                "  1 frame results: 1 view",
            ],
        )

    def test_cli_skip_frame_count_mismatches_processes_valid_views_and_reports_tally(self) -> None:
        valid_view = self._make_view_sequence(view_id="study/series_a")
        short_view = self._make_view_sequence(view_id="study/series_b")
        short_view.frames = short_view.frames[:1]
        processed_view_ids: list[str] = []

        def fake_run_single_view(view_sequence, **kwargs):
            processed_view_ids.append(view_sequence.view_id)
            return None

        argv = [
            "run_temporal_fusion.py",
            "--results-root",
            "InputResults",
            "--output-root",
            "OutputTemporal",
            "--expected-frame-count",
            "2",
            "--skip-frame-count-mismatches",
        ]
        stdout = io.StringIO()
        with (
            patch.object(sys, "argv", argv),
            patch.object(temporal_cli, "load_view_sequences", return_value=[valid_view, short_view]),
            patch.object(temporal_cli, "_run_single_view", side_effect=fake_run_single_view),
            redirect_stdout(stdout),
        ):
            exit_code = temporal_cli.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(processed_view_ids, ["study/series_a"])
        output = stdout.getvalue()
        self.assertIn("Processed 1 views.", output)
        self.assertIn("Skipped 1 view with frame-count mismatches (expected 2 frame results).", output)
        self.assertIn("  1 frame results: 1 view", output)

    def _make_frame_result(
        self,
        *,
        image_name: str,
        frame_index: int,
        skeleton_points: list[list[int]],
        observations: list[LesionObservation] | None = None,
    ) -> FrameLevelResult:
        return FrameLevelResult(
            result_path=Path(f"Output/{image_name.replace('.png', '_stenosis_results.json')}"),
            image_path=f"Input_Original/{image_name}",
            mask_path=f"Input_Mask/{image_name.removesuffix('.png')}_mask.png",
            image_name=image_name,
            image_stem=image_name.removesuffix(".png"),
            view_id="study/series_a",
            frame_index=frame_index,
            width=600,
            height=800,
            skeleton_points_xy=np.asarray(skeleton_points, dtype=np.int32),
            observations=observations or [],
        )

    def _make_observation(
        self,
        *,
        frame_index: int,
        image_name: str,
        point_xy: list[int],
        degree: float,
        registered_point_xy: list[float] | None = None,
        reference_centerline_point_xy: list[int] | None = None,
        centerline_component_id: int | None = None,
        centerline_position: float | None = None,
    ) -> LesionObservation:
        return LesionObservation(
            frame_index=frame_index,
            image_name=image_name,
            point_xy=np.asarray(point_xy, dtype=np.int32),
            degree=degree,
            severity="moderate" if degree > 0.5 else "mild",
            registered_point_xy=None if registered_point_xy is None else np.asarray(registered_point_xy, dtype=np.float64),
            reference_centerline_point_xy=(
                None if reference_centerline_point_xy is None else np.asarray(reference_centerline_point_xy, dtype=np.int32)
            ),
            centerline_component_id=centerline_component_id,
            centerline_position=centerline_position,
        )

    def _make_view_result_for_rendering(self) -> ViewLevelResult:
        first_frame = self._make_frame_result(
            image_name="slice_0001.png",
            frame_index=1,
            skeleton_points=[[1, 1], [2, 1], [3, 1], [4, 1], [5, 1]],
        )
        reference_frame = self._make_frame_result(
            image_name="slice_0002.png",
            frame_index=2,
            skeleton_points=[[1, 1], [2, 1], [3, 1], [4, 1], [5, 1]],
        )
        lesion_track = LesionTrack(
            track_id=3,
            observations=[
                self._make_observation(
                    frame_index=1,
                    image_name="slice_0001.png",
                    point_xy=[3, 2],
                    degree=0.52,
                    registered_point_xy=[3.0, 2.0],
                    reference_centerline_point_xy=[3, 1],
                    centerline_component_id=1,
                    centerline_position=0.50,
                ),
                self._make_observation(
                    frame_index=2,
                    image_name="slice_0002.png",
                    point_xy=[4, 2],
                    degree=0.60,
                    registered_point_xy=[4.0, 2.0],
                    reference_centerline_point_xy=[4, 1],
                    centerline_component_id=1,
                    centerline_position=0.75,
                ),
            ],
        )
        subtle_track = LesionTrack(
            track_id=5,
            observations=[
                self._make_observation(
                    frame_index=1,
                    image_name="slice_0001.png",
                    point_xy=[2, 3],
                    degree=0.41,
                    registered_point_xy=[2.0, 3.0],
                    reference_centerline_point_xy=[2, 2],
                    centerline_component_id=2,
                    centerline_position=0.20,
                )
            ],
        )
        final_lesion = PersistentLesion(
            lesion_id=1,
            track_id=3,
            supporting_frames=lesion_track.supporting_frames,
            frame_indices=lesion_track.frame_indices,
            supporting_frame_count=2,
            total_frame_count=2,
            persistence_ratio=1.0,
            median_degree=0.56,
            max_degree=0.60,
            median_registered_point_xy=np.asarray([3.5, 2.0], dtype=np.float64),
            median_centerline_point_xy=np.asarray([4, 1], dtype=np.int32),
            median_centerline_position=0.625,
            dominant_centerline_component_id=1,
            degree_std=0.04,
            image_position_std_px=0.5,
            centerline_position_std=0.125,
            max_frame_gap=1,
        )
        subtle_lesion = PersistentLesion(
            lesion_id=2,
            track_id=5,
            supporting_frames=subtle_track.supporting_frames,
            frame_indices=subtle_track.frame_indices,
            supporting_frame_count=1,
            total_frame_count=2,
            persistence_ratio=0.5,
            median_degree=0.41,
            max_degree=0.41,
            median_registered_point_xy=np.asarray([2.0, 3.0], dtype=np.float64),
            median_centerline_point_xy=np.asarray([2, 2], dtype=np.int32),
            median_centerline_position=0.20,
            dominant_centerline_component_id=2,
            degree_std=0.0,
            image_position_std_px=0.0,
            centerline_position_std=0.0,
            max_frame_gap=0,
        )
        return ViewLevelResult(
            view_id="study/series_a",
            frames=[first_frame, reference_frame],
            reference_selection=ReferenceFrameSelection(
                frame=reference_frame,
                method="largest_centerline_support",
                score_name="skeleton_point_count",
                score_value=5.0,
                support_point_count=5,
                support_coverage_ratio=0.01,
                middle_frame_distance=0,
                reason="synthetic",
            ),
            registrations=[],
            tracks=[lesion_track, subtle_track],
            persistent_lesions=[final_lesion, subtle_lesion],
            final_lesion=final_lesion,
            min_supporting_frames=1,
            min_persistence_ratio=0.5,
            final_selection_rule="test",
        )

    def _make_view_sequence(self, *, view_id: str = "study/series_a"):
        first_frame = self._make_frame_result(
            image_name="slice_0001.png",
            frame_index=1,
            skeleton_points=[[1, 1], [2, 1], [3, 1]],
        )
        second_frame = self._make_frame_result(
            image_name="slice_0002.png",
            frame_index=2,
            skeleton_points=[[1, 1], [2, 1], [3, 1]],
        )
        first_frame.view_id = view_id
        second_frame.view_id = view_id
        first_frame.result_path = Path("Output/study/series_a/slice_0001_stenosis_results.json")
        second_frame.result_path = Path("Output/study/series_a/slice_0002_stenosis_results.json")
        return temporal_cli.ViewSequence(view_id=view_id, frames=[first_frame, second_frame])


if __name__ == "__main__":
    unittest.main()

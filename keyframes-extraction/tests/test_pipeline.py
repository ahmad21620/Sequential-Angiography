from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from angio_keyframes.backends import create_backend, get_cuda_unavailable_reason, is_cuda_available
from angio_keyframes.discovery import discover_frame_directories
from angio_keyframes.images import enhance_vessel_image, list_image_files, load_grayscale_image
from angio_keyframes.models import KeyframeCandidate
from angio_keyframes.pipeline import (
    build_baseline_image,
    compute_contrast_fill_score,
    extract_keyframes_from_directory,
    extract_keyframes_from_root,
    smooth_score_curve,
    select_keyframe_window,
    write_keyframes,
)


def make_candidates(scores: list[float]) -> list[KeyframeCandidate]:
    return [
        KeyframeCandidate(
            frame_index=index,
            name=f"frame_{index:05d}.png",
            source_path=Path(f"frame_{index:05d}.png"),
            score=score,
        )
        for index, score in enumerate(scores)
    ]


class PipelineTests(unittest.TestCase):
    def test_create_backend_returns_cpu_backend(self) -> None:
        backend = create_backend("cpu")

        self.assertEqual(backend.name, "cpu")

    def test_cuda_backend_request_validates_availability(self) -> None:
        if is_cuda_available():
            backend = create_backend("cuda")
            self.assertEqual(backend.name, "cuda")
            return

        reason = get_cuda_unavailable_reason()
        self.assertIsNotNone(reason)
        with self.assertRaisesRegex(RuntimeError, "CUDA|cuda"):
            create_backend("cuda")

    def test_load_grayscale_image_uses_direct_grayscale_read(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "frame.png"
            color_image = np.dstack(
                (
                    np.full((4, 4), 10, dtype=np.uint8),
                    np.full((4, 4), 80, dtype=np.uint8),
                    np.full((4, 4), 150, dtype=np.uint8),
                )
            )
            cv2.imwrite(str(image_path), color_image)

            loaded = load_grayscale_image(image_path)
            expected = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

            self.assertIsNotNone(expected)
            np.testing.assert_array_equal(loaded, expected)

    def test_build_baseline_image_uses_pixelwise_median(self) -> None:
        frames = [
            np.array([[10, 90], [20, 80]], dtype=np.uint8),
            np.array([[30, 70], [40, 60]], dtype=np.uint8),
            np.array([[20, 100], [50, 50]], dtype=np.uint8),
        ]

        baseline = build_baseline_image(frames)

        np.testing.assert_array_equal(
            baseline,
            np.array([[20.0, 90.0], [40.0, 60.0]], dtype=np.float32),
        )

    def test_discover_frame_directories_accepts_direct_image_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.full((4, 4, 3), 10, dtype=np.uint8)
            cv2.imwrite(str(root / "frame_00000.png"), image)

            discovered = discover_frame_directories(root)

            self.assertEqual(discovered, [root])

    def test_discover_frame_directories_finds_nested_image_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir) / "dataset"
            first_sequence_dir = dataset_root / "Study_1" / "Series_1"
            second_sequence_dir = dataset_root / "Study_1" / "Series_2"
            first_sequence_dir.mkdir(parents=True)
            second_sequence_dir.mkdir(parents=True)

            image = np.full((4, 4, 3), 10, dtype=np.uint8)
            cv2.imwrite(str(first_sequence_dir / "slice_0000.png"), image)
            cv2.imwrite(str(second_sequence_dir / "slice_0000.png"), image)

            discovered = discover_frame_directories(dataset_root)

            self.assertEqual(discovered, [first_sequence_dir, second_sequence_dir])

    def test_list_image_files_ignores_extract_complete_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.full((4, 4, 3), 10, dtype=np.uint8)
            cv2.imwrite(str(root / ".extract_complete.png"), image)
            cv2.imwrite(str(root / "slice_0000.png"), image)

            discovered = list_image_files(root)

            self.assertEqual([path.name for path in discovered], ["slice_0000.png"])

    def test_compute_contrast_fill_score_is_higher_for_stronger_vessel_response(self) -> None:
        baseline = np.zeros((3, 3), dtype=np.float32)
        low_fill_frame = np.full((3, 3), 5, dtype=np.uint8)
        high_fill_frame = np.full((3, 3), 40, dtype=np.uint8)

        low_fill_score = compute_contrast_fill_score(baseline, low_fill_frame)
        high_fill_score = compute_contrast_fill_score(baseline, high_fill_frame)

        self.assertGreater(high_fill_score, low_fill_score)
        self.assertEqual(low_fill_score, 5.0)
        self.assertEqual(high_fill_score, 40.0)

    def test_vessel_enhancement_gives_dark_line_a_higher_score(self) -> None:
        baseline_frame = np.full((64, 64), 180, dtype=np.uint8)
        vessel_frame = baseline_frame.copy()
        cv2.line(vessel_frame, (8, 32), (56, 32), color=40, thickness=3)

        enhanced_baseline = enhance_vessel_image(baseline_frame)
        enhanced_vessel = enhance_vessel_image(vessel_frame)
        scoring_baseline = build_baseline_image([enhanced_baseline])

        baseline_score = compute_contrast_fill_score(scoring_baseline, enhanced_baseline)
        vessel_score = compute_contrast_fill_score(scoring_baseline, enhanced_vessel)

        self.assertEqual(enhanced_baseline.dtype, np.uint8)
        self.assertEqual(enhanced_vessel.dtype, np.uint8)
        self.assertGreater(vessel_score, baseline_score)

    def test_select_keyframe_window_returns_contiguous_peak_window(self) -> None:
        candidates = make_candidates([0.0, 1.0, 2.0, 6.0, 10.0, 8.0, 3.0, 1.0, 0.0])

        selected = select_keyframe_window(candidates, limit=4, smoothing_window=3)

        self.assertEqual([candidate.name for candidate in selected], [
            "frame_00002.png",
            "frame_00003.png",
            "frame_00004.png",
            "frame_00005.png",
        ])

    def test_select_keyframe_window_centered_mode_preserves_even_window_behavior(self) -> None:
        candidates = make_candidates([0.0] * 20 + [10.0] + [0.0] * 4)

        selected = select_keyframe_window(
            candidates,
            limit=8,
            smoothing_window=1,
            window_mode="centered",
        )

        self.assertEqual(
            [candidate.frame_index for candidate in selected],
            [16, 17, 18, 19, 20, 21, 22, 23],
        )

    def test_select_keyframe_window_leading_mode_ends_at_peak(self) -> None:
        candidates = make_candidates([0.0] * 20 + [10.0] + [0.0] * 4)

        selected = select_keyframe_window(
            candidates,
            limit=8,
            smoothing_window=1,
            window_mode="leading",
        )

        self.assertEqual(
            [candidate.frame_index for candidate in selected],
            [13, 14, 15, 16, 17, 18, 19, 20],
        )

    def test_select_keyframe_window_returns_all_frames_for_short_sequence(self) -> None:
        candidates = make_candidates([1.0, 3.0, 5.0])

        selected = select_keyframe_window(candidates, limit=5, smoothing_window=5)

        self.assertEqual([candidate.name for candidate in selected], [
            "frame_00000.png",
            "frame_00001.png",
            "frame_00002.png",
        ])

    def test_select_keyframe_window_clips_near_start(self) -> None:
        candidates = make_candidates([10.0, 9.0, 3.0, 1.0, 0.0])

        selected = select_keyframe_window(candidates, limit=4, smoothing_window=3)

        self.assertEqual([candidate.name for candidate in selected], [
            "frame_00000.png",
            "frame_00001.png",
            "frame_00002.png",
            "frame_00003.png",
        ])

    def test_select_keyframe_window_leading_mode_shifts_forward_near_start(self) -> None:
        candidates = make_candidates([1.0, 10.0, 3.0, 1.0, 0.0])

        selected = select_keyframe_window(
            candidates,
            limit=4,
            smoothing_window=1,
            window_mode="leading",
        )

        self.assertEqual([candidate.frame_index for candidate in selected], [0, 1, 2, 3])

    def test_select_keyframe_window_clips_near_end(self) -> None:
        candidates = make_candidates([0.0, 1.0, 3.0, 9.0, 10.0])

        selected = select_keyframe_window(candidates, limit=4, smoothing_window=3)

        self.assertEqual([candidate.name for candidate in selected], [
            "frame_00001.png",
            "frame_00002.png",
            "frame_00003.png",
            "frame_00004.png",
        ])

    def test_select_keyframe_window_leading_mode_handles_peak_near_end(self) -> None:
        candidates = make_candidates([0.0, 1.0, 3.0, 9.0, 10.0])

        selected = select_keyframe_window(
            candidates,
            limit=4,
            smoothing_window=1,
            window_mode="leading",
        )

        self.assertEqual([candidate.frame_index for candidate in selected], [1, 2, 3, 4])

    def test_select_keyframe_window_leading_mode_returns_all_frames_for_short_sequence(self) -> None:
        candidates = make_candidates([1.0, 3.0, 5.0])

        selected = select_keyframe_window(
            candidates,
            limit=5,
            smoothing_window=1,
            window_mode="leading",
        )

        self.assertEqual([candidate.frame_index for candidate in selected], [0, 1, 2])

    def test_select_keyframe_window_uses_earliest_peak_on_tie(self) -> None:
        candidates = make_candidates([0.0, 10.0, 0.0, 10.0, 0.0])

        selected = select_keyframe_window(candidates, limit=1, smoothing_window=1)

        self.assertEqual([candidate.name for candidate in selected], ["frame_00001.png"])

    def test_smooth_score_curve_matches_expected_centered_average(self) -> None:
        scores = np.array([0.0, 1.0, 2.0, 6.0, 10.0], dtype=np.float32)

        smoothed_scores = smooth_score_curve(scores, smoothing_window=3)

        np.testing.assert_allclose(
            smoothed_scores,
            np.array([0.5, 1.0, 3.0, 6.0, 8.0], dtype=np.float32),
        )

    def test_write_keyframes_rereads_selected_source_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "frame_00000.png"
            output_dir = root / "output"

            original_image = np.full((4, 4), 10, dtype=np.uint8)
            updated_image = np.full((4, 4), 90, dtype=np.uint8)
            cv2.imwrite(str(source_path), original_image)

            candidates = [
                KeyframeCandidate(
                    frame_index=0,
                    name=source_path.name,
                    source_path=source_path,
                    score=1.0,
                )
            ]

            cv2.imwrite(str(source_path), updated_image)
            write_keyframes(candidates, output_dir, overwrite=True)

            written = load_grayscale_image(output_dir / source_path.name)
            np.testing.assert_array_equal(written, updated_image)

    def test_extract_keyframes_from_directory_skips_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            frames_dir = root / "frames"
            output_dir = root / "keyFrames"
            frames_dir.mkdir()
            output_dir.mkdir()

            for index in range(3):
                image = np.full((16, 16, 3), 180, dtype=np.uint8)
                cv2.imwrite(str(frames_dir / f"frame_{index:05d}.png"), image)

            existing_output = np.full((16, 16), 55, dtype=np.uint8)
            cv2.imwrite(str(output_dir / "frame_00042.png"), existing_output)

            result = extract_keyframes_from_directory(
                frames_dir=frames_dir,
                output_dir=output_dir,
                backend="cuda",
                skip_existing=True,
            )

            self.assertTrue(result.skipped)
            self.assertEqual(result.selected_count, 1)
            self.assertEqual(sorted(path.name for path in output_dir.iterdir()), ["frame_00042.png"])

    def test_extract_keyframes_writes_contiguous_peak_window(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            coro_dir = Path(temp_dir) / "Patient_1" / "Coro_1"
            frames_dir = coro_dir / "frames"
            output_dir = Path(temp_dir) / "mirrored_output" / "Patient_1" / "Coro_1" / "keyFrames"
            frames_dir.mkdir(parents=True)

            vessel_segments = [
                (),
                (),
                (),
                (((8, 32), (56, 32), 3),),
                (((8, 24), (56, 24), 3), ((8, 40), (56, 40), 3)),
                (
                    ((8, 10), (56, 10), 3),
                    ((8, 18), (56, 18), 3),
                    ((8, 32), (56, 32), 3),
                    ((8, 46), (56, 46), 3),
                ),
                (((8, 24), (56, 24), 3), ((8, 40), (56, 40), 3)),
                (((8, 32), (56, 32), 3),),
            ]

            for index, segments in enumerate(vessel_segments):
                image = np.full((64, 64, 3), 180, dtype=np.uint8)
                for start, end, thickness in segments:
                    cv2.line(image, start, end, color=(40, 40, 40), thickness=thickness)
                cv2.imwrite(str(frames_dir / f"frame_{index:05d}.png"), image)

            result = extract_keyframes_from_directory(
                frames_dir,
                output_dir=output_dir,
                limit=4,
                baseline_frames=3,
                smoothing_window=3,
                overwrite=True,
            )

            self.assertEqual(result.selected_count, 4)
            self.assertEqual(
                sorted(path.name for path in result.output_dir.iterdir()),
                ["frame_00003.png", "frame_00004.png", "frame_00005.png", "frame_00006.png"],
            )

    def test_extract_keyframes_uses_available_frames_for_short_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            coro_dir = Path(temp_dir) / "Patient_1" / "Coro_1"
            frames_dir = coro_dir / "frames"
            output_dir = Path(temp_dir) / "mirrored_output" / "Patient_1" / "Coro_1" / "keyFrames"
            frames_dir.mkdir(parents=True)

            for index in range(2):
                image = np.full((64, 64, 3), 180, dtype=np.uint8)
                if index == 1:
                    cv2.line(image, (8, 32), (56, 32), color=(40, 40, 40), thickness=3)
                cv2.imwrite(str(frames_dir / f"frame_{index:05d}.png"), image)

            result = extract_keyframes_from_directory(
                frames_dir,
                output_dir=output_dir,
                limit=2,
                baseline_frames=3,
                smoothing_window=3,
                overwrite=True,
            )

            self.assertEqual(result.selected_count, 2)
            self.assertEqual(
                sorted(path.name for path in result.output_dir.iterdir()),
                ["frame_00000.png", "frame_00001.png"],
            )

    def test_extract_keyframes_from_root_writes_mirrored_tree_with_keyframes_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir) / "dataset"
            frames_dir = dataset_root / "Patient_1" / "Coro_1" / "frames"
            output_root = Path(temp_dir) / "dataset_keyFrames"
            frames_dir.mkdir(parents=True)

            for index in range(5):
                image = np.full((64, 64, 3), 180, dtype=np.uint8)
                if index >= 2:
                    cv2.line(image, (8, 32), (56, 32), color=(40, 40, 40), thickness=index - 1)
                cv2.imwrite(str(frames_dir / f"frame_{index:05d}.png"), image)

            results = extract_keyframes_from_root(
                input_path=dataset_root,
                output_root=output_root,
                limit=3,
                baseline_frames=2,
                smoothing_window=3,
                overwrite=True,
            )

            expected_output_dir = output_root / "Patient_1" / "Coro_1"

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].output_dir, expected_output_dir)
            self.assertTrue(expected_output_dir.is_dir())
            self.assertFalse((output_root / "Patient_1" / "Coro_1" / "frames").exists())
            self.assertEqual(len(list(expected_output_dir.iterdir())), 3)

    def test_extract_keyframes_from_root_preserves_non_frames_directory_hierarchy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir) / "dataset"
            sequence_dir = dataset_root / "Patient_1" / "Series_1"
            output_root = Path(temp_dir) / "dataset_keyFrames"
            sequence_dir.mkdir(parents=True)

            marker_image = np.full((64, 64, 3), 255, dtype=np.uint8)
            cv2.imwrite(str(sequence_dir / ".extract_complete.png"), marker_image)

            for index in range(5):
                image = np.full((64, 64, 3), 180, dtype=np.uint8)
                if index >= 2:
                    cv2.line(image, (8, 32), (56, 32), color=(40, 40, 40), thickness=index - 1)
                cv2.imwrite(str(sequence_dir / f"slice_{index:04d}.png"), image)

            results = extract_keyframes_from_root(
                input_path=dataset_root,
                output_root=output_root,
                limit=3,
                baseline_frames=2,
                smoothing_window=3,
                overwrite=True,
            )

            expected_output_dir = output_root / "Patient_1" / "Series_1"

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].output_dir, expected_output_dir)
            self.assertTrue(expected_output_dir.is_dir())
            self.assertFalse((expected_output_dir / ".extract_complete.png").exists())
            self.assertEqual(len(list(expected_output_dir.iterdir())), 3)

    def test_extract_keyframes_from_cadica_root_uses_selected_frame_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cadica_root = Path(temp_dir) / "CADICA"
            output_root = Path(temp_dir) / "cadica_keyframes"
            video_inputs = [
                (cadica_root / "selectedVideos" / "p1" / "v1" / "input", 2),
                (cadica_root / "selectedVideos" / "p1" / "v2" / "input", 3),
            ]

            for video_input, selected_count in video_inputs:
                video_input.mkdir(parents=True)
                patient_id = video_input.parent.parent.name
                video_id = video_input.parent.name
                selected_names = []
                for frame_index in range(5):
                    image = np.full((16, 16, 3), 180, dtype=np.uint8)
                    if frame_index >= 2:
                        cv2.line(image, (2, 8), (13, 8), color=(40, 40, 40), thickness=frame_index)
                    image_name = f"{patient_id}_{video_id}_{frame_index:05d}.png"
                    cv2.imwrite(str(video_input / image_name), image)
                    if len(selected_names) < selected_count:
                        selected_names.append(image_name)
                selected_frames_path = video_input.parent / f"{patient_id}_{video_id}_selectedFrames.txt"
                selected_frames_path.write_text("\n".join(selected_names), encoding="utf-8")

            results = extract_keyframes_from_root(
                input_path=cadica_root,
                output_root=output_root,
                baseline_frames=1,
                smoothing_window=1,
                overwrite=True,
                cadica_selected_frame_counts=True,
            )

            output_counts = {
                result.output_dir.relative_to(output_root).as_posix(): result.selected_count
                for result in results
            }

            self.assertEqual(output_counts, {"p1/v1": 2, "p1/v2": 3})
            self.assertFalse((output_root / "selectedVideos").exists())
            self.assertEqual(len(list((output_root / "p1" / "v1").glob("*.png"))), 2)
            self.assertEqual(len(list((output_root / "p1" / "v2").glob("*.png"))), 3)

    def test_extract_keyframes_from_root_copies_patient_metadata_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir) / "dataset"
            patient_dir = dataset_root / "Patient_1"
            frames_dir = patient_dir / "Coro_1" / "frames"
            output_root = Path(temp_dir) / "dataset_keyFrames"
            frames_dir.mkdir(parents=True)
            views_path = patient_dir / "views.json"
            views_path.write_text('{"view":"LAO"}', encoding="utf-8")
            patient_path = patient_dir / "patient.json"
            patient_path.write_text('{"patient_id":"Patient_1"}', encoding="utf-8")

            for index in range(4):
                image = np.full((16, 16, 3), 180, dtype=np.uint8)
                cv2.imwrite(str(frames_dir / f"frame_{index:05d}.png"), image)

            extract_keyframes_from_root(
                input_path=dataset_root,
                output_root=output_root,
                overwrite=True,
            )

            mirrored_views_path = output_root / "Patient_1" / "views.json"
            self.assertTrue(mirrored_views_path.is_file())
            self.assertEqual(mirrored_views_path.read_text(encoding="utf-8"), '{"view":"LAO"}')
            mirrored_patient_path = output_root / "Patient_1" / "patient.json"
            self.assertTrue(mirrored_patient_path.is_file())
            self.assertEqual(
                mirrored_patient_path.read_text(encoding="utf-8"),
                '{"patient_id":"Patient_1"}',
            )

    def test_extract_keyframes_from_root_skips_existing_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir) / "dataset"
            frames_dir = dataset_root / "Patient_1" / "Coro_1" / "frames"
            output_root = Path(temp_dir) / "dataset_keyFrames"
            output_dir = output_root / "Patient_1" / "Coro_1"
            frames_dir.mkdir(parents=True)
            output_dir.mkdir(parents=True)

            for index in range(4):
                image = np.full((16, 16, 3), 180, dtype=np.uint8)
                cv2.imwrite(str(frames_dir / f"frame_{index:05d}.png"), image)

            existing_output = np.full((16, 16), 90, dtype=np.uint8)
            cv2.imwrite(str(output_dir / "frame_00099.png"), existing_output)

            results = extract_keyframes_from_root(
                input_path=dataset_root,
                output_root=output_root,
                skip_existing=True,
            )

            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].skipped)
            self.assertEqual(results[0].selected_count, 1)
            self.assertEqual(sorted(path.name for path in output_dir.iterdir()), ["frame_00099.png"])

    def test_extract_keyframes_from_root_copies_patient_metadata_files_when_skipping_existing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir) / "dataset"
            patient_dir = dataset_root / "Patient_1"
            frames_dir = patient_dir / "Coro_1" / "frames"
            output_root = Path(temp_dir) / "dataset_keyFrames"
            output_dir = output_root / "Patient_1" / "Coro_1"
            frames_dir.mkdir(parents=True)
            output_dir.mkdir(parents=True)
            (patient_dir / "views.json").write_text('{"view":"RAO"}', encoding="utf-8")
            (patient_dir / "patient.json").write_text('{"patient_id":"Patient_1"}', encoding="utf-8")

            for index in range(4):
                image = np.full((16, 16, 3), 180, dtype=np.uint8)
                cv2.imwrite(str(frames_dir / f"frame_{index:05d}.png"), image)

            existing_output = np.full((16, 16), 90, dtype=np.uint8)
            cv2.imwrite(str(output_dir / "frame_00099.png"), existing_output)

            extract_keyframes_from_root(
                input_path=dataset_root,
                output_root=output_root,
                skip_existing=True,
            )

            mirrored_views_path = output_root / "Patient_1" / "views.json"
            self.assertTrue(mirrored_views_path.is_file())
            self.assertEqual(mirrored_views_path.read_text(encoding="utf-8"), '{"view":"RAO"}')
            mirrored_patient_path = output_root / "Patient_1" / "patient.json"
            self.assertTrue(mirrored_patient_path.is_file())
            self.assertEqual(
                mirrored_patient_path.read_text(encoding="utf-8"),
                '{"patient_id":"Patient_1"}',
            )

    def test_multi_worker_execution_matches_single_worker_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir) / "dataset"
            single_output_root = Path(temp_dir) / "single_output"
            multi_output_root = Path(temp_dir) / "multi_output"
            sequence_roots = [
                dataset_root / "Patient_1" / "Coro_1" / "frames",
                dataset_root / "Patient_2" / "Coro_2" / "frames",
            ]

            for sequence_index, frames_dir in enumerate(sequence_roots):
                frames_dir.mkdir(parents=True)
                for frame_index in range(6):
                    image = np.full((64, 64, 3), 180, dtype=np.uint8)
                    if frame_index >= 2:
                        offset = 8 + (sequence_index * 6)
                        cv2.line(
                            image,
                            (8, 20 + offset),
                            (56, 20 + offset),
                            color=(40, 40, 40),
                            thickness=frame_index - 1,
                        )
                    cv2.imwrite(str(frames_dir / f"frame_{frame_index:05d}.png"), image)

            extract_keyframes_from_root(
                input_path=dataset_root,
                output_root=single_output_root,
                limit=3,
                baseline_frames=2,
                smoothing_window=3,
                workers=1,
                overwrite=True,
            )
            extract_keyframes_from_root(
                input_path=dataset_root,
                output_root=multi_output_root,
                limit=3,
                baseline_frames=2,
                smoothing_window=3,
                workers=2,
                overwrite=True,
            )

            single_paths = sorted(
                path.relative_to(single_output_root)
                for path in single_output_root.rglob("*.png")
            )
            multi_paths = sorted(
                path.relative_to(multi_output_root)
                for path in multi_output_root.rglob("*.png")
            )

            self.assertEqual(single_paths, multi_paths)

            for relative_path in single_paths:
                single_image = load_grayscale_image(single_output_root / relative_path)
                multi_image = load_grayscale_image(multi_output_root / relative_path)
                np.testing.assert_array_equal(single_image, multi_image)

    def test_extract_keyframes_from_root_updates_tqdm_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir) / "dataset"
            sequence_roots = [
                dataset_root / "Patient_1" / "Coro_1" / "frames",
                dataset_root / "Patient_2" / "Coro_2" / "frames",
            ]

            for frames_dir in sequence_roots:
                frames_dir.mkdir(parents=True)
                for frame_index in range(3):
                    image = np.full((16, 16, 3), 180, dtype=np.uint8)
                    cv2.imwrite(str(frames_dir / f"frame_{frame_index:05d}.png"), image)

            class FakeTqdm:
                instances: list["FakeTqdm"] = []

                def __init__(self, *args: object, **kwargs: object) -> None:
                    self.total = kwargs["total"]
                    self.desc = kwargs["desc"]
                    self.unit = kwargs["unit"]
                    self.dynamic_ncols = kwargs["dynamic_ncols"]
                    self.updates: list[int] = []
                    FakeTqdm.instances.append(self)

                def __enter__(self) -> "FakeTqdm":
                    return self

                def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
                    return False

                def update(self, value: int = 1) -> None:
                    self.updates.append(value)

            with patch("angio_keyframes.pipeline.tqdm", FakeTqdm):
                extract_keyframes_from_root(
                    input_path=dataset_root,
                    output_root=Path(temp_dir) / "output",
                    overwrite=True,
                )

            self.assertEqual(len(FakeTqdm.instances), 1)
            progress = FakeTqdm.instances[0]
            self.assertEqual(progress.total, 2)
            self.assertEqual(progress.desc, "Sequences")
            self.assertEqual(progress.unit, "seq")
            self.assertTrue(progress.dynamic_ncols)
            self.assertEqual(sum(progress.updates), 2)

    def test_overwrite_and_skip_existing_are_mutually_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "overwrite and skip_existing"):
                extract_keyframes_from_root(
                    input_path=Path(temp_dir),
                    output_root=Path(temp_dir) / "output",
                    overwrite=True,
                    skip_existing=True,
                )

    def test_cuda_backend_rejects_multi_worker_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "cpu backend"):
                extract_keyframes_from_root(
                    input_path=Path(temp_dir),
                    output_root=Path(temp_dir) / "output",
                    backend="cuda",
                    workers=2,
                )

    def test_cpu_and_cuda_backends_select_same_frames(self) -> None:
        if not is_cuda_available():
            self.skipTest(get_cuda_unavailable_reason() or "CUDA backend is unavailable.")

        with tempfile.TemporaryDirectory() as temp_dir:
            frames_dir = Path(temp_dir) / "Patient_1" / "Coro_1" / "frames"
            frames_dir.mkdir(parents=True)

            vessel_segments = [
                (),
                (),
                (((8, 32), (56, 32), 3),),
                (((8, 24), (56, 24), 3), ((8, 40), (56, 40), 3)),
                (
                    ((8, 10), (56, 10), 3),
                    ((8, 18), (56, 18), 3),
                    ((8, 32), (56, 32), 3),
                    ((8, 46), (56, 46), 3),
                ),
                (((8, 24), (56, 24), 3), ((8, 40), (56, 40), 3)),
            ]

            for index, segments in enumerate(vessel_segments):
                image = np.full((64, 64, 3), 180, dtype=np.uint8)
                for start, end, thickness in segments:
                    cv2.line(image, start, end, color=(40, 40, 40), thickness=thickness)
                cv2.imwrite(str(frames_dir / f"frame_{index:05d}.png"), image)

            cpu_backend = create_backend("cpu")
            cuda_backend = create_backend("cuda")

            cpu_candidates = cpu_backend.score_frame_directory(frames_dir, baseline_frames=3)
            cuda_candidates = cuda_backend.score_frame_directory(frames_dir, baseline_frames=3)

            cpu_selected = select_keyframe_window(cpu_candidates, limit=4, smoothing_window=3)
            cuda_selected = select_keyframe_window(cuda_candidates, limit=4, smoothing_window=3)

            self.assertEqual(
                [candidate.frame_index for candidate in cpu_selected],
                [candidate.frame_index for candidate in cuda_selected],
            )


if __name__ == "__main__":
    unittest.main()

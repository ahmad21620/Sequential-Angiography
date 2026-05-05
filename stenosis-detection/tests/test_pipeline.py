from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stenosis_detection.pipeline import (
    PipelineConfig,
    _filter_stenosis_points_near_branch_points,
    _load_mask_image,
    run_stenosis_detection_variants,
)


class PipelineMaskLoadingTests(unittest.TestCase):
    def test_load_mask_returns_clean_binary_mask_and_removes_artifacts(self) -> None:
        mask = np.zeros((40, 60), dtype=np.uint8)
        mask[10:20, 10:15] = 255
        mask[2, 2] = 255
        mask[5, 5] = 80
        mask[-1, :] = 255

        config = PipelineConfig(
            resize_height=40,
            resize_width=60,
            min_component_area=4,
            border_margin_px=1,
            border_artifact_max_height=2,
            border_artifact_min_width_ratio=0.5,
        )

        mask_gray, binary_mask = self._load_temp_mask(mask, config)

        self.assertEqual(mask_gray.dtype, np.uint8)
        self.assertEqual(binary_mask.dtype, np.bool_)
        self.assertEqual(set(np.unique(mask_gray).tolist()), {0, 255})
        self.assertTrue(binary_mask[12, 12])
        self.assertFalse(binary_mask[2, 2])
        self.assertFalse(binary_mask[5, 5])
        self.assertFalse(np.any(binary_mask[-1, :]))

    def test_load_mask_preserves_non_artifact_border_touching_component(self) -> None:
        mask = np.zeros((40, 60), dtype=np.uint8)
        mask[25:40, 22:32] = 255

        config = PipelineConfig(
            resize_height=40,
            resize_width=60,
            min_component_area=4,
            border_margin_px=1,
            border_artifact_max_height=2,
            border_artifact_min_width_ratio=0.5,
        )

        mask_gray, binary_mask = self._load_temp_mask(mask, config)

        self.assertEqual(set(np.unique(mask_gray).tolist()), {0, 255})
        self.assertTrue(np.any(binary_mask[25:40, 22:32]))
        self.assertTrue(binary_mask[-1, 26])

    def test_load_mask_removes_long_bottom_line_even_with_vertical_stub(self) -> None:
        mask = np.zeros((40, 80), dtype=np.uint8)
        mask[8:18, 8:14] = 255
        mask[-1, :] = 255
        mask[-10:, -1] = 255

        config = PipelineConfig(
            resize_height=40,
            resize_width=80,
            min_component_area=4,
            border_margin_px=1,
            border_artifact_max_height=2,
            border_artifact_min_width_ratio=0.5,
        )

        _, binary_mask = self._load_temp_mask(mask, config)

        self.assertTrue(np.any(binary_mask[8:18, 8:14]))
        self.assertFalse(np.any(binary_mask[-1, :]))
        self.assertFalse(np.any(binary_mask[-10:, -1]))

    def test_load_mask_removes_wide_ten_pixel_bottom_artifact_with_defaults(self) -> None:
        mask = np.zeros((800, 600), dtype=np.uint8)
        mask[100:140, 100:112] = 255
        mask[790:800, 29:600] = 255
        mask[793:800, 0:17] = 255

        config = PipelineConfig(
            resize_height=800,
            resize_width=600,
        )

        _, binary_mask = self._load_temp_mask(mask, config)

        self.assertTrue(np.any(binary_mask[100:140, 100:112]))
        self.assertFalse(np.any(binary_mask[790:800, 29:600]))
        self.assertFalse(np.any(binary_mask[790:800, :]))

    def test_load_mask_preserves_wide_thick_border_touching_component(self) -> None:
        mask = np.zeros((40, 80), dtype=np.uint8)
        mask[-12:, 10:70] = 255

        config = PipelineConfig(
            resize_height=40,
            resize_width=80,
            min_component_area=4,
            border_margin_px=1,
            border_artifact_max_height=2,
            border_artifact_min_width_ratio=0.5,
        )

        _, binary_mask = self._load_temp_mask(mask, config)

        self.assertTrue(np.all(binary_mask[-12:, 10:70]))

    def test_load_mask_uses_nearest_neighbor_resize(self) -> None:
        mask = np.asarray(
            [
                [0, 255],
                [0, 0],
            ],
            dtype=np.uint8,
        )
        config = PipelineConfig(
            resize_height=4,
            resize_width=4,
            min_component_area=1,
            remove_border_artifacts=False,
        )

        mask_gray, binary_mask = self._load_temp_mask(mask, config)

        expected = np.asarray(
            [
                [0, 0, 255, 255],
                [0, 0, 255, 255],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(mask_gray, expected)
        np.testing.assert_array_equal(binary_mask, expected.astype(bool))

    def test_filter_stenosis_points_near_branch_points_removes_nearby_candidates(self) -> None:
        stenosis_points_xy = np.asarray([[10, 10], [20, 20], [40, 40]], dtype=np.int32)
        stenosis_degrees = np.asarray([0.7, 0.8, 0.9], dtype=np.float64)
        branch_points_xy = np.asarray([[12, 10], [35, 35]], dtype=np.int32)

        filtered_points_xy, filtered_degrees = _filter_stenosis_points_near_branch_points(
            stenosis_points_xy,
            stenosis_degrees,
            branch_points_xy,
            exclusion_distance=5.0,
        )

        np.testing.assert_array_equal(filtered_points_xy, np.asarray([[20, 20], [40, 40]], dtype=np.int32))
        np.testing.assert_array_equal(filtered_degrees, np.asarray([0.8, 0.9], dtype=np.float64))

    def test_branch_point_filter_can_be_disabled(self) -> None:
        stenosis_points_xy = np.asarray([[10, 10]], dtype=np.int32)
        stenosis_degrees = np.asarray([0.7], dtype=np.float64)
        branch_points_xy = np.asarray([[10, 10]], dtype=np.int32)

        filtered_points_xy, filtered_degrees = _filter_stenosis_points_near_branch_points(
            stenosis_points_xy,
            stenosis_degrees,
            branch_points_xy,
            exclusion_distance=0.0,
        )

        np.testing.assert_array_equal(filtered_points_xy, stenosis_points_xy)
        np.testing.assert_array_equal(filtered_degrees, stenosis_degrees)

    def test_detection_variants_can_vary_radius_and_threshold_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "p1_v1_00001.png"
            mask_path = root / "p1_v1_00001_mask.png"
            image = np.zeros((16, 16), dtype=np.uint8)
            mask = np.zeros((16, 16), dtype=np.uint8)
            cv2.line(mask, (2, 8), (13, 8), color=255, thickness=1)
            self.assertTrue(cv2.imwrite(str(image_path), image))
            self.assertTrue(cv2.imwrite(str(mask_path), mask))

            configs = [
                PipelineConfig(
                    resize_height=16,
                    resize_width=16,
                    remove_border_artifacts=False,
                    radius_outside_fraction_threshold=0.05,
                    radius_min_outside_samples=2,
                    stenosis_threshold=0.25,
                    average_radius_threshold=4.0,
                ),
                PipelineConfig(
                    resize_height=16,
                    resize_width=16,
                    remove_border_artifacts=False,
                    radius_outside_fraction_threshold=0.10,
                    radius_min_outside_samples=3,
                    stenosis_threshold=0.35,
                    average_radius_threshold=5.0,
                ),
            ]

            results = run_stenosis_detection_variants(image_path, mask_path, configs)

        self.assertEqual([result.config for result in results], configs)

    def _load_temp_mask(self, mask: np.ndarray, config: PipelineConfig) -> tuple[np.ndarray, np.ndarray]:
        with tempfile.TemporaryDirectory() as temp_dir:
            mask_path = Path(temp_dir) / "mask.png"
            self.assertTrue(cv2.imwrite(str(mask_path), mask))
            return _load_mask_image(mask_path, config)


if __name__ == "__main__":
    unittest.main()

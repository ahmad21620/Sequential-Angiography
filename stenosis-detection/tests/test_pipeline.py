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

from stenosis_detection.pipeline import PipelineConfig, _load_mask_image


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

    def _load_temp_mask(self, mask: np.ndarray, config: PipelineConfig) -> tuple[np.ndarray, np.ndarray]:
        with tempfile.TemporaryDirectory() as temp_dir:
            mask_path = Path(temp_dir) / "mask.png"
            self.assertTrue(cv2.imwrite(str(mask_path), mask))
            return _load_mask_image(mask_path, config)


if __name__ == "__main__":
    unittest.main()

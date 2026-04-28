from __future__ import annotations

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

from stenosis_detection.pipeline import PipelineConfig, StenosisDetectionResult
from stenosis_detection.visualization import save_detection_outputs


class VisualizationOutputTests(unittest.TestCase):
    def test_save_detection_outputs_writes_cleaned_mask_and_radius_debug(self) -> None:
        original_mask_gray = np.zeros((6, 6), dtype=np.uint8)
        original_mask_gray[1:4, 2:4] = 255
        original_mask_gray[-1, :] = 255

        cleaned_mask_gray = np.zeros((6, 6), dtype=np.uint8)
        cleaned_mask_gray[1:4, 2:4] = 255

        result = StenosisDetectionResult(
            image_path=Path("slice_0001.png"),
            mask_path=Path("slice_0001_mask.png"),
            config=PipelineConfig(resize_height=6, resize_width=6),
            original_image_bgr=np.zeros((6, 6, 3), dtype=np.uint8),
            original_mask_gray=original_mask_gray,
            mask_gray=cleaned_mask_gray,
            binary_mask=cleaned_mask_gray > 0,
            skeleton_mask=cleaned_mask_gray > 0,
            skeleton_points_rc=np.asarray([[2, 3], [3, 3], [4, 3]], dtype=np.int32),
            skeleton_points_xy=np.asarray([[3, 2], [3, 3], [3, 4]], dtype=np.int32),
            point_data={(2, 3): 1.0, (3, 3): 2.0, (4, 3): 3.0},
            segmentation_points_xy=np.zeros((0, 2), dtype=np.int32),
            filtered_segmentation_points_xy=np.zeros((0, 2), dtype=np.int32),
            stenosis_points_xy=np.zeros((0, 2), dtype=np.int32),
            stenosis_degrees=np.zeros((0,), dtype=np.float64),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_files = save_detection_outputs(result, temp_dir)

            for key in (
                "loaded_mask",
                "cleaned_mask",
                "centerline",
                "segmentation_points",
                "stenosis_mask",
                "stenosis_original",
                "radius_debug",
                "results_json",
            ):
                self.assertIn(key, output_files)
                self.assertTrue(output_files[key].exists(), key)

            cleaned_mask = cv2.imread(str(output_files["cleaned_mask"]), cv2.IMREAD_GRAYSCALE)
            np.testing.assert_array_equal(cleaned_mask, cleaned_mask_gray)

            centerline = cv2.imread(str(output_files["centerline"]), cv2.IMREAD_COLOR)
            self.assertEqual(centerline[-1, 0].tolist(), [0, 0, 0])

            payload = json.loads(output_files["results_json"].read_text(encoding="utf-8"))
            self.assertNotIn("debug", payload)
            self.assertNotIn("point_data", payload)

    def test_save_detection_outputs_can_write_json_only(self) -> None:
        result = StenosisDetectionResult(
            image_path=Path("slice_0001.png"),
            mask_path=Path("slice_0001_mask.png"),
            config=PipelineConfig(resize_height=4, resize_width=4),
            original_image_bgr=np.zeros((4, 4, 3), dtype=np.uint8),
            original_mask_gray=np.zeros((4, 4), dtype=np.uint8),
            mask_gray=np.zeros((4, 4), dtype=np.uint8),
            binary_mask=np.zeros((4, 4), dtype=bool),
            skeleton_mask=np.zeros((4, 4), dtype=bool),
            skeleton_points_rc=np.zeros((0, 2), dtype=np.int32),
            skeleton_points_xy=np.zeros((0, 2), dtype=np.int32),
            point_data={},
            segmentation_points_xy=np.zeros((0, 2), dtype=np.int32),
            filtered_segmentation_points_xy=np.zeros((0, 2), dtype=np.int32),
            stenosis_points_xy=np.zeros((0, 2), dtype=np.int32),
            stenosis_degrees=np.zeros((0,), dtype=np.float64),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_files = save_detection_outputs(result, temp_dir, write_debug_images=False)

            self.assertEqual(list(output_files), ["results_json"])
            self.assertTrue(output_files["results_json"].exists())
            self.assertEqual(len(list(Path(temp_dir).iterdir())), 1)


if __name__ == "__main__":
    unittest.main()

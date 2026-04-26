from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stenosis_detection.radius import MoMforSeg1


class RadiusExtractionTests(unittest.TestCase):
    def test_one_black_pixel_inside_vessel_does_not_collapse_radius(self) -> None:
        clean_mask = self._circle_mask(radius=6)
        noisy_mask = clean_mask.copy()
        noisy_mask[15, 17] = 0

        clean_radius = MoMforSeg1(16, 16, 10.0, clean_mask)
        noisy_radius = MoMforSeg1(16, 16, 10.0, noisy_mask)

        self.assertEqual(noisy_radius, clean_radius)
        self.assertGreater(noisy_radius, 3.0)

    def test_real_boundary_still_stops_radius_search(self) -> None:
        mask = self._circle_mask(radius=6)

        radius = MoMforSeg1(16, 16, 10.0, mask)

        self.assertGreaterEqual(radius, 5.0)
        self.assertLessEqual(radius, 7.0)

    def _circle_mask(self, radius: int) -> np.ndarray:
        mask = np.zeros((31, 31), dtype=np.uint8)
        cv2.circle(mask, (15, 15), radius, 255, thickness=-1)
        return mask


if __name__ == "__main__":
    unittest.main()

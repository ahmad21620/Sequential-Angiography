from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stenosis_detection.parameter_sweep import build_frame_sweep_variants, build_temporal_sweep_variants
from stenosis_detection.pipeline import PipelineConfig


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


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stenosis_detection import cli
from stenosis_detection.pipeline import PipelineConfig


class StenosisDetectionCliTests(unittest.TestCase):
    def test_parser_defaults_match_pipeline_config_defaults(self) -> None:
        parser = cli.build_parser()
        config = cli._build_pipeline_config(parser.parse_args([]))

        self.assertEqual(config, PipelineConfig())

    def test_parser_passes_mask_cleanup_and_radius_options_to_config(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(
            [
                "--mask-threshold",
                "101",
                "--min-component-area",
                "42",
                "--no-remove-border-artifacts",
                "--border-margin-px",
                "7",
                "--border-artifact-max-height",
                "5",
                "--border-artifact-min-width-ratio",
                "0.75",
                "--radius-vessel-threshold",
                "111",
                "--radius-outside-fraction-threshold",
                "0.12",
                "--radius-min-outside-samples",
                "9",
                "--branch-point-exclusion-distance",
                "6.5",
            ]
        )

        config = cli._build_pipeline_config(args)

        self.assertEqual(config.mask_threshold, 101)
        self.assertEqual(config.min_component_area, 42)
        self.assertFalse(config.remove_border_artifacts)
        self.assertEqual(config.border_margin_px, 7)
        self.assertEqual(config.border_artifact_max_height, 5)
        self.assertEqual(config.border_artifact_min_width_ratio, 0.75)
        self.assertEqual(config.radius_vessel_threshold, 111)
        self.assertEqual(config.radius_outside_fraction_threshold, 0.12)
        self.assertEqual(config.radius_min_outside_samples, 9)
        self.assertEqual(config.branch_point_exclusion_distance, 6.5)

    def test_parser_accepts_workers_option(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(["--workers", "4"])

        self.assertEqual(args.workers, 4)

    def test_builds_threshold_variants_from_comma_lists(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(
            [
                "--stenosis-thresholds",
                "0.2,0.3",
                "--average-radius-thresholds",
                "3.0,4.0",
            ]
        )

        variants = cli._build_threshold_variants(args, cli._build_pipeline_config(args))

        self.assertIsNotNone(variants)
        self.assertEqual(
            [name for name, _ in variants],
            [
                "stenosis_threshold_0p2__average_radius_threshold_3",
                "stenosis_threshold_0p2__average_radius_threshold_4",
                "stenosis_threshold_0p3__average_radius_threshold_3",
                "stenosis_threshold_0p3__average_radius_threshold_4",
            ],
        )
        self.assertEqual([config.stenosis_threshold for _, config in variants], [0.2, 0.2, 0.3, 0.3])
        self.assertEqual([config.average_radius_threshold for _, config in variants], [3.0, 4.0, 3.0, 4.0])


if __name__ == "__main__":
    unittest.main()

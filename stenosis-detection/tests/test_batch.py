from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stenosis_detection.batch import TreeProcessingJob, process_tree
from stenosis_detection.pipeline import PipelineConfig


class BatchProcessingTests(unittest.TestCase):
    def test_process_tree_saves_each_threshold_variant(self) -> None:
        job = TreeProcessingJob(
            image_path=Path("images/case_001/view_01/slice_0001.png"),
            mask_path=Path("masks/case_001/view_01/slice_0001_mask.png"),
            relative_dir=Path("case_001/view_01"),
            image_stem="slice_0001",
        )
        variants = [
            ("stenosis_threshold_0p2__average_radius_threshold_4", PipelineConfig(stenosis_threshold=0.2)),
            ("stenosis_threshold_0p3__average_radius_threshold_4", PipelineConfig(stenosis_threshold=0.3)),
        ]
        fake_results = [object(), object()]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            with (
                patch("stenosis_detection.batch.run_stenosis_detection_variants", return_value=fake_results) as run_variants,
                patch("stenosis_detection.batch.save_detection_outputs", Mock()) as save_outputs,
            ):
                summary = process_tree(
                    [job],
                    output_root,
                    images_root="images",
                    masks_root="masks",
                    config=PipelineConfig(),
                    workers=1,
                    threshold_variants=variants,
                    write_debug_images=False,
                )

        self.assertEqual(summary.processed, 1)
        self.assertEqual(summary.threshold_variants, [name for name, _ in variants])
        run_variants.assert_called_once()
        self.assertEqual(save_outputs.call_count, 2)
        saved_dirs = [call.args[1] for call in save_outputs.call_args_list]
        self.assertEqual(
            saved_dirs,
            [
                output_root / "stenosis_threshold_0p2__average_radius_threshold_4" / "case_001" / "view_01",
                output_root / "stenosis_threshold_0p3__average_radius_threshold_4" / "case_001" / "view_01",
            ],
        )
        self.assertTrue(all(call.kwargs["write_debug_images"] is False for call in save_outputs.call_args_list))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from angio_keyframes.cli import main


class CliTests(unittest.TestCase):
    def test_positional_input_path_keeps_default_extraction_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_root = Path(temp_dir) / "dataset"
            output_root = Path(temp_dir) / "keyframes"

            with patch("angio_keyframes.cli.extract_keyframes_from_root", return_value=[]) as extract:
                exit_code = main(
                    [
                        str(input_root),
                        "--output-root",
                        str(output_root),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(extract.call_args.kwargs["input_path"], input_root)
            self.assertEqual(extract.call_args.kwargs["output_root"], output_root)
            self.assertEqual(extract.call_args.kwargs["frames_dirname"], "frames")
            self.assertEqual(extract.call_args.kwargs["window_mode"], "centered")
            self.assertFalse(extract.call_args.kwargs["cadica_selected_frame_counts"])

    def test_named_input_root_is_passed_to_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_root = Path(temp_dir) / "selectedVideos"
            output_root = Path(temp_dir) / "keyframes"

            with patch("angio_keyframes.cli.extract_keyframes_from_root", return_value=[]) as extract:
                exit_code = main(
                    [
                        "--input-root",
                        str(input_root),
                        "--output-root",
                        str(output_root),
                        "--frames-dirname",
                        "input",
                        "--cadica-selected-frame-counts",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(extract.call_args.kwargs["input_path"], input_root)
            self.assertEqual(extract.call_args.kwargs["output_root"], output_root)
            self.assertEqual(extract.call_args.kwargs["frames_dirname"], "input")
            self.assertTrue(extract.call_args.kwargs["cadica_selected_frame_counts"])

    def test_window_mode_is_passed_to_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_root = Path(temp_dir) / "dataset"
            output_root = Path(temp_dir) / "keyframes"

            with patch("angio_keyframes.cli.extract_keyframes_from_root", return_value=[]) as extract:
                exit_code = main(
                    [
                        str(input_root),
                        "--output-root",
                        str(output_root),
                        "--window-mode",
                        "leading",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(extract.call_args.kwargs["window_mode"], "leading")

    def test_cadica_selected_frame_counts_default_to_input_frames_dirname(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_root = Path(temp_dir) / "CADICA"
            output_root = Path(temp_dir) / "keyframes"

            with patch("angio_keyframes.cli.extract_keyframes_from_root", return_value=[]) as extract:
                exit_code = main(
                    [
                        "--input-root",
                        str(input_root),
                        "--output-root",
                        str(output_root),
                        "--cadica-selected-frame-counts",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(extract.call_args.kwargs["frames_dirname"], "input")
            self.assertTrue(extract.call_args.kwargs["cadica_selected_frame_counts"])


if __name__ == "__main__":
    unittest.main()

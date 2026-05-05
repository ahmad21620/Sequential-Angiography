from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from angio_keyframes.cadica_benchmark import read_selected_frames, run_cadica_keyframe_benchmark


class CadicaKeyframeBenchmarkTests(unittest.TestCase):
    def test_benchmark_compares_extracted_frames_to_cadica_selected_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            cadica_root = temp_root / "CADICA"
            extracted_root = temp_root / "extracted"
            output_root = temp_root / "benchmark"
            self._write_selected_frames(cadica_root / "selectedVideos" / "p1" / "v1" / "p1_v1_selectedFrames.txt", "p1_v1_00001.png, 3")
            self._write_selected_frames(cadica_root / "selectedVideos" / "p1" / "v2" / "p1_v2_selectedFrames.txt", "p1_v2_00002.png")
            self._write_image_placeholder(extracted_root / "p1" / "v1" / "p1_v1_00001.png")
            self._write_image_placeholder(extracted_root / "p1" / "v1" / "p1_v1_00004.png")

            result = run_cadica_keyframe_benchmark(
                cadica_root=cadica_root,
                extracted_keyframes_root=extracted_root,
                output_root=output_root,
            )

            self.assertTrue((output_root / "cadica_keyframe_frame_rows.csv").is_file())
            self.assertTrue((output_root / "cadica_keyframe_frame_rows.jsonl").is_file())
            self.assertTrue((output_root / "cadica_keyframe_video_rows.csv").is_file())
            self.assertTrue((output_root / "cadica_keyframe_summary.json").is_file())
            self.assertTrue((output_root / "missed_keyframes.csv").is_file())
            self.assertTrue((output_root / "extra_keyframes.csv").is_file())

            frame_outcomes = {
                (row["patient_id"], row["video_id"], int(row["frame_id"])): row["outcome"]
                for row in result.frame_rows
            }
            self.assertEqual(frame_outcomes[("p1", "v1", 1)], "TP")
            self.assertEqual(frame_outcomes[("p1", "v1", 3)], "FN")
            self.assertEqual(frame_outcomes[("p1", "v1", 4)], "FP")
            self.assertEqual(frame_outcomes[("p1", "v2", 2)], "FN")

            video_by_id = {row["video_id"]: row for row in result.video_rows}
            self.assertEqual(video_by_id["v1"]["gt_selected_count"], 2)
            self.assertEqual(video_by_id["v1"]["predicted_count"], 2)
            self.assertEqual(video_by_id["v1"]["matched_count"], 1)
            self.assertEqual(video_by_id["v1"]["missed_frame_ids"], [3])
            self.assertEqual(video_by_id["v1"]["extra_frame_ids"], [4])
            self.assertFalse(video_by_id["v1"]["exact_match"])
            self.assertEqual(video_by_id["v2"]["predicted_count"], 0)

            metrics = result.summary["metrics"]
            self.assertEqual(metrics["video_count"], 2)
            self.assertEqual(metrics["total_gt_selected_frames"], 3)
            self.assertEqual(metrics["total_predicted_keyframes"], 2)
            self.assertEqual(metrics["matched_frames"], 1)
            self.assertEqual(metrics["missed_frames"], 2)
            self.assertEqual(metrics["extra_frames"], 1)
            self.assertAlmostEqual(metrics["precision"] or 0.0, 0.5)
            self.assertAlmostEqual(metrics["recall"] or 0.0, 1 / 3)

            self.assertEqual(len(self._read_csv(output_root / "missed_keyframes.csv")), 2)
            self.assertEqual(len(self._read_csv(output_root / "extra_keyframes.csv")), 1)
            summary_payload = json.loads((output_root / "cadica_keyframe_summary.json").read_text(encoding="utf-8"))
            self.assertIn("CADICA selectedFrames", summary_payload["note"])

    def test_read_selected_frames_accepts_filenames_and_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            selected_frames_path = Path(temp_dir) / "p1_v1_selectedFrames.txt"
            selected_frames_path.write_text("p1_v1_00001.png\n2, 00003", encoding="utf-8")

            self.assertEqual(read_selected_frames(selected_frames_path), {1, 2, 3})

    def _write_selected_frames(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _write_image_placeholder(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")

    def _read_csv(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()

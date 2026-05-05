from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stenosis_detection.cadica import load_cadica_annotations, prepare_cadica_for_pipeline
from stenosis_detection.cadica.annotations import read_groundtruth_boxes, read_selected_frames


class CadicaPreparationTests(unittest.TestCase):
    def test_load_cadica_annotations_labels_selected_videos_conservatively(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cadica_root = self._write_tiny_cadica_tree(Path(temp_dir) / "CADICA")

            frames = load_cadica_annotations(
                cadica_root,
                frame_scope="all_selected_videos",
                negative_frame_scope="selected",
            )

            labels = {(frame.video_id, frame.frame_id): frame.frame_label for frame in frames}
            self.assertEqual(len(frames), 4)
            self.assertEqual(labels[("v1", 1)], "positive")
            self.assertEqual(labels[("v1", 2)], "unknown")
            self.assertEqual(labels[("v2", 1)], "negative")
            self.assertEqual(labels[("v2", 2)], "unknown")
            self.assertTrue(frames[0].is_selected_frame)
            self.assertEqual(frames[0].prepared_image_name, "slice_00001.png")
            self.assertEqual(frames[0].boxes[0].category, "stenosis")
            self.assertEqual(frames[0].boxes[0].x, 10.0)

            selected_frames = read_selected_frames(cadica_root / "selectedVideos" / "p1" / "v1" / "p1_v1_selectedFrames.txt")
            self.assertEqual(selected_frames, {1, 2})

            boxes = read_groundtruth_boxes(cadica_root / "selectedVideos" / "p1" / "v1" / "groundtruth", "p1", "v1")
            self.assertEqual(len(boxes[1]), 1)
            self.assertEqual(boxes[1][0].w, 30.0)

    def test_prepare_cadica_writes_pipeline_tree_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            cadica_root = self._write_tiny_cadica_tree(temp_root / "CADICA")
            output_root = temp_root / "prepared"

            outputs = prepare_cadica_for_pipeline(
                cadica_root,
                output_root,
                frame_scope="all_selected_videos",
                negative_frame_scope="selected",
                copy_mode="copy",
            )

            self.assertEqual(outputs["keyframes_root"], output_root / "keyframes")
            self.assertTrue((output_root / "keyframes" / "p1" / "v1" / "slice_00001.png").is_file())
            self.assertTrue((output_root / "keyframes" / "p1" / "v1" / "slice_00002.png").is_file())
            self.assertTrue((output_root / "keyframes" / "p1" / "v2" / "slice_00001.png").is_file())
            self.assertTrue((output_root / "keyframes" / "p1" / "patient.json").is_file())
            self.assertTrue((output_root / "keyframes" / "p1" / "views.json").is_file())

            rows = self._read_csv(output_root / "manifest.csv")
            self.assertEqual(len(rows), 4)
            row_by_video_frame = {(row["video_id"], int(row["frame_id"])): row for row in rows}
            positive_row = row_by_video_frame[("v1", 1)]
            self.assertEqual(positive_row["prepared_relative_path"], "keyframes/p1/v1/slice_00001.png")
            self.assertEqual(positive_row["prepared_image_name"], "slice_00001.png")
            self.assertEqual(positive_row["video_label"], "lesion")
            self.assertEqual(positive_row["frame_label"], "positive")
            self.assertEqual(positive_row["box_count"], "1")
            self.assertEqual(json.loads(positive_row["gt_boxes_json"])[0]["category"], "stenosis")
            self.assertEqual(row_by_video_frame[("v2", 1)]["frame_label"], "negative")
            self.assertEqual(row_by_video_frame[("v2", 2)]["frame_label"], "unknown")

            jsonl_rows = [
                json.loads(line)
                for line in (output_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertEqual(len(jsonl_rows), 4)
            self.assertEqual(jsonl_rows[0]["prepared_image_name"], "slice_00001.png")

            summary = json.loads((output_root / "preparation_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["patient_count"], 1)
            self.assertEqual(summary["video_count"], 2)
            self.assertEqual(summary["frame_label_counts"], {"negative": 1, "positive": 1, "unknown": 2})

            views = json.loads((output_root / "keyframes" / "p1" / "views.json").read_text(encoding="utf-8"))
            self.assertTrue(views["metadata"]["projection_angles_placeholder"])
            self.assertEqual(
                views["views"][0]["temporal_fusion_json"],
                "../../stenosis_temporal_results/p1/v1/view_temporal_fusion.json",
            )

    def _write_tiny_cadica_tree(self, cadica_root: Path) -> Path:
        patient_dir = cadica_root / "selectedVideos" / "p1"
        lesion_video = patient_dir / "v1"
        nonlesion_video = patient_dir / "v2"
        (lesion_video / "input").mkdir(parents=True)
        (lesion_video / "groundtruth").mkdir()
        (nonlesion_video / "input").mkdir(parents=True)
        (cadica_root / "nonselectedVideos").mkdir(parents=True)

        (patient_dir / "lesionVideos.txt").write_text("v1\n", encoding="utf-8")
        (patient_dir / "nonlesionVideos.txt").write_text("2\n", encoding="utf-8")
        (lesion_video / "p1_v1_selectedFrames.txt").write_text("p1_v1_00001.png, 00002\n", encoding="utf-8")
        (nonlesion_video / "p1_v2_selectedFrames.txt").write_text("p1_v2_00001.png\n", encoding="utf-8")

        for frame_id in (1, 2):
            (lesion_video / "input" / f"p1_v1_{frame_id:05d}.png").write_bytes(f"lesion-{frame_id}".encode())
            (nonlesion_video / "input" / f"p1_v2_{frame_id:05d}.png").write_bytes(f"nonlesion-{frame_id}".encode())

        (lesion_video / "groundtruth" / "p1_v1_00001.txt").write_text("10;20;30;40;stenosis\n", encoding="utf-8")
        return cadica_root

    def _read_csv(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()

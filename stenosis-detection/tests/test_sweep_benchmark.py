from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_sweep_benchmark import run_sweep_benchmark
from stenosis_detection.yolo.schema import YoloDetection, YoloDetectorMetadata, build_yolo_frame_payload


class SweepBenchmarkTests(unittest.TestCase):
    def test_run_sweep_benchmark_writes_outputs_for_selected_levels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            sweep_root = temp_root / "sweep"
            weak_labels = temp_root / "weak_labels.jsonl"
            output_root = temp_root / "benchmark"
            frame_variant = "radius_outside_fraction_threshold_0p1"
            temporal_variant = "min_supporting_frames_2__min_persistence_ratio_0p2"

            _write_weak_labels(weak_labels)
            _write_frame_result(
                sweep_root
                / "frame_results"
                / frame_variant
                / "case_1"
                / "view_1"
                / "slice_0001_stenosis_results.json"
            )
            _write_temporal_result(
                sweep_root
                / "temporal_results"
                / frame_variant
                / temporal_variant
                / "case_1"
                / "view_1"
                / "view_temporal_fusion.json"
            )
            _write_multiview_result(
                sweep_root
                / "multiview_results"
                / frame_variant
                / temporal_variant
                / "case_1"
                / "case_multiview_fusion.json"
            )

            summary = run_sweep_benchmark(
                sweep_root=sweep_root,
                weak_labels_path=weak_labels,
                output_root=output_root,
                levels=("frame", "temporal", "multiview"),
                workers=2,
            )

            self.assertEqual(summary["completed_jobs"], 3)
            self.assertEqual(summary["failed_jobs"], 0)
            self.assertEqual(summary["config"]["workers"], 2)
            self.assertTrue((output_root / "frame" / frame_variant / "frame_summary.json").is_file())
            self.assertTrue(
                (
                    output_root
                    / "temporal"
                    / frame_variant
                    / temporal_variant
                    / "temporal_sequence_summary.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    output_root
                    / "multiview"
                    / frame_variant
                    / temporal_variant
                    / "multiview_case_summary.json"
                ).is_file()
            )
            self.assertTrue((output_root / "sweep_benchmark_summary.json").is_file())

    def test_run_sweep_benchmark_accepts_yolo_frame_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            sweep_root = temp_root / "yolo_sweep"
            weak_labels = temp_root / "weak_labels.jsonl"
            output_root = temp_root / "benchmark"
            frame_variant = "detector_yolo__yolo_conf_0p25__yolo_iou_0p70__yolo_imgsz_1024"

            _write_weak_labels(weak_labels)
            _write_yolo_frame_result(
                sweep_root
                / "frame_results"
                / frame_variant
                / "case_1"
                / "view_1"
                / "slice_0001_stenosis_results.json"
            )

            summary = run_sweep_benchmark(
                sweep_root=sweep_root,
                weak_labels_path=weak_labels,
                output_root=output_root,
                levels=("frame",),
                write_threshold_sweep_report=True,
            )

            self.assertEqual(summary["completed_jobs"], 1)
            self.assertEqual(summary["failed_jobs"], 0)
            frame_summary = json.loads(
                (output_root / "frame" / frame_variant / "frame_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(frame_summary["summary"]["TP"], 1)
            self.assertTrue((output_root / "frame" / frame_variant / "threshold_sweep_frame.csv").is_file())
            frame_rows = [
                json.loads(line)
                for line in (output_root / "frame" / frame_variant / "frame_rows.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(frame_rows[0]["score"], 0.83)
            self.assertEqual(frame_rows[0]["max_stenosis_degree"], 0.83)


def _write_weak_labels(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "case_id": "case_1",
                "stenosis_exists": "yes",
                "severity": "moderate",
            }
        ),
        encoding="utf-8",
    )


def _write_frame_result(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "frame": {
                    "image_stem": "slice_0001",
                    "image_name": "slice_0001.png",
                    "frame_index": 1,
                },
                "stenosis_points": [
                    {
                        "degree": 0.7,
                        "severity": "moderate",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_yolo_frame_result(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    detection = YoloDetection(
        bbox_xyxy_zero_based=(99.0, 199.0, 144.0, 239.0),
        confidence=0.83,
        class_id=0,
        class_name="Stenosis",
        image_width=512,
        image_height=512,
    )
    payload = build_yolo_frame_payload(
        image_path=Path("work/keyframes/case_1/view_1/slice_0001.png"),
        mask_path=None,
        detector=YoloDetectorMetadata(
            name="yolov8",
            weights="runs/stenosis/best.pt",
            imgsz=1024,
            conf=0.25,
            iou=0.7,
            device="cpu",
        ),
        view_id="case_1/view_1",
        image_width=512,
        image_height=512,
        detections=[detection],
    )
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_temporal_result(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "view_id": "case_1/view_1",
                "fusion": {
                    "persistent_lesion_count": 1,
                },
                "final_lesion": {
                    "severity": "moderate",
                    "degrees": {
                        "median": 0.7,
                        "max": 0.8,
                    },
                    "persistence_ratio": 0.5,
                    "supporting_frame_count": 2,
                    "total_frame_count": 3,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_multiview_result(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "case_id": "case_1",
                "final_case_lesion": {
                    "severity": "moderate",
                    "degrees": {
                        "median": 0.7,
                        "max": 0.8,
                    },
                    "total_score": 0.9,
                    "distinct_supporting_view_ids": ["view_1"],
                },
                "confidence": {
                    "score": 0.8,
                    "label": "high",
                },
                "supporting_views": [{"view_id": "view_1"}],
                "fusion_metadata": {
                    "total_candidate_count": 1,
                },
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()

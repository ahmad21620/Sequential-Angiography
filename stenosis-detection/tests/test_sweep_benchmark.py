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
            )

            self.assertEqual(summary["completed_jobs"], 3)
            self.assertEqual(summary["failed_jobs"], 0)
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

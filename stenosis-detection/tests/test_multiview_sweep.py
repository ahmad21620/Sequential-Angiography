from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_multiview_sweep import run_multiview_sweep


class MultiViewSweepTests(unittest.TestCase):
    def test_multiview_sweep_runs_all_variants_with_one_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            sweep_root = temp_root / "sweep"
            case_tree = temp_root / "cases"
            frame_variants = ["frame_a", "frame_b"]
            temporal_variants = ["temporal_a", "temporal_b"]
            case_ids = ["case_1", "case_2"]

            for case_id in case_ids:
                _write_case(case_tree / case_id, case_id=case_id)
            for frame_variant in frame_variants:
                for temporal_variant in temporal_variants:
                    for case_id in case_ids:
                        _write_temporal_result(
                            sweep_root
                            / "temporal_results"
                            / frame_variant
                            / temporal_variant
                            / case_id
                            / "view_01"
                            / "view_temporal_fusion.json"
                        )

            summary = run_multiview_sweep(
                sweep_root=sweep_root,
                case_root_tree=case_tree,
                workers=2,
                skip_existing=True,
                write_images=False,
            )
            skipped_summary = run_multiview_sweep(
                sweep_root=sweep_root,
                case_root_tree=case_tree,
                workers=2,
                skip_existing=True,
                write_images=False,
            )

            self.assertEqual(summary["total_jobs"], 8)
            self.assertEqual(summary["processed_cases"], 8)
            self.assertEqual(summary["skipped_cases"], 0)
            self.assertEqual(summary["failed_cases"], 0)
            self.assertEqual(skipped_summary["processed_cases"], 0)
            self.assertEqual(skipped_summary["skipped_cases"], 8)
            self.assertTrue((sweep_root / "multiview_results" / "multiview_sweep_summary.json").is_file())
            self.assertTrue(
                (
                    sweep_root
                    / "multiview_results"
                    / "frame_a"
                    / "temporal_a"
                    / "case_1"
                    / "case_multiview_fusion.json"
                ).is_file()
            )


def _write_case(case_root: Path, *, case_id: str) -> None:
    case_root.mkdir(parents=True)
    view_root = case_root / "view_01"
    view_root.mkdir()
    (view_root / "slice_0001.png").write_bytes(b"placeholder")
    (case_root / "views.json").write_text(
        json.dumps(
            {
                "case_id": case_id,
                "views": [
                    {
                        "view_id": "view_01",
                        "sequence_id": "view_01",
                        "rao_lao": 0.0,
                        "cra_cau": 0.0,
                        "temporal_fusion_json": "stale/path.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_temporal_result(path: Path) -> None:
    path.parent.mkdir(parents=True)
    lesion = {
        "lesion_id": 1,
        "track_id": 10,
        "severity": "moderate",
        "supporting_frame_count": 2,
        "total_frame_count": 3,
        "persistence_ratio": 0.5,
        "frame_indices": [1, 2],
        "degrees": {
            "median": 0.7,
            "max": 0.8,
        },
        "stability": {
            "degree_std": 0.02,
            "max_frame_gap": 1,
        },
        "positions": {
            "median_centerline_position": 0.43,
            "dominant_centerline_component_id": 2,
        },
    }
    path.write_text(
        json.dumps(
            {
                "view_id": "view_01",
                "fusion": {
                    "persistent_lesion_count": 1,
                },
                "final_lesion": lesion,
                "persistent_lesions": [lesion],
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()

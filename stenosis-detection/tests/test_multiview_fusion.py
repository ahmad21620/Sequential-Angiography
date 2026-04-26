from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_multiview_fusion as multiview_cli

from stenosis_detection.multiview import (
    load_multiview_case,
    load_multiview_case_input,
    run_multiview_fusion,
    save_multiview_case_result,
    save_multiview_visualization_outputs,
)


class MultiViewFusionTests(unittest.TestCase):
    def test_loader_reads_case_input_and_resolves_relative_paths(self) -> None:
        fixture_dir = self._fixture_dir("single_view_case")

        case_input = load_multiview_case_input(fixture_dir / "multiview_input.json")

        self.assertEqual(case_input.case_id, "case_single_view")
        self.assertEqual(len(case_input.views), 1)
        view_input = case_input.views[0]
        self.assertEqual(view_input.view_id, "view_01")
        self.assertEqual(view_input.sequence_id, "seq_01")
        self.assertAlmostEqual(view_input.rao_lao, 30.0)
        self.assertAlmostEqual(view_input.cra_cau, -10.0)
        self.assertEqual(
            view_input.temporal_fusion_json_path,
            (fixture_dir / "outputs" / "case_single_view" / "view_01_temporal_fusion.json").resolve(),
        )

    def test_case_root_resolution_uses_views_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            case_root = Path(temp_dir)
            views_json_path = case_root / "views.json"
            views_json_path.write_text('{"case_id": "case", "views": []}', encoding="utf-8")

            resolved_path = multiview_cli._resolve_case_input_path(
                argparse.Namespace(case_root=str(case_root), input_json=None)
            )

            self.assertEqual(resolved_path, views_json_path)

    def test_tree_mode_writes_mirrored_case_outputs(self) -> None:
        fixture_dir = self._fixture_dir("single_view_case")
        payload = json.loads((fixture_dir / "multiview_input.json").read_text(encoding="utf-8"))
        for view in payload["views"]:
            view["temporal_fusion_json"] = str((fixture_dir / view["temporal_fusion_json"]).resolve())

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            case_root = temp_root / "cases" / "case_a"
            output_root = temp_root / "outputs"
            case_root.mkdir(parents=True)
            (case_root / "views.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

            exit_code = multiview_cli._run_tree_mode(
                temp_root / "cases",
                output_root,
                multiview_cli.MultiViewFusionConfig(),
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_root / "case_a" / "case_multiview_fusion.json").is_file())
            self.assertTrue((output_root / "case_a" / "case_multiview_fusion_summary.png").is_file())
            self.assertTrue((output_root / "case_a" / "case_multiview_fusion_support_matrix.png").is_file())

    def test_loader_accepts_sequence_based_temporal_view_id_with_friendly_view_id(self) -> None:
        fixture_dir = self._fixture_dir("single_view_case")
        temporal_payload_path = fixture_dir / "outputs" / "case_single_view" / "view_01_temporal_fusion.json"
        temporal_payload = json.loads(temporal_payload_path.read_text(encoding="utf-8"))
        temporal_payload["view_id"] = "Input/images_root/case_single_view/seq_01"

        case_payload = {
            "case_id": "case_single_view",
            "views": [
                {
                    "view_id": "view_01",
                    "sequence_id": "seq_01",
                    "rao_lao": 30.0,
                    "cra_cau": -10.0,
                    "temporal_fusion_json": "view_01_temporal_fusion.json",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            case_input_path = temp_root / "views.json"
            temporal_output_path = temp_root / "view_01_temporal_fusion.json"
            case_input_path.write_text(json.dumps(case_payload, indent=2), encoding="utf-8")
            temporal_output_path.write_text(json.dumps(temporal_payload, indent=2), encoding="utf-8")

            loaded_case = load_multiview_case(case_input_path)

            self.assertEqual(loaded_case.case_id, "case_single_view")
            self.assertEqual(loaded_case.views[0].view_input.view_id, "view_01")
            self.assertEqual(loaded_case.views[0].view_input.sequence_id, "seq_01")

    def test_single_view_case_returns_valid_case_result_with_lower_confidence(self) -> None:
        case_result = run_multiview_fusion(self._load_case("single_view_case"))

        self.assertEqual(case_result.case_id, "case_single_view")
        self.assertEqual(case_result.view_count, 1)
        self.assertEqual(case_result.fusion_metadata.total_candidate_count, 1)
        self.assertEqual(case_result.confidence.label, "medium")
        self.assertAlmostEqual(case_result.confidence.score, 0.632)
        self.assertEqual(case_result.supporting_views, [])
        self.assertIsNotNone(case_result.final_case_lesion)

        final_case_lesion = case_result.final_case_lesion
        if final_case_lesion is None:
            self.fail("Expected a final case lesion.")
        self.assertEqual(final_case_lesion.primary_view_id, "view_01")
        self.assertEqual(final_case_lesion.severity, "moderate")
        self.assertAlmostEqual(final_case_lesion.median_degree, 0.70)
        self.assertEqual(final_case_lesion.supporting_view_ids, [])

    def test_distinct_angle_views_with_strong_support_increase_confidence(self) -> None:
        single_view_result = run_multiview_fusion(self._load_case("single_view_case"))
        distinct_support_result = run_multiview_fusion(self._load_case("distinct_support_case"))

        self.assertGreater(distinct_support_result.confidence.score, single_view_result.confidence.score)
        self.assertEqual(distinct_support_result.confidence.label, "high")
        self.assertIsNotNone(distinct_support_result.final_case_lesion)

        final_case_lesion = distinct_support_result.final_case_lesion
        if final_case_lesion is None:
            self.fail("Expected a final case lesion.")
        self.assertEqual(final_case_lesion.primary_view_id, "view_01")
        self.assertEqual(final_case_lesion.supporting_view_ids, ["view_02"])
        self.assertEqual(final_case_lesion.distinct_supporting_view_ids, ["view_02"])
        self.assertGreater(final_case_lesion.distinct_view_support_score, 0.10)

    def test_duplicate_angle_views_do_not_count_as_strong_independent_support(self) -> None:
        duplicate_support_result = run_multiview_fusion(self._load_case("duplicate_support_case"))
        distinct_support_result = run_multiview_fusion(self._load_case("distinct_support_case"))

        self.assertLess(duplicate_support_result.confidence.score, distinct_support_result.confidence.score)
        self.assertEqual(duplicate_support_result.confidence.label, "medium")
        self.assertIsNotNone(duplicate_support_result.final_case_lesion)

        final_case_lesion = duplicate_support_result.final_case_lesion
        if final_case_lesion is None:
            self.fail("Expected a final case lesion.")
        self.assertEqual(final_case_lesion.supporting_view_ids, ["view_02"])
        self.assertEqual(final_case_lesion.distinct_supporting_view_ids, [])
        self.assertLess(final_case_lesion.distinct_view_support_score, 0.04)

    def test_case_with_no_persistent_lesions_returns_clean_no_final_lesion_result(self) -> None:
        case_result = run_multiview_fusion(self._load_case("no_persistent_case"))

        self.assertEqual(case_result.case_id, "case_no_persistent")
        self.assertEqual(case_result.view_count, 2)
        self.assertEqual(case_result.fusion_metadata.total_candidate_count, 0)
        self.assertIsNone(case_result.final_case_lesion)
        self.assertEqual(case_result.confidence.label, "low")
        self.assertEqual(case_result.confidence.score, 0.0)
        self.assertEqual(case_result.supporting_views, [])
        self.assertEqual([summary.candidate_count for summary in case_result.per_view_summary], [0, 0])

    def test_output_json_serialization_uses_expected_top_level_keys(self) -> None:
        output_path = self._fixture_dir("single_view_case") / "_generated_case_multiview_fusion.json"
        self.addCleanup(output_path.unlink, missing_ok=True)

        case_result = run_multiview_fusion(self._load_case("single_view_case"))
        save_multiview_case_result(case_result, output_path)
        payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(
            list(payload),
            [
                "case_id",
                "view_count",
                "views",
                "per_view_summary",
                "final_case_lesion",
                "confidence",
                "supporting_views",
                "fusion_metadata",
            ],
        )
        self.assertEqual(payload["case_id"], "case_single_view")
        self.assertEqual(payload["view_count"], 1)
        self.assertEqual(len(payload["views"]), 1)
        self.assertEqual(len(payload["per_view_summary"]), 1)
        self.assertIsInstance(payload["final_case_lesion"], dict)
        self.assertEqual(payload["confidence"]["label"], "medium")
        self.assertEqual(payload["supporting_views"], [])
        self.assertIn("selection_rule", payload["fusion_metadata"])

    def test_visualization_outputs_are_written_without_temporal_thumbnails(self) -> None:
        case_result = run_multiview_fusion(self._load_case("distinct_support_case"))

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "case_multiview_fusion.json"
            save_multiview_case_result(case_result, output_path)

            output_paths = save_multiview_visualization_outputs(case_result, output_path)

            self.assertTrue(output_paths["summary_png"].is_file())
            self.assertGreater(output_paths["summary_png"].stat().st_size, 0)
            self.assertTrue(output_paths["support_matrix_png"].is_file())
            self.assertGreater(output_paths["support_matrix_png"].stat().st_size, 0)

    def _load_case(self, case_name: str):
        fixture_dir = self._fixture_dir(case_name)
        return load_multiview_case(fixture_dir / "multiview_input.json")

    def _fixture_dir(self, case_name: str) -> Path:
        return Path(__file__).resolve().parent / "data" / "multiview" / case_name


if __name__ == "__main__":
    unittest.main()

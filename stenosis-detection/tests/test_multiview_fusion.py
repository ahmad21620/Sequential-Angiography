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
    LesionCandidateScore,
    LoadedMultiViewCase,
    LoadedMultiViewView,
    MultiViewFusionConfig,
    MultiViewPerViewSummary,
    MultiViewViewInput,
    ViewLevelLesionCandidate,
    compute_distinct_view_support_score,
    compute_projection_group_diversity_weight,
    is_projection_group_distinct_support,
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
        self.assertIsNone(view_input.projection_group)
        self.assertEqual(
            view_input.temporal_fusion_json_path,
            (fixture_dir / "outputs" / "case_single_view" / "view_01_temporal_fusion.json").resolve(),
        )

    def test_loader_reads_optional_projection_group_metadata(self) -> None:
        case_payload = {
            "case_id": "case_projection",
            "views": [
                {
                    "view_id": "view_01",
                    "sequence_id": "seq_01",
                    "rao_lao": 0.0,
                    "cra_cau": 0.0,
                    "angle_status": "missing",
                    "projection_group": "LCA",
                    "projection_groups": ["LCA"],
                    "coronary_side": "left",
                    "projection_status": "known",
                    "temporal_fusion_json": "view_temporal_fusion.json",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            case_input_path = Path(temp_dir) / "views.json"
            case_input_path.write_text(json.dumps(case_payload), encoding="utf-8")

            case_input = load_multiview_case_input(case_input_path)

            view_input = case_input.views[0]
            self.assertEqual(view_input.angle_status, "missing")
            self.assertEqual(view_input.projection_group, "LCA")
            self.assertEqual(view_input.projection_groups, ["LCA"])
            self.assertEqual(view_input.coronary_side, "left")
            self.assertEqual(view_input.projection_status, "known")

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
            case_tree = temp_root / "cases"
            output_root = temp_root / "outputs"
            for case_id in ("case_a", "case_b"):
                case_root = case_tree / case_id
                case_root.mkdir(parents=True)
                (case_root / "view_01").mkdir()
                (case_root / "view_01" / "slice_0001.png").write_bytes(b"placeholder")
                (case_root / "views.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

            exit_code = multiview_cli._run_tree_mode(
                case_tree,
                output_root,
                multiview_cli.MultiViewFusionConfig(),
                workers=2,
            )
            skipped_exit_code = multiview_cli._run_tree_mode(
                case_tree,
                output_root,
                multiview_cli.MultiViewFusionConfig(),
                workers=2,
                skip_existing=True,
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(skipped_exit_code, 0)
            self.assertTrue((output_root / "case_a" / "case_multiview_fusion.json").is_file())
            self.assertTrue((output_root / "case_a" / "case_multiview_fusion_summary.png").is_file())
            self.assertTrue((output_root / "case_a" / "case_multiview_fusion_support_matrix.png").is_file())
            self.assertTrue((output_root / "case_b" / "case_multiview_fusion.json").is_file())

    def test_tree_mode_can_use_separate_temporal_results_root(self) -> None:
        fixture_dir = self._fixture_dir("single_view_case")
        payload = json.loads((fixture_dir / "multiview_input.json").read_text(encoding="utf-8"))
        payload["views"][0]["temporal_fusion_json"] = "stale/path/that/should/be/ignored.json"
        temporal_payload = (fixture_dir / "outputs" / "case_single_view" / "view_01_temporal_fusion.json").read_text(
            encoding="utf-8"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            case_tree = temp_root / "cases"
            case_root = case_tree / "case_a"
            temporal_root = temp_root / "temporal"
            temporal_output_path = temporal_root / "case_a" / "view_01" / "view_temporal_fusion.json"
            output_root = temp_root / "outputs"

            case_root.mkdir(parents=True)
            (case_root / "view_01").mkdir()
            (case_root / "view_01" / "slice_0001.png").write_bytes(b"placeholder")
            (case_root / "views.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temporal_output_path.parent.mkdir(parents=True)
            temporal_output_path.write_text(temporal_payload, encoding="utf-8")

            loaded_case = load_multiview_case(
                case_root / "views.json",
                temporal_results_root=temporal_root,
                case_root_tree=case_tree,
            )
            exit_code = multiview_cli._run_tree_mode(
                case_tree,
                output_root,
                multiview_cli.MultiViewFusionConfig(),
                temporal_results_root=temporal_root,
            )

            self.assertEqual(loaded_case.views[0].view_input.temporal_fusion_json_path, temporal_output_path.resolve())
            self.assertEqual(exit_code, 0)
            self.assertTrue((output_root / "case_a" / "case_multiview_fusion.json").is_file())

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
            (temp_root / "view_01").mkdir()
            (temp_root / "view_01" / "slice_0001.png").write_bytes(b"placeholder")
            case_input_path.write_text(json.dumps(case_payload, indent=2), encoding="utf-8")
            temporal_output_path.write_text(json.dumps(temporal_payload, indent=2), encoding="utf-8")

            loaded_case = load_multiview_case(case_input_path)

            self.assertEqual(loaded_case.case_id, "case_single_view")
            self.assertEqual(loaded_case.views[0].view_input.view_id, "view_01")
            self.assertEqual(loaded_case.views[0].view_input.sequence_id, "seq_01")

    def test_loader_skips_views_missing_from_case_root_when_view_dirs_exist(self) -> None:
        fixture_dir = self._fixture_dir("distinct_support_case")
        payload = json.loads((fixture_dir / "multiview_input.json").read_text(encoding="utf-8"))
        for view in payload["views"]:
            view["temporal_fusion_json"] = str((fixture_dir / view["temporal_fusion_json"]).resolve())

        with tempfile.TemporaryDirectory() as temp_dir:
            case_root = Path(temp_dir)
            existing_view_dir = case_root / "view_01"
            existing_view_dir.mkdir()
            (existing_view_dir / "slice_0001.png").write_bytes(b"placeholder")
            (case_root / "views.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

            loaded_case = load_multiview_case(case_root / "views.json")

            self.assertEqual(loaded_case.case_id, "case_distinct_support")
            self.assertEqual([view.view_input.view_id for view in loaded_case.views], ["view_01"])

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

    def test_projection_group_fusion_same_group_uses_duplicate_weight(self) -> None:
        primary_view = self._projection_view("view_01", "LCA", "left")
        support_view = self._projection_view("view_02", "LCA", "left")

        score, supporting_view_ids, distinct_supporting_view_ids = compute_distinct_view_support_score(
            primary_view,
            [self._support_summary(support_view, candidate_score=0.8)],
            support_score_scale=1.0,
            max_support_score=1.0,
            view_diversity_mode="projection_group",
        )

        self.assertAlmostEqual(compute_projection_group_diversity_weight(primary_view, support_view), 0.25)
        self.assertAlmostEqual(score, 0.20)
        self.assertEqual(supporting_view_ids, ["view_02"])
        self.assertEqual(distinct_supporting_view_ids, [])

    def test_projection_group_fusion_lca_vs_lca2_uses_left_compatible_weight(self) -> None:
        primary_view = self._projection_view("view_01", "LCA", "left")
        support_view = self._projection_view("view_02", "LCA2", "left")

        self.assertAlmostEqual(compute_projection_group_diversity_weight(primary_view, support_view), 0.65)
        self.assertTrue(is_projection_group_distinct_support(primary_view, support_view))

    def test_projection_group_fusion_lca_vs_rca_has_no_support(self) -> None:
        primary_view = self._projection_view("view_01", "LCA", "left")
        support_view = self._projection_view("view_02", "RCA", "right")

        score, supporting_view_ids, distinct_supporting_view_ids = compute_distinct_view_support_score(
            primary_view,
            [self._support_summary(support_view, candidate_score=0.8)],
            support_score_scale=1.0,
            max_support_score=1.0,
            view_diversity_mode="projection_group",
        )

        self.assertEqual(compute_projection_group_diversity_weight(primary_view, support_view), 0.0)
        self.assertEqual(score, 0.0)
        self.assertEqual(supporting_view_ids, [])
        self.assertEqual(distinct_supporting_view_ids, [])

    def test_projection_group_fusion_ambiguous_left_group_is_compatible_not_distinct(self) -> None:
        primary_view = self._projection_view("view_01", "LCA", "left")
        ambiguous_view = self._projection_view(
            "view_02",
            "unknown",
            "left",
            projection_groups=["LCA", "LCA2"],
            projection_status="ambiguous",
        )

        self.assertAlmostEqual(compute_projection_group_diversity_weight(primary_view, ambiguous_view), 0.65)
        self.assertFalse(is_projection_group_distinct_support(primary_view, ambiguous_view))

    def test_split_by_coronary_side_writes_side_results_and_excludes_unknown_views(self) -> None:
        loaded_case = LoadedMultiViewCase(
            case_id="case_projection",
            views=[
                self._loaded_projection_view("left_01", "LCA", "left"),
                self._loaded_projection_view("left_02", "LCA2", "left"),
                self._loaded_projection_view("right_01", "RCA", "right"),
                self._loaded_projection_view(
                    "unknown_01",
                    "unknown",
                    "unknown",
                    projection_groups=[],
                    projection_status="missing",
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "case_multiview_fusion.json"
            payload = multiview_cli._run_split_case(
                loaded_case,
                output_path,
                MultiViewFusionConfig(view_diversity_mode="projection_group"),
            )

            self.assertTrue(output_path.is_file())
            self.assertEqual(payload["case_id"], "case_projection")
            self.assertEqual(sorted(payload["side_results"]), ["left", "right"])
            self.assertEqual(len(payload["unknown_views"]), 1)
            left_result = payload["side_results"]["left"]
            right_result = payload["side_results"]["right"]
            self.assertEqual([view["view_id"] for view in left_result["views"]], ["left_01", "left_02"])
            self.assertEqual([view["view_id"] for view in right_result["views"]], ["right_01"])
            self.assertNotIn("right_01", left_result["supporting_views"])

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

    def _projection_view(
        self,
        view_id: str,
        projection_group: str,
        coronary_side: str,
        *,
        projection_groups: list[str] | None = None,
        projection_status: str = "known",
    ) -> MultiViewViewInput:
        groups = projection_groups
        if groups is None:
            groups = [] if projection_group == "unknown" else [projection_group]
        return MultiViewViewInput(
            view_id=view_id,
            sequence_id=view_id,
            rao_lao=0.0,
            cra_cau=0.0,
            temporal_fusion_json_path=Path(f"{view_id}.json"),
            angle_status="missing",
            projection_group=projection_group,
            projection_groups=groups,
            coronary_side=coronary_side,
            projection_status=projection_status,
        )

    def _support_summary(
        self,
        view_input: MultiViewViewInput,
        *,
        candidate_score: float,
    ) -> MultiViewPerViewSummary:
        candidate = self._candidate()
        return MultiViewPerViewSummary(
            view_input=view_input,
            candidate_source="final_lesion",
            candidate_count=1,
            best_candidate=candidate,
            best_candidate_score=LesionCandidateScore(
                base_score=candidate_score,
                stability_adjustment=0.0,
                candidate_score=candidate_score,
            ),
        )

    def _loaded_projection_view(
        self,
        view_id: str,
        projection_group: str,
        coronary_side: str,
        *,
        projection_groups: list[str] | None = None,
        projection_status: str = "known",
    ) -> LoadedMultiViewView:
        return LoadedMultiViewView(
            view_input=self._projection_view(
                view_id,
                projection_group,
                coronary_side,
                projection_groups=projection_groups,
                projection_status=projection_status,
            ),
            final_lesion=self._candidate(),
            persistent_lesions=[],
        )

    def _candidate(self) -> ViewLevelLesionCandidate:
        return ViewLevelLesionCandidate(
            lesion_id=1,
            track_id=1,
            severity="moderate",
            supporting_frame_count=2,
            total_frame_count=3,
            persistence_ratio=0.67,
            frame_indices=[1, 2],
            median_degree=0.70,
            max_degree=0.80,
            degree_std=0.05,
            max_frame_gap=1,
        )


if __name__ == "__main__":
    unittest.main()

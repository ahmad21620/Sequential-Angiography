from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stenosis_detection import (
    DISTINCT_VIEW_ANGLE_DISTANCE_DEGREES,
    DUPLICATE_VIEW_ANGLE_DISTANCE_DEGREES,
    HIGH_CONFIDENCE_THRESHOLD,
    LoadedMultiViewCase,
    LoadedMultiViewView,
    MEDIUM_CONFIDENCE_THRESHOLD,
    SUPPORT_SCORE_SCALE,
    MultiViewCaseResult,
    MultiViewFusionConfig,
    MultiViewLoadError,
    load_multiview_case,
    run_multiview_fusion,
    save_multiview_case_result,
    save_multiview_visualization_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run case-level multi-view fusion on temporal fusion outputs.",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--case-root",
        help="Directory containing 'views.json' for one case.",
    )
    input_group.add_argument(
        "--input-json",
        help="Direct path to one case-level 'views.json' file.",
    )
    input_group.add_argument(
        "--case-root-tree",
        help="Root directory containing one or more case folders with 'views.json'.",
    )

    parser.add_argument(
        "--output",
        help="Path to the output JSON file for the fused case-level result.",
    )
    parser.add_argument(
        "--output-root",
        help="Root directory where mirrored case-level fusion outputs will be written in tree mode.",
    )
    parser.add_argument(
        "--temporal-results-root",
        help=(
            "Optional root containing temporal fusion outputs. When provided, "
            "multi-view fusion resolves each view's temporal JSON under this root "
            "instead of using temporal_fusion_json paths from views.json."
        ),
    )
    parser.add_argument(
        "--duplicate-angle-distance",
        type=float,
        default=DUPLICATE_VIEW_ANGLE_DISTANCE_DEGREES,
        help="Angle distance threshold, in degrees, below which two views are treated as duplicate evidence.",
    )
    parser.add_argument(
        "--distinct-angle-distance",
        type=float,
        default=DISTINCT_VIEW_ANGLE_DISTANCE_DEGREES,
        help="Angle distance threshold, in degrees, at which support is treated as fully distinct.",
    )
    parser.add_argument(
        "--support-score-scale",
        type=float,
        default=SUPPORT_SCORE_SCALE,
        help="Scale factor applied to cross-view support from other views.",
    )
    parser.add_argument(
        "--medium-confidence-threshold",
        type=float,
        default=MEDIUM_CONFIDENCE_THRESHOLD,
        help="Minimum confidence score labeled as medium.",
    )
    parser.add_argument(
        "--high-confidence-threshold",
        type=float,
        default=HIGH_CONFIDENCE_THRESHOLD,
        help="Minimum confidence score labeled as high.",
    )
    parser.add_argument(
        "--view-diversity-mode",
        choices=["angle", "projection_group", "auto"],
        default="angle",
        help="Cross-view diversity mode. Use projection_group or auto for CADICA projection groups.",
    )
    parser.add_argument(
        "--split-by-coronary-side",
        action="store_true",
        help="Run separate left/right fusions using coronary_side metadata and report unknown-side views.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.duplicate_angle_distance <= 0.0:
        parser.error("--duplicate-angle-distance must be positive.")
    if args.distinct_angle_distance <= args.duplicate_angle_distance:
        parser.error("--distinct-angle-distance must be greater than --duplicate-angle-distance.")
    if args.support_score_scale < 0.0:
        parser.error("--support-score-scale must be >= 0.0.")
    if not 0.0 <= args.medium_confidence_threshold <= 1.0:
        parser.error("--medium-confidence-threshold must be in the range [0.0, 1.0].")
    if not 0.0 <= args.high_confidence_threshold <= 1.0:
        parser.error("--high-confidence-threshold must be in the range [0.0, 1.0].")
    if args.medium_confidence_threshold > args.high_confidence_threshold:
        parser.error("--medium-confidence-threshold must be <= --high-confidence-threshold.")
    if args.case_root_tree is not None:
        if args.output_root is None:
            parser.error("--case-root-tree requires --output-root.")
        if args.output is not None:
            parser.error("--output is only supported for single-case mode. Use --output-root with --case-root-tree.")
    elif args.output is None:
        parser.error("--output is required for single-case mode.")
    if args.temporal_results_root is not None:
        temporal_results_root = Path(args.temporal_results_root)
        if not temporal_results_root.exists():
            parser.error(f"--temporal-results-root does not exist: {temporal_results_root}")
        if not temporal_results_root.is_dir():
            parser.error(f"--temporal-results-root is not a directory: {temporal_results_root}")

    try:
        config = _build_fusion_config(args)
        temporal_results_root = None if args.temporal_results_root is None else Path(args.temporal_results_root)
        if args.case_root_tree is not None:
            return _run_tree_mode(
                Path(args.case_root_tree),
                Path(args.output_root),
                config,
                temporal_results_root=temporal_results_root,
                split_by_coronary_side=args.split_by_coronary_side,
            )

        _run_one_case(
            _resolve_case_input_path(args),
            Path(args.output),
            config,
            temporal_results_root=temporal_results_root,
            split_by_coronary_side=args.split_by_coronary_side,
        )
        return 0
    except (FileNotFoundError, NotADirectoryError, MultiViewLoadError, ValueError) as exc:
        print(f"Multi-view fusion failed: {exc}", file=sys.stderr)
        return 1


def _resolve_case_input_path(args: argparse.Namespace) -> Path:
    if args.case_root is not None:
        case_root = Path(args.case_root)
        if not case_root.exists():
            raise FileNotFoundError(f"Case root does not exist: {case_root}")
        if not case_root.is_dir():
            raise NotADirectoryError(f"Case root is not a directory: {case_root}")
        return case_root / "views.json"

    return Path(args.input_json)


def _build_fusion_config(args: argparse.Namespace) -> MultiViewFusionConfig:
    return MultiViewFusionConfig(
        duplicate_view_angle_distance_degrees=args.duplicate_angle_distance,
        distinct_view_angle_distance_degrees=args.distinct_angle_distance,
        support_score_scale=args.support_score_scale,
        medium_confidence_threshold=args.medium_confidence_threshold,
        high_confidence_threshold=args.high_confidence_threshold,
        view_diversity_mode=args.view_diversity_mode,
    )


def _run_tree_mode(
    case_root_tree: Path,
    output_root: Path,
    config: MultiViewFusionConfig,
    *,
    temporal_results_root: Path | None = None,
    split_by_coronary_side: bool = False,
) -> int:
    case_inputs = _discover_case_input_paths(case_root_tree)
    processed = 0
    failed = 0

    for case_input_path in case_inputs:
        output_path = _build_tree_output_path(case_input_path, case_root_tree, output_root)
        try:
            _run_one_case(
                case_input_path,
                output_path,
                config,
                temporal_results_root=temporal_results_root,
                case_root_tree=case_root_tree,
                split_by_coronary_side=split_by_coronary_side,
            )
            processed += 1
        except (FileNotFoundError, NotADirectoryError, MultiViewLoadError, ValueError) as exc:
            failed += 1
            print(f"Failed case '{case_input_path}': {exc}", file=sys.stderr)

    print(f"Processed cases: {processed}")
    print(f"Failed cases: {failed}")
    return 0 if failed == 0 else 1


def _discover_case_input_paths(case_root_tree: Path) -> list[Path]:
    if not case_root_tree.exists():
        raise FileNotFoundError(f"Case root tree does not exist: {case_root_tree}")
    if not case_root_tree.is_dir():
        raise NotADirectoryError(f"Case root tree is not a directory: {case_root_tree}")

    case_inputs = sorted(path for path in case_root_tree.rglob("views.json") if path.is_file())
    if not case_inputs:
        raise FileNotFoundError(f"No views.json files were found under: {case_root_tree}")
    return case_inputs


def _build_tree_output_path(case_input_path: Path, case_root_tree: Path, output_root: Path) -> Path:
    relative_case_dir = case_input_path.parent.relative_to(case_root_tree)
    return output_root / relative_case_dir / "case_multiview_fusion.json"


def _run_one_case(
    case_input_path: Path,
    output_path: Path,
    config: MultiViewFusionConfig,
    *,
    temporal_results_root: Path | None = None,
    case_root_tree: Path | None = None,
    split_by_coronary_side: bool = False,
) -> MultiViewCaseResult | dict[str, object]:
    multiview_case = load_multiview_case(
        case_input_path,
        temporal_results_root=temporal_results_root,
        case_root_tree=case_root_tree,
    )
    print(f"Loaded case '{multiview_case.case_id}' with {multiview_case.view_count} views.")

    if split_by_coronary_side:
        return _run_split_case(multiview_case, output_path, config)

    case_result = run_multiview_fusion(multiview_case, config=config)
    _print_fusion_summary(case_result)

    saved_path = save_multiview_case_result(case_result, output_path)
    print(f"Saved case-level result: {saved_path}")
    _save_visualizations(case_result, saved_path)
    return case_result


def _run_split_case(
    multiview_case: LoadedMultiViewCase,
    output_path: Path,
    config: MultiViewFusionConfig,
) -> dict[str, object]:
    side_groups = _split_views_by_coronary_side(multiview_case)
    side_results: dict[str, object] = {}
    skipped_sides: list[str] = []

    for side in ("left", "right"):
        side_views = side_groups[side]
        if not side_views:
            skipped_sides.append(side)
            continue
        side_case = LoadedMultiViewCase(
            case_id=f"{multiview_case.case_id}:{side}",
            views=side_views,
        )
        side_result = run_multiview_fusion(side_case, config=config)
        side_results[side] = side_result.to_dict()
        print(f"{side.title()} side: {len(side_views)} views, confidence={side_result.confidence.label}.")

    unknown_views = [view.view_input.to_dict() for view in side_groups["unknown"]]
    payload: dict[str, object] = {
        "case_id": multiview_case.case_id,
        "split_by_coronary_side": True,
        "view_diversity_mode": config.view_diversity_mode,
        "side_results": side_results,
        "unknown_views": unknown_views,
        "skipped_sides": skipped_sides,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Unknown-side views reported but not fused: {len(unknown_views)}")
    print(f"Saved split multi-view result: {output_path}")
    return payload


def _split_views_by_coronary_side(multiview_case: LoadedMultiViewCase) -> dict[str, list[LoadedMultiViewView]]:
    groups: dict[str, list[LoadedMultiViewView]] = {"left": [], "right": [], "unknown": []}
    for view in multiview_case.views:
        side = (view.view_input.coronary_side or "unknown").strip().lower()
        if side not in {"left", "right"}:
            side = "unknown"
        groups[side].append(view)
    return groups


def _save_visualizations(case_result: MultiViewCaseResult, output_path: Path) -> None:
    try:
        visualization_paths = save_multiview_visualization_outputs(case_result, output_path)
    except Exception as exc:
        print(f"Warning: failed to save multi-view visualizations: {exc}", file=sys.stderr)
        return

    print(f"Saved multi-view summary visualization: {visualization_paths['summary_png']}")
    print(f"Saved multi-view support matrix: {visualization_paths['support_matrix_png']}")


def _print_fusion_summary(case_result: MultiViewCaseResult) -> None:
    final_case_lesion = case_result.final_case_lesion
    supporting_views_text = ", ".join(case_result.supporting_views) if case_result.supporting_views else "none"

    print(f"Lesion candidates: {case_result.fusion_metadata.total_candidate_count}")
    if final_case_lesion is None:
        print("Final severity: none")
        print("Final degree: n/a")
    else:
        print(f"Final severity: {final_case_lesion.severity}")
        print(f"Final degree: {final_case_lesion.median_degree:.3f}")

    print(
        f"Confidence: {case_result.confidence.label} "
        f"({case_result.confidence.score:.3f})"
    )
    print(f"Supporting views: {supporting_views_text}")


if __name__ == "__main__":
    raise SystemExit(main())
